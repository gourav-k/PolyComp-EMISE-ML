"""
Sensitivity analysis service: Sobol indices and SHAP values over a BoTorch GP
surrogate that matches the Modeling page's recipe exactly.

The GP fit is delegated to services.modeling.fit_botorch_gp_full, which is the
same code path used for the Modeling page's final-fit (after CV). Sensitivity
analyses therefore see an identical surrogate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.modeling import (
    BOTORCH_AVAILABLE,
    _build_set_gp_covar_module,
    _is_log_feature,
    _transform_minmax,
    fit_botorch_gp_full,
)

try:
    import torch
    import gpytorch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

SALIB_AVAILABLE = True
_salib_import_error: str | None = None
try:
    from SALib.sample import sobol as _sobol_sample
    from SALib.analyze import sobol as _sobol_analyze
except Exception as _err:
    SALIB_AVAILABLE = False
    _salib_import_error = str(_err)

SHAP_AVAILABLE = True
_shap_import_error: str | None = None
try:
    import shap as _shap
except Exception as _err:
    SHAP_AVAILABLE = False
    _shap_import_error = str(_err)


def _kernel_supported(kernel: str, n_features: int) -> bool:
    if kernel == "set":
        return n_features >= 2 and n_features % 2 == 0
    return True


def fit_surrogate(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    kernel: str = "standard",
) -> dict[str, Any]:
    """Fit a BoTorch GP surrogate on (features, target) from df.

    kernel:
      - "standard": min-max normalised inputs, learned noise (matches the
        Modeling page's "GP Regression (botorch)" spec).
      - "set":      no input normalisation, MeanSet kernel (matches the
        Modeling page's "Set GP" spec). Requires an even feature count.
    """
    if not BOTORCH_AVAILABLE:
        return {"error": "botorch is not installed (install the [bo] extra)."}

    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_features:
        return {"error": "No numeric input features available."}
    if not pd.api.types.is_numeric_dtype(df[target]):
        return {"error": f"Target column '{target}' is not numeric."}
    if not _kernel_supported(kernel, len(numeric_features)):
        return {"error": f"Set-GP kernel requires an even feature count (got {len(numeric_features)})."}

    keep = numeric_features + [target]
    working = df[keep].dropna(subset=[target]).copy()
    if len(working) < 6:
        return {"error": f"Too few rows for surrogate fit: {len(working)} (need ≥ 6)."}

    X_df = working[numeric_features].copy()
    y = working[target].to_numpy(float)

    if kernel == "set":
        fit = fit_botorch_gp_full(
            X_df, y, numeric_features, target,
            covar_module_factory=_build_set_gp_covar_module,
            normalize_inputs=False,
            use_fixed_noise=False,
        )
    else:
        fit = fit_botorch_gp_full(
            X_df, y, numeric_features, target,
            covar_module_factory=None,
            normalize_inputs=True,
            use_fixed_noise=False,
        )

    if "error" in fit:
        return fit

    fit["features"] = numeric_features
    fit["X_train_orig"] = X_df.reset_index(drop=True)
    fit["y_train"] = y
    fit["kernel"] = kernel
    return fit


def _apply_log_transforms(X: np.ndarray, features: list[str]) -> np.ndarray:
    X = X.copy().astype(float)
    for j, c in enumerate(features):
        if _is_log_feature(c):
            X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
    return X


def _predict_mean(surrogate: dict[str, Any], X_model_space: np.ndarray) -> np.ndarray:
    """Predict posterior mean from points already in model input space.

    Model input space = log-transformed (for log features), and min-max
    normalised iff surrogate["normalize_inputs"] is True.
    """
    model = surrogate["model"]
    device = surrogate["device"]
    dtype = surrogate["dtype"]
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        X_t = torch.tensor(X_model_space, dtype=dtype, device=device)
        return model.posterior(X_t).mean.squeeze(-1).cpu().numpy()


def compute_sobol(
    surrogate: dict[str, Any],
    n_samples: int = 512,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute first-order (S1) and total-order (ST) Sobol indices via Saltelli.

    Mirrors sobol.ipynb: bounds are taken from the surrogate's training inputs
    in model space; samples are pushed through the GP posterior mean.
    """
    if not SALIB_AVAILABLE:
        return {"error": f"SALib not installed ({_salib_import_error})."}
    if "error" in surrogate:
        return surrogate

    features = surrogate["features"]
    X_log = surrogate["X_log"]
    x_min = surrogate["x_min"]
    denom = surrogate["denom"]
    normalize = surrogate["normalize_inputs"]

    if normalize and x_min is not None:
        X_bounds = _transform_minmax(X_log, x_min, denom)
    else:
        X_bounds = X_log

    bounds: list[list[float]] = []
    for i in range(len(features)):
        lo = float(X_bounds[:, i].min())
        hi = float(X_bounds[:, i].max())
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, lo + 0.5
        bounds.append([lo, hi])

    # SALib internals call pd.unique on names; an ndarray sidesteps a pandas
    # 2.x quirk where plain lists trip an assertion.
    problem = {
        "num_vars": len(features),
        "names": np.array(features),
        "bounds": bounds,
    }

    param_values = _sobol_sample.sample(problem, N=n_samples, calc_second_order=False, scramble=True, seed=seed)
    Y = _predict_mean(surrogate, param_values)
    Si = _sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    return {
        "features": features,
        "S1": np.asarray(Si["S1"], dtype=float),
        "S1_conf": np.asarray(Si["S1_conf"], dtype=float),
        "ST": np.asarray(Si["ST"], dtype=float),
        "ST_conf": np.asarray(Si["ST_conf"], dtype=float),
        "n_samples": int(n_samples),
        "kernel": surrogate["kernel"],
    }


def compute_shap(
    surrogate: dict[str, Any],
    n_background: int = 30,
    n_samples: int = 128,
) -> dict[str, Any]:
    """Compute KernelSHAP values in original feature space.

    The predictor applies log/min-max transforms internally so SHAP
    attributions stay interpretable on the user's original units. Mirrors
    shap_gp.ipynb.
    """
    if not SHAP_AVAILABLE:
        return {"error": f"shap not installed ({_shap_import_error})."}
    if "error" in surrogate:
        return surrogate

    features = surrogate["features"]
    X_orig_df = surrogate["X_train_orig"]
    X_orig = X_orig_df.to_numpy(float)
    normalize = surrogate["normalize_inputs"]
    x_min = surrogate["x_min"]
    denom = surrogate["denom"]

    def predict_orig(X_in: np.ndarray) -> np.ndarray:
        X = _apply_log_transforms(np.asarray(X_in, dtype=float), features)
        if normalize and x_min is not None:
            X = _transform_minmax(X, x_min, denom)
        return _predict_mean(surrogate, X)

    bg_k = min(int(n_background), len(X_orig))
    background = _shap.kmeans(X_orig, bg_k)
    explainer = _shap.KernelExplainer(predict_orig, background)
    shap_values = explainer.shap_values(X_orig, nsamples=int(n_samples), silent=True)

    return {
        "features": features,
        "shap_values": np.asarray(shap_values, dtype=float),
        "feature_values": X_orig_df.copy(),
        "expected": float(explainer.expected_value),
        "n_samples": int(n_samples),
        "n_background": int(bg_k),
        "kernel": surrogate["kernel"],
    }
