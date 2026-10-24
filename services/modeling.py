"""
Model comparison service.

Mirrors GP-BO/model_comparison_kfold_eaton.ipynb exactly for model specs,
preprocessing, and cross-validation:
  - KFold(n_splits=5, shuffle=True, random_state=42) with pooled OOF predictions
  - ColumnTransformer: log1p + StandardScaler on "Electrical conductivity*" inputs,
    StandardScaler on the rest (for linear/Ridge/GP), log1p + passthrough for trees
  - Per-model pipelines and hyperparameters match the notebook
  - Test split: uses `Main Folder == "Eaton-new"` as holdout when that column is
    present; otherwise falls back to a random 80/20 train_test_split
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as app_config

try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

TABPFN_AVAILABLE = True
_tabpfn_import_error: str | None = None
try:
    _tabpfn_key = getattr(app_config, "TABPFN_API_KEY", "") or ""
    if _tabpfn_key:
        os.environ["TABPFN_TOKEN"] = _tabpfn_key
    from tabpfn import TabPFNRegressor
except Exception as _tabpfn_err:
    TABPFN_AVAILABLE = False
    _tabpfn_import_error = str(_tabpfn_err)

BOTORCH_AVAILABLE = True
_botorch_import_error: str | None = None
try:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import torch
    import gpytorch
    from botorch import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize as OutcomeStandardize
    from gpytorch.constraints.constraints import GreaterThan
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors.torch_priors import LogNormalPrior
except Exception as _bo_err:
    BOTORCH_AVAILABLE = False
    _botorch_import_error = str(_bo_err)


SEED = 42
if BOTORCH_AVAILABLE:
    torch.manual_seed(SEED)


def _is_log_feature(name: str) -> bool:
    """Columns that should be log1p-transformed before scaling (matches notebook)."""
    return name.strip().lower().startswith("electrical conductivity")


def _make_preprocessor(features: list[str], scale: bool) -> "ColumnTransformer":
    log_cols = [c for c in features if _is_log_feature(c)]
    other_cols = [c for c in features if c not in log_cols]

    if scale:
        return ColumnTransformer(
            transformers=[
                (
                    "log_scale",
                    Pipeline(
                        [
                            ("log1p", FunctionTransformer(np.log1p, validate=False)),
                            ("std", StandardScaler()),
                        ]
                    ),
                    log_cols,
                ),
                ("scale", StandardScaler(), other_cols),
            ],
            remainder="drop",
        )
    return ColumnTransformer(
        transformers=[
            ("log_only", FunctionTransformer(np.log1p, validate=False), log_cols),
            ("pass", "passthrough", other_cols),
        ],
        remainder="drop",
    )


def _build_model_specs(features: list[str]) -> dict[str, Any]:
    sklearn_gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(len(features)),
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))

    specs: dict[str, Any] = {
        "Linear Regression": Pipeline(
            [
                ("prep", _make_preprocessor(features, scale=True)),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("prep", _make_preprocessor(features, scale=True)),
                ("model", Ridge(alpha=1.0, random_state=SEED)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", _make_preprocessor(features, scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GP Regression (sklearn)": Pipeline(
            [
                ("prep", _make_preprocessor(features, scale=True)),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=sklearn_gp_kernel,
                        alpha=1e-6,
                        normalize_y=True,
                        n_restarts_optimizer=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    if XGBOOST_AVAILABLE:
        specs["XGBoost"] = Pipeline(
            [
                ("prep", _make_preprocessor(features, scale=False)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=600,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return specs


def _fit_minmax_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = np.where((x_max - x_min) < 1e-12, 1.0, (x_max - x_min))
    return x_min, denom


def _transform_minmax(X: np.ndarray, x_min: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.clip((X - x_min) / denom, 0.0, 1.0)


def _prepare_botorch_train_arrays(
    X_df: pd.DataFrame, y_np: np.ndarray, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tmp = X_df.copy()
    tmp[target] = y_np
    agg = tmp.groupby(features, dropna=False)[target].agg(["mean", "var", "count"]).reset_index()
    agg["var"] = agg["var"].fillna(0.0)
    agg["yvar"] = (agg["var"] / agg["count"]).clip(lower=1e-6, upper=1e2)

    X = agg[features].to_numpy(float)
    y = agg["mean"].to_numpy(float)
    yvar = agg["yvar"].to_numpy(float)

    for j, c in enumerate(features):
        if _is_log_feature(c):
            X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
    return X, y, yvar


def _prepare_botorch_eval_arrays(X_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = X_df[features].to_numpy(float)
    for j, c in enumerate(features):
        if _is_log_feature(c):
            X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
    return X


if BOTORCH_AVAILABLE:

    class MeanSetKernel(gpytorch.kernels.Kernel):
        """Set-aware RBF kernel averaging element-pair similarities (matches notebook)."""

        is_stationary = True
        has_lengthscale = True

        def __init__(self, elem_shape: int, **kwargs):
            super().__init__(**kwargs)
            self.elem_shape = int(elem_shape)

        def forward(self, x1, x2, diag=False, **params):
            # Batch-shape-preserving implementation. x1 / x2 may have arbitrary
            # leading batch dims (BoTorch passes (b, q, d) tensors during
            # optimize_acqf), so we never collapse them with reshape(-1, ...).
            elem_shape = self.elem_shape
            n1 = x1.shape[-2]
            n2 = x2.shape[-2]
            set1_card = x1.shape[-1] // elem_shape
            set2_card = x2.shape[-1] // elem_shape
            batch_shape = x1.shape[:-2]

            # (*batch, n, d) -> (*batch, n, set_card, elem_shape)
            x1_sets = x1.reshape(*batch_shape, n1, set1_card, elem_shape)
            x2_sets = x2.reshape(*x2.shape[:-2], n2, set2_card, elem_shape)

            # ARD lengthscale broadcasts over the trailing elem_shape dim.
            x1_sets = x1_sets / self.lengthscale
            x2_sets = x2_sets / self.lengthscale

            # Flatten the set-element axis into the data axis so a single
            # covar_dist call computes all (i, j) element pairs at once.
            x1_flat = x1_sets.reshape(*batch_shape, n1 * set1_card, elem_shape)
            x2_flat = x2_sets.reshape(*x2.shape[:-2], n2 * set2_card, elem_shape)

            if diag:
                dists = self.covar_dist(x1_flat, x2_flat, diag=True, **params)
                rbf = torch.exp(-0.5 * dists ** 2)
                rbf = rbf.reshape(*batch_shape, n1, set1_card)
                return rbf.mean(dim=-1)

            dists = self.covar_dist(x1_flat, x2_flat, diag=False, **params)
            rbf = torch.exp(-0.5 * dists ** 2)
            rbf = rbf.reshape(*batch_shape, n1, set1_card, n2, set2_card)
            return rbf.mean(dim=(-3, -1))


def _build_set_gp_covar_module(input_dim: int):
    """Set-GP scale * MeanSet kernel (matches notebook's _build_set_gp_covar_module)."""
    if input_dim % 2 != 0:
        raise ValueError(f"Set GP expects two equal feature sets; got {input_dim} features.")
    elem_shape = input_dim // 2
    lengthscale_prior = LogNormalPrior(loc=2.0 + np.log(input_dim) / 2.0, scale=1.0)
    lengthscale_constraint = GreaterThan(25e-2, transform=None, initial_value=np.sqrt(input_dim))
    return gpytorch.kernels.ScaleKernel(
        MeanSetKernel(
            elem_shape=elem_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=elem_shape,
        )
    )


def _build_botorch_model(
    train_X_t,
    train_Y_t,
    train_Yvar_t=None,
    use_fixed_noise: bool = True,
    covar_module_factory=None,
):
    kwargs: dict[str, Any] = {
        "train_X": train_X_t,
        "train_Y": train_Y_t,
        "outcome_transform": OutcomeStandardize(m=1),
    }
    if use_fixed_noise and train_Yvar_t is not None:
        kwargs["train_Yvar"] = train_Yvar_t
    if covar_module_factory is not None:
        kwargs["covar_module"] = covar_module_factory(train_X_t.shape[-1])
    return SingleTaskGP(**kwargs)


def _fit_botorch_with_retries(model, max_retries: int = 5, base_jitter: float = 1e-8):
    last_err: Exception | None = None
    for attempt in range(max_retries):
        jitter = base_jitter * (10 ** attempt)
        try:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                jitter_ctx = gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter)
            except TypeError:
                jitter_ctx = gpytorch.settings.cholesky_jitter(float=jitter, double=jitter)
            with jitter_ctx:
                fit_gpytorch_mll(mll)
            return mll, model, {"attempt": attempt, "jitter": jitter, "error": None}
        except Exception as err:
            last_err = err
    return None, None, {"attempt": max_retries, "jitter": None, "error": str(last_err)}


def fit_botorch_gp_full(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    features: list[str],
    target: str,
    covar_module_factory=None,
    normalize_inputs: bool = True,
    use_fixed_noise: bool = False,
):
    """Fit a single BoTorch GP on the full training set.

    Returns the same recipe used inside _eval_botorch_gp's final-fit block, so
    surrogates used by sensitivity analyses match the Modeling page exactly.

    Returns dict with keys: model, x_min, denom, X_log (pre-normalization,
    post-log feature matrix used for downstream Sobol bounds), device, dtype.
    Returns {"error": ...} on failure.
    """
    if not BOTORCH_AVAILABLE:
        return {"error": f"botorch unavailable ({_botorch_import_error})"}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    X_ag, y_ag, yvar_ag = _prepare_botorch_train_arrays(X_tr_df, y_tr, features, target)
    if normalize_inputs:
        x_min, denom = _fit_minmax_scaler(X_ag)
        X_ag_n = _transform_minmax(X_ag, x_min, denom)
    else:
        X_ag_n = X_ag
        x_min, denom = None, None

    X_ag_t = torch.tensor(X_ag_n, dtype=dtype, device=device)
    y_ag_t = torch.tensor(y_ag, dtype=dtype, device=device).unsqueeze(-1)
    yvar_ag_t = torch.tensor(yvar_ag, dtype=dtype, device=device).unsqueeze(-1)

    model = _build_botorch_model(
        X_ag_t, y_ag_t, yvar_ag_t,
        use_fixed_noise=use_fixed_noise,
        covar_module_factory=covar_module_factory,
    )
    _, fit_model, fit_info = _fit_botorch_with_retries(model)
    if fit_model is None:
        model = _build_botorch_model(
            X_ag_t, y_ag_t, train_Yvar_t=None,
            use_fixed_noise=False,
            covar_module_factory=covar_module_factory,
        )
        _, fit_model, fit_info = _fit_botorch_with_retries(model)
    if fit_model is None:
        return {"error": f"botorch fit failed: {fit_info['error']}"}

    fit_model.eval()
    return {
        "model": fit_model,
        "x_min": x_min,
        "denom": denom,
        "X_log": X_ag,
        "device": device,
        "dtype": dtype,
        "normalize_inputs": normalize_inputs,
    }


def _eval_botorch_gp(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_hold_df: pd.DataFrame,
    y_hold: np.ndarray,
    features: list[str],
    target: str,
    n_splits: int,
    random_state: int,
    covar_module_factory=None,
    normalize_inputs: bool = True,
    use_fixed_noise: bool = False,
    loocv: bool = False,
) -> dict[str, Any]:
    if not BOTORCH_AVAILABLE:
        return {"error": f"botorch unavailable ({_botorch_import_error})"}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    kf = LeaveOneOut() if loocv else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y_tr), dtype=float)
    fold_r2: list[float] = []

    for tr_idx, va_idx in kf.split(X_tr_df):
        X_fold_train = X_tr_df.iloc[tr_idx]
        y_fold_train = y_tr[tr_idx]
        X_fold_val = X_tr_df.iloc[va_idx]

        X_ag, y_ag, yvar_ag = _prepare_botorch_train_arrays(X_fold_train, y_fold_train, features, target)
        X_val = _prepare_botorch_eval_arrays(X_fold_val, features)

        if normalize_inputs:
            x_min, denom = _fit_minmax_scaler(X_ag)
            X_ag_n = _transform_minmax(X_ag, x_min, denom)
            X_val_n = _transform_minmax(X_val, x_min, denom)
        else:
            X_ag_n = X_ag
            X_val_n = X_val

        X_ag_t = torch.tensor(X_ag_n, dtype=dtype, device=device)
        y_ag_t = torch.tensor(y_ag, dtype=dtype, device=device).unsqueeze(-1)
        yvar_ag_t = torch.tensor(yvar_ag, dtype=dtype, device=device).unsqueeze(-1)
        X_val_t = torch.tensor(X_val_n, dtype=dtype, device=device)

        model = _build_botorch_model(
            X_ag_t, y_ag_t, yvar_ag_t,
            use_fixed_noise=use_fixed_noise,
            covar_module_factory=covar_module_factory,
        )
        _, fit_model, fit_info = _fit_botorch_with_retries(model)
        if fit_model is None:
            model = _build_botorch_model(
                X_ag_t, y_ag_t, train_Yvar_t=None,
                use_fixed_noise=False,
                covar_module_factory=covar_module_factory,
            )
            _, fit_model, fit_info = _fit_botorch_with_retries(model)
        if fit_model is None:
            return {"error": f"botorch fold fit failed: {fit_info['error']}"}

        fit_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_mean = fit_model.posterior(X_val_t).mean.squeeze(-1).detach().cpu().numpy()
        oof_pred[va_idx] = pred_mean
        if len(va_idx) >= 2:
            fold_r2.append(float(r2_score(y_tr[va_idx], pred_mean)))

    cv_r2 = float(r2_score(y_tr, oof_pred))
    cv_r2_std = float(np.std(fold_r2)) if fold_r2 else 0.0

    X_ag_all, y_ag_all, yvar_ag_all = _prepare_botorch_train_arrays(X_tr_df, y_tr, features, target)
    if normalize_inputs:
        x_min_all, denom_all = _fit_minmax_scaler(X_ag_all)
        X_ag_all_n = _transform_minmax(X_ag_all, x_min_all, denom_all)
    else:
        X_ag_all_n = X_ag_all
        x_min_all, denom_all = None, None

    X_ag_all_t = torch.tensor(X_ag_all_n, dtype=dtype, device=device)
    y_ag_all_t = torch.tensor(y_ag_all, dtype=dtype, device=device).unsqueeze(-1)
    yvar_ag_all_t = torch.tensor(yvar_ag_all, dtype=dtype, device=device).unsqueeze(-1)

    model_final = _build_botorch_model(
        X_ag_all_t, y_ag_all_t, yvar_ag_all_t,
        use_fixed_noise=use_fixed_noise,
        covar_module_factory=covar_module_factory,
    )
    _, model_final_fit, fit_info = _fit_botorch_with_retries(model_final)
    if model_final_fit is None:
        model_final = _build_botorch_model(
            X_ag_all_t, y_ag_all_t, train_Yvar_t=None,
            use_fixed_noise=False,
            covar_module_factory=covar_module_factory,
        )
        _, model_final_fit, fit_info = _fit_botorch_with_retries(model_final)
    if model_final_fit is None:
        return {"error": f"botorch final fit failed: {fit_info['error']}"}

    test_r2 = float("nan")
    y_pred_hold: list[float] = []
    if len(X_hold_df) > 0:
        X_hold = _prepare_botorch_eval_arrays(X_hold_df, features)
        if normalize_inputs:
            X_hold_n = _transform_minmax(X_hold, x_min_all, denom_all)
        else:
            X_hold_n = X_hold
        X_hold_t = torch.tensor(X_hold_n, dtype=dtype, device=device)
        model_final_fit.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            hold_pred = model_final_fit.posterior(X_hold_t).mean.squeeze(-1).detach().cpu().numpy()
        y_pred_hold = hold_pred.tolist()
        test_r2 = float(r2_score(y_hold, hold_pred))

    return {
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "test_r2": test_r2,
        "y_pred": y_pred_hold,
        "oof_pred": oof_pred.tolist(),
    }


def _eval_tabpfn(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_hold_df: pd.DataFrame,
    y_hold: np.ndarray,
    n_splits: int,
    random_state: int,
    loocv: bool = False,
) -> dict[str, Any]:
    if not TABPFN_AVAILABLE:
        return {"error": f"tabpfn unavailable ({_tabpfn_import_error})"}

    device = "cpu"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    if loocv:
        kf = LeaveOneOut()
    else:
        n_splits = max(2, min(n_splits, len(X_tr_df)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y_tr), dtype=float)
    fold_r2: list[float] = []

    for tr_idx, va_idx in kf.split(X_tr_df):
        model = TabPFNRegressor(device=device)
        model.fit(X_tr_df.iloc[tr_idx], y_tr[tr_idx])
        pred = model.predict(X_tr_df.iloc[va_idx])
        oof_pred[va_idx] = pred
        if len(va_idx) >= 2:
            fold_r2.append(float(r2_score(y_tr[va_idx], pred)))

    cv_r2 = float(r2_score(y_tr, oof_pred))
    cv_r2_std = float(np.std(fold_r2)) if fold_r2 else 0.0

    test_r2 = float("nan")
    y_pred_hold: list[float] = []
    if len(X_hold_df) > 0:
        model_full = TabPFNRegressor(device=device)
        model_full.fit(X_tr_df, y_tr)
        hold_pred = model_full.predict(X_hold_df)
        y_pred_hold = list(map(float, hold_pred))
        test_r2 = float(r2_score(y_hold, hold_pred))

    return {
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "test_r2": test_r2,
        "y_pred": y_pred_hold,
        "oof_pred": oof_pred.tolist(),
    }


def _split_train_test(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (train_df, test_df, split_mode). Uses Eaton-new holdout if available."""
    if "Main Folder" in df.columns:
        is_eaton = df["Main Folder"].astype(str).str.lower().eq("eaton-new")
        if is_eaton.any() and (~is_eaton).sum() >= 6:
            return df.loc[~is_eaton].copy(), df.loc[is_eaton].copy(), "eaton-new holdout"
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), "random 80/20"


def run_model_comparison(
    df: pd.DataFrame,
    input_cols: list[str],
    output_col: str,
    random_state: int = SEED,
    output_dir: Path | None = None,
    split_mode: str = "adaptive_auto",
    log_target: bool = False,
) -> dict[str, Any]:
    """Fit and evaluate model suite using the notebook's exact recipe."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn is not installed. Run: pip install scikit-learn"}

    numeric_inputs = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_inputs:
        return {"error": "None of the selected input columns are numeric."}
    if not pd.api.types.is_numeric_dtype(df[output_col]):
        return {"error": f"Output column '{output_col}' is not numeric."}

    # Row filtering (NaN-target drop, fillna on inputs, high-variance group
    # removal, duplicate aggregation) happens at the Select stage. This service
    # consumes the dataset as-is.
    keep_cols = numeric_inputs + [output_col]
    if "Main Folder" in df.columns:
        keep_cols = keep_cols + ["Main Folder"]
    working = df[keep_cols].copy()
    if log_target:
        # Model log1p(target); metrics and plots are then in log space.
        working[output_col] = np.log1p(working[output_col].clip(lower=0.0))
    if len(working) < 6:
        return {"error": f"Too few rows in dataset: {len(working)} (need ≥ 6)."}

    loocv_mode = split_mode == "loocv"
    # Pooled modes use every row for CV (no holdout); metrics/plots are out-of-fold.
    pooled_mode = loocv_mode or split_mode == "pooled_kfold"
    if pooled_mode:
        train_df = working.reset_index(drop=True)
        test_df = working.iloc[0:0].copy()
        split_mode_used = "LOOCV" if loocv_mode else "pooled K-fold"
    else:
        train_df, test_df, split_mode_used = _split_train_test(working, random_state)
    n_hv_removed = 0

    X_train_df = train_df[numeric_inputs].copy()
    y_train = train_df[output_col].to_numpy(dtype=float)
    X_test_df = test_df[numeric_inputs].copy()
    y_test = test_df[output_col].to_numpy(dtype=float)

    train_csv_path: str | None = None
    test_csv_path: str | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "training_data.csv"
        train_df.to_csv(train_path, index=False)
        train_csv_path = str(train_path)
        if not pooled_mode and len(test_df) > 0:
            test_path = output_dir / "test_data.csv"
            test_df.to_csv(test_path, index=False)
            test_csv_path = str(test_path)

    n_train = len(X_train_df)
    if loocv_mode:
        if n_train < 2:
            return {"error": f"Too few rows for LOOCV: {n_train}."}
        k = n_train
    else:
        cv_folds_cfg = max(2, int(getattr(app_config, "CV_FOLDS", 5)))
        k = min(cv_folds_cfg, n_train)
        if k < 2:
            return {"error": f"Too few training rows for CV: {n_train}."}

    def _make_cv_splitter():
        if loocv_mode:
            return LeaveOneOut()
        return KFold(n_splits=k, shuffle=True, random_state=random_state)

    enabled_models = set(getattr(app_config, "ENABLED_MODELS", []) or [])

    model_specs = _build_model_specs(numeric_inputs)

    results: list[dict[str, Any]] = []

    for name, estimator in model_specs.items():
        if name not in enabled_models:
            continue
        try:
            kf = _make_cv_splitter()
            oof_pred = np.zeros(n_train, dtype=float)
            fold_r2: list[float] = []

            for tr_idx, va_idx in kf.split(X_train_df):
                model = clone(estimator)
                model.fit(X_train_df.iloc[tr_idx], y_train[tr_idx])
                fold_pred = model.predict(X_train_df.iloc[va_idx])
                oof_pred[va_idx] = fold_pred
                if len(va_idx) >= 2:
                    fold_r2.append(float(r2_score(y_train[va_idx], fold_pred)))

            cv_r2 = float(r2_score(y_train, oof_pred))
            cv_r2_std = float(np.std(fold_r2)) if fold_r2 else 0.0

            if pooled_mode:
                y_pred = []
                test_r2 = float("nan")
            else:
                model_full = clone(estimator)
                model_full.fit(X_train_df, y_train)
                y_pred = model_full.predict(X_test_df)
                test_r2 = float(r2_score(y_test, y_pred)) if len(y_test) else float("nan")

            if pooled_mode:
                y_true_plot = y_train.tolist()
                y_pred_plot = oof_pred.tolist()
            else:
                y_true_plot = y_test.tolist()
                y_pred_plot = y_pred.tolist() if len(y_test) else []
            results.append(
                {
                    "name": name,
                    "cv_r2": cv_r2,
                    "cv_r2_std": cv_r2_std,
                    "test_r2": test_r2,
                    "y_test": y_true_plot,
                    "y_pred": y_pred_plot,
                    "kind": "available",
                    "badge": "Available",
                    "badge_color": "positive",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "cv_r2": None,
                    "cv_r2_std": None,
                    "test_r2": None,
                    "y_test": [],
                    "y_pred": [],
                    "kind": "error",
                    "badge": f"Error: {exc}",
                    "badge_color": "negative",
                }
            )

    # GP models — mirror notebook's gp_kfold_eaton_nlpd.ipynb gp_model_specs:
    #   - Regular GP: unit-cube min-max inputs, learned noise (no fixed Yvar)
    #   - Set GP:     no input normalization, learned noise, MeanSet kernel
    set_gp_supported = (len(numeric_inputs) % 2 == 0) and len(numeric_inputs) >= 2
    gp_specs: list[dict[str, Any]] = [
        {
            "name": "GP Regression (botorch)",
            "covar_module_factory": None,
            "normalize_inputs": True,
            "use_fixed_noise": False,
        },
    ]
    if BOTORCH_AVAILABLE and set_gp_supported:
        gp_specs.append({
            "name": "Set GP",
            "covar_module_factory": _build_set_gp_covar_module,
            "normalize_inputs": False,
            "use_fixed_noise": False,
        })

    for spec in gp_specs:
        gp_name = spec["name"]
        if gp_name not in enabled_models:
            continue
        if not BOTORCH_AVAILABLE:
            results.append({
                "name": gp_name,
                "cv_r2": None, "cv_r2_std": None, "test_r2": None,
                "y_test": [], "y_pred": [],
                "kind": "not_installed",
                "badge": f"Not installed ({_botorch_import_error})" if _botorch_import_error else "Not installed",
                "badge_color": "warning",
            })
            continue
        try:
            bo_res = _eval_botorch_gp(
                X_train_df, y_train, X_test_df, y_test,
                numeric_inputs, output_col, n_splits=k, random_state=random_state,
                covar_module_factory=spec["covar_module_factory"],
                normalize_inputs=spec["normalize_inputs"],
                use_fixed_noise=spec["use_fixed_noise"],
                loocv=loocv_mode,
            )
            if "error" in bo_res:
                results.append({
                    "name": gp_name,
                    "cv_r2": None, "cv_r2_std": None, "test_r2": None,
                    "y_test": [], "y_pred": [],
                    "kind": "error",
                    "badge": f"Error: {bo_res['error']}",
                    "badge_color": "negative",
                })
            else:
                if pooled_mode:
                    y_true_plot = y_train.tolist()
                    y_pred_plot = bo_res.get("oof_pred", [])
                else:
                    y_true_plot = y_test.tolist()
                    y_pred_plot = bo_res["y_pred"]
                results.append({
                    "name": gp_name,
                    "cv_r2": bo_res["cv_r2"],
                    "cv_r2_std": bo_res["cv_r2_std"],
                    "test_r2": bo_res["test_r2"],
                    "y_test": y_true_plot,
                    "y_pred": y_pred_plot,
                    "kind": "available",
                    "badge": "Available",
                    "badge_color": "positive",
                })
        except Exception as exc:
            results.append({
                "name": gp_name,
                "cv_r2": None, "cv_r2_std": None, "test_r2": None,
                "y_test": [], "y_pred": [],
                "kind": "error",
                "badge": f"Error: {exc}",
                "badge_color": "negative",
            })

    if BOTORCH_AVAILABLE and not set_gp_supported and "Set GP" in enabled_models:
        results.append({
            "name": "Set GP",
            "cv_r2": None, "cv_r2_std": None, "test_r2": None,
            "y_test": [], "y_pred": [],
            "kind": "not_installed",
            "badge": f"Requires even feature count (got {len(numeric_inputs)})",
            "badge_color": "warning",
        })

    # TabPFN — direct fit/predict (the model handles its own preprocessing)
    tabpfn_name = "TabPFN"
    if tabpfn_name not in enabled_models:
        pass
    elif TABPFN_AVAILABLE:
        try:
            tp_res = _eval_tabpfn(
                X_train_df, y_train, X_test_df, y_test,
                n_splits=k, random_state=random_state, loocv=loocv_mode,
            )
            if "error" in tp_res:
                results.append({
                    "name": tabpfn_name,
                    "cv_r2": None, "cv_r2_std": None, "test_r2": None,
                    "y_test": [], "y_pred": [],
                    "kind": "error",
                    "badge": f"Error: {tp_res['error']}",
                    "badge_color": "negative",
                })
            else:
                if pooled_mode:
                    y_true_plot = y_train.tolist()
                    y_pred_plot = tp_res.get("oof_pred", [])
                else:
                    y_true_plot = y_test.tolist()
                    y_pred_plot = tp_res["y_pred"]
                results.append({
                    "name": tabpfn_name,
                    "cv_r2": tp_res["cv_r2"],
                    "cv_r2_std": tp_res["cv_r2_std"],
                    "test_r2": tp_res["test_r2"],
                    "y_test": y_true_plot,
                    "y_pred": y_pred_plot,
                    "kind": "available",
                    "badge": "Available",
                    "badge_color": "positive",
                })
        except Exception as exc:
            results.append({
                "name": tabpfn_name,
                "cv_r2": None, "cv_r2_std": None, "test_r2": None,
                "y_test": [], "y_pred": [],
                "kind": "error",
                "badge": f"Error: {exc}",
                "badge_color": "negative",
            })
    else:
        results.append({
            "name": tabpfn_name,
            "cv_r2": None, "cv_r2_std": None, "test_r2": None,
            "y_test": [], "y_pred": [],
            "kind": "not_installed",
            "badge": f"Not installed ({_tabpfn_import_error})" if _tabpfn_import_error else "Not installed",
            "badge_color": "warning",
        })

    if not XGBOOST_AVAILABLE and "XGBoost" in enabled_models:
        results.append(
            {
                "name": "XGBoost",
                "cv_r2": None,
                "cv_r2_std": None,
                "test_r2": None,
                "y_test": [],
                "y_pred": [],
                "kind": "not_installed",
                "badge": "Not installed",
                "badge_color": "warning",
            }
        )

    return {
        "models": results,
        "k_folds": k,
        "n_train": n_train,
        "n_test": len(X_test_df),
        "numeric_inputs": numeric_inputs,
        "output_col": output_col,
        "log_target": log_target,
        "split_mode": split_mode_used,
        "loocv": loocv_mode,
        "pooled": pooled_mode,
        "high_var_rows_removed": n_hv_removed,
        "train_csv_path": train_csv_path,
        "test_csv_path": test_csv_path,
    }
