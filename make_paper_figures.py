"""Regenerate the paper's Figure 4 (a-e) and Figure 6 (a-d) panels as individual plots.

Data: prep_outputs/app_input_max2fillers.xlsx (from `python prep_handoff.py 2`) pushed through the
workbench
pipeline (preprocessing -> featurization -> thickness/SE-per-mm filters from config.py),
no Select-page filter, raw EMI SE per mm target. Models/sensitivity reuse services/.

Panel sizes (mm) and font sizes (pt) are in SPEC below.
Run:  python make_paper_figures.py            (full run, caches results)
                                   python make_paper_figures.py --replot   (re-plot from cache after style edits)
Outputs (svg + png) and figure_stats.json go to paper-figures/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper-figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402

import services.modeling as M  # noqa: E402
from services.featurization import featurize_dataset  # noqa: E402
from services.preprocessing import run_preprocessing  # noqa: E402
from services.selection_filters import apply_selection_filters  # noqa: E402
from services.sensitivity import compute_shap, compute_sobol, fit_surrogate  # noqa: E402

SEED = 42
N_FOLDS = int(config.CV_FOLDS)  # UI default (20)
# Paper feature order: filler 1 block, then filler 2 block (halves = the two set elements).
FEATURES = [
    "Electrical conductivity", "Aspect ratio", "Density", "Particle size", "Filler Composition",
    "Electrical conductivity 2", "Aspect ratio 2", "Density 2", "Particle size 2", "Filler 2 Composition",
]
LABELS = ["EC1", "AR1", "D1", "PS1", "FC1", "EC2", "AR2", "D2", "PS2", "FC2"]
TARGET = "EMI_SE_per_mm"
TARGET_LABEL = "EMI SE per mm"

MM = 1 / 25.4
# Final physical canvas size (mm) and font size (pt) of every panel, measured from the layout in
# the manuscript figure layout.
# Place the exported SVGs at 100% scale. Edit here to change a panel.
SPEC = {  # name: (width_mm, height_mm, font_pt)
    "fig4a_radviz": (89.5, 89.5, 9.5),
    "fig4b_histogram": (101.6, 101.6, 11.0),
    "fig4c_parity_set_gp": (84.0, 67.0, 8.5),
    "fig4d_shap_beeswarm_set_gp": (89.0, 67.0, 9.0),
    "fig4e_sobol_set_gp": (189.8, 81.4, 9.6),
    "fig6a_cv_r2": (85.9, 85.9, 10.1),
    "fig6b_permutation_invariance_score": (82.2, 82.2, 9.7),
    "fig6c_set_gp_permutation_parity": (89.5, 89.5, 10.6),
    "fig6d_regular_gp_permutation_parity": (89.5, 89.5, 10.6),
}
RADVIZ_VMAX = None  # set a number to clip the RadViz colour scale (not needed once MAX_SE_PER_MM caps the data)
ANCHOR_PT = 12.0   # RadViz anchor labels (editable text in the layout)
R2_TEXT_PT = 13.3  # "R2 = x" annotation on the parity plot
BASE_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "figure.dpi": 300,
    "axes.linewidth": 1, "xtick.major.size": 2, "xtick.major.width": 1.0, "ytick.major.size": 2,
    "ytick.major.width": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "lines.linewidth": 1.5, "svg.fonttype": "none",
}


def rc_for(name):
    f = SPEC[name][2]
    return {**BASE_RC, "font.size": f, "axes.titlesize": f, "axes.labelsize": f, "legend.fontsize": f,
            "xtick.labelsize": f, "ytick.labelsize": f}


def new_fig(name):
    w, h, _ = SPEC[name]
    plt.rcParams.update(rc_for(name))
    return plt.subplots(figsize=(w * MM, h * MM))

STATS: dict = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg")  # no bbox_inches="tight": keep the exact canvas size
    fig.savefig(OUT / f"{name}.png", format="png", dpi=300)
    plt.close(fig)
    print("saved", name)


# ───────────────────────────── data (mirrors UI Load + Select pages) ─────────────────────────────
def load_dataset() -> pd.DataFrame:
    raw = pd.read_excel(HERE / "prep_outputs" / "app_input_max2fillers.xlsx")
    pre = run_preprocessing(raw)
    feat = featurize_dataset(pre.df)
    df = (feat.df if not feat.df.empty else pre.df).copy()
    th = pd.to_numeric(df["sample_thickness_value_mm"], errors="coerce")
    df["sample_thickness_value_mm"] = th
    df["EMI shielding value"] = pd.to_numeric(df["EMI shielding value"], errors="coerce")
    keep = th.notna()
    if config.MIN_THICKNESS_MM is not None:
        keep &= th >= config.MIN_THICKNESS_MM
    if config.MAX_THICKNESS_MM is not None:
        keep &= th <= config.MAX_THICKNESS_MM
    df = df[keep].reset_index(drop=True)
    df[TARGET] = np.where(df.sample_thickness_value_mm > 0, df["EMI shielding value"] / df.sample_thickness_value_mm, np.nan)
    if config.MIN_SE_PER_MM is not None:
        df = df[df[TARGET] > config.MIN_SE_PER_MM].reset_index(drop=True)
    if config.MAX_SE_PER_MM is not None:
        df = df[df[TARGET] <= config.MAX_SE_PER_MM].reset_index(drop=True)
    df, _ = apply_selection_filters(df, FEATURES, TARGET, filter_duplicates=config.SELECTION_FILTER_ENABLED)
    STATS["rows_raw_upload"] = len(raw)
    STATS["rows_after_preprocessing"] = pre.rows_out
    STATS["rows_after_featurization"] = feat.rows_out
    STATS["max_se_per_mm_cap"] = config.MAX_SE_PER_MM
    STATS["rows_training"] = len(df)
    return df


# ───────────────────────────── GP helpers ─────────────────────────────
GP_SPECS = {
    "Set GP": dict(covar_module_factory=M._build_set_gp_covar_module, normalize_inputs=False),
    "Regular GP": dict(covar_module_factory=None, normalize_inputs=True),
}


def fit_gp(X_df, y, spec):
    fit = M.fit_botorch_gp_full(X_df, y, FEATURES, TARGET, use_fixed_noise=False, **spec)
    if "error" in fit:
        raise RuntimeError(fit["error"])
    return fit


def gp_predict(fit, X_df):
    X = M._prepare_botorch_eval_arrays(X_df, FEATURES)
    if fit["normalize_inputs"]:
        X = M._transform_minmax(X, fit["x_min"], fit["denom"])
    with torch.no_grad():
        post = fit["model"].posterior(torch.tensor(X, dtype=fit["dtype"], device=fit["device"]))
    return post.mean.squeeze(-1).cpu().numpy(), post.variance.squeeze(-1).clamp(min=0).sqrt().cpu().numpy()


def kfold_oof(X_df, y, spec, k):
    oof, std = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in KFold(n_splits=min(k, len(y)), shuffle=True, random_state=SEED).split(X_df):
        fit = fit_gp(X_df.iloc[tr], y[tr], spec)
        oof[va], std[va] = gp_predict(fit, X_df.iloc[va])
    return oof, std


# ───────────────────────────── radviz ─────────────────────────────
def radviz_coordinates(X):
    X = np.asarray(X, dtype=float)
    rng = X.max(0) - X.min(0)
    rng[rng == 0] = 1.0
    W = (X - X.min(0)) / rng
    ang = 2 * np.pi * np.arange(X.shape[1]) / X.shape[1]
    anchors = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (W @ anchors) / s, anchors


def fig4a(df):
    coords, anchors = radviz_coordinates(df[FEATURES])
    fig, ax = new_fig("fig4a_radviz")
    f = SPEC["fig4a_radviz"][2]
    th = np.linspace(0, 2 * np.pi, 300)
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(r * np.cos(th), r * np.sin(th), color="lightgray", lw=0.8 if r < 1 else 1.2, ls="--" if r < 1 else "-", zorder=0)
        if r < 1:
            ax.text(r + 0.01, 0.01, f"{r:.2f}", fontsize=f - 1.5, color="gray", va="bottom")
    for ax_x, ax_y in anchors:
        ax.plot([0, ax_x], [0, ax_y], color="lightgray", lw=0.6, ls="--", zorder=0)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df[TARGET], cmap="viridis", s=10, alpha=0.8, zorder=2, vmax=RADVIZ_VMAX)
    for (x, y), name in zip(anchors, LABELS):
        ax.plot(x, y, "o", color="tomato", ms=7, zorder=3)
        ax.text(x * 1.2, y * 1.2, name, ha="center", va="center", color="tomato", fontsize=ANCHOR_PT)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.0, shrink=0.8, extend="max" if RADVIZ_VMAX is not None else "neither")
    cb.set_label(TARGET_LABEL, fontsize=f)
    cb.ax.tick_params(labelsize=f)
    fig.tight_layout(pad=0.3)
    save(fig, "fig4a_radviz")
    STATS["single_filler_samples"] = int((df["Filler 2 Composition"].fillna(0) <= 0).sum())
    STATS["two_filler_samples"] = int((df["Filler 2 Composition"].fillna(0) > 0).sum())


def fig4b(df):
    fig, ax = new_fig("fig4b_histogram")
    ax.hist(df[TARGET], bins=20, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(TARGET_LABEL)
    ax.set_ylabel("Count")
    fig.tight_layout(pad=0.4)
    save(fig, "fig4b_histogram")
    y = df[TARGET]
    STATS["target_dB_per_mm"] = {k: float(v) for k, v in y.describe().items()}
    STATS["target_dB_per_mm"]["n_above_300"] = int((y > 300).sum())


def fig4c(y_true, y_pred, y_std):
    fig, ax = new_fig("fig4c_parity_set_gp")
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", color="black", alpha=0.7, markersize=3, elinewidth=0.6, capsize=1.5, capthick=0.6)
    lo = min(y_true.min(), (y_pred - y_std).min())
    hi = max(y_true.max(), (y_pred + y_std).max())
    m = (hi - lo) * 0.05
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", lw=0.9)
    ax.set_xlim(lo - m, hi + m)
    ax.set_ylim(lo - m, hi + m)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.text(0.05, 0.95, f"R$^2$ = {r2_score(y_true, y_pred):.2f}", transform=ax.transAxes, fontsize=R2_TEXT_PT, va="top")
    fig.tight_layout(pad=0.4)
    save(fig, "fig4c_parity_set_gp")


def fig6a(cv_r2):
    names = list(cv_r2)
    f = SPEC["fig6a_cv_r2"][2]
    fig, ax = new_fig("fig6a_cv_r2")
    ax.bar(np.arange(len(names)), [cv_r2[n] for n in names], width=0.5, color="steelblue", edgecolor="black")
    for i, n in enumerate(names):
        ax.text(i, max(cv_r2[n], 0) + 0.02, f"R$^2$ = {cv_r2[n]:.2f}", ha="center", fontsize=f - 1.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(f"{N_FOLDS}-fold CV R$^2$")
    ax.axhline(0.0, color="black")
    ax.set_ylim(min(0, min(cv_r2.values()) - 0.1), 1)
    fig.tight_layout(pad=0.4)
    save(fig, "fig6a_cv_r2")


def permutation_test(df):
    """70/30 split (as in gp_permutation_invariance.ipynb); permute = swap filler-1 and filler-2 blocks."""
    train, test = train_test_split(df, test_size=0.30, random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    perm_cols = FEATURES[5:] + FEATURES[:5]
    X_orig = test[FEATURES].copy()
    X_perm = test[perm_cols].copy()
    X_perm.columns = FEATURES
    y_test = test[TARGET].to_numpy(float)
    out = {}
    for name, spec in GP_SPECS.items():
        fit = fit_gp(train[FEATURES], train[TARGET].to_numpy(float), spec)
        po, _ = gp_predict(fit, X_orig)
        pp, _ = gp_predict(fit, X_perm)
        out[name] = dict(po=po, pp=pp, y=y_test, pi_fraction=float(np.mean(np.abs(po - pp) < 1e-2)),
                         consistency_rmse=float(np.sqrt(np.mean((po - pp) ** 2))),
                         test_r2_orig=float(r2_score(y_test, po)), test_r2_perm=float(r2_score(y_test, pp)))
    STATS["permutation_split"] = {"n_train": len(train), "n_test": len(test), "delta": 1e-2,
                                  **{k: {m: v[m] for m in ("pi_fraction", "consistency_rmse", "test_r2_orig", "test_r2_perm")} for k, v in out.items()}}
    return out


def fig6b(perm):
    names = list(perm)
    fig, ax = new_fig("fig6b_permutation_invariance_score")
    x = np.arange(len(names))
    vals = [perm[n]["pi_fraction"] for n in names]
    ax.bar(x, vals, width=0.5, color="mediumseagreen")
    for xi, v in zip(x, vals):
        if v == 0:
            ax.plot(xi, 0.03, marker="x", color="red", ms=7, mew=1.5, ls="none")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Permutation invariance score")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(pad=0.4)
    save(fig, "fig6b_permutation_invariance_score")


def fig6cd(perm, name, fname):
    r = perm[name]
    fig, ax = new_fig(fname)
    yt, po, pp = r["y"], r["po"], r["pp"]
    lo, hi = min(yt.min(), po.min(), pp.min()), max(yt.max(), po.max(), pp.max())
    m = (hi - lo) * 0.05
    ax.scatter(yt, po, color="steelblue", alpha=0.75, s=14, label="Original", zorder=3)
    ax.scatter(yt, pp, color="coral", alpha=0.75, s=14, label="Permuted", zorder=3, marker="^")
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", lw=1.0)
    ax.set_xlim(lo - m, hi + m)
    ax.set_ylim(lo - m, hi + m)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.text(0.05, 0.62, name, transform=ax.transAxes)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    save(fig, fname)


# ───────────────────────────── sensitivity ─────────────────────────────
def fig4d(sh):
    name = "fig4d_shap_beeswarm_set_gp"
    w, h, f = SPEC[name]
    plt.rcParams.update(rc_for(name))
    X_df = pd.DataFrame(np.asarray(sh["feature_values"], float), columns=FEATURES)
    shap.summary_plot(sh["shap_values"], X_df, feature_names=LABELS, show=False, plot_type="dot", plot_size=(w * MM, h * MM))
    fig = plt.gcf()
    for ax in fig.axes:  # shap hard-codes its font sizes; bring them to the panel font size
        ax.tick_params(axis="both", labelsize=f)
        ax.xaxis.label.set_size(f)
        ax.yaxis.label.set_size(f)
    fig.axes[0].set_xlabel(fig.axes[0].get_xlabel(), fontsize=f)
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() in ("High", "Low", "Feature value"):
            t.set_fontsize(f)
    fig.tight_layout(pad=0.4)
    save(fig, name)
    mean_abs = np.abs(sh["shap_values"]).mean(0)
    STATS["shap_mean_abs_ranking"] = {LABELS[i]: float(mean_abs[i]) for i in np.argsort(-mean_abs)}


def fig4e(so):
    fig, ax = new_fig("fig4e_sobol_set_gp")
    x, w = np.arange(len(FEATURES)), 0.35
    ax.bar(x - w / 2, so["S1"], w, yerr=so["S1_conf"], capsize=3, label="S1 (first-order)", color="steelblue")
    ax.bar(x + w / 2, so["ST"], w, yerr=so["ST_conf"], capsize=3, label="ST (total-order)", color="coral")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("Sobol Index")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig4e_sobol_set_gp")
    STATS["sobol"] = {l: {"S1": float(a), "ST": float(b)} for l, a, b in zip(LABELS, so["S1"], so["ST"])}
    STATS["sobol_ST_ranking"] = [LABELS[i] for i in np.argsort(-so["ST"])]
    STATS["sobol_n_samples"] = so["n_samples"]


CACHE = HERE / ".figure_cache.pkl"
SOBOL_N = 16184  # Saltelli base samples (paper notebook used 16184); seeded for reproducibility


def compute(df):
    """Slow part (GP CV, permutation test, SHAP, Sobol). Cached so style changes only need a re-plot."""
    y = df[TARGET].to_numpy(float)
    X = df[FEATURES].copy()
    cv = {}
    for name, spec in GP_SPECS.items():
        oof, std = kfold_oof(X, y, spec, N_FOLDS)
        cv[name] = dict(oof=oof, std=std, r2=float(r2_score(y, oof)))
    perm = permutation_test(df)
    sur = fit_surrogate(df, FEATURES, TARGET, kernel="set")
    if "error" in sur:
        raise RuntimeError(sur["error"])
    np.random.seed(SEED)
    sh = compute_shap(sur, n_background=30, n_samples=512)
    so = compute_sobol(sur, n_samples=SOBOL_N, seed=SEED)
    if "error" in so:
        raise RuntimeError(so["error"])
    return dict(y=y, cv=cv, perm=perm, stats_perm=STATS["permutation_split"],
                shap=dict(shap_values=sh["shap_values"], feature_values=sh["feature_values"].to_numpy(float)), sobol=so)


def main():
    import pickle
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = load_dataset()
    df[FEATURES] = df[FEATURES].fillna(0.0)
    print("training rows:", len(df))
    if "--replot" in sys.argv and CACHE.exists():
        res = pickle.load(open(CACHE, "rb"))
        STATS["permutation_split"] = res["stats_perm"]
    else:
        res = compute(df)
        pickle.dump(res, open(CACHE, "wb"))

    fig4a(df)
    fig4b(df)
    c = res["cv"]["Set GP"]
    fig4c(res["y"], c["oof"], c["std"])
    cv_r2 = {k: v["r2"] for k, v in res["cv"].items()}
    STATS["cv"] = {"folds": N_FOLDS, "pooled_oof_r2": cv_r2}
    fig6a(cv_r2)
    fig6b(res["perm"])
    fig6cd(res["perm"], "Set GP", "fig6c_set_gp_permutation_parity")
    fig6cd(res["perm"], "Regular GP", "fig6d_regular_gp_permutation_parity")
    fig4d(res["shap"])
    fig4e(res["sobol"])
    (OUT / "figure_stats.json").write_text(json.dumps(STATS, indent=2))
    print("done")


if __name__ == "__main__":
    main()
