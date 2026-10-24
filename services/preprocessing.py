"""
Stage 2A preprocessing pipeline.

Mirrors the logic in data_analysis.ipynb and visualization_feature_space.ipynb:
  1. Column normalisation  (strip whitespace, standardise known aliases)
  2. Required-column filtering  (drop rows missing key columns)
  3. Deduplication  (remove high-variance repeat measurements, CV > threshold)
  4. Thickness parsing  (convert mixed units → mm)
  5. EMI SE per mm  (derived feature)
  6. Exact-duplicate removal

Featurisation (filler-name → numeric feature vectors) is a separate step that
requires the filler_feature_vectors.xlsx asset and is NOT done here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column-name alias maps — maps common variants to a canonical name
# ---------------------------------------------------------------------------

_SE_ALIASES = [
    "shielding effectiveness (db)", "shielding_effectiveness value",
    "shielding effectiveness db", "shielding_effectiveness_1 value",
    "emi shielding value", "emi se (db)", "emi_se_db",
    "shielding effectiveness dB",
]

_THICKNESS_ALIASES = [
    "sample thickness", "thickness", "thickness_mm",
    "sample_thickness_value_mm", "thickness (mm)",
]

_FILLER_NAME_ALIASES = ["filler name", "filler 1 name", "filler1 name"]
_FILLER_COMP_ALIASES = ["filler composition", "filler 1 composition", "filler1 composition"]
_FILLER2_NAME_ALIASES = ["filler 2 name", "filler2 name"]
_FILLER2_COMP_ALIASES = ["filler 2 composition", "filler2 composition"]
_FILLER3_NAME_ALIASES = ["filler 3 name", "filler3 name"]
_FILLER3_COMP_ALIASES = ["filler 3 composition", "filler3 composition"]


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first column in *df* whose lower-stripped name matches any alias."""
    lower_map = {c.strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    cols_added: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_in - self.rows_out,
            "cols_added": self.cols_added,
            "warnings": self.warnings,
            "log": self.log,
        }


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names."""
    renamed = {c: c.strip() for c in df.columns if c != c.strip()}
    if renamed:
        df = df.rename(columns=renamed)
        log.append(f"Stripped whitespace from {len(renamed)} column name(s).")
    return df


def filter_required_columns(
    df: pd.DataFrame,
    log: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Drop rows missing any of: SE value, sample thickness, or (filler name + composition)."""
    n_before = len(df)

    filler_col = _find_col(df, _FILLER_NAME_ALIASES)
    comp_col = _find_col(df, _FILLER_COMP_ALIASES)
    se_col = _find_col(df, _SE_ALIASES)
    thickness_col = _find_col(df, _THICKNESS_ALIASES)

    if se_col is None:
        warnings.append(
            "Could not detect an EMI shielding effectiveness column. "
            "Skipping required-column filter — all rows retained."
        )
        return df

    # Drop rows where the SE value is missing
    df = df[df[se_col].notna()].copy()

    # Drop rows where sample thickness is missing (required for SE/mm computation)
    if thickness_col is not None:
        before_thick = len(df)
        df = df[df[thickness_col].notna()].copy()
        n_thick_dropped = before_thick - len(df)
        if n_thick_dropped:
            log.append(f"Required-column filter: dropped {n_thick_dropped} row(s) with missing thickness value.")
    else:
        warnings.append("No thickness column detected — cannot enforce thickness requirement.")

    # Also drop rows where filler name AND composition are both missing (if detected)
    if filler_col and comp_col:
        mask_missing = df[filler_col].isna() & df[comp_col].isna()
        df = df[~mask_missing]

    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(f"Required-column filter: dropped {n_dropped} total row(s) with missing required values.")
    else:
        log.append("Required-column filter: no rows dropped.")
    return df


def remove_exact_duplicates(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(f"Exact-duplicate removal: dropped {n_dropped} exact-duplicate row(s).")
    else:
        log.append("Exact-duplicate removal: no exact duplicates found.")
    return df


def deduplicate_high_variance(
    df: pd.DataFrame,
    log: list[str],
    warnings: list[str],
    cv_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Mirrors data_analysis.ipynb deduplication:
    - Build group_key = sorted tuple of (Filler Name, Filler 2 Name, Filler 3 Name)
    - Inside each filler-name group, subgroup by (Filler Composition,
      Filler 2 Composition, Filler 3 Composition)
    - Drop every row whose composition subgroup has std/mean(SE) > cv_threshold.
    Surviving duplicates are kept as-is (notebook does not average).
    """
    filler_col = _find_col(df, _FILLER_NAME_ALIASES)
    comp_col = _find_col(df, _FILLER_COMP_ALIASES)
    se_col = _find_col(df, _SE_ALIASES)

    if not (filler_col and comp_col and se_col):
        warnings.append(
            "Could not identify filler/composition/SE columns for deduplication. "
            "Skipping high-variance deduplication."
        )
        return df

    filler2_col = _find_col(df, _FILLER2_NAME_ALIASES)
    filler2_comp_col = _find_col(df, _FILLER2_COMP_ALIASES)
    filler3_col = _find_col(df, _FILLER3_NAME_ALIASES)
    filler3_comp_col = _find_col(df, _FILLER3_COMP_ALIASES)

    df = df.copy()
    df[se_col] = pd.to_numeric(df[se_col], errors="coerce")
    df = df[df[se_col].notna()]
    n_before = len(df)

    name_cols = [c for c in [filler_col, filler2_col, filler3_col] if c is not None]
    comp_cols = [c for c in [comp_col, filler2_comp_col, filler3_comp_col] if c is not None]

    # sorted-tuple filler-name key (order-independent, matches notebook)
    def _norm(x: Any) -> str:
        return str(x).strip().lower() if pd.notna(x) else "none"

    df["__group_key"] = df[name_cols].apply(lambda r: tuple(sorted(_norm(v) for v in r)), axis=1)

    drop_indices: list[int] = []
    n_bad_groups = 0
    for _, grp in df.groupby("__group_key", dropna=False):
        stats = (
            grp.groupby(comp_cols, dropna=False)[se_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(0.0)
        stats["ratio"] = stats["std"] / stats["mean"].abs().clip(lower=1e-12)
        bad = stats[(stats["count"] > 1) & (stats["ratio"] > cv_threshold)]
        if bad.empty:
            continue
        bad_keys = set(map(tuple, bad[comp_cols].to_numpy()))
        mask = grp[comp_cols].apply(tuple, axis=1).isin(bad_keys)
        drop_indices.extend(grp.index[mask].tolist())
        n_bad_groups += len(bad)

    if drop_indices:
        df = df.drop(index=drop_indices)
    df = df.drop(columns="__group_key").reset_index(drop=True)
    n_dropped = n_before - len(df)

    log.append(
        f"High-variance deduplication (std/mean > {cv_threshold}): "
        f"dropped {n_bad_groups} composition subgroup(s), {n_dropped} row(s) total."
    )
    return df


# ---------------------------------------------------------------------------
# Thickness parsing  (from data_analysis.ipynb Cell 23)
# ---------------------------------------------------------------------------

_THICKNESS_UNIT_FACTORS = {
    "nm": 1e-6,
    "um": 1e-3, "μm": 1e-3, "µm": 1e-3,
    "micrometer": 1e-3, "micrometers": 1e-3, "micron": 1e-3, "microns": 1e-3,
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4,
}
_THICKNESS_RE = re.compile(
    r"([\d]+(?:[.,][\d]+)?)\s*"
    r"(nm|um|μm|µm|micromet(?:er|ers)|microns?|mm|cm|in(?:ch(?:es)?)?|m)\b",
    re.IGNORECASE,
)


def _parse_thickness_value(raw: Any) -> tuple[float | None, bool]:
    """Convert a raw thickness cell value to mm.

    Returns (value_mm, had_unit). value_mm is None if unparsable.
    had_unit is True only when an explicit unit token was matched in the cell.
    """
    if pd.isna(raw):
        return None, False
    if isinstance(raw, (int, float)):
        return float(raw), False  # assume mm
    s = str(raw).strip().lower().replace(",", ".")
    match = _THICKNESS_RE.search(s)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        for unit_key, factor in _THICKNESS_UNIT_FACTORS.items():
            if unit_key == unit or unit.startswith(unit_key):
                return value * factor, True
    # Try bare number
    try:
        return float(s), False
    except ValueError:
        return None, False


def parse_thickness(
    df: pd.DataFrame,
    log: list[str],
    warnings: list[str],
    cols_added: list[str],
) -> pd.DataFrame:
    """Parse the thickness column into a clean float column in mm."""
    thickness_col = _find_col(df, _THICKNESS_ALIASES)
    if thickness_col is None:
        warnings.append("No thickness column detected — skipping thickness parsing.")
        return df

    out_col = "sample_thickness_mm"
    if out_col in df.columns:
        # Already present (e.g., file was already processed)
        log.append(f"Thickness column '{out_col}' already present — skipping parse.")
        return df

    df = df.copy()

    n_parsed_unitless = 0
    # If the source column is already numeric (e.g., 'thickness_mm'), just rename/copy
    if pd.api.types.is_numeric_dtype(df[thickness_col]):
        df[out_col] = pd.to_numeric(df[thickness_col], errors="coerce")
        n_parsed_unitless = int(df[out_col].notna().sum())
    else:
        parsed = df[thickness_col].apply(_parse_thickness_value)
        df[out_col] = [v for v, _ in parsed]
        n_parsed_unitless = int(sum(1 for v, had_unit in parsed if v is not None and not had_unit))

    n_parsed = df[out_col].notna().sum()
    n_total = len(df)
    log.append(
        f"Thickness parsing: parsed {n_parsed}/{n_total} rows from '{thickness_col}' → '{out_col}' (mm)."
    )
    if n_parsed_unitless:
        warnings.append(
            "No units found for sample thickness; unit assumed to be mm"
            + (f" ({n_parsed_unitless} row(s))." if n_parsed_unitless != n_parsed else ".")
        )
    cols_added.append(out_col)
    return df


def filter_missing_parsed_thickness(
    df: pd.DataFrame,
    log: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Drop rows whose parsed thickness (sample_thickness_mm) is missing or non-positive."""
    if "sample_thickness_mm" not in df.columns:
        warnings.append("Parsed thickness column missing — cannot enforce post-parse thickness filter.")
        return df
    n_before = len(df)
    thickness_numeric = pd.to_numeric(df["sample_thickness_mm"], errors="coerce")
    df = df[thickness_numeric.notna() & (thickness_numeric > 0)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(
            f"Post-parse thickness filter: dropped {n_dropped} row(s) with unparsable or non-positive thickness."
        )
    else:
        log.append("Post-parse thickness filter: no rows dropped.")
    return df


def compute_se_per_mm(
    df: pd.DataFrame,
    log: list[str],
    warnings: list[str],
    cols_added: list[str],
) -> pd.DataFrame:
    """Compute EMI SE per mm = SE / thickness_mm, then filter implausible values."""
    se_col = _find_col(df, _SE_ALIASES)
    out_col = "EMI_SE_per_mm"

    if out_col in df.columns:
        log.append(f"'{out_col}' already present — skipping.")
        return df

    if se_col is None:
        warnings.append("No SE column detected — cannot compute SE per mm.")
        return df

    thickness_col = "sample_thickness_mm" if "sample_thickness_mm" in df.columns else _find_col(df, _THICKNESS_ALIASES)
    if thickness_col is None:
        warnings.append("No thickness column detected — cannot compute SE per mm.")
        return df

    df = df.copy()
    se_numeric = pd.to_numeric(df[se_col], errors="coerce")
    thickness_numeric = pd.to_numeric(df[thickness_col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(thickness_numeric > 0, se_numeric / thickness_numeric, np.nan)

    df[out_col] = ratio
    n_computed = np.sum(~np.isnan(ratio))
    log.append(f"EMI SE per mm: computed for {n_computed}/{len(df)} rows → '{out_col}'.")
    cols_added.append(out_col)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing(df: pd.DataFrame) -> PreprocessingResult:
    """
    Apply the full preprocessing pipeline to *df* and return a PreprocessingResult.

    Steps (in order):
      1. Normalise column names
      2. Remove exact duplicates
      3. Filter rows missing required columns
      4. Deduplicate high-variance repeat measurements
      5. Parse thickness → mm
      6. Compute EMI SE per mm
    """
    result = PreprocessingResult(df=df.copy(), rows_in=len(df))
    log = result.log
    warnings = result.warnings
    cols_added = result.cols_added

    result.df = normalise_columns(result.df, log)
    result.df = remove_exact_duplicates(result.df, log)
    result.df = filter_required_columns(result.df, log, warnings)
    result.df = deduplicate_high_variance(result.df, log, warnings)
    result.df = parse_thickness(result.df, log, warnings, cols_added)
    result.df = filter_missing_parsed_thickness(result.df, log, warnings)
    result.df = compute_se_per_mm(result.df, log, warnings, cols_added)

    result.rows_out = len(result.df)
    return result
