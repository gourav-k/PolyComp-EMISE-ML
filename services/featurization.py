"""
Stage 2A featurization.

Maps filler names (categorical) to numeric feature vectors using two bundled
asset files:
  - filler_feature_vectors.xlsx  (sheet: 'updated')  — per-filler properties
  - canonization.xlsx            (sheet: 'Sheet1')    — name-alias → canonical name

Output DataFrame has the same column layout produced by the notebook:
  Source, Main Folder,
  Filler Name, Filler 2 Name, Filler 3 Name,
  Electrical conductivity, Shape, Magnetic permeability, Density, Particle size,
  Filler Composition,
  Electrical conductivity 2 … Filler 2 Composition,
  Electrical conductivity 3 … Filler Composition 3,
  sample_thickness_value_mm,
  EMI shielding value,
  Aspect ratio, Aspect ratio 2, Aspect ratio 3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CANONIZATION_PATH, FILLER_FEATURE_VECTORS_PATH

# ---------------------------------------------------------------------------
# Constants (match the notebook exactly)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["Electrical conductivity", "Shape", "Magnetic permeability", "Density", "Particle size"]
DEFAULT_FEATURE = [0, "none", 0, 0, 0]   # used for "none" fillers
SHAPE_TO_ASPECT = {"Sphere": 1, "Fiber": 0, "Flakes": 0.5}

# ---------------------------------------------------------------------------
# Asset loading (cached so assets are read only once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_feature_df() -> pd.DataFrame:
    df = pd.read_excel(FILLER_FEATURE_VECTORS_PATH, sheet_name="updated")
    df = df[["Filler name"] + FEATURE_COLS].copy()
    df = df.dropna(subset=["Filler name"])
    df["Filler name"] = df["Filler name"].str.strip().str.lower()
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_canon_df() -> pd.DataFrame:
    df = pd.read_excel(CANONIZATION_PATH, sheet_name="Sheet1")
    # keep only the two mapping columns
    df = df[["Name annotated DB", "Name feature DB"]].dropna(subset=["Name annotated DB"])
    df["Name annotated DB"] = df["Name annotated DB"].str.strip().str.lower()
    df["Name feature DB"] = df["Name feature DB"].str.strip().str.lower()
    return df.reset_index(drop=True)


def check_assets() -> tuple[bool, str]:
    """Return (ok, message). Call on startup."""
    missing = []
    for p in (FILLER_FEATURE_VECTORS_PATH, CANONIZATION_PATH):
        if not Path(p).exists():
            missing.append(str(p))
    if missing:
        return False, "Missing featurization asset files: " + ", ".join(missing)
    return True, "Featurization assets OK."


# ---------------------------------------------------------------------------
# Helpers (mirror the notebook helpers exactly)
# ---------------------------------------------------------------------------

def _is_none_like(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in {"none", ""}:
        return True
    return False


def _safe_lower(x: Any) -> Any:
    return x.strip().lower() if isinstance(x, str) else x


def _lookup_canonical(name: Any, canon_df: pd.DataFrame) -> str | None:
    """Return canonical name, 'none' for none-like, or None if not found."""
    if _is_none_like(name):
        return "none"
    name = _safe_lower(name)
    rows = canon_df.loc[canon_df["Name annotated DB"] == name, "Name feature DB"]
    if rows.empty:
        # No canon entry: fall back to an exact feature-DB name.
        return name if name in set(_load_feature_df()["Filler name"]) else None
    val = rows.iloc[0]
    return val if pd.notna(val) else None


def _lookup_features(canon_name: str | None, feat_df: pd.DataFrame) -> list | None:
    """Return feature list, DEFAULT_FEATURE for 'none', None if not in DB."""
    if canon_name == "none":
        return DEFAULT_FEATURE
    if canon_name is None:
        return None
    rows = feat_df.loc[feat_df["Filler name"] == canon_name, FEATURE_COLS]
    if rows.empty:
        return None
    return rows.iloc[0].tolist()


# ---------------------------------------------------------------------------
# Column-name aliases for the input DataFrame
# ---------------------------------------------------------------------------

def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lmap = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lmap:
            return lmap[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeaturizationResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    rows_missing_feature: int = 0
    rows_problem: int = 0
    missing_filler_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def featurize_dataset(df: pd.DataFrame) -> FeaturizationResult:
    """
    Apply filler-name → feature-vector mapping to *df*.

    Returns a FeaturizationResult whose .df has numeric columns for all three
    filler slots plus Aspect ratio columns, matching the notebook output layout.

    Rows where any primary filler cannot be featurized are excluded and
    reported in .missing_filler_names.
    """
    result = FeaturizationResult(df=pd.DataFrame(), rows_in=len(df))
    log = result.log

    ok, msg = check_assets()
    if not ok:
        log.append(f"ERROR: {msg}")
        result.df = df.copy()
        return result

    feat_df = _load_feature_df()
    canon_df = _load_canon_df()

    # Detect required columns with aliases
    col_map: dict[str, str | None] = {
        "filler1_name": _find(df, ["Filler Name", "Filler 1 Name"]),
        "filler1_comp": _find(df, ["Filler Composition", "Filler 1 Composition"]),
        "filler2_name": _find(df, ["Filler 2 Name", "Filler2 Name"]),
        "filler2_comp": _find(df, ["Filler 2 Composition", "Filler2 Composition"]),
        "filler3_name": _find(df, ["Filler 3 Name", "Filler3 Name"]),
        "filler3_comp": _find(df, ["Filler 3 Composition", "Filler3 Composition"]),
        "se_value": _find(df, [
            "shielding_effectiveness Value", "shielding effectiveness Value",
            "EMI shielding value", "Shielding Effectiveness dB",
            "EMI_SE_dB", "SE (dB)",
        ]),
        "thickness": _find(df, [
            "sample_thickness_value_mm", "sample_thickness_mm",
            "Thickness_mm", "thickness_mm",
        ]),
        "source": _find(df, ["Source"]),
        "main_folder": _find(df, ["Main Folder"]),
    }

    if col_map["filler1_name"] is None:
        log.append("ERROR: Could not find a primary filler name column — featurization skipped.")
        result.df = df.copy()
        return result

    if col_map["se_value"] is None:
        log.append("ERROR: Could not find an EMI shielding value column — featurization skipped.")
        result.df = df.copy()
        return result

    converted: list[list] = []
    missing_feature: list[Any] = []
    problem: list[Any] = []
    missing_names_set: set[str] = set()

    for _, sample in df.iterrows():
        try:
            f1_name = _safe_lower(sample.get(col_map["filler1_name"], "none"))
            f2_name = _safe_lower(sample.get(col_map["filler2_name"], "none") if col_map["filler2_name"] else "none")
            f3_name = _safe_lower(sample.get(col_map["filler3_name"], "none") if col_map["filler3_name"] else "none")

            f1_wt = sample.get(col_map["filler1_comp"], 0) if col_map["filler1_comp"] else 0
            f2_wt = sample.get(col_map["filler2_comp"], 0) if col_map["filler2_comp"] else 0
            f3_wt = sample.get(col_map["filler3_comp"], 0) if col_map["filler3_comp"] else 0

            c1 = _lookup_canonical(f1_name, canon_df)
            c2 = _lookup_canonical(f2_name, canon_df)
            c3 = _lookup_canonical(f3_name, canon_df)

            v1 = _lookup_features(c1, feat_df)
            v2 = _lookup_features(c2, feat_df)
            v3 = _lookup_features(c3, feat_df)

            real_missing = (
                v1 is None
                or (v2 is None and c2 != "none")
                or (v3 is None and c3 != "none")
            )
            if real_missing:
                if v1 is None and not _is_none_like(f1_name):
                    missing_names_set.add(str(f1_name))
                if v2 is None and c2 != "none" and not _is_none_like(f2_name):
                    missing_names_set.add(str(f2_name))
                if v3 is None and c3 != "none" and not _is_none_like(f3_name):
                    missing_names_set.add(str(f3_name))
                missing_feature.append(sample)
                continue

            # Use zeros for missing secondary/tertiary slots
            v2 = v2 or DEFAULT_FEATURE
            v3 = v3 or DEFAULT_FEATURE

            source_val = sample.get(col_map["source"], "") if col_map["source"] else ""
            folder_val = sample.get(col_map["main_folder"], "") if col_map["main_folder"] else ""
            se_val = sample.get(col_map["se_value"], np.nan)
            thickness_val = sample.get(col_map["thickness"], np.nan) if col_map["thickness"] else np.nan

            converted.append([
                source_val, folder_val,
                str(f1_name), str(f2_name), str(f3_name),
                *v1, f1_wt,
                *v2, f2_wt,
                *v3, f3_wt,
                thickness_val,
                se_val,
            ])

        except Exception as exc:
            problem.append(sample)
            log.append(f"  Row error: {exc}")

    # Build output DataFrame
    out_cols = [
        "Source", "Main Folder",
        "Filler Name", "Filler 2 Name", "Filler 3 Name",
        "Electrical conductivity", "Shape", "Magnetic permeability", "Density", "Particle size",
        "Filler Composition",
        "Electrical conductivity 2", "Shape 2", "Magnetic permeability 2", "Density 2", "Particle size 2",
        "Filler 2 Composition",
        "Electrical conductivity 3", "Shape 3", "Magnetic permeability 3", "Density 3", "Particle size 3",
        "Filler Composition 3",
        "sample_thickness_value_mm",
        "EMI shielding value",
    ]

    out_df = pd.DataFrame(converted, columns=out_cols)

    # Add Aspect ratio columns (Shape → numeric, matching notebook)
    for suffix in ["", " 2", " 3"]:
        shape_col = f"Shape{suffix}"
        aspect_col = f"Aspect ratio{suffix}"
        out_df[aspect_col] = out_df[shape_col].map(SHAPE_TO_ASPECT).fillna(0)

    result.df = out_df
    result.rows_out = len(out_df)
    result.rows_missing_feature = len(missing_feature)
    result.rows_problem = len(problem)
    result.missing_filler_names = sorted(missing_names_set)

    log.append(
        f"Featurization: {result.rows_in} rows in → {result.rows_out} featurized, "
        f"{result.rows_missing_feature} skipped (filler not in DB), "
        f"{result.rows_problem} errors."
    )
    if result.missing_filler_names:
        log.append(
            f"  Filler names not found in DB: {', '.join(result.missing_filler_names)}"
        )

    return result
