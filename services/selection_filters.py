"""
Per-selection dataset filtering.

Applied once on the Select page when the user commits an input/output choice.
After this runs, modeling.py and bo.py operate on the already-cleaned frame
and perform no further row-level filtering or aggregation.

Filters (in order):
  1. Drop rows where the chosen output column is NaN / non-numeric.
  2. Fill NaN in chosen numeric inputs with 0.0 (matches notebook recipe).
  3. Drop rows whose duplicate-feature group has std(target)/|mean(target)| > 0.20.
  4. Aggregate duplicate feature rows: per unique (input_cols) tuple, replace
     the target with the per-group mean. This is the same aggregation the GP
     fit used to do internally; doing it here means modeling and BO see one
     row per unique input combination.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


HIGH_VAR_RATIO_THRESHOLD = 0.20


def apply_selection_filters(
    df: pd.DataFrame,
    input_cols: list[str],
    output_col: str,
    ratio_threshold: float = HIGH_VAR_RATIO_THRESHOLD,
    filter_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (filtered_df, summary). Operates on numeric inputs only.

    The returned DataFrame keeps every column from the original, but rows are
    pruned and duplicate feature combinations are collapsed to one row using
    per-group target means. Non-input/non-output columns of the surviving
    "first" row of each group are kept as-is.
    """
    summary: dict[str, Any] = {
        "rows_in": len(df),
        "rows_after_target_dropna": 0,
        "rows_high_var_removed": 0,
        "rows_aggregated_away": 0,
        "rows_out": 0,
        "input_cols": list(input_cols),
        "output_col": output_col,
    }

    if not input_cols or not output_col or output_col not in df.columns:
        summary["rows_out"] = len(df)
        return df.copy(), summary

    numeric_inputs = [c for c in input_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_inputs or not pd.api.types.is_numeric_dtype(df[output_col]):
        summary["rows_out"] = len(df)
        summary["numeric_inputs"] = numeric_inputs
        return df.copy(), summary

    work = df.copy()
    work[numeric_inputs] = work[numeric_inputs].fillna(0.0)
    work = work.dropna(subset=[output_col]).reset_index(drop=True)
    summary["rows_after_target_dropna"] = len(work)

    # High-variance group filter
    if filter_duplicates and len(work) > 0:
        stats = (
            work.groupby(numeric_inputs, dropna=False)[output_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(0.0)
        stats["ratio"] = stats["std"] / stats["mean"].abs().clip(lower=1e-12)
        bad = stats.query("count > 1 and ratio > @ratio_threshold")
        if not bad.empty:
            bad_keys = set(map(tuple, bad[numeric_inputs].to_numpy()))
            keep_mask = ~work[numeric_inputs].apply(tuple, axis=1).isin(bad_keys)
            summary["rows_high_var_removed"] = int((~keep_mask).sum())
            work = work.loc[keep_mask].reset_index(drop=True)

    # Aggregate duplicate feature rows: per unique input combo, replace target
    # with the group mean and keep the first row's non-input columns.
    rows_before_agg = len(work)
    if filter_duplicates and len(work) > 0:
        target_mean = (
            work.groupby(numeric_inputs, dropna=False)[output_col]
            .mean()
            .rename(output_col)
        )
        first_rows = (
            work.groupby(numeric_inputs, dropna=False, as_index=False)
            .first()
        )
        first_rows = first_rows.drop(columns=[output_col])
        work = first_rows.merge(target_mean.reset_index(), on=numeric_inputs, how="left")
        # Restore original column order
        original_order = [c for c in df.columns if c in work.columns]
        work = work[original_order].reset_index(drop=True)

    summary["rows_aggregated_away"] = rows_before_agg - len(work)
    summary["rows_out"] = len(work)
    summary["numeric_inputs"] = numeric_inputs
    return work, summary
