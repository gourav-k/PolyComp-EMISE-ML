"""Convert EMI_Dataset_Handoff/Consolidated_EMI_Dataset.xlsx into an app-ready workbook.

Run from anywhere:  python prep_handoff.py [max_fillers]   (default 3; the paper uses 2)
Outputs go to prep_outputs/.
"""
from pathlib import Path

import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "raw" / "Consolidated_EMI_Dataset.xlsx"
CANON = HERE / "data" / "assets" / "canonization.xlsx"
OUT = HERE / "prep_outputs"

BASES = {"wt%"}  # composition bases to keep (the app has one composition number per filler)
MAX_FILLERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # app supports filler slots 1-3 only
SUFFIX = "" if MAX_FILLERS == 3 else f"_max{MAX_FILLERS}fillers"
OUT_NAME = f"app_input{SUFFIX}.xlsx"
UNMATCHED_NAME = f"unmatched_fillers{SUFFIX}.csv"
TO_MM = {"mm": 1.0, "um": 1e-3, "μm": 1e-3, "µm": 1e-3, "cm": 10.0, "nm": 1e-6}


def phase1_se_filter(df):
    """Keep dB shielding-effectiveness rows reported as a single number."""
    keep = (
        (df["SE metric"] == "Shielding effectiveness (dB)")
        & (df["SE unit"].str.strip().str.lower() == "db")
        & (df["SE value interpretation"] == "Single number")
    )
    return df[keep].copy()


def phase2_numeric(df):
    """SE -> float; thickness -> mm. Rows that don't parse are dropped."""
    df["Shielding Effectiveness dB"] = pd.to_numeric(df["SE value"], errors="coerce")
    factor = df["Thickness unit"].str.strip().str.lower().map(TO_MM)
    df["Thickness_mm"] = pd.to_numeric(df["Thickness value"], errors="coerce") * factor
    return df[df["Shielding Effectiveness dB"].notna() & (df["Thickness_mm"] > 0)].copy()


def phase3_fillers(df):
    """1-3 fillers, all with a numeric amount in an allowed basis; rename to app columns."""
    n = df[[f"Filler {i}" for i in range(1, 6)]].notna().sum(axis=1)
    df = df[(n >= 1) & (n <= MAX_FILLERS)].copy()
    ok = pd.Series(True, index=df.index)
    for i in range(1, MAX_FILLERS + 1):
        df[f"Filler {i} amount"] = pd.to_numeric(df[f"Filler {i} amount"], errors="coerce")
        present = df[f"Filler {i}"].notna()
        good = df[f"Filler {i} amount"].notna() & df[f"Filler {i} basis"].isin(BASES)
        ok &= ~present | good
    df = df[ok]
    for i in range(1, MAX_FILLERS + 1):
        df = df.rename(columns={f"Filler {i}": f"Filler {i} Name", f"Filler {i} amount": f"Filler {i} Composition"})
    return df


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_excel(SRC)
    print(f"read {len(df)} rows")

    df = phase1_se_filter(df)
    print(f"phase 1 (dB, single number): {len(df)}")
    df.to_csv(OUT / "phase1_se_filtered.csv", index=False)

    df = phase2_numeric(df)
    print(f"phase 2 (numeric SE, thickness in mm): {len(df)}")

    df = phase3_fillers(df)
    print(f"phase 3 (1-{MAX_FILLERS} fillers, basis in {sorted(BASES)}): {len(df)}")

    df = df.rename(columns={"Source collection": "Source", "Record ID": "Main Folder"})
    cols = (
        ["Source", "Main Folder", "DOI"]
        + [c for i in range(1, MAX_FILLERS + 1) for c in (f"Filler {i} Name", f"Filler {i} Composition")]
        + ["Thickness_mm", "Shielding Effectiveness dB"]
    )
    df[cols].to_excel(OUT / OUT_NAME, index=False)

    # Filler names the app's canonization table can't resolve -> those rows get dropped at featurization.
    known = set(pd.read_excel(CANON)["Name annotated DB"].dropna().str.strip().str.lower())
    names = pd.concat([df[f"Filler {i} Name"] for i in range(1, MAX_FILLERS + 1)]).dropna().str.strip().str.lower()
    unmatched = names[~names.isin(known)].value_counts().rename_axis("filler").reset_index(name="count")
    unmatched.to_csv(OUT / UNMATCHED_NAME, index=False)
    print(f"wrote {OUT / OUT_NAME}; {len(unmatched)} filler names missing from canonization.xlsx")


if __name__ == "__main__":
    main()
