"""Settings used to generate the paper figures (subset of the EMI workbench config)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "data" / "assets"
FILLER_FEATURE_VECTORS_PATH = ASSETS_DIR / "filler_feature_vectors.xlsx"
CANONIZATION_PATH = ASSETS_DIR / "canonization.xlsx"

# Row filters applied after featurization (thickness in mm; SE per mm in dB/mm). None disables a bound.
MIN_THICKNESS_MM = 0.1
MAX_THICKNESS_MM = 100.0
MIN_SE_PER_MM = 0.0
MAX_SE_PER_MM = 300.0  # removes 4 extreme samples (356, 414, 530, 620 dB/mm)

# Duplicate-group filter/averaging (services.selection_filters); off for the paper.
SELECTION_FILTER_ENABLED = False

# Pooled out-of-fold cross-validation folds.
CV_FOLDS = 20

# Models run by services.modeling.run_model_comparison (not used by the figure script).
ENABLED_MODELS = ["Linear Regression", "Ridge Regression", "Random Forest", "XGBoost",
                  "GP Regression (botorch)", "Set GP"]

# Optional TabPFN token, read from the environment only. Never commit a token.
import os
TABPFN_API_KEY = os.environ.get("TABPFN_TOKEN", "")
