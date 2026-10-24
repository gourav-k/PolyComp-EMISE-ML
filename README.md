# Code for "Natural Language Processing based Data Curation and ML for Polymer Composite Design"

Regenerates the machine-learning figures of the paper (Fig. 4a-e and Fig. 6a-d) from the consolidated EMI shielding dataset.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `prep_handoff.py` | Filters `data/raw/Consolidated_EMI_Dataset.xlsx` (dB single-number SE, numeric SE and thickness in mm, 1-2 fillers, all filler amounts in wt%) and writes `prep_outputs/app_input_max2fillers.xlsx` (620 rows). |
| 2 | `services/preprocessing.py` | Exact-duplicate removal, required-column filter, high-variance duplicate removal (std/mean of SE > 0.15), thickness parsing to mm. |
| 3 | `services/featurization.py` | Maps filler names to feature vectors through `data/assets/canonization.xlsx` and `data/assets/filler_feature_vectors.xlsx` (sheet `updated`); drops samples with unknown fillers. |
| 4 | `make_paper_figures.py` | Applies the thickness ([0.1, 100] mm) and SE/mm ((0, 300] dB/mm) filters from `config.py` (282 training rows), then fits the models, runs cross-validation, SHAP, Sobol and the permutation-invariance test, and writes the figures. |

Supporting modules: `services/modeling.py` (Set GP / regular GP with botorch, cross-validation), `services/sensitivity.py` (SHAP and Sobol), `services/selection_filters.py` (optional duplicate filter, disabled for the paper), `config.py` (all thresholds).

## Run

```bash
pip install -r requirements.txt
python prep_handoff.py 2          # 2 = maximum number of fillers per sample
python make_paper_figures.py      # full run (cached to .figure_cache.pkl)
python make_paper_figures.py --replot   # redraw from the cache after style edits
```

Outputs go to `paper-figures/` (SVG and PNG for every panel, plus `figure_stats.json` with R^2 values, split sizes, Sobol indices and row counts at each filtering step). A full run takes several minutes on a CPU.

## Inputs

The Excel inputs are not stored in this repository (`data/**/*.xlsx` is git-ignored). Place them at:

* `data/raw/Consolidated_EMI_Dataset.xlsx` - consolidated dataset (1748 records, 284 articles; from `EMI_Dataset_Handoff.zip`); see `data/raw/ML_Dataset_Handoff.md` for column definitions.
* `data/assets/filler_feature_vectors.xlsx` - filler feature-vector database (electrical conductivity, shape, density, particle size; sheet `updated`).
* `data/assets/canonization.xlsx` - filler-name canonicalization table.

## License

MIT, see `LICENSE`.

## Notes

* The random seed is 42 everywhere; the Sobol analysis uses 16184 Saltelli base samples.
* Cross-validation is pooled out-of-fold with 20 folds (`CV_FOLDS` in `config.py`). The permutation test uses a 70/30 random split (197 train, 85 test).
* Fonts: figures use Arial if installed, otherwise DejaVu Sans.
