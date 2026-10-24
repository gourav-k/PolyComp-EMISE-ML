# Code for "Natural Language Processing based Data Curation and ML for Polymer Composite Design"

Gaussian process-based predictive model for EMI Shielding effectiveness of polymer composites.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `prep_handoff.py` | Filters raw data (`data/raw/Consolidated_EMI_Dataset.xlsx`) (dB single-number SE, numeric SE and thickness in mm, 1-2 fillers, all filler amounts in wt%) and writes `prep_outputs/app_input_max2fillers.xlsx` (620 rows). |
| 2 | `services/preprocessing.py` | Exact-duplicate removal, required-column filter, high-variance duplicate removal (std/mean of SE > 0.15), thickness parsing to mm. |
| 3 | `services/featurization.py` | Maps filler names to feature vectors through `data/assets/canonization.xlsx` and `data/assets/filler_feature_vectors.xlsx` (sheet `updated`); drops samples with unknown fillers. |
| 4 | `make_paper_figures.py` | Applies the thickness ([0.1, 100] mm) and SE/mm ((0, 300] dB/mm) filters from `config.py` (282 training rows), then fits the models, runs cross-validation, SHAP, Sobol and the permutation-invariance test, and writes the figures. |

Supporting modules: `services/modeling.py` (Set GP / regular GP with botorch, cross-validation), `services/sensitivity.py` (SHAP and Sobol), `services/selection_filters.py` (optional duplicate filter, disabled for the paper), `config.py` (all thresholds).

## Models

**Inputs and target.** Each sample is a set of two fillers (zero-padded when only one is present), each described by electrical conductivity, aspect ratio, density, particle size and weight fraction (10 features in total). The target is EMI shielding effectiveness divided by sample thickness (dB/mm). Matrix properties and thickness are not model inputs.

**Permutation-invariant (Set) GP.** Ordinary GP kernels treat the two filler blocks as ordered, so swapping filler 1 and filler 2 changes the prediction even though the composite is physically identical. The Set GP removes this by using a set kernel (Gartner et al., 2002) that averages a base kernel over all pairs of filler elements of two samples:

k_set(X, X') = 1 / (|X| |X'|) * sum_i sum_j k_base(x_i, x'_j)

Because the sum runs over all pairs, the kernel (and therefore the posterior mean and variance) does not change when the fillers are reordered. Here the base kernel k_base is an RBF kernel with a per-feature (ARD) lengthscale shared by both fillers, wrapped in a scale kernel; the GP is built with BoTorch and fit by marginal likelihood (`MeanSetKernel` and `_build_set_gp_covar_module` in `services/modeling.py`). Predictions come with a predictive standard deviation, which is used for the +/-1 sigma bars in the parity plot.

**Regular GP (baseline).** A standard BoTorch `SingleTaskGP` on the same 10 features (min-max scaled), with a Matérn kernel. It is accurate but not order-invariant, which is what the permutation test in Fig. 6 demonstrates.

**Evaluation.** Accuracy is the pooled out-of-fold R^2 from 20-fold cross-validation. Permutation invariance is tested on a 70/30 split (197 train, 85 test) by swapping the filler-1 and filler-2 feature blocks of the test samples; the score is the fraction of test points whose prediction changes by less than 0.01. Interpretation uses KernelSHAP and Sobol total-order indices computed on the fitted Set GP (`services/sensitivity.py`).

## Run

```bash
pip install -r requirements.txt
python prep_handoff.py 2          # 2 = maximum number of fillers per sample
python make_paper_figures.py      # full run (cached to .figure_cache.pkl)
python make_paper_figures.py --replot   # redraw from the cache after style edits
```
A full run takes several minutes on a CPU.

## Inputs

The Excel inputs are not stored in this repository (`data/**/*.xlsx` is git-ignored). When available, place them at:

* `data/raw/Consolidated_EMI_Dataset.xlsx` - consolidated dataset (1748 records, 284 articles; from `EMI_Dataset_Handoff.zip`); see `data/raw/ML_Dataset_Handoff.md` for column definitions.
* `data/assets/filler_feature_vectors.xlsx` - filler feature-vector database (electrical conductivity, shape, density, particle size; sheet `updated`).
* `data/assets/canonization.xlsx` - filler-name canonicalization table.

## License

MIT, see `LICENSE`.

## Notes

* The random seed is fixed everywhere; the Sobol analysis uses 16184 Saltelli base samples.
* Cross-validation is pooled out-of-fold with 20 folds (`CV_FOLDS` in `config.py`). The permutation test uses a 70/30 random split (197 train, 85 test).
