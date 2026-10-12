# Greening-Resilience Decoupling and Precipitation Variability Across Global Vegetation

Scientific processing, analysis and plotting code, with small intermediate result tables and figure source data. This repository contains the current MODIS SummaryQA=0/1 analysis, not superseded QA or scaling experiments.

## Included

- `code/`: the scientific scripts and their local dependencies, preserved byte for byte from the analysis workspace.
- `source_data/`: regional trajectories, OLS/HAC statistics, area fractions, RF summaries, VIF retention, PDP curves, lag/window/EVI comparisons, exploratory ENSO results, and spatial-validation tables.
- `figure_inputs/`: compact MATLAB input arrays and the actual unscaled interval arrays used for the main figures.
- `metadata/`: predictor definitions, input templates, recorded software versions and selected verification records.
- `reference_matlab_code/`: original author plotting references, retained for provenance.
- `scripts/`: lightweight integrity checks; these do not retrain models.

## Data and time conventions

MOD13C1 Collection 6.1 NDVI/EVI are handled in physical units without a second scale-factor application. SummaryQA classes 0 (good) and 1 (marginal) are retained; snow/ice, cloudy and historical-estimate categories are excluded. kNDVI is `tanh(NDVI**2)`.

Residual preprocessing uses the common 2000-2025 record. TAC is a robust lagged regression slope, not a newly calculated Pearson correlation. Missing observations retain their collection positions. The 22 window endpoints, 2005-01-01 through 2026-01-01, are labelled by reference years 2004-2025. The primary RF response uses the first 19 endpoints, reference years 2004-2022, to reduce end-of-record moving-mean effects.

The primary RF analysis uses **229,056 forest** and **494,701 non-forest** pixels. Eligibility uses ordinary pixel-wise OLS `P < 0.05`, QA01 missingness below 50%, and finite candidate predictors within the retained original vegetation/missingness domain. Class 1 is non-negative TAC trend (declining TAC-inferred resilience); class 0 is negative trend. IQR and VIF filtering are training-only. VIF threshold 10 excludes P and Ta in the primary refits, leaving 20 fitted predictors.

The source rasters, raw satellite download stacks, trained model binaries, RF pixel-level input tables, complete global pixel caches, account credentials, manuscripts and private reviewer correspondence are **not** included. Public satellite and covariate sources must be obtained under their own terms; their provenance is recorded in `metadata/predictor_dictionary.csv`. This lightweight repository intentionally omits the RF input cohort; reproducing model training requires preparing the source inputs using the included scripts. Citation and dependency rights remain with the respective authors/providers; no new software licence has been assigned by this upload.

## Inspect the included data

```powershell
python scripts/check_release.py
```

The integrity check uses the Python standard library. Only small result tables and compact figure arrays are bundled; raw observations, pixel-level training tables and trained models are not uploaded. No file or repository payload exceeds 100 MB, and no Git LFS is required.

## Workflow

1. Obtain MOD13C1 data with `MOD13C1_NDVI_EVI_QA_local_20261003_GEE.js`; obtain covariates with the ERA5 and non-ERA5 GEE scripts. Set actual paths in the input manifests.
2. Check downloaded bands, dates and grid alignment using `audit_local_MOD13C1_inputs_20261003.py`. Review the robust-regression conformance and preprocessing checks.
3. Run `recompute_QA01_22periods_20261005.py` for unified residuals, TAC, greenness and RF input preparation.
4. Run `train_RF_QA01_unified22_20261005.py` and `pdp_RF_QA01_unified22_20261005.py` for the primary 30 refits per vegetation group and probability PDPs. These are 30 fixed-parameter refits, not 30 independent tuning searches.
5. Run `analyse_plot_QA01_unified22_20261005.py` for the main summaries and HAC analysis. **Then run `integrate_lag_results_20261006.py`** for the fixed four-lag intersection reported in the paper; pairwise-lag summaries are not interchangeable with it.
6. Run `run_RF_QA01_spatial5_20261006.py` and its verification script for nested spatial validation; run the ENSO entry points for the exploratory monthly analysis.
7. Use the MATLAB preparation and plotting scripts, and the final redraw scripts, for figure production. `run_RF_QA01_full22_20261010.py` is the optional full-period RF sensitivity entry point; it does not replace the first-19-period primary model.

The original calculations used Windows and the package versions in `requirements-recorded.txt`. Scripts retain historical QA0 helper names and some author-machine paths. These support modules are invoked by the QA01 entry points with the approved current settings. The path inventory documents required adaptation. Native MATLAB requires a licensed installation; map scripts use Natural Earth coastlines. This is an auditable source snapshot, not a claim of a fresh end-to-end reproduction on every platform.

## Interpretation and uncertainty

Precipitation variability remains among important predictors, but ranks are conditional on the chosen cohort, period and importance estimator. Random-holdout accuracy measures within-grid interpolation. The spatial five-fold analysis uses 500 km projected blocks without a minimum-distance buffer and does not eliminate all spatial dependence. Thirty-refit ranges and block-bootstrap intervals do not represent complete training uncertainty. Regional HAC does not correct individual pixel OLS tests or their derived area fractions. EVI and higher lags reveal sensitivity rather than universal agreement; ENSO results do not establish event causation.

Fig. 1d and Fig. 2 use fivefold display magnification of interval half-widths, as disclosed in the captions. `figure_inputs/` includes the **unmagnified** arrays; do not interpret the enlarged display bands as actual confidence intervals.

## Verification

Scientific source copies are hash-checked; new packaging helpers receive syntax and data-value checks. This upload does not rerun the global satellite preprocessing or RF training. File hashes and the selected numerical verification records are provided for traceability.
