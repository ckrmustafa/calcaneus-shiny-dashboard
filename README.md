# Calcaneus Morphometry — Machine Learning Dashboard

Publication-grade R/Shiny pipeline for classifying calcaneal clinical findings
(Haglund deformity, heel spur, Achilles tendon pathology) from morphometric
measurements (Gissane angle, Böhler angle, age, sex).

## Files

| File | Purpose |
|------|---------|
| `data_prep.R` | Reads the raw `.xlsx`/`.csv`, standardizes Turkish/English labels (VAR/YOK, YES/NO, K/E, F/M), fixes typos, applies clinical plausibility filters and writes `calcaneus_clean.csv`. |
| `app.R` | Shiny dashboard: 10 caret classifiers, 5-fold CV, in-pipeline K-NN imputation (leakage-free), SMOTE/ROSE/up/down-sampling, train/test metrics (CSV export), 600-DPI figures (ROC, comparison bars, feature importance, white-box CART), a correlation module (pairwise Pearson/Spearman coefficients with p-values, significance-starred 600-DPI heatmap and CSV export) and an English documentation tab. |

## Setup

```r
install.packages(c(
  "shiny", "shinydashboard", "DT", "caret", "pROC", "dplyr", "tidyr",
  "ggplot2", "randomForest", "xgboost", "gbm", "e1071", "rpart",
  "rpart.plot", "naivebayes", "nnet", "MASS", "readxl", "readr",
  "stringr", "janitor", "themis", "ROSE"
))
```

## Run

```r
source("data_prep.R")        # 1) produce calcaneus_clean.csv
shiny::runApp("app.R")       # 2) launch the dashboard
```

Place `HAM_VERI.xlsx` in the same folder (or upload any raw file via the
sidebar — the app re-runs the same standardization logic).

## Notes

- Missing values are **not** imputed in `data_prep.R`. In the app, K-NN
  imputation is **fitted on the training partition only** and merely applied
  to the test set, so no information leaks from test into train. (caret runs
  `sampling` before `preProcess` inside each resample, and SMOTE/ROSE reject
  missing values — therefore imputation must precede resampling.)
- Balancing (SMOTE/ROSE/up/down) is applied *within* resamples, so synthetic
  observations never enter validation folds.
- The positive class is "Yes" (first factor level) for Sensitivity/AUC.
- All figures download as 600-DPI PNG; the metrics table downloads as CSV.
- Figures are **title-free** (journals typically require captions in the
  manuscript, not inside the graphic) and use tight, canvas-filling margins.
- The **Multi-Endpoint Batch** tab runs the identical pipeline across all
  binary endpoints in one click and merges them into *composite* figures —
  a multi-panel ROC figure (one panel per endpoint) and an endpoint × model
  test-AUC heatmap — plus a consolidated all-endpoints metrics CSV. This
  keeps the manuscript within journal figure-count limits: one composite
  figure instead of one figure per endpoint.
- The **Prediction (New Case)** tab predicts class and probability for an
  unseen case with every trained model; empty fields are imputed with the
  training-fitted K-NN imputer. Held-out test-set predictions download as CSV
  from the Metrics tab.
- **Every download filename** embeds the target, train/test proportion,
  tuneLength and seed — e.g.
  `roc_curves_600dpi_hd_r_train75_tune3_seed2025_2026-10-03.png` — so each
  figure and table is traceable to the exact settings that produced it.
  Model-specific exports also carry the algorithm name
  (e.g. `feature_importance_RF_...`).

## Correlation module

The **Correlations** tab computes pairwise Pearson or Spearman coefficients
(with two-sided p-values) for all numeric variables of the loaded dataset and
renders a significance-starred heatmap (`* p<0.05, ** p<0.01, *** p<0.001`).
The heatmap downloads as a 600-DPI PNG and the full coefficient table as CSV.

## Data

The de-identified clinical dataset analysed with this pipeline (1,037 records;
sex, age, right/left Gissane and Böhler angles, six binary findings) is
deposited on Zenodo together with an archived copy of this code:

> [Zenodo DOI to be inserted upon deposit]

The raw input file (`HAM_VERI.xlsx` / `HAM_VERI.csv`) is not part of this
repository; place it next to `data_prep.R` locally, or upload any raw file
through the dashboard sidebar.

## Citation

If you use this pipeline, please cite the accompanying manuscript and the
archived code:

> Çakır M, Kaştan Ö, Oral O. Calcaneus morphometry and a reproducible
> machine-learning pipeline (manuscript, 2026). Code archived on Zenodo:
> [Zenodo DOI to be inserted upon deposit].

## License

This project is released under the MIT License (see `LICENSE`).

## Authors

- Mustafa Çakır — İskenderun Technical University
