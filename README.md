# Uncertainty-Aware Bus ETA Prediction Under Drift

Reproducibility code for the master's thesis *Uncertainty-Aware Bus ETA Prediction Under Drift with Segment-Level Risk Decomposition* (Düzgün & Abdulsamad Abdulhakim, Jönköping University, 2026).

The pipeline trains XGBoost point predictors at route and segment level, wraps them with conformal prediction via [`calibrated_explanations`](https://github.com/Moffran/calibrated_explanations), and evaluates coverage and interval efficiency on a real GTFS dataset from Astana, Kazakhstan (Mansurova et al., 2025) under temporal drift.

Three experiments map to the thesis research questions:

| Experiment | RQ | Question |
|---|---|---|
| 1 | RQ1 | Does temporal drift break static conformal coverage? |
| 2 | RQ2 | Do online recalibration and adaptive NCMs help? (4×4 factorial) |
| 3 | RQ3 | Can segment-level decomposition narrow intervals and attribute uncertainty? |

## Setup

Python 3.10. From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS/Linux: `source venv/bin/activate`.)

## Data

This repo does **not** redistribute the raw segment-level CSV. Download `segment_level_data.csv` (~160 MB, ~786K records over 55 days, 3 Astana bus routes) from the original publication by Mansurova et al. (2025) and place it at the project root before running Phase 1:

```
reproducibility/
└── segment_level_data.csv   <-- put it here
```

> Source: Mansurova et al. (2025). *Astana bus trajectory dataset*. <!-- TODO: replace with the canonical URL / DOI from the dataset publication -->

The GTFS reference files (`agency.txt`, `routes.txt`, `stops.txt`, `stop_times.txt`, `trips.txt`, `calendar_dates.txt`) under [`data/gtfs_data/`](data/gtfs_data/) are small and are committed to the repo.

## Run order

Execute the notebooks in [`notebooks/`](notebooks/) top to bottom:

1. `Phase1_Preprocessing.ipynb` — duplicate/anomalous-date filtering, drop direction-1 / segment-1 artefacts, route-level aggregation, temporal split. Writes `outputs/processed_data/{segment_cleaned,route_level}.parquet`.
2. `Phase2_Feature_Engineering.ipynb` — temporal, spatial, and 7-day historical features. Writes `outputs/processed_data/{route_features,segment_features}.parquet`.
3. `Phase3_Baseline_XGBoost.ipynb` — randomized hyperparameter search with forward-chaining temporal CV, then full-train retrain. Writes `outputs/models/{route_xgboost_model,segment_xgboost_model,xgboost_hyperparameters}.json`. The search cells are cache-aware: if `xgboost_hyperparameters.json` already exists, the search is skipped and saved params are reused. Delete that file to force a fresh search.
4. `evalaution.ipynb` — single-document reproduction of the thesis results: Static CP (Exp 1), the 16-config grand comparison (Exp 2), and segment-level re-calibration with attribution (Exp 3).

## Layout

```
.
├── data/                  Raw segment CSV and GTFS reference files
├── notebooks/             Phase 1-3 + evaluation notebooks
├── utils/                 Preprocessing, feature engineering, CP wrappers, metrics
├── outputs/
│   ├── processed_data/    Parquet feature files (created by Phase 1-2)
│   ├── models/            Saved XGBoost weights + hyperparameter JSON
│   └── tables/            LaTeX tables consumed by V4.tex
└── V4.tex                 Thesis source
```

## Reproducibility

All random operations use seed 42 (XGBoost, RandomizedSearchCV, KNN difficulty estimator). Hardware used for the reported runs: AMD Ryzen 7 7435HS, 16 GB RAM, CPU only. Phase 3 search takes ~30 minutes; the evaluation notebook (16-config CP loop) takes ~15 minutes.

## License & citation

Code released under the MIT License. If you use this work, please cite the thesis and the underlying dataset:

- Düzgün, K., & Abdulsamad Abdulhakim, M. (2026). *Uncertainty-Aware Bus ETA Prediction Under Drift with Segment-Level Risk Decomposition*. Master's thesis, Jönköping University.
- Mansurova et al. (2025). Astana bus trajectory dataset.
