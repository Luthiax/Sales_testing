# Generated evaluation artifacts

`python run_analysis.py` writes:

- `ridge_cv_results.csv` — every candidate Ridge alpha and its training cross-validation RMSE;
- `test_metrics.json` — the selected alpha and one-time holdout metrics.

These small text artifacts are committed as reproducibility evidence. The binary model bundle is excluded from Git.
