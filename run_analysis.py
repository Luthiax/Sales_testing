"""Train and evaluate the retail return-rate model.

Model selection happens only on the training partition. The untouched test
partition is used once for the final performance estimate.
"""

import json
import logging
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import RetailDataPipeline
from src.feature_engineering import RetailFeatureEngineer
from src.financial_sim import FinancialRiskSimulator


RANDOM_STATE = 42
TARGET = "return_rate"
# The source describes customer_sentiment as estimated satisfaction, not a
# verified pre-purchase field. Exclude it and its derivative from the primary
# model to avoid an unsupported availability/leakage assumption.
EXCLUDED_FEATURES = ["customer_sentiment", "value_perception"]


def main():
    logging.info("Running return-rate analysis pipeline...")

    data_pipeline = RetailDataPipeline(file_path="RetailStoreProductSalesDataset.csv")
    engineer = RetailFeatureEngineer()
    raw_data = data_pipeline.load_and_clean()
    engineered_data = engineer.construct_features(raw_data)

    X = engineered_data.drop(columns=[TARGET, *EXCLUDED_FEATURES])
    y = engineered_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Scaling is inside the Pipeline so it is refit within every CV fold.
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"ridge__alpha": np.logspace(-3, 3, 25)},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    cv_rmse = -search.best_score_
    best_alpha = search.best_params_["ridge__alpha"]

    print("\n=========================================")
    print("      MODEL PERFORMANCE — CLEAN HOLDOUT  ")
    print("=========================================")
    print(f"Training CV RMSE      : {cv_rmse:.5f}")
    print(f"Selected Ridge alpha : {best_alpha:.6g}")
    print(f"Holdout test R²      : {r2:.4f}")
    print(f"Holdout test RMSE    : {rmse:.5f}")
    print("=========================================\n")

    # Save the complete preprocessing + model object, not the Ridge step alone.
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    artifact_path = model_dir / "ridge_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": model,
            "feature_names": list(X.columns),
            "target": TARGET,
            "excluded_features": EXCLUDED_FEATURES,
            "random_state": RANDOM_STATE,
        },
        artifact_path,
    )
    logging.info("Saved reproducible model bundle to %s", artifact_path)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    cv_results = pd.DataFrame(
        {
            "ridge_alpha": search.cv_results_["param_ridge__alpha"],
            "mean_cv_rmse": -search.cv_results_["mean_test_score"],
            "std_cv_rmse": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort_values("rank")
    cv_results.to_csv(artifacts_dir / "ridge_cv_results.csv", index=False)
    metrics = {
        "model": "Ridge",
        "selected_alpha": float(best_alpha),
        "training_cv_rmse": float(cv_rmse),
        "holdout_r2": float(r2),
        "holdout_rmse": float(rmse),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "excluded_features": EXCLUDED_FEATURES,
        "random_state": RANDOM_STATE,
    }
    artifact_payload = {
        "environment": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
            "seaborn": version("seaborn"),
        },
        "metrics": metrics,
    }
    with (artifacts_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact_payload, handle, indent=2)

    coefficients = pd.DataFrame(
        {
            "Feature": X.columns,
            "Coefficient": model.named_steps["ridge"].coef_,
        }
    ).sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(10, 5))
    colors = ["#5cb85c" if value < 0 else "#d9534f" for value in coefficients["Coefficient"]]
    sns.barplot(
        x="Coefficient",
        y="Feature",
        data=coefficients,
        hue="Feature",
        palette=dict(zip(coefficients["Feature"], colors)),
        legend=False,
    )
    plt.title(
        "Standardized Ridge Coefficients (Associations, Not Causal Effects)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Coefficient on Standardized Feature")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("coefficients_plot.png", dpi=300)
    plt.close()

    residuals = pd.Series(y_test.to_numpy() - y_pred, index=X_test.index)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#2b5c8f", bins=30)
    plt.title("Holdout Residual Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Actual Return Rate − Predicted Return Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("residuals_plot.png", dpi=300)
    plt.close()

    diagnostic_df = X_test.copy()
    diagnostic_df["absolute_error"] = residuals.abs()
    diagnostic_df["price_tier"] = pd.qcut(
        diagnostic_df["price"],
        q=3,
        labels=["Low-Cost", "Mid-Tier", "Premium"],
    )
    print("\n--- HOLDOUT MEAN ABSOLUTE ERROR BY PRICE TIER ---")
    print(diagnostic_df.groupby("price_tier", observed=False)["absolute_error"].mean())

    # This is a standalone sensitivity scenario. It is not an observed model impact.
    simulator = FinancialRiskSimulator(capex=50_000, opex_annual=15_000, discount_rate=0.10)
    base_npv = simulator.simulate_3year_npv(
        return_handling_cost=20.0,
        target_reduction=0.10,
        total_returns_annual=30_000,
    )
    breakeven = simulator.breakeven_reduction(
        return_handling_cost=20.0,
        total_returns_annual=30_000,
    )
    print("\n--- ILLUSTRATIVE BUSINESS SCENARIO (NOT MEASURED IMPACT) ---")
    print(f"3-year NPV: ${base_npv:,.2f}")
    print(f"Breakeven reduction: {breakeven:.2%}")


if __name__ == "__main__":
    main()
