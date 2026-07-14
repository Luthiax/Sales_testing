"""
run_business_ROI.py
--------------------
Computes and visualises exploratory figures and an illustrative business case.

The NPV grid is a transparent sensitivity analysis based on user-supplied
assumptions. It is not an estimate of measured or model-caused savings.

Outputs
-------
  business_1_npv_roi_matrix.png  — 3-Year NPV sensitivity grid ($000s)
  eda_5_sentiment_decile.png     — Average return rate by customer sentiment decile
  model_1_risk_distribution.png  — Actual return-rate distribution with risk thresholds
"""

import matplotlib
matplotlib.use('Agg')   # headless backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.financial_sim import FinancialRiskSimulator

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ── 0. Load the synthetic dataset ────────────────────────────────────────────
def _load_data():
    """Load the source CSV and remove its exported row-index column."""
    df = pd.read_csv("RetailStoreProductSalesDataset.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# ── 1. NPV Sensitivity Matrix ─────────────────────────────────────────────────
def generate_npv_matrix(simulator: FinancialRiskSimulator, total_returns_annual: int):
    """
    Builds a 4×4 sensitivity grid of 3-Year NPV (in $000s) by varying:
      - handling cost per return : $5, $10, $15, $20
      - relative return reduction : 5 %, 10 %, 15 %, 20 %

    All NPVs are computed by the FinancialRiskSimulator using the same
    capex/opex/discount-rate assumptions as run_analysis.py.
    """
    cost_levels      = [5, 10, 15, 20]
    reduction_levels = [0.05, 0.10, 0.15, 0.20]

    rows = []
    for cost in cost_levels:
        row = []
        for red in reduction_levels:
            npv = simulator.simulate_3year_npv(
                return_handling_cost=cost,
                target_reduction=red,
                total_returns_annual=total_returns_annual,
            )
            row.append(round(npv / 1000, 1))   # convert to $000s
        rows.append(row)

    npv_grid = np.array(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        npv_grid,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        xticklabels=["5% Reduction", "10% Reduction", "15% Reduction", "20% Reduction"],
        yticklabels=["USD 5/return", "USD 10/return", "USD 15/return", "USD 20/return"],
        cbar_kws={"label": "3-Year NPV (USD 000s)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Illustrative 3-Year NPV Sensitivity Grid (USD 000s)\n"
        f"Assumptions: {total_returns_annual:,} annual returns · USD 50k capex · "
        "USD 15k/year opex · 10% discount rate",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Assumed Relative Reduction in Returns")
    ax.set_ylabel("Average Cost per Return (Logistics + Handling)")
    ax.tick_params(axis="y", labelrotation=0)

    plt.tight_layout()
    plt.savefig("business_1_npv_roi_matrix.png", dpi=300)
    plt.close()
    print("Saved  business_1_npv_roi_matrix.png")


# ── 2. Sentiment Decile Plot ──────────────────────────────────────────────────
def generate_decile_plot(df: pd.DataFrame):
    """
    Groups the ACTUAL dataset by customer_sentiment decile and plots
    the mean return_rate for each decile (in %).
    """
    df = df.copy()
    df["sentiment_decile"] = pd.qcut(
        df["customer_sentiment"], q=10, labels=False
    ) + 1                           # 1-indexed so label reads "Decile 1 = Worst"
    summary = (
        df.groupby("sentiment_decile", observed=False)["return_rate"]
        .mean()
        .mul(100)                   # fraction → percentage
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        summary["sentiment_decile"],
        summary["return_rate"],
        marker="o",
        linewidth=2.5,
        color="#e7543d",
        markersize=8,
    )
    ax.set_title(
        "Synthetic Return Rate by Customer Sentiment Decile\n"
        "(Decile 1 = Lowest Sentiment, Decile 10 = Highest Sentiment)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Customer Sentiment Decile")
    ax.set_ylabel("Average Return Rate (%)")
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    plt.savefig("eda_5_sentiment_decile.png", dpi=300)
    plt.close()
    print("Saved  eda_5_sentiment_decile.png")


# ── 3. Return-Rate Distribution ───────────────────────────────────────────────
def generate_risk_distribution(df: pd.DataFrame):
    """Plots the ACTUAL return_rate distribution from the dataset (as %)."""
    return_pct = df["return_rate"] * 100   # fraction → %

    mean_val = return_pct.mean()
    p95_val  = return_pct.quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(return_pct, bins=40, color="#7a8b99", kde=True, ax=ax, edgecolor="white")

    ax.axvline(mean_val, color="#f39c12", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.2f}%")
    ax.axvline(p95_val,  color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Top 5% Risk Threshold (P95): {p95_val:.2f}%")

    ax.set_title("Distribution of Simulated Product Return Rates", fontweight="bold", pad=12)
    ax.set_xlabel("Return Rate (%)")
    ax.set_ylabel("Number of Transactions")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    plt.savefig("model_1_risk_distribution.png", dpi=300)
    plt.close()
    print("Saved  model_1_risk_distribution.png")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading synthetic dataset …")
    df_full = _load_data()

    # Simulator with the same financial parameters as run_analysis.py
    simulator = FinancialRiskSimulator(capex=50_000, opex_annual=15_000, discount_rate=0.10)
    TOTAL_RETURNS = 30_000   # enterprise-scale baseline

    print("\nGenerating exploratory and business-scenario visuals …")
    generate_npv_matrix(simulator, total_returns_annual=TOTAL_RETURNS)
    generate_decile_plot(df_full)
    generate_risk_distribution(df_full)

    # Quick sanity-print of the key number for the README
    npv_base = simulator.simulate_3year_npv(
        return_handling_cost=20.0,
        target_reduction=0.10,
        total_returns_annual=TOTAL_RETURNS,
    )
    print(f"\n  Base-case NPV ($20/return, 10% reduction, {TOTAL_RETURNS:,} returns/yr): "
          f"${npv_base:,.0f}")
    print("\nAll exploratory and scenario outputs generated successfully.")
