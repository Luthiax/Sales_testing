# Predicting Retail Product Return Risk — and Translating It into Dollars

> How much money would a retailer save if it could predict and proactively reduce product return rates? This project answers that question end-to-end: from raw transaction data, to a predictive model, to a 3-year Net Present Value (NPV) estimate of deploying the model.

This is a **business-analytics** project, not a pure ML benchmark. The model is a means to an end — the end is a defensible dollar figure a category manager or COO could act on.

---

## Business question

Product returns cost retailers between **5% and 15% of total sales** in handling, restocking, markdowns and reverse logistics. This project builds a predictive model that flags high-return-risk products from *pre-purchase* signals (price positioning, promotion intensity, marketing efficiency, customer sentiment, etc.), then quantifies the financial upside of acting on those predictions across realistic operating-cost scenarios.

---

## Key results

| Metric | Value |
|---|---|
| Best model — Ridge regression (α = 2.976) | **R² = 0.6938**, RMSE = 0.0099 (return-rate fraction) |
| Target distribution | Mean return rate 6.34 %, 95ᵗʰ-percentile (tail-risk) 9.15 % |
| Top *positive* drivers of return risk (raise returns) | `marketing_efficiency`, `discount`, `ad_spend` |
| Top *negative* drivers (suppress returns) | `price`, `customer_sentiment`, `value_perception`, `competitor_price` |
| 3-year NPV @ $20/return, 10 % reduction, 30 k returns/yr | **+$61,908** |
| Breakeven reduction needed at $20/return handling cost | **5.85 %** — i.e. the model only needs to cut returns by ~6 % to pay back its $50 k capex |

**Interpretation for a non-technical stakeholder:** even a conservative 10 % reduction in returns is comfortably profitable at a $20 per-return handling cost, and the project breaks even once the model enables a ~6 % reduction. At lower handling costs the bar is higher ($15 → 7.8 %, $10 → 11.7 %), which sharpens where the model is worth deploying.

---

## Project structure

```
Retail/
├── README.md                       # ← you are here
├── requirements.txt                # pinned Python dependencies
├── .gitignore
├── RetailStoreProductSalesDataset.csv  # source data (Kaggle, see data/README.md)
│
├── src/                             # reusable, importable modules
│   ├── __init__.py
│   ├── data_pipeline.py             # load + clean + train/test split + scale
│   ├── feature_engineering.py       # business-driven feature construction
│   └── financial_sim.py             # NPV + breakeven engine
│
├── run_eda.py                       # produces the 4 EDA plots
├── run_analysis.py                  # trains, evaluates, dumps diagnostics
├── run_business_ROI.py              # builds the NPV grid + dashboards FROM the model
│
├── Retail_Return_Risk_Analysis.ipynb  # narrative walk-through (the interview artifact)
│
└── *.png                            # generated charts (referenced in this README)
```

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-3.0-informational)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![xgboost](https://img.shields.io/badge/xgboost-✓-red)
![lightgbm](https://img.shields.io/badge/lightgbm-✓-green)
![matplotlib](https://img.shields.io/badge/matplotlib-✓-blueviolet)

Methodology: ridge regression with standardized features (chosen for explainability via coefficients — see *Model selection rationale* below), 80/20 holdout, 5-fold cross-validation in the notebook benchmark, NPV simulation under a sensitivity grid of handling costs and reduction rates.

---

## How to run

```bash
git clone https://github.com/leonardoflores-data/retail-return-risk.git
cd retail-return-risk
pip install -r requirements.txt

# 1. Exploratory plots
python run_eda.py

# 2. Train model + diagnostics (writes coefficients_plot.png, residuals_plot.png)
python run_analysis.py

# 3. NPV business-case grid (writes business_1_npv_roi_matrix.png, etc.)
python run_business_ROI.py
```

Open `Retail_Return_Risk_Analysis.ipynb` for the full narrative walk-through including the model benchmark (Ridge vs. Random Forest vs. XGBoost vs. LightGBM).

---

## Dataset

Source: **Kaggle — "Retail Store Product Sales Dataset"** (15,000 rows × 11 columns). Used under Kaggle's community license for educational/portfolio purposes. See `data/README.md` for the full column dictionary, encoding notes and a leakage discussion.

---

## Modeling choices — explained for reviewers

**Why Ridge and not the tuned XGBoost / LightGBM?**
In the benchmark notebook, the regularized linear model was within ~0.01 R² of the tree ensembles. For a *business-deployment* context, however, explainability via coefficients (which factors drive returns and in which direction) is more important than squeezing an extra fraction of a percent of R². Ridge lets a category manager read "discount raises returns by β" directly off a bar chart; a tree model requires SHAP to do the same. The 0.01 R² trade was therefore not worth the loss of stakeholder trust.

**Avoiding leakage.**
Each engineered feature uses only signals observable *before* a return decision is made. `customer_sentiment` is treated as a pre-purchase product rating (consistent with the dataset documentation on Kaggle); this assumption is stated explicitly in the notebook so a reviewer can challenge it.

**Adopting a 3-year horizon for NPV.**
Retail analytics tooling typically depreciates over ~3 years — capex of $50,000 covers data infrastructure + model development; opex of $15,000/yr covers monitoring + retraining. The 10 % discount rate reflects a mid-cap retailer's WACC.

---

## About me

I'm **Leonardo Flores**, a bilingual (English/Spanish) business analytics professional based in Lima, Peru, with a specialization in Business Analytics and AI. This project is part of my data-analytics portfolio — see my other repositories for cohort/retention analysis, A/B testing, customer segmentation, and an automation pipeline built on Peruvian public data.

🔗 [LinkedIn placeholder — add link] · 📧 leonardo.[your-email]@gmail.com

