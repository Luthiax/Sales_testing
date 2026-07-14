# Retail Return-Rate Modeling — Synthetic Portfolio Case Study

This project demonstrates a defensible analytics workflow: inspect a dataset, define what can and cannot be predicted, select a model using training data only, evaluate once on a holdout set, and keep predictive performance separate from an illustrative financial scenario.

> **Scope:** the source is a **synthetic simulation dataset**, not retailer transaction data. Results show performance within that simulation. They do not establish real-world accuracy, causal effects, or realized savings.

## Decision question

Can simulated product-level signals help estimate a simulated return rate, and under what operating assumptions could a return-reduction initiative have a positive three-year NPV?

The two parts answer different questions:

1. **Predictive analysis:** how accurately can Ridge regression estimate `return_rate` on an untouched holdout sample?
2. **Business-case sensitivity:** if a retailer achieved an assumed reduction in annual returns, what would the NPV be at different handling costs?

The model does **not** prove that it will cause the assumed reduction.

## Clean-run results

Reproduced locally with Python 3.12.10 using the declared dependencies:

| Metric | Conservative primary model |
|---|---:|
| Model | Ridge |
| Selected `alpha` | 10 |
| Five-fold training CV RMSE | 0.01563 |
| Untouched holdout R² | **0.2213** |
| Untouched holdout RMSE | **0.01582** |

The evaluation design ensures that:

- scaling occurs inside each cross-validation fold;
- Ridge `alpha` is selected using only the training partition;
- the test partition is used once for final evaluation;
- the saved artifact contains preprocessing and the model together;
- `customer_sentiment` and its derivative are excluded from the primary model because the source does not verify that sentiment is available before the return outcome.

The earlier public R² of about 0.69 is withdrawn. It depended heavily on treating the ambiguous satisfaction field as a safe pre-purchase predictor. The conservative R² of 0.2213 is weaker but answers a more defensible question. It also shows that the available simulated operational fields explain only a limited share of return-rate variation.

## Illustrative financial scenario

The financial calculation is reproducible, but it is a scenario—not measured impact.

| Assumption | Base case |
|---|---:|
| Annual returns | 30,000 |
| Average handling cost | USD 20 per return |
| Assumed relative reduction | 10% |
| Up-front cost | USD 50,000 |
| Annual operating cost | USD 15,000 |
| Discount rate / horizon | 10% / 3 years |
| Resulting 3-year NPV | **USD 61,908** |
| Breakeven reduction | **5.85%** |

Formula:

```text
annual gross savings = annual returns × cost per return × assumed reduction
NPV = -capex + Σ[(annual gross savings - annual opex) / (1 + discount rate)^year]
```

These inputs are illustrative. A real proposal would need retailer-specific return volume, fully loaded return cost, intervention cost, adoption, and an experiment or pilot measuring incremental reduction.

## Data

Source: [Kaggle — Retail Store Product Sales Simulation Dataset](https://www.kaggle.com/datasets/mabubakrsiddiq/retail-store-product-sales-simulation-dataset), 15,000 simulated rows. The dataset page identifies the data as synthetic and lists an Apache 2.0 license.

Important limitations:

- no product, store, customer, transaction, or date identifiers;
- each row's real operational grain cannot be verified beyond the simulation description;
- `customer_sentiment` is described as estimated satisfaction, not a confirmed pre-purchase feature;
- relationships may reflect the generator's formula rather than retail behavior;
- a random split measures interpolation within this simulation, not performance across future stores or time periods.

See [`data/README.md`](data/README.md) for the observed schema, ranges, and quality checks.

## Methodology

1. Drop the exported row-index column.
2. Create four ratio features from the simulated fields.
3. Exclude `customer_sentiment` and `value_perception` from the primary model.
4. Create a reproducible 80/20 train/test split.
5. Use a scikit-learn `Pipeline` so standardization is learned separately inside each cross-validation fold.
6. Select Ridge regularization strength with five-fold CV on training data only.
7. Evaluate R² and RMSE once on the untouched test set.
8. Report standardized coefficients as **model associations**, not causal drivers.
9. Calculate NPV independently across explicit scenario assumptions.

Ridge is appropriate here as a transparent regularized baseline. It can handle correlated inputs better than ordinary least squares, but correlated raw and ratio features still make individual coefficients unstable. Coefficient signs and sizes should therefore not be presented as business causality.

## Run locally

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/Luthiax/Sales_testing.git
cd Sales_testing
python -m venv .venv

# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the source CSV from the Kaggle link above and place it at:

```text
RetailStoreProductSalesDataset.csv
```

Then run:

```bash
python run_eda.py
python run_analysis.py
python run_business_ROI.py
python -m unittest discover -s tests
```

`run_analysis.py` writes the complete preprocessing/model bundle to `models/ridge_pipeline.joblib`, cross-validation evidence to `artifacts/ridge_cv_results.csv`, and final metrics to `artifacts/test_metrics.json`. Generated binaries and the raw CSV are excluded from Git.

## Repository map

```text
Retail/
├── README.md
├── requirements.txt
├── data/README.md
├── artifacts/
│   ├── ridge_cv_results.csv
│   └── test_metrics.json
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   └── financial_sim.py
├── tests/test_financial_sim.py
├── run_eda.py
├── run_analysis.py
├── run_business_ROI.py
└── Retail_Return_Risk_Analysis.ipynb
```

## What this project demonstrates

- leakage-aware model evaluation;
- honest separation of prediction, causality, and financial assumptions;
- reproducible preprocessing and model packaging;
- business sensitivity analysis with visible formulas;
- clear communication of data limitations.

It does not demonstrate production deployment, live scoring, causal return reduction, or validated retailer ROI.

## Interview explanation

The concise version:

> “I used a synthetic dataset to demonstrate a rigorous modeling workflow, not to claim real retail performance. I excluded an ambiguous satisfaction field from the primary model, tuned Ridge only inside the training data, and evaluated once on a holdout. Separately, I built a transparent NPV sensitivity model. The USD 61.9k is an illustrative scenario under stated assumptions, not savings proven by the ML model. With real data, my next step would be time-based validation followed by a controlled operational pilot.”

## About

I'm **Leonardo Flores**, a bilingual English/Spanish operations and business-analytics professional based in Lima, Peru. This project is part of a portfolio focused on translating analytical work into clear business decisions while stating evidence limits honestly.

[LinkedIn](https://www.linkedin.com/in/leonardo-floresg/) · [GitHub](https://github.com/Luthiax)
