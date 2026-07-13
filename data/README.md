# Dataset — Retail Store Product Sales

**Source:** [Kaggle — "Retail Store Product Sales Dataset"](https://www.kaggle.com/datasets) (15,000 rows × 11 columns).
**License:** Kaggle Community License — used here for educational / portfolio purposes only.
**File:** `../RetailStoreProductSalesDataset.csv` (kept out of git via `.gitignore`; download from Kaggle to reproduce).

---

## Column dictionary

| Column | Type | Range | Meaning |
|---|---|---|---|
| `price` | float | ~$45–$55 | Unit price of the product (USD) |
| `discount` | float | ~5–7 | Discount applied at time of purchase |
| `promotion_intensity` | float | ~0–5 | Strength of the promotional campaign on the product |
| `footfall` | float | ~150–400 | Number of shoppers passing the product display |
| `ad_spend` | float | ~$2,400–$2,700 | Local advertising spend for the product |
| `competitor_price` | float | ~$43–$57 | Closest competitor's price for the same category |
| `stock_level` | float | ~900–1,400 | Units in stock at time of sale |
| `weather_index` | float | 1–10 | Daily weather favourability score (higher = nicer) |
| `customer_sentiment` | float | ~0.4–1.6 | Pre-purchase product rating signal (higher = better) |
| `return_rate` | float | 0–1 | **Target.** Fraction of units returned post-purchase (mean ≈ 6.3 %) |
| `Unnamed: 0` | int | 0–14,999 | Row index — dropped during loading (kaggle artefact) |

---

## Engineered features (built in `src/feature_engineering.py`)

| Feature | Formula | Business meaning |
|---|---|---|
| `price_competitiveness` | `price / competitor_price` | >1 = priced above market |
| `real_discount_ratio` | `discount / price` | True depth of the discount |
| `marketing_efficiency` | `footfall / ad_spend` | Shoppers reached per $ of ad spend |
| `value_perception` | `customer_sentiment / price` | Sentiment per dollar of price |

---

## Leakage discussion (important for reviewers)

Every feature used for prediction is observable **before** a return decision is made:

- `customer_sentiment` is interpreted as a *pre-purchase* product-rating signal (consistent with the Kaggle dataset card). If a reviewer can show it is instead a *post-return* satisfaction score, it should be dropped and the model re-run — this would be a strong sign of leakage and an honest thing to flag.
- `return_rate` is the target and is never used as a feature.
- The train/test split uses `random_state=42` so the split is reproducible; no shuffling happens across the split boundary.

This assumption is stated explicitly in the project notebook and in the main `README.md` so a reviewer can challenge it directly.
