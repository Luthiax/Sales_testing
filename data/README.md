# Data card — Retail Store Product Sales Simulation

**Source:** [Kaggle — Retail Store Product Sales Simulation Dataset](https://www.kaggle.com/datasets/mabubakrsiddiq/retail-store-product-sales-simulation-dataset)

**Source description:** synthetic/simulated retail data

**Source license:** Apache 2.0, as listed on the dataset page

**Local file:** `../RetailStoreProductSalesDataset.csv` (excluded from Git)

## Local integrity snapshot

The supplied CSV contains 15,000 rows, 10 analytical columns, and one exported index column. The current local copy has no missing values and no exact duplicate rows.

| Column | Observed local range | Conservative interpretation |
|---|---:|---|
| `price` | 20.2126 to 79.2481 | Simulated product price |
| `discount` | 0 to 18.7577 | Simulated discount field; unit is not established locally |
| `promotion_intensity` | -1.0497 to 6.9667 | Simulated promotion index |
| `footfall` | 70.1524 to 348.7952 | Simulated footfall measure |
| `ad_spend` | 2,097.0203 to 2,764.8192 | Simulated advertising-spend field |
| `competitor_price` | 16.6570 to 85.8046 | Simulated competitor price |
| `stock_level` | 1,083.6779 to 1,301.0904 | Simulated stock-level measure |
| `weather_index` | 1.8895 to 13.4517 | Simulated weather index |
| `customer_sentiment` | -0.2207 to 1.3473 | Estimated satisfaction/sentiment field |
| `return_rate` | 0 to 0.1860 | **Target:** simulated return-rate fraction |
| `Unnamed: 0` | 0 to 14,999 | Exported row index; removed during loading |

These are observed ranges, not promised business constraints. Negative values in `promotion_intensity` and `customer_sentiment`, plus `weather_index` values above 10, are retained because the source does not provide validated domain bounds.

## Engineered fields

| Feature | Formula | Intended interpretation |
|---|---|---|
| `price_competitiveness` | `price / competitor_price` | Relative price position |
| `real_discount_ratio` | `discount / price` | Discount relative to price |
| `marketing_efficiency` | `footfall / ad_spend` | Simulated footfall per spend unit |
| `value_perception` | `customer_sentiment / price` | Sentiment relative to price |

Ratio features share information with their source columns. Ridge regularization reduces instability, but individual coefficient values remain sensitive to multicollinearity.

## Feature-availability and leakage decision

The source describes `customer_sentiment` as estimated customer satisfaction. It does not establish that this measure exists before purchase or before a return occurs. Treating it as a pre-purchase product rating would be an unsupported assumption.

For that reason, the primary model excludes:

- `customer_sentiment`;
- `value_perception`, because it is derived from `customer_sentiment`.

`return_rate` is always excluded from predictors. The exported row index is also removed.

## Limits on interpretation

- There are no product, store, customer, transaction, or date identifiers.
- The dataset cannot support claims about future weeks, unseen stores, SKU targeting, customer-level decisions, or temporal drift.
- A random holdout is the only practical split supported by the available schema, but it estimates performance only within this simulation.
- Correlations and fitted coefficients describe statistical association; they do not identify causal levers.
- The data cannot validate the separate assumptions used in the NPV scenario.

To reproduce the project, download the CSV from the source link and keep its filename as `RetailStoreProductSalesDataset.csv` in the repository root.
