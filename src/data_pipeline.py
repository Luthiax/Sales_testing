"""Loading and minimal cleaning for the synthetic retail dataset."""

import logging

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class RetailDataPipeline:
    """Load the CSV and remove its exported row-index column.

    Train/test splitting and preprocessing live with the model code so the
    scikit-learn Pipeline can prevent cross-validation leakage.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_clean(self) -> pd.DataFrame:
        logging.info("Loading synthetic dataset from: %s", self.file_path)
        df = pd.read_csv(self.file_path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            logging.info("Removed exported row-index column 'Unnamed: 0'.")

        required = {
            "price",
            "discount",
            "promotion_intensity",
            "footfall",
            "ad_spend",
            "competitor_price",
            "stock_level",
            "weather_index",
            "customer_sentiment",
            "return_rate",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if df.empty:
            raise ValueError("Dataset contains no rows.")
        if df[list(required)].isna().any().any():
            raise ValueError("Required analytical columns contain missing values.")

        return df
