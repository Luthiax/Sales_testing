import pandas as pd
import logging

class RetailFeatureEngineer:
    """Builds business-driven derived features from the raw retail signals."""
    
    def __init__(self):
        pass

    def construct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Constructing derived retail features...")
        df = df.copy()
        
        # 1. Price Competitiveness Ratio
        df['price_competitiveness'] = df['price'] / df['competitor_price']
        
        # 2. Real Discount Ratio
        df['real_discount_ratio'] = df['discount'] / df['price']
        
        # 3. Marketing Footfall Efficiency
        df['marketing_efficiency'] = df['footfall'] / df['ad_spend']
        
        # 4. Value Perception Metric
        df['value_perception'] = df['customer_sentiment'] / df['price']
        
        logging.info(f"Feature engineering complete. Total features now: {df.shape[1]}")
        return df
    