import pandas as pd
import logging

class RetailFeatureEngineer:
    """Worker #2: Responsible for building smart, industry-level business features."""
    
    def __init__(self):
        pass

    def construct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Constructing high-level retail domain features...")
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
    