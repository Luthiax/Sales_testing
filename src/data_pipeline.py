import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up clean logging to keep track of operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RetailDataPipeline:
    """Worker #1: Responsible for loading data safely and preparing training sets."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.scaler = StandardScaler()
        
    def load_and_clean(self) -> pd.DataFrame:
        logging.info(f"Loading raw dataset from: {self.file_path}")
        df = pd.read_csv(self.file_path)
        
        # Guard clause: Drop accidental index columns to prevent data leakage
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            logging.info("Removed accidental 'Unnamed: 0' index column to secure data integrity.")
            
        return df

    def split_and_scale(self, df: pd.DataFrame, target_col: str = 'return_rate'):
        """Splits data into train/test sets and scales numerical values perfectly."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit scale on training data ONLY to prevent target/data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to clean DataFrames to keep column names intact
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        logging.info(f"Data successfully split. Train shape: {X_train_scaled_df.shape}, Test shape: {X_test_scaled_df.shape}")
        return X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_test
    