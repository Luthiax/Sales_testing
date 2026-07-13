import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom workers from our src library
from src.data_pipeline import RetailDataPipeline
from src.feature_engineering import RetailFeatureEngineer
from src.financial_sim import FinancialRiskSimulator

def main():
    logging.info("Initializing Master Optimization Pipeline...")
    
    # 1. Pipeline & Engineering Phases
    pipeline = RetailDataPipeline(file_path="RetailStoreProductSalesDataset.csv")
    engineer = RetailFeatureEngineer()
    
    raw_data = pipeline.load_and_clean()
    engineered_data = engineer.construct_features(raw_data)
    
    # Split and scale features cleanly
    X_train, X_test, y_train, y_test, raw_test_features = pipeline.split_and_scale(engineered_data)
    
    # 2. Model Training Phase
    logging.info("Training regularized Champion Ridge Regression Model (alpha=2.976)...")
    model = Ridge(alpha=2.976, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Validation Metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n=========================================")
    print("      MODEL TEST PERFORMANCE METRICS     ")
    print("=========================================")
    print(f"Holdout Test R² Score : {r2:.4f} (Explains {r2*100:.1f}% of variance)")
    print(f"Holdout Test RMSE     : {rmse:.5f}")
    print("=========================================\n")
    
    # 4. Save the Model Asset for Production Use
    joblib.dump(model, 'models/champion_ridge_model.pkl')
    logging.info("Model saved to disk at 'models/champion_ridge_model.pkl'")
    
    # 5. NEW: Generate & Save Feature Importance (Coefficients) Graph
    logging.info("Generating Model Feature Coefficients plot...")
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(10, 5))
    # Green bars suppress return rate risk, Red bars accelerate it
    colors = ['#5cb85c' if x < 0 else '#d9534f' for x in coefficients['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette=colors)
    plt.title('Ridge Regression Coefficients (Drivers of Return Rate Risk)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Coefficient Value (Impact Weight)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('coefficients_plot.png', dpi=300) # Saves directly to your workspace directory
    plt.close()
    logging.info("Saved 'coefficients_plot.png' to your workspace.")

    # 6. NEW: Generate & Save Model Error Diagnostics (Residuals) Graph
    logging.info("Generating Model Error Diagnostics (Residuals) plot...")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='#2b5c8f', bins=30)
    plt.title('Model Residuals Distribution (Error Analysis)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Prediction Error (Actual Rate - Predicted Rate)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=300) # Saves directly to your workspace directory
    plt.close()
    logging.info("Saved 'residuals_plot.png' to your workspace.")

    # 7. Segmented Error Diagnostics Analysis
    diagnostic_df = raw_test_features.copy()
    diagnostic_df['absolute_error'] = abs(residuals)
    diagnostic_df['price_tier'] = pd.qcut(diagnostic_df['price'], q=3, labels=['Low-Cost', 'Mid-Tier', 'Premium'])
    
    print("\n--- MEAN ABSOLUTE ERROR BY PRICE TIER ---")
    print(diagnostic_df.groupby('price_tier', observed=False)['absolute_error'].mean().to_string(index=False))
    print("------------------------------------------\n")
    
    # 8. Corporate Financial NPV Analysis (Simulating 30,000 annual returns for Enterprise Scale)
    simulator = FinancialRiskSimulator(capex=50000, opex_annual=15000, discount_rate=0.10)
    simulator.run_breakeven_analysis(total_returns_annual=30000)
    
    npv_20 = simulator.simulate_3year_npv(return_handling_cost=20.0, target_reduction=0.10, total_returns_annual=30000)
    print(f"\nFinal Executive Projection: At a $20 return cost (Enterprise Scale), 3-Year Project NPV is: ${npv_20:,.2f}\n")

if __name__ == "__main__":
    main()
    