import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Full Exploratory Data Analysis (EDA) Graphic Suite Generation...")
    
    # Load dataset
    df = pd.read_csv('RetailStoreProductSalesDataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Set plotting style for clean corporate presentation
    sns.set_theme(style="whitegrid")
    
    # ----------------------------------------------------
    # GRAPH 1: Pearson Correlation Matrix (The original one!)
    # ----------------------------------------------------
    logging.info("Generating Graph 1: Pearson Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    
    # Target correlation for return_rate specifically to recreate your exact bar plot
    return_corr = correlation_matrix['return_rate'].drop('return_rate').sort_values(ascending=True)
    
    plt.figure(figsize=(11, 6))
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in return_corr.values]
    ax = sns.barplot(x=return_corr.values, y=return_corr.index, palette=colors)
    
    # Add values directly on bars
    for i, v in enumerate(return_corr.values):
        ax.text(v + (0.01 if v >= 0 else -0.04), i, f"{v:.2f}", va='center', fontsize=10, fontweight='bold')
        
    plt.title('Which Features Drive Return Risk? (Pearson Correlation)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Correlation Coefficient (Negative = Reduces Returns | Positive = Increases Returns)', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_1_pearson_correlation.png', dpi=300)
    plt.close()

    # ----------------------------------------------------
    # GRAPH 2: Distribution of the Target Variable (Return Rate)
    # ----------------------------------------------------
    logging.info("Generating Graph 2: Target Variable Distribution...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['return_rate'], kde=True, color='#2c3e50', bins=30)
    plt.title('Distribution of Retail Product Return Rates', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Return Rate (Percentage)', fontsize=12)
    plt.ylabel('Frequency Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_2_return_rate_distribution.png', dpi=300)
    plt.close()

    # ----------------------------------------------------
    # GRAPH 3: Strategic Scatter Plot (Customer Sentiment vs Return Rate)
    # ----------------------------------------------------
    logging.info("Generating Graph 3: Customer Sentiment vs Return Rate Scatter Plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='customer_sentiment', y='return_rate', alpha=0.6, color='#2980b9')
    sns.regplot(data=df, x='customer_sentiment', y='return_rate', scatter=False, color='#c0392b', line_kws={"linewidth": 2})
    plt.title('Impact of Customer Sentiment on Product Return Rates', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Customer Sentiment Score (Higher = Better)', fontsize=12)
    plt.ylabel('Return Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_3_sentiment_vs_return.png', dpi=300)
    plt.close()

    # ----------------------------------------------------
    # GRAPH 4: Price Competitiveness vs Return Rate
    # ----------------------------------------------------
    logging.info("Generating Graph 4: Price vs Competitor Price Analysis...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='price', y='competitor_price', hue='return_rate', palette='viridis', alpha=0.7)
    plt.title('Price Positioning vs Competitor Pricing Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Our Store Price ($)', fontsize=12)
    plt.ylabel('Competitor Price ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_4_price_positioning.png', dpi=300)
    plt.close()

    logging.info("Successfully generated and saved all 4 EDA graphs to your root workspace folder!")

if __name__ == "__main__":
    main()
    