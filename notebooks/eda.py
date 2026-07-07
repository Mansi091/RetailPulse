import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ingestion.ingestion import DataIngestion
from src.config.settings import RAW_DATA_PATH, CLEANED_DATA_PATH
from src.database.connection import get_engine

def main():
    print("Starting Exploratory Data Analysis...")
    
    df = None
    # Check if raw data path exists
    if os.path.exists(RAW_DATA_PATH):
        print(f"Loading raw data from {RAW_DATA_PATH}...")
        ingestion = DataIngestion(RAW_DATA_PATH)
        df = ingestion.load_data()
    else:
        print(f"Raw data not found at {RAW_DATA_PATH}. Checking PostgreSQL database...")
        try:
            engine = get_engine()
            df = pd.read_sql('SELECT * FROM online_retail_clean', engine)
            print(f"Loaded {df.shape[0]} rows of cleaned transaction data from database.")
        except Exception as e:
            print(f"Database query failed ({e}). Checking local clean CSV...")
            if os.path.exists(CLEANED_DATA_PATH):
                df = pd.read_csv(CLEANED_DATA_PATH)
                print(f"Loaded {df.shape[0]} rows from local CSV: {CLEANED_DATA_PATH}")
            else:
                print("No raw data, database table, or clean CSV available. Cannot run EDA.")
                return

    # 1. Missing values analysis
    print("\n1. Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_pct})
    print(missing_df)
    
    # Plot missing values
    plt.figure(figsize=(8, 5))
    missing_pct.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Missing Values per Column')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    os.makedirs("notebooks/plots", exist_ok=True)
    plt.savefig("notebooks/plots/missing_values.png")
    print("Saved missing values plot to notebooks/plots/missing_values.png")
    
    # 2. Transaction distribution analysis
    print("\n2. Transaction Statistics:")
    numeric_cols = [c for c in ['Quantity', 'UnitPrice', 'Revenue'] if c in df.columns]
    print(df[numeric_cols].describe())
    
    # 3. Revenue calculation and time trend
    if 'Revenue' not in df.columns and 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Filter valid sales (Quantity > 0, Price > 0, non-null customer if raw)
    if 'CustomerID' in df.columns:
        valid_sales = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'].notnull())].copy()
    else:
        valid_sales = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
        
    valid_sales['InvoiceMonth'] = valid_sales['InvoiceDate'].dt.to_period('M')
    
    if 'Revenue' in valid_sales.columns:
        monthly_revenue = valid_sales.groupby('InvoiceMonth')['Revenue'].sum()
        
        plt.figure(figsize=(12, 6))
        # Convert period index to string for plotting support
        monthly_revenue.index = monthly_revenue.index.astype(str)
        monthly_revenue.plot(kind='line', marker='o', color='green')
        plt.title('Monthly Revenue Trend (Cleaned Data)')
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("notebooks/plots/monthly_revenue.png")
        print("Saved monthly revenue plot to notebooks/plots/monthly_revenue.png")
    
    # 4. Quantity and UnitPrice distributions (log scaled to handle outliers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(np.log1p(valid_sales['Quantity']), bins=30, color='coral', edgecolor='black')
    ax1.set_title('Distribution of Log(Quantity)')
    ax1.set_xlabel('Log(Quantity + 1)')
    ax1.set_ylabel('Count')
    
    ax2.hist(np.log1p(valid_sales['UnitPrice']), bins=30, color='orchid', edgecolor='black')
    ax2.set_title('Distribution of Log(UnitPrice)')
    ax2.set_xlabel('Log(UnitPrice + 1)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig("notebooks/plots/feature_distributions.png")
    print("Saved feature distributions plot to notebooks/plots/feature_distributions.png")
    
    print("\nExploratory Data Analysis completed successfully!")

if __name__ == "__main__":
    main()
