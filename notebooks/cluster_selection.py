import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.database.connection import get_engine
from src.config.settings import CLEANED_DATA_PATH

def main():
    # 1. Load Data (try DB first, then CSV)
    df = None
    try:
        engine = get_engine()
        df = pd.read_sql('SELECT "CustomerID", "InvoiceDate", "InvoiceNo", "Revenue" FROM online_retail_clean', engine)
        print(f"Loaded {df.shape[0]} rows from PostgreSQL database.")
    except Exception as e:
        print(f"Database connection failed ({e}). Loading from CSV instead...")
        
    if df is None:
        if os.path.exists(CLEANED_DATA_PATH):
            df = pd.read_csv(CLEANED_DATA_PATH)
            print(f"Loaded {df.shape[0]} rows from local CSV: {CLEANED_DATA_PATH}")
        else:
            # If clean file doesn't exist, we can ingest and clean
            from src.ingestion.ingestion import DataIngestion
            from src.config.settings import RAW_DATA_PATH
            from src.transformation.transform import DataTransformer
            
            print("CSV and DB not available. Running ingestion/transformation on raw data...")
            ingestion = DataIngestion(RAW_DATA_PATH)
            raw_df = ingestion.load_data()
            transformer = DataTransformer(raw_df)
            transformer.clean_data()
            df = transformer.create_features()
            print(f"Ingested and cleaned raw data. Resulting shape: {df.shape}")

    # 2. RFM Calculation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    max_date = df["InvoiceDate"].max()
    reference_date = max_date + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
    rfm = rfm.drop(columns=["LastPurchase"])

    # Preprocess
    rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    rfm_features["Monetary"] = rfm_features["Monetary"].clip(lower=0.01)
    rfm_log = np.log1p(rfm_features)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # Evaluate K
    inertias, silhouettes = [], []
    K_range = range(2, 9)

    print("Running KMeans iterations...")
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(rfm_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, km.labels_))

    # Plot and save
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Elbow Method)', color=color)
    ax1.plot(K_range, inertias, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K_range, silhouettes, 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('K-Means Cluster Selection (Inertia & Silhouette Score)')
    fig.tight_layout()

    # Save the plot
    os.makedirs("notebooks", exist_ok=True)
    plot_path = "notebooks/cluster_selection_plot.png"
    plt.savefig(plot_path)
    print(f"Saved cluster selection plot to {plot_path}")

    # Print findings
    print("\nK-Means Evaluation Summary:")
    for k, inertia, sil in zip(K_range, inertias, silhouettes):
        print(f"K={k} | Inertia: {inertia:.2f} | Silhouette Score: {sil:.4f}")

if __name__ == "__main__":
    main()
