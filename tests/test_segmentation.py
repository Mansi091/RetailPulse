import pandas as pd
import numpy as np
import pytest
from src.ml.segmentation import CustomerSegmenter

def test_calculate_rfm():
    mock_data = pd.DataFrame([
        # Customer 1: 3 purchases, high spend, recent
        {"CustomerID": 1001.0, "InvoiceNo": "5001", "InvoiceDate": "2026-01-01 10:00:00", "Revenue": 100.0},
        {"CustomerID": 1001.0, "InvoiceNo": "5002", "InvoiceDate": "2026-01-02 10:00:00", "Revenue": 150.0},
        {"CustomerID": 1001.0, "InvoiceNo": "5003", "InvoiceDate": "2026-01-03 10:00:00", "Revenue": 50.0},
        # Customer 2: 1 purchase, low spend, older
        {"CustomerID": 1002.0, "InvoiceNo": "5004", "InvoiceDate": "2025-12-01 10:00:00", "Revenue": 10.0},
    ])
    
    segmenter = CustomerSegmenter()
    rfm = segmenter.calculate_rfm(mock_data)
    
    assert len(rfm) == 2
    assert set(rfm.columns) == {"CustomerID", "Recency", "Frequency", "Monetary"}
    
    # Check values
    cust1 = rfm[rfm["CustomerID"] == 1001.0].iloc[0]
    cust2 = rfm[rfm["CustomerID"] == 1002.0].iloc[0]
    
    assert cust1["Frequency"] == 3
    assert cust1["Monetary"] == 300.0
    assert cust2["Frequency"] == 1
    assert cust2["Monetary"] == 10.0
    assert cust1["Recency"] < cust2["Recency"]

def test_preprocess_and_segment():
    # Make sure we have enough varied points for 4 clusters
    rfm_mock = pd.DataFrame([
        {"CustomerID": 1.0, "Recency": 1, "Frequency": 100, "Monetary": 10000.0},
        {"CustomerID": 2.0, "Recency": 2, "Frequency": 80, "Monetary": 8000.0},
        {"CustomerID": 3.0, "Recency": 30, "Frequency": 5, "Monetary": 200.0},
        {"CustomerID": 4.0, "Recency": 40, "Frequency": 4, "Monetary": 150.0},
        {"CustomerID": 5.0, "Recency": 120, "Frequency": 2, "Monetary": 50.0},
        {"CustomerID": 6.0, "Recency": 150, "Frequency": 1, "Monetary": 20.0},
        {"CustomerID": 7.0, "Recency": 250, "Frequency": 1, "Monetary": 10.0},
        {"CustomerID": 8.0, "Recency": 300, "Frequency": 1, "Monetary": 5.0},
    ])
    
    segmenter = CustomerSegmenter()
    segmented = segmenter.preprocess_and_segment(rfm_mock)
    
    assert "Segment" in segmented.columns
    assert "Cluster" in segmented.columns
    assert segmented["Segment"].isnull().sum() == 0
    assert set(segmented["Segment"].unique()).issubset(
        {"VIP / Champions", "Loyal / Active", "At-Risk", "Hibernating / Lost"}
    )
