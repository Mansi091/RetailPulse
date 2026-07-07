import pandas as pd
import pytest
from src.validation.validation import DataValidator

def test_data_validator_report():
    mock_data = pd.DataFrame([
        {"CustomerID": 12345.0, "Description": "Test Product", "Quantity": 10, "UnitPrice": 2.5, "InvoiceNo": "536365"},
        {"CustomerID": None, "Description": "Test Product 2", "Quantity": 5, "UnitPrice": 1.0, "InvoiceNo": "536366"}, # null customer ID
        {"CustomerID": 12346.0, "Description": None, "Quantity": -2, "UnitPrice": 5.0, "InvoiceNo": "C536367"}, # null desc, negative quantity, cancelled order
        {"CustomerID": 12347.0, "Description": "Promo Item", "Quantity": 1, "UnitPrice": -1.0, "InvoiceNo": "536368"}, # negative price
    ])
    
    # Add a duplicate of row 0
    mock_data = pd.concat([mock_data, mock_data.iloc[[0]]], ignore_index=True)
    
    validator = DataValidator(mock_data)
    report = validator.generate_report()
    
    assert report["total_rows"] == 5
    assert report["null_customer_id"] == 1
    assert report["null_customer_id_pct"] == 20.0
    assert "Excluded" in report["decision_null_customer_id"]
    
    assert report["negative_quantity"] == 1
    assert report["negative_quantity_pct"] == 20.0
    
    assert report["duplicate_rows"] == 1
    assert report["duplicate_rows_pct"] == 20.0
