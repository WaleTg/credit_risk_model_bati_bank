#unit testing script
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from src.data_processing import preprocess_data

def test_preprocess_data():
    # Creating a minimal dummy dataframe resembling my raw data
    data = {
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 300],
        "TransactionId": [101, 102, 103],
        "TransactionStartTime": ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"]
    }
    df = pd.DataFrame(data)
    
    # Calling preprocessing function
    X, customer_ids, preprocessor, df_agg = preprocess_data(df)

    # Assert the output shape and type
    assert X.shape[0] == len(customer_ids)
    assert len(customer_ids) == 2  # because there are 2 unique CustomerId
    
   
