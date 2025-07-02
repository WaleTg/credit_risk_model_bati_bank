import pandas as pd
from src.data_processing import TransactionAggregator

def test_transaction_aggregator_transform():
    # Sample raw transactions data
    data = {
        "CustomerId": [1, 1, 2, 2, 2],
        "TransactionId": [101, 102, 201, 202, 203],
        "Amount": [100, 200, 50, 60, 70],
        "TransactionStartTime": [
            "2023-01-01 10:00:00",
            "2023-01-02 11:00:00",
            "2023-01-01 09:00:00",
            "2023-01-03 12:00:00",
            "2023-01-04 13:00:00"
        ]
    }
    df = pd.DataFrame(data)

    aggregator = TransactionAggregator()
    df_agg = aggregator.transform(df)

    # Check the number of unique customers aggregated
    assert df_agg["CustomerId"].nunique() == 2

    # Check total amount for CustomerId 1
    cust1_total = df_agg.loc[df_agg["CustomerId"] == 1, "total_amount"].values[0]
    assert cust1_total == 300  # 100 + 200

    # Check transaction count for CustomerId 2
    cust2_count = df_agg.loc[df_agg["CustomerId"] == 2, "transaction_count"].values[0]
    assert cust2_count == 3

    # Check if median hour is calculated correctly (should be median of hours for CustomerId 2)
    cust2_hour_median = df_agg.loc[df_agg["CustomerId"] == 2, "hour_median"].values[0]
    assert cust2_hour_median == 12  # Median of [9, 12, 13] is 12
