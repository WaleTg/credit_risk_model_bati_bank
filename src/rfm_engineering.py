# src/rfm_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df, snapshot_date=None):
    """Compute Recency, Frequency, and Monetary for each CustomerId."""
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    }).reset_index()

    return rfm


def cluster_customers(rfm_df, n_clusters=3, random_state=42):
    """Cluster customers based on RFM values and return labeled DataFrame."""
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    rfm_df["cluster"] = kmeans.fit_predict(scaled_rfm)

    # Identify high-risk cluster: lowest freq + lowest monetary
    cluster_summary = rfm_df.groupby("cluster")[features].mean()
    high_risk_cluster = cluster_summary.sort_values(["Frequency", "Monetary"]).index[0]

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    return rfm_df[["CustomerId", "is_high_risk"]]
