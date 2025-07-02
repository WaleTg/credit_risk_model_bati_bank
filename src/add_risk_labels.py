import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rfm(df, snapshot_date):
    # Convert and remove timezone from TransactionStartTime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    df["TransactionStartTime"] = df["TransactionStartTime"].dt.tz_localize(None)

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,  # Recency
        "TransactionId": "count",                                          # Frequency
        "Amount": "sum"                                                    # Monetary
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    }).reset_index()

    return rfm

def main():
    logger.info("[INFO] Loading raw data for RFM...")
    df_raw = pd.read_csv("data/raw/data.csv")  # You must keep the raw CSV for this

    snapshot_date = pd.to_datetime("2025-07-01").tz_localize(None)

    rfm = calculate_rfm(df_raw, snapshot_date)

    logger.info("[INFO] Scaling RFM...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    logger.info("[INFO] Running KMeans clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    logger.info("[INFO] Assigning high-risk label...")
    risk_cluster = rfm.groupby("cluster")[["Frequency", "Monetary"]].mean().idxmin().min()
    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype(int)

    logger.info("[INFO] Merging risk labels with processed features...")
    df_processed = pd.read_csv("data/processed/processed.csv")
    merged = pd.merge(df_processed, rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

    logger.info("[INFO] Saving final processed dataset with risk labels...")
    merged.to_csv("data/processed/processed_with_risk.csv", index=False)
    logger.info("[DONE] Saved: data/processed/processed_with_risk.csv")

if __name__ == "__main__":
    main()
