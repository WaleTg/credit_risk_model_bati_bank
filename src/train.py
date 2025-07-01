# src/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rfm_engineering import calculate_rfm, cluster_customers
from src.data_processing import preprocess_data

import pandas as pd
from src.data_processing import preprocess_data

if __name__ == "__main__":
    print("[INFO] Reading data...")
    df = pd.read_csv("/Users/tagesehandiso/Downloads/game/Kifya/data.csv")
    
    print("[INFO] Preprocessing data...")
    X, customer_ids = preprocess_data(df)
    
    print("[DONE] Feature matrix shape:", X.shape)


# After preprocess_data(df)...
rfm = calculate_rfm(df)
rfm_labeled = cluster_customers(rfm)

# Merge with processed features
processed_df = pd.concat([customer_ids.reset_index(drop=True), pd.DataFrame(X)], axis=1)

final_df = processed_df.merge(rfm_labeled, on="CustomerId", how="left")

print("[DONE] Final dataset shape:", final_df.shape)
print("Sample high risk labels:\n", final_df["is_high_risk"].value_counts())
