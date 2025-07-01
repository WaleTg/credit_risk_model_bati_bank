# src/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import preprocess_data

import pandas as pd
from src.data_processing import preprocess_data

if __name__ == "__main__":
    print("[INFO] Reading data...")
    df = pd.read_csv("/Users/tagesehandiso/Downloads/game/Kifya/data.csv")
    
    print("[INFO] Preprocessing data...")
    X, customer_ids = preprocess_data(df)
    
    print("[DONE] Feature matrix shape:", X.shape)
