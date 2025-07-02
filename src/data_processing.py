
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import os

class TransactionAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"], errors='coerce')
        agg = X.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_amount=("Amount", "std"),
            hour_median=("TransactionStartTime", lambda x: x.dt.hour.median()),
            day_median=("TransactionStartTime", lambda x: x.dt.day.median()),
            month_median=("TransactionStartTime", lambda x: x.dt.month.median()),
            year_median=("TransactionStartTime", lambda x: x.dt.year.median())
        ).reset_index()
        return agg

def build_preprocessing_pipeline(df):
    numeric_cols = [
        "total_amount", "avg_amount", "transaction_count", "std_amount",
        "hour_median", "day_median", "month_median", "year_median"
    ]
    categorical_cols = []

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor

def preprocess_data(df):
    aggregator = TransactionAggregator()
    df_agg = aggregator.transform(df)
    preprocessor = build_preprocessing_pipeline(df_agg)
    X = preprocessor.fit_transform(df_agg)
    return X, df_agg["CustomerId"], preprocessor, df_agg

if __name__ == "__main__":
    df_raw = pd.read_csv("/Users/tagesehandiso/Downloads/game/Kifya/data.csv")

    X, customer_ids, preprocessor, df_agg = preprocess_data(df_raw)

    numeric_cols = [
        "total_amount", "avg_amount", "transaction_count", "std_amount",
        "hour_median", "day_median", "month_median", "year_median"
    ]
    categorical_cols = []

    feature_names_num = numeric_cols
    feature_names_cat = []

    feature_names = feature_names_num + list(feature_names_cat)

    if hasattr(X, "toarray"):
        X = X.toarray()

    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df["CustomerId"] = customer_ids.values

    os.makedirs("data/processed", exist_ok=True)

    processed_df.to_csv("data/processed/processed.csv", index=False)
    print("[DONE] Processed data saved to data/processed/processed.csv")
