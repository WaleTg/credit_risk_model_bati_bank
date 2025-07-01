# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction features per CustomerId."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
    # Identify columns after aggregation
    numeric_cols = [
        "total_amount", "avg_amount", "transaction_count", "std_amount",
        "hour_median", "day_median", "month_median", "year_median"
    ]

    categorical_cols = []  # You can add `CountryCode`, `ChannelId`, etc. from raw data

    # Preprocessors
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor

def preprocess_data(df):
    """Complete pipeline: aggregation + transformation."""
    aggregator = TransactionAggregator()
    df_agg = aggregator.transform(df)
    preprocessor = build_preprocessing_pipeline(df_agg)
    X = preprocessor.fit_transform(df_agg)
    return X, df_agg["CustomerId"]
