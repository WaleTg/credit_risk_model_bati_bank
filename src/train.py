import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

def main():
    logger.info("[INFO] Reading processed dataset with risk labels...")
    df = pd.read_csv("data/processed/processed_with_risk.csv")
    if df.columns.duplicated().any():
        logger.info("[INFO] Found duplicated columns. Removing duplicates...")
        df = df.loc[:, ~df.columns.duplicated()]
    # Fix duplicated is_high_risk columns if they exist
    if "is_high_risk_x" in df.columns and "is_high_risk_y" in df.columns:
        logger.info("[INFO] Found duplicated is_high_risk columns. Keeping 'is_high_risk_x'.")
        df = df.rename(columns={"is_high_risk_x": "is_high_risk"})
        df = df.drop(columns=["is_high_risk_y"])
    elif "is_high_risk_x" in df.columns:
        df = df.rename(columns={"is_high_risk_x": "is_high_risk"})
    elif "is_high_risk_y" in df.columns:
        df = df.rename(columns={"is_high_risk_y": "is_high_risk"})

    required_cols = {"CustomerId", "is_high_risk"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Drop non-numeric or identifier columns that cause errors
    non_feature_cols = [
        "CustomerId", "is_high_risk",
        "TransactionId", "BatchId", "AccountId", "SubscriptionId",
        "CurrencyCode", "CountryCode", "ProviderId", "ProductId",
        "ProductCategory", "ChannelId", "TransactionStartTime",
        "PricingStrategy", "FraudResult"
    ]

    # Keep only columns that exist (some might not be present)
    non_feature_cols = [col for col in non_feature_cols if col in df.columns]

    feature_cols = [
    "total_amount", "avg_amount", "transaction_count", "std_amount",
    "hour_median", "day_median", "month_median", "year_median"
                                                                ]
    X = df[feature_cols]

    y = df["is_high_risk"]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Train/test split with stratification for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    mlflow.set_experiment("Credit Risk Model")

    with mlflow.start_run(run_name="LogisticRegression") as run_lr:
        logger.info("[INFO] Training Logistic Regression...")
        #lr = LogisticRegression(max_iter=1000, random_state=42)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)

        lr.fit(X_train, y_train)

        metrics = evaluate_model(lr, X_test, y_test)
        mlflow.sklearn.log_model(lr, "logistic_regression_model")
        mlflow.log_metrics(metrics)

    with mlflow.start_run(run_name="RandomForest") as run_rf:
        logger.info("[INFO] Training Random Forest with Grid Search...")
        #rf = RandomForestClassifier(random_state=42)
        rf = RandomForestClassifier(class_weight="balanced", random_state=42)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        logger.info(f"Best RF params: {grid_search.best_params_}")

        metrics_rf = evaluate_model(best_rf, X_test, y_test)

        mlflow.sklearn.log_model(best_rf, "random_forest_model")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics_rf)

    logger.info("[DONE] Training complete and models logged to MLflow.")

if __name__ == "__main__":
    main()
