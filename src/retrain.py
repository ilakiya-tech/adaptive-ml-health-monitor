"""
retrain.py

Retrain the churn model, version it, and log metrics.

Features:
- Load or create dataset (data/churn.csv)
- Train RandomForestClassifier
- Auto-increment model version: model_v1.pkl -> model_v2.pkl -> ...
- Evaluate accuracy, precision, recall, f1
- Save new model under models/model_v{n}.pkl
- Append metrics to data/metrics.csv
"""

import os
from pathlib import Path
from datetime import datetime
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.datasets import make_classification
import joblib


# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (adaptive-ml-health-monitor/)
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "churn.csv"
METRICS_FILE = DATA_DIR / "metrics.csv"

MODEL_PREFIX = "model_v"
MODEL_EXTENSION = ".pkl"


# -------------------------------------------------------------------
# Dataset utilities
# -------------------------------------------------------------------

def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """
    Fallback: create a synthetic churn-like dataset if churn.csv does not exist.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42,
    )

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    return df


def load_or_create_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from data/churn.csv if it exists, otherwise create and save a synthetic one.

    Assumes: last column is the target.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        print(f"[INFO] Loaded existing dataset from {DATA_FILE}")
    else:
        print("[WARN] Dataset not found. Creating a synthetic dataset...")
        df = create_synthetic_dataset()
        df.to_csv(DATA_FILE, index=False)
        print(f"[INFO] Synthetic dataset saved to {DATA_FILE}")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 target column.")

    # Assume last column is target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[INFO] Features shape: {X.shape}, Target column: '{target_col}'")
    return X, y


# -------------------------------------------------------------------
# Model versioning utilities
# -------------------------------------------------------------------

def get_existing_model_versions() -> list[int]:
    """
    Return a sorted list of existing model version numbers.
    Looks for files named model_v{n}.pkl in models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    versions = []
    for path in MODELS_DIR.glob(f"{MODEL_PREFIX}*{MODEL_EXTENSION}"):
        match = re.search(rf"{MODEL_PREFIX}(\d+){MODEL_EXTENSION}$", path.name)
        if match:
            versions.append(int(match.group(1)))

    return sorted(versions)


def get_next_model_path() -> tuple[int, Path]:
    """
    Determine the next model version and its file path.
    If no models exist, start with model_v1.pkl.
    """
    versions = get_existing_model_versions()
    next_version = 1 if not versions else max(versions) + 1
    model_path = MODELS_DIR / f"{MODEL_PREFIX}{next_version}{MODEL_EXTENSION}"
    return next_version, model_path


# -------------------------------------------------------------------
# Training & evaluation
# -------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, dict]:
    """
    Train RandomForestClassifier and compute evaluation metrics.
    Returns:
        model, metrics_dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    print("[INFO] Training RandomForest model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return model, metrics


# -------------------------------------------------------------------
# Persistence & logging
# -------------------------------------------------------------------

def save_model(model, model_path: Path) -> None:
    """
    Save the trained model to models/model_v{n}.pkl.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[INFO] Saved model to {model_path}")


def log_metrics(model_version: int, metrics: dict) -> None:
    """
    Append a new metrics row to data/metrics.csv.
    Columns: timestamp, model_version, accuracy, precision, recall, f1
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    df_row = pd.DataFrame([row])

    if METRICS_FILE.exists():
        existing = pd.read_csv(METRICS_FILE)
        # Optional: keep only common columns if schema differs slightly
        common_cols = [c for c in existing.columns if c in df_row.columns]
        if common_cols:
            existing = existing[common_cols]
            df_row = df_row[common_cols]
        df_out = pd.concat([existing, df_row], ignore_index=True)
    else:
        df_out = df_row

    df_out.to_csv(METRICS_FILE, index=False)
    print(f"[INFO] Logged metrics to {METRICS_FILE}")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    print("=" * 60)
    print("       Adaptive ML Health Monitor - Retrain Pipeline")
    print("=" * 60)

    # 1) Load or create dataset
    X, y = load_or_create_dataset()

    # 2) Determine next model version and path
    next_version, model_path = get_next_model_path()
    print(f"[INFO] Next model version will be: v{next_version}")
    print(f"[INFO] Model file: {model_path.name}")

    # 3) Train model and get metrics
    model, metrics = train_model(X, y)

    # 4) Save model
    save_model(model, model_path)

    # 5) Log metrics
    log_metrics(next_version, metrics)

    # 6) Pretty print results
    print("\n[RESULT] Retraining completed.")
    print(f"         Model version: v{next_version}")
    print(f"         Saved at     : {model_path}")
    print("         Metrics:")
    for k, v in metrics.items():
        print(f"           - {k:9s}: {v:.4f}")

    print("\n[OK] Done")


if __name__ == "__main__":
    main()