"""
Configuration settings for the adaptive ML health monitoring system.
Contains all constants, paths, and hyperparameters used across the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data file paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation.csv"
PRODUCTION_DATA_PATH = DATA_DIR / "production.csv"

# Model file paths
MODEL_PATH = MODEL_DIR / "current_model.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Feature configuration
FEATURE_COLUMNS = []  # TODO: Define feature columns
TARGET_COLUMN = "target"  # TODO: Update target column name
CATEGORICAL_FEATURES = []  # TODO: Define categorical features
NUMERICAL_FEATURES = []  # TODO: Define numerical features

# Model hyperparameters
MODEL_TYPE = "random_forest"  # TODO: Choose model type
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model-specific parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    # TODO: Add more hyperparameters
}

# Drift detection settings
DRIFT_DETECTION_ENABLED = True
DRIFT_THRESHOLD = 0.1  # Threshold for data drift detection
CONCEPT_DRIFT_THRESHOLD = 0.05  # Threshold for concept drift
MIN_SAMPLES_FOR_DRIFT = 100  # Minimum samples needed for drift detection
DRIFT_CHECK_INTERVAL = 3600  # Seconds between drift checks

# Performance monitoring settings
PERFORMANCE_METRICS = ["accuracy", "precision", "recall", "f1_score"]
PERFORMANCE_THRESHOLD = 0.75  # Minimum acceptable performance
PERFORMANCE_CHECK_INTERVAL = 1800  # Seconds between performance checks
ALERT_THRESHOLD = 0.70  # Threshold to trigger alerts

# Retraining settings
AUTO_RETRAIN_ENABLED = True
MIN_SAMPLES_FOR_RETRAIN = 1000  # Minimum new samples before retraining
RETRAIN_WINDOW_SIZE = 5000  # Number of recent samples to use for retraining
RETRAIN_TRIGGER_CONDITIONS = {
    "drift_detected": True,
    "performance_degraded": True,
    "time_based": False,  # TODO: Implement time-based retraining
}
RETRAIN_INTERVAL_DAYS = 7  # Days between scheduled retraining

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
METRICS_LOG_PATH = LOGS_DIR / "metrics.log"
DRIFT_LOG_PATH = LOGS_DIR / "drift.log"
RETRAIN_LOG_PATH = LOGS_DIR / "retrain.log"

# Dashboard settings
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_INTERVAL = 60  # Seconds

# Database/Storage settings (if needed)
USE_DATABASE = False  # TODO: Set to True if using database
DATABASE_URL = None  # TODO: Add database connection string

# Notification settings
ENABLE_ALERTS = False  # TODO: Enable email/slack alerts
ALERT_EMAIL = None  # TODO: Add alert email
ALERT_SLACK_WEBHOOK = None  # TODO: Add Slack webhook URL