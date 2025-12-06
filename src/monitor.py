"""
Performance monitoring module for the adaptive ML health monitoring system.
Evaluates model performance and logs metrics over time.
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project paths using pathlib
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
METRICS_LOG_PATH = PROJECT_ROOT / "data" / "metrics.csv"


def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    directories = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def load_or_create_data() -> pd.DataFrame:
    """
    Load CSV dataset from data/churn.csv or create synthetic data if not found.
    Reuses the same logic as train.py to ensure consistency.
    
    Returns:
        DataFrame containing the dataset with features and target column.
    """
    # Try to load existing dataset
    if DATA_PATH.exists():
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        try:
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Falling back to synthetic data generation")
    else:
        logger.info(f"Dataset not found at {DATA_PATH}. Creating synthetic dataset.")
    
    # Create synthetic binary classification dataset
    logger.info("Generating synthetic binary classification dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced classes (70% class 0, 30% class 1)
        flip_y=0.01,  # Add 1% noise
        random_state=42
    )
    
    # Create DataFrame with meaningful column names
    feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['target'] = y
    
    # Save synthetic dataset for future use
    ensure_directories()
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Synthetic dataset saved to {DATA_PATH} with shape: {df.shape}")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.
    Uses the same parameters as train.py for consistency.
    
    Args:
        df: Input DataFrame containing features and target.
        test_size: Proportion of dataset to include in test split.
        random_state: Random state for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info("Splitting data into train and test sets...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution in splits
    )
    
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def load_model(model_path: Path):
    """
    Load the trained model from disk.
    
    Args:
        model_path: Path to the saved model file.
    
    Returns:
        Loaded model object.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    logger.info(f"Loading model from {model_path}")
    
    if not model_path.exists():
        error_msg = f"Model file not found at {model_path}. Please train the model first using src/train.py"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model on test set...")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Log metrics to console
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("=" * 60)
    
    return metrics


def log_metrics_to_csv(metrics: dict, model_path: Path, log_path: Path):
    """
    Log evaluation metrics to CSV file with timestamp.
    Creates the file with headers if it doesn't exist.
    
    Args:
        metrics: Dictionary of metric names and values.
        model_path: Path to the model that was evaluated.
        log_path: Path to the metrics log CSV file.
    """
    logger.info(f"Logging metrics to {log_path}")
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the log entry
    log_entry = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score']
    }
    
    # Check if metrics log file exists
    if log_path.exists():
        # Append to existing file
        df_existing = pd.read_csv(log_path)
        df_new = pd.DataFrame([log_entry])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(log_path, index=False)
        logger.info(f"Metrics appended to existing log file")
    else:
        # Create new file with headers
        df_new = pd.DataFrame([log_entry])
        df_new.to_csv(log_path, index=False)
        logger.info(f"New metrics log file created at {log_path}")
    
    logger.info(f"Logged entry: {log_entry}")


def display_metrics_summary(log_path: Path):
    """
    Display a summary of historical metrics if available.
    
    Args:
        log_path: Path to the metrics log CSV file.
    """
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            df_metrics = pd.read_csv(log_path)
            
            if len(df_metrics) > 1:
                logger.info("\n" + "=" * 60)
                logger.info("HISTORICAL METRICS SUMMARY")
                logger.info("=" * 60)
                logger.info(f"Total evaluations: {len(df_metrics)}")
                logger.info(f"Average accuracy:  {df_metrics['accuracy'].mean():.4f}")
                logger.info(f"Latest accuracy:   {df_metrics['accuracy'].iloc[-1]:.4f}")
                logger.info(f"Best accuracy:     {df_metrics['accuracy'].max():.4f}")
                logger.info(f"Worst accuracy:    {df_metrics['accuracy'].min():.4f}")
                logger.info("=" * 60 + "\n")
        except Exception as e:
            logger.warning(f"Could not display metrics summary: {e}")


def run_monitoring():
    """
    Main monitoring pipeline that orchestrates the entire monitoring process.
    
    Steps:
        1. Ensure directories exist
        2. Load or create dataset
        3. Split data into train/test
        4. Load trained model
        5. Evaluate model on test set
        6. Print accuracy to console
        7. Log metrics to CSV file
        8. Display historical metrics summary
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL PERFORMANCE MONITORING")
    logger.info("=" * 60)
    
    try:
        # Step 1: Ensure necessary directories exist
        ensure_directories()
        
        # Step 2: Load or create dataset
        df = load_or_create_data()
        
        # Step 3: Split data (using same parameters as training)
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Step 4: Load trained model
        model = load_model(MODEL_PATH)
        
        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 6: Print clear accuracy message
        print("\n" + "=" * 60)
        print(f"   MODEL ACCURACY: {metrics['accuracy']:.2%}")
        print("=" * 60 + "\n")
        
        # Step 7: Log metrics to CSV
        log_metrics_to_csv(metrics, MODEL_PATH, METRICS_LOG_PATH)
        
        # Step 8: Display historical metrics summary
        display_metrics_summary(METRICS_LOG_PATH)
        
        logger.info("=" * 60)
        logger.info("MODEL MONITORING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return metrics
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure the model has been trained first by running: python src/train.py")
        raise
    
    except Exception as e:
        logger.error(f"Error in monitoring pipeline: {e}")
        raise


if __name__ == "__main__":
    # Execute the monitoring pipeline
    run_monitoring()