"""
Model training module for the adaptive ML health monitoring system.
Handles initial model training, evaluation, and saving.
"""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    directories = [
        Path("data"),
        Path("models"),
        Path("logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def load_or_create_data() -> pd.DataFrame:
    """
    Load CSV dataset from data/churn.csv or create synthetic data if not found.
    
    Returns:
        DataFrame containing the dataset with features and target column.
    """
    data_path = Path("data/churn.csv")
    
    # Try to load existing dataset
    if data_path.exists():
        logger.info(f"Loading existing dataset from {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Falling back to synthetic data generation")
    else:
        logger.info(f"Dataset not found at {data_path}. Creating synthetic dataset.")
    
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
    df.to_csv(data_path, index=False)
    logger.info(f"Synthetic dataset saved to {data_path} with shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
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
    
    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    logger.info(f"Training target distribution:\n{y_train.value_counts()}")
    logger.info(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features.
        y_train: Training target.
    
    Returns:
        Trained RandomForestClassifier model.
    """
    logger.info("Training Random Forest classifier...")
    
    # Initialize model with reasonable defaults
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    logger.info("Model training completed successfully")
    logger.info(f"Number of trees: {model.n_estimators}")
    logger.info(f"Number of features: {model.n_features_in_}")
    
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Log metrics
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("=" * 60)
    
    # Print detailed classification report
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return metrics


def save_model(model: RandomForestClassifier, metrics: dict, model_path: Path = Path("models/model_v1.pkl")):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save.
        metrics: Evaluation metrics to save alongside model.
        model_path: Path where to save the model.
    """
    logger.info(f"Saving model to {model_path}")
    
    # Ensure models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model using joblib
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model successfully saved to {model_path}")
        
        # Save metrics to a separate file
        metrics_path = model_path.parent / "model_v1_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Model Training Metrics\n")
            f.write("=" * 40 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        logger.info(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def train_initial_model():
    """
    Main training pipeline that orchestrates the entire training process.
    
    Steps:
        1. Ensure directories exist
        2. Load or create dataset
        3. Split data into train/test
        4. Train model
        5. Evaluate model
        6. Save model and metrics
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Ensure necessary directories exist
        ensure_directories()
        
        # Step 2: Load or create dataset
        df = load_or_create_data()
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 6: Save model
        save_model(model, metrics)
        
        logger.info("=" * 60)
        logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    # Execute the training pipeline
    train_initial_model()