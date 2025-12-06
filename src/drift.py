"""
Data and concept drift detection module for the adaptive ML health monitoring system.
Monitors performance degradation by comparing baseline and current metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project paths using pathlib
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_LOG_PATH = PROJECT_ROOT / "data" / "metrics.csv"

# Define drift thresholds
ACCURACY_DROP_THRESHOLD = 0.05  # 5% absolute drop triggers drift alert
MIN_SAMPLES_FOR_DRIFT = 2  # Minimum number of metric entries needed


def load_metrics_history(metrics_path: Path) -> Optional[pd.DataFrame]:
    """
    Load historical metrics from CSV file.
    
    Args:
        metrics_path: Path to the metrics CSV file.
    
    Returns:
        DataFrame containing metrics history, or None if file doesn't exist.
    """
    logger.info(f"Loading metrics history from {metrics_path}")
    
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found at {metrics_path}")
        return None
    
    try:
        df = pd.read_csv(metrics_path)
        logger.info(f"Loaded {len(df)} metric entries")
        return df
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return None


def validate_metrics_data(df: pd.DataFrame) -> bool:
    """
    Validate that metrics DataFrame has sufficient data for drift detection.
    
    Args:
        df: DataFrame containing metrics history.
    
    Returns:
        True if data is valid, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("Metrics DataFrame is empty")
        return False
    
    if len(df) < MIN_SAMPLES_FOR_DRIFT:
        logger.warning(f"Not enough data points. Found {len(df)}, need at least {MIN_SAMPLES_FOR_DRIFT}")
        return False
    
    # Check if required column exists
    if 'accuracy' not in df.columns:
        logger.error("'accuracy' column not found in metrics data")
        return False
    
    return True


def calculate_drift(df: pd.DataFrame, threshold: float = ACCURACY_DROP_THRESHOLD) -> Dict[str, any]:
    """
    Calculate drift by comparing baseline and latest accuracy.
    
    Args:
        df: DataFrame containing metrics history.
        threshold: Accuracy drop threshold for drift detection.
    
    Returns:
        Dictionary containing drift detection results.
    """
    logger.info("Calculating drift metrics...")
    
    # Get baseline accuracy (first row)
    baseline_accuracy = df['accuracy'].iloc[0]
    
    # Get latest accuracy (last row)
    latest_accuracy = df['accuracy'].iloc[-1]
    
    # Calculate drop
    drop = baseline_accuracy - latest_accuracy
    
    # Determine if drift is detected
    drift_detected = drop >= threshold
    
    # Log the results
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"Latest accuracy:   {latest_accuracy:.4f}")
    logger.info(f"Accuracy drop:     {drop:.4f}")
    logger.info(f"Drift threshold:   {threshold:.4f}")
    
    # Prepare result dictionary
    result = {
        "drift_detected": drift_detected,
        "baseline_accuracy": float(baseline_accuracy),
        "latest_accuracy": float(latest_accuracy),
        "drop": float(drop),
        "threshold": float(threshold),
        "num_evaluations": len(df),
        "baseline_timestamp": df['timestamp'].iloc[0] if 'timestamp' in df.columns else None,
        "latest_timestamp": df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
    }
    
    return result


def display_drift_results(result: Dict[str, any]):
    """
    Display drift detection results in a formatted, user-friendly way.
    
    Args:
        result: Dictionary containing drift detection results.
    """
    print("\n" + "=" * 70)
    print("[DRIFT DETECTION RESULTS]")
    print("=" * 70)
    
    if result.get("drift_detected"):
        print("[WARNING] DRIFT DETECTED! Model performance has degraded.")
        print("-" * 70)
        print(f"   Baseline Accuracy:  {result['baseline_accuracy']:.2%}")
        print(f"   Latest Accuracy:    {result['latest_accuracy']:.2%}")
        print(f"   Accuracy Drop:      {result['drop']:.2%} (Threshold: {result['threshold']:.2%})")
        print(f"   Total Evaluations:  {result['num_evaluations']}")
        print("-" * 70)
        print("[ACTION REQUIRED] Consider retraining the model!")
    else:
        print("[OK] NO SIGNIFICANT DRIFT DETECTED")
        print("-" * 70)
        print(f"   Baseline Accuracy:  {result['baseline_accuracy']:.2%}")
        print(f"   Latest Accuracy:    {result['latest_accuracy']:.2%}")
        print(f"   Accuracy Drop:      {result['drop']:.2%} (Threshold: {result['threshold']:.2%})")
        print(f"   Total Evaluations:  {result['num_evaluations']}")
        print("-" * 70)
        print("[INFO] Model performance is stable!")
    
    print("=" * 70 + "\n")


def log_drift_detection(result: Dict[str, any], log_path: Path):
    """
    Log drift detection results to a file for historical tracking.
    
    Args:
        result: Dictionary containing drift detection results.
        log_path: Path to the drift log file.
    """
    drift_log_path = log_path.parent / "drift_log.csv"
    
    try:
        from datetime import datetime
        
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'drift_detected': result['drift_detected'],
            'baseline_accuracy': result['baseline_accuracy'],
            'latest_accuracy': result['latest_accuracy'],
            'drop': result['drop'],
            'threshold': result['threshold']
        }
        
        # Append to drift log
        df_new = pd.DataFrame([log_entry])
        
        if drift_log_path.exists():
            df_existing = pd.read_csv(drift_log_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(drift_log_path, index=False)
        else:
            df_new.to_csv(drift_log_path, index=False)
        
        logger.info(f"Drift detection logged to {drift_log_path}")
    
    except Exception as e:
        logger.warning(f"Could not log drift detection: {e}")


def check_drift(metrics_path: Path = METRICS_LOG_PATH, 
                threshold: float = ACCURACY_DROP_THRESHOLD) -> Dict[str, any]:
    """
    Main drift detection pipeline.
    
    Loads metrics history, validates data, calculates drift, and reports results.
    
    Args:
        metrics_path: Path to the metrics CSV file.
        threshold: Accuracy drop threshold for drift detection.
    
    Returns:
        Dictionary containing drift detection results, or None if insufficient data.
    """
    logger.info("=" * 70)
    logger.info("STARTING DRIFT DETECTION")
    logger.info("=" * 70)
    
    # Step 1: Load metrics history
    df = load_metrics_history(metrics_path)
    
    # Step 2: Validate metrics data
    if not validate_metrics_data(df):
        message = "Not enough data to detect drift yet. Please run monitor.py multiple times to collect more metrics."
        print("\n" + "=" * 70)
        print(f"   {message}")
        print("=" * 70 + "\n")
        logger.info(message)
        logger.info("=" * 70)
        return None
    
    # Step 3: Calculate drift
    result = calculate_drift(df, threshold)
    
    # Step 4: Display results
    display_drift_results(result)
    
    # Step 5: Log drift detection
    log_drift_detection(result, metrics_path)
    
    logger.info("=" * 70)
    logger.info("DRIFT DETECTION COMPLETED")
    logger.info("=" * 70)
    
    return result


def get_drift_summary(metrics_path: Path = METRICS_LOG_PATH) -> Optional[Dict[str, any]]:
    """
    Get a summary of drift trends over time.
    
    Args:
        metrics_path: Path to the metrics CSV file.
    
    Returns:
        Dictionary containing drift summary statistics.
    """
    df = load_metrics_history(metrics_path)
    
    if not validate_metrics_data(df):
        return None
    
    try:
        summary = {
            'total_evaluations': len(df),
            'baseline_accuracy': float(df['accuracy'].iloc[0]),
            'latest_accuracy': float(df['accuracy'].iloc[-1]),
            'mean_accuracy': float(df['accuracy'].mean()),
            'min_accuracy': float(df['accuracy'].min()),
            'max_accuracy': float(df['accuracy'].max()),
            'std_accuracy': float(df['accuracy'].std()),
            'trend': 'improving' if df['accuracy'].iloc[-1] > df['accuracy'].iloc[0] else 'degrading'
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error calculating drift summary: {e}")
        return None


if __name__ == "__main__":
    # Execute the drift detection pipeline
    result = check_drift()
    
    # Print the result dictionary
    if result:
        print("[INFO] Result Dictionary:")
        print(result)
        print()
        
        # Optionally display drift summary
        print("[INFO] Drift Summary:")
        summary = get_drift_summary()
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
    else:
        print("[WARNING] Drift detection could not be performed. Need more metric data.")