"""
End-to-end pipeline orchestrator for the adaptive ML health monitoring system.
Coordinates training, monitoring, drift detection, and retraining.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
DRIFT_LOG_PATH = PROJECT_ROOT / "data" / "drift_log.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "metrics.csv"


def ensure_initial_model():
    """
    Ensure that an initial model exists. Train one if it doesn't.
    """
    if not MODEL_PATH.exists():
        logger.info("No trained model found. Training initial model...")
        try:
            result = subprocess.run(
                [sys.executable, "src/train.py"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Initial model training completed")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Model training failed: {e.stderr}")
            raise
    else:
        logger.info(f"Model already exists at {MODEL_PATH}")


def run_monitoring() -> bool:
    """
    Run the monitoring script to evaluate model performance.
    
    Returns:
        True if monitoring succeeded, False otherwise.
    """
    logger.info("Running performance monitoring...")
    try:
        result = subprocess.run(
            [sys.executable, "src/monitor.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Monitoring completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Monitoring failed: {e.stderr}")
        return False


def run_drift_detection() -> Optional[Dict]:
    """
    Run drift detection to check for model degradation.
    
    Returns:
        Dictionary with drift results, or None if detection failed.
    """
    logger.info("Running drift detection...")
    try:
        result = subprocess.run(
            [sys.executable, "src/drift.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Drift detection completed")
        logger.info(result.stdout)
        
        if DRIFT_LOG_PATH.exists():
            import pandas as pd
            df = pd.read_csv(DRIFT_LOG_PATH)
            if len(df) > 0:
                latest = df.iloc[-1].to_dict()
                return latest
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Drift detection failed: {e.stderr}")
        return None


def run_retraining() -> bool:
    """
    Run model retraining script.
    
    Returns:
        True if retraining succeeded, False otherwise.
    """
    logger.info("Running model retraining...")
    try:
        result = subprocess.run(
            [sys.executable, "src/retrain.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Retraining completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e.stderr}")
        return False


def run_pipeline(force_retrain: bool = False):
    """
    Execute the complete ML pipeline.
    
    Steps:
        1. Ensure initial model exists
        2. Run monitoring
        3. Check for drift
        4. Retrain if drift detected or forced
    
    Args:
        force_retrain: If True, skip drift check and retrain immediately.
    """
    logger.info("=" * 70)
    logger.info("STARTING ML PIPELINE")
    logger.info("=" * 70)
    
    try:
        ensure_initial_model()
        
        monitoring_success = run_monitoring()
        if not monitoring_success:
            logger.warning("Monitoring failed, but continuing pipeline...")
        
        if force_retrain:
            logger.info("Force retraining requested")
            retrain_success = run_retraining()
            if retrain_success:
                logger.info("Forced retraining completed successfully")
            else:
                logger.error("Forced retraining failed")
        else:
            drift_result = run_drift_detection()
            
            if drift_result and drift_result.get('drift_detected'):
                logger.warning("Drift detected! Triggering retraining...")
                retrain_success = run_retraining()
                if retrain_success:
                    logger.info("Retraining completed successfully")
                    run_monitoring()
                else:
                    logger.error("Retraining failed")
            else:
                logger.info("No drift detected. Model is performing well.")
        
        logger.info("=" * 70)
        logger.info("ML PIPELINE COMPLETED")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model retraining regardless of drift detection"
    )
    
    args = parser.parse_args()
    run_pipeline(force_retrain=args.force_retrain)