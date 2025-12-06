"""
Streamlit dashboard for Adaptive ML Health Monitor.
Displays metrics, drift detection results, and model versions.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="ML Health Monitor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

METRICS_FILE = DATA_DIR / "metrics.csv"
DRIFT_LOG_FILE = DATA_DIR / "drift_log.csv"
PIPELINE_SCRIPT = SRC_DIR / "pipeline.py"
RETRAIN_SCRIPT = SRC_DIR / "retrain.py"


# Helper functions

def load_metrics():
    """
    Load metrics history from CSV file.
    Returns DataFrame or None if file doesn't exist.
    """
    if not METRICS_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(METRICS_FILE)
        
        # Try to convert timestamp to datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


def load_drift_log():
    """
    Load drift detection log from CSV file.
    Returns DataFrame or None if file doesn't exist.
    """
    if not DRIFT_LOG_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(DRIFT_LOG_FILE)
        
        # Try to convert timestamp to datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Error loading drift log: {e}")
        return None


def list_models():
    """
    Scan models directory and return list of model files with metadata.
    Returns list of dictionaries with model information.
    """
    if not MODELS_DIR.exists():
        return []
    
    models = []
    
    for model_file in MODELS_DIR.glob("model_v*.pkl"):
        try:
            # Parse version number
            version_str = model_file.stem.replace("model_v", "")
            version = int(version_str)
            
            # Get file stats
            stat = model_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            models.append({
                'filename': model_file.name,
                'version': version,
                'size_mb': size_mb,
                'modified': modified,
                'path': model_file
            })
        except:
            continue
    
    # Sort by version number
    models.sort(key=lambda x: x['version'])
    
    return models


def get_drift_column(df):
    """
    Detect which column contains drift detection boolean.
    Returns column name or None if not found.
    """
    possible_columns = ['drift', 'drift_detected', 'is_drift', 'drift_flag']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None


def run_pipeline():
    """
    Execute the health check pipeline script.
    Returns (success: bool, output: str, error: str).
    """
    try:
        result = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Pipeline execution timed out after 5 minutes"
    except Exception as e:
        return False, "", str(e)


def run_manual_retrain():
    """
    Execute the manual retrain script.
    Returns (success: bool, output: str, error: str).
    """
    try:
        result = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Retrain execution timed out after 5 minutes"
    except Exception as e:
        return False, "", str(e)


# Display functions

def display_metrics_view():
    """Display metrics history and charts."""
    st.header("Model Performance Metrics")
    
    df = load_metrics()
    
    if df is None or df.empty:
        st.warning("No metrics logged yet. Run monitor.py to generate metrics.")
        return
    
    # Latest metrics summary
    st.subheader("Latest Metrics")
    latest = df.iloc[-1]
    
    cols = st.columns(4)
    
    if 'accuracy' in df.columns:
        cols[0].metric("Accuracy", f"{latest['accuracy']:.4f}")
    
    if 'precision' in df.columns:
        cols[1].metric("Precision", f"{latest['precision']:.4f}")
    
    if 'recall' in df.columns:
        cols[2].metric("Recall", f"{latest['recall']:.4f}")
    
    if 'f1_score' in df.columns or 'f1' in df.columns:
        f1_col = 'f1_score' if 'f1_score' in df.columns else 'f1'
        cols[3].metric("F1 Score", f"{latest[f1_col]:.4f}")
    
    # Metrics over time chart
    st.subheader("Metrics Over Time")
    
    chart_data = pd.DataFrame()
    
    if 'timestamp' in df.columns:
        chart_data['timestamp'] = df['timestamp']
    
    if 'accuracy' in df.columns:
        chart_data['accuracy'] = df['accuracy']
    
    if 'f1_score' in df.columns:
        chart_data['f1_score'] = df['f1_score']
    elif 'f1' in df.columns:
        chart_data['f1'] = df['f1']
    
    if not chart_data.empty:
        if 'timestamp' in chart_data.columns:
            chart_data = chart_data.set_index('timestamp')
        st.line_chart(chart_data)
    
    # Full metrics table
    st.subheader("All Metrics History")
    st.dataframe(df, use_container_width=True)


def display_drift_view():
    """Display drift detection history and status."""
    st.header("Drift Detection")
    
    df = load_drift_log()
    
    if df is None or df.empty:
        st.info("No drift events logged yet. Run drift.py to check for drift.")
        return
    
    # Latest drift status
    st.subheader("Latest Drift Check")
    latest = df.iloc[-1]
    
    cols = st.columns(3)
    
    # Timestamp
    if 'timestamp' in df.columns:
        cols[0].metric("Last Check", str(latest['timestamp']))
    
    # Drift detected status
    drift_col = get_drift_column(df)
    if drift_col:
        drift_detected = latest[drift_col]
        
        # Handle different types
        if isinstance(drift_detected, bool):
            drift_status = "Yes" if drift_detected else "No"
        elif isinstance(drift_detected, str):
            drift_status = "Yes" if drift_detected.lower() in ['true', 'yes', '1'] else "No"
        else:
            drift_status = "Yes" if drift_detected else "No"
        
        if drift_status == "Yes":
            cols[1].metric("Drift Detected", drift_status, delta="Warning", delta_color="inverse")
        else:
            cols[1].metric("Drift Detected", drift_status, delta="OK", delta_color="normal")
    
    # Accuracy comparison
    if 'baseline_accuracy' in df.columns and 'latest_accuracy' in df.columns:
        baseline = latest['baseline_accuracy']
        current = latest['latest_accuracy']
        cols[2].metric(
            "Accuracy Drop",
            f"{(baseline - current):.4f}",
            delta=f"{current:.4f} vs {baseline:.4f}"
        )
    elif 'baseline_accuracy' in df.columns and 'current_accuracy' in df.columns:
        baseline = latest['baseline_accuracy']
        current = latest['current_accuracy']
        cols[2].metric(
            "Accuracy Drop",
            f"{(baseline - current):.4f}",
            delta=f"{current:.4f} vs {baseline:.4f}"
        )
    
    # Drift over time chart
    if 'drop' in df.columns and 'threshold' in df.columns:
        st.subheader("Accuracy Drop vs Threshold")
        
        chart_data = pd.DataFrame({
            'drop': df['drop'],
            'threshold': df['threshold']
        })
        
        if 'timestamp' in df.columns:
            chart_data['timestamp'] = df['timestamp']
            chart_data = chart_data.set_index('timestamp')
        
        st.line_chart(chart_data)
    
    # Full drift log table
    st.subheader("All Drift Events")
    st.dataframe(df, use_container_width=True)


def display_models_view():
    """Display available model versions."""
    st.header("Model Versions")
    
    models = list_models()
    
    if not models:
        st.warning("No model files found in models/ directory.")
        return
    
    st.subheader("Available Models")
    
    # Create table data
    table_data = []
    latest_version = max(m['version'] for m in models)
    
    for model in models:
        is_latest = model['version'] == latest_version
        
        row = {
            'Version': f"v{model['version']}",
            'Filename': model['filename'],
            'Size (MB)': f"{model['size_mb']:.2f}",
            'Modified': model['modified'].strftime('%Y-%m-%d %H:%M:%S'),
            'Status': 'Current' if is_latest else ''
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Highlight current model
    current_model = [m for m in models if m['version'] == latest_version][0]
    st.success(f"Current production model: {current_model['filename']} (v{current_model['version']})")
    
    # Model statistics
    st.subheader("Model Statistics")
    cols = st.columns(3)
    cols[0].metric("Total Models", len(models))
    cols[1].metric("Latest Version", f"v{latest_version}")
    cols[2].metric("Total Size", f"{sum(m['size_mb'] for m in models):.2f} MB")


def display_all_views():
    """Display all views in one page."""
    display_metrics_view()
    st.divider()
    display_drift_view()
    st.divider()
    display_models_view()


# Main application

def main():
    # Sidebar
    st.sidebar.title("Adaptive ML Health Monitor")
    st.sidebar.markdown("---")
    
    # View selector
    view = st.sidebar.selectbox(
        "Select View",
        ["All", "Metrics", "Drift", "Models"]
    )
    
    st.sidebar.markdown("---")
    
    # Pipeline runner
    st.sidebar.subheader("Pipeline Control")
    
    if st.sidebar.button("Run Health Check Pipeline", type="primary"):
        with st.spinner("Running health check pipeline..."):
            success, stdout, stderr = run_pipeline()
        
        if success:
            st.sidebar.success("Pipeline run completed.")
            
            # Reload data
            st.rerun()
        else:
            st.sidebar.error("Pipeline execution failed.")
        
        # Show output in expander
        with st.sidebar.expander("Pipeline Output"):
            if stdout:
                st.text("STDOUT:")
                st.code(stdout)
            if stderr:
                st.text("STDERR:")
                st.code(stderr)
    
    # Manual retrain button
    if st.sidebar.button("Manual Retrain Model"):
        with st.spinner("Running manual retrain..."):
            success, stdout, stderr = run_manual_retrain()
        
        if success:
            st.sidebar.success("Manual retrain completed successfully.")
            
            # Reload data
            st.rerun()
        else:
            st.sidebar.error("Manual retrain failed. See logs for details.")
        
        # Show output in expander
        with st.sidebar.expander("Retrain Output"):
            if stdout:
                st.text("STDOUT:")
                st.code(stdout)
            if stderr:
                st.text("STDERR:")
                st.code(stderr)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Click 'Run Health Check Pipeline' to execute monitor, drift detection, and optional retraining.")
    
    # Main content
    st.title("Adaptive ML Health Monitor Dashboard")
    
    # Display selected view
    if view == "All":
        display_all_views()
    elif view == "Metrics":
        display_metrics_view()
    elif view == "Drift":
        display_drift_view()
    elif view == "Models":
        display_models_view()


if __name__ == "__main__":
    main()