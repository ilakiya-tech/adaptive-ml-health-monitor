import subprocess
import sys
import pandas as pd
from pathlib import Path

def ensure_initial_model():
    """
    Check if the initial model exists. If not, automatically run train.py.
    This is critical for Streamlit Cloud deployments where models/ is empty.
    """
    model_path = Path("models/model_v1.pkl")
    
    print("\n[PRE-CHECK] Verifying initial model exists...")
    
    if model_path.exists():
        print(f"[OK] Found existing model at {model_path}")
        return
    
    print(f"[WARN] Model not found at {model_path}")
    print("[INFO] Running train.py to create initial model...")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[OK] train.py completed successfully")
        if result.stdout:
            print(result.stdout)
        
        # Verify model was created
        if model_path.exists():
            print(f"[OK] Initial model created at {model_path}")
        else:
            print("[ERROR] train.py completed but model file was not created")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print("[ERROR] train.py failed with error:")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error running train.py: {e}")
        sys.exit(1)


def run_pipeline():
    """
    Main pipeline that orchestrates monitoring, drift detection, and retraining.
    """
    print("=" * 60)
    print("ADAPTIVE ML HEALTH MONITOR - PIPELINE STARTED")
    print("=" * 60)
    
    # Pre-check: Ensure initial model exists (critical for Streamlit Cloud)
    ensure_initial_model()
    
    # Step 1: Run monitor.py
    print("\n[STEP 1] Running monitor.py...")
    try:
        result = subprocess.run(
            [sys.executable, "src/monitor.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[OK] monitor.py completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[ERROR] monitor.py failed with error:")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error running monitor.py: {e}")
        sys.exit(1)
    
    # Step 2: Run drift.py
    print("\n[STEP 2] Running drift.py...")
    try:
        result = subprocess.run(
            [sys.executable, "src/drift.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[OK] drift.py completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[ERROR] drift.py failed with error:")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error running drift.py: {e}")
        sys.exit(1)
    
    # Step 3: Read drift log
    print("\n[STEP 3] Reading drift_log.csv...")
    drift_log_path = Path("data/drift_log.csv")
    
    if not drift_log_path.exists():
        print("[ERROR] drift_log.csv not found. Cannot proceed with drift detection.")
        sys.exit(1)
    
    try:
        drift_df = pd.read_csv(drift_log_path)
        
        if drift_df.empty:
            print("[ERROR] drift_log.csv is empty. Cannot detect drift.")
            sys.exit(1)
        
        print(f"[OK] Loaded drift_log.csv with {len(drift_df)} rows")
        
        # Step 4: Detect drift
        print("\n[STEP 4] Detecting drift...")
        
        # Get the latest row
        latest_row = drift_df.iloc[-1]
        
        # Try different possible column names for drift detection
        drift_column_names = ['drift', 'drift_detected', 'is_drift', 'drift_flag']
        drift_detected = False
        drift_column_found = None
        
        for col_name in drift_column_names:
            if col_name in drift_df.columns:
                drift_column_found = col_name
                drift_value = latest_row[col_name]
                
                # Handle different data types
                if isinstance(drift_value, bool):
                    drift_detected = drift_value
                elif isinstance(drift_value, str):
                    drift_detected = drift_value.lower() in ['true', 'yes', '1', 'drift']
                elif isinstance(drift_value, (int, float)):
                    drift_detected = bool(drift_value)
                
                break
        
        if drift_column_found is None:
            print(f"[WARN] Warning: No drift column found. Checked columns: {drift_column_names}")
            print(f"Available columns: {list(drift_df.columns)}")
            print("Assuming no drift detected.")
            drift_detected = False
        else:
            print(f"[OK] Drift column found: '{drift_column_found}'")
            print(f"  Latest drift status: {drift_detected}")
        
        # Step 5: Trigger retraining if drift detected
        if drift_detected:
            print("\n[STEP 5] Drift detected! Triggering retraining...")
            try:
                # Import and call retrain.py's main function
                sys.path.insert(0, str(Path("src").resolve()))
                import retrain
                
                print("Starting retrain.main()...")
                retrain.main()
                print("[OK] Retraining completed successfully")
                
            except Exception as e:
                print(f"[ERROR] Error during retraining: {e}")
                sys.exit(1)
        else:
            print("\n[STEP 5] No drift detected. Skipping retraining.")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] Error processing drift_log.csv: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
```
