# Adaptive ML Health Monitor

A production-ready MLOps system for monitoring machine learning model performance, detecting drift, and automatically triggering model retraining when performance degrades.

## Overview

This project implements a complete ML monitoring and maintenance pipeline that tracks model performance over time, detects when the model starts to degrade (drift), and automatically retrains new model versions to maintain accuracy. It includes a FastAPI prediction service, automated health checks, and a Streamlit dashboard for visualization and control.

## Features

- **Model Training**: Train baseline RandomForest classifier with versioned model storage
- **Prediction API**: FastAPI service with health checks and batch prediction support
- **Performance Monitoring**: Automated evaluation and metric logging (accuracy, precision, recall, F1)
- **Drift Detection**: Compare baseline vs. current performance to identify model degradation
- **Automated Retraining**: Trigger retraining automatically when drift is detected
- **Model Versioning**: Auto-increment model versions (model_v1.pkl → model_v2.pkl → ...)
- **Interactive Dashboard**: Streamlit UI for metrics visualization, drift analysis, and pipeline control
- **End-to-End Pipeline**: Orchestrated workflow from monitoring to retraining

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE ML HEALTH MONITOR                    │
└─────────────────────────────────────────────────────────────────┘

[1] TRAINING PHASE
    ├── train.py → Train initial model → model_v1.pkl
    └── Save to models/ directory

[2] PREDICTION SERVICE
    ├── predict.py (FastAPI) → Load model → Serve predictions
    └── Endpoints: /health, /predict, /predict/batch

[3] MONITORING PIPELINE
    ├── monitor.py → Evaluate model → Log metrics.csv
    ├── drift.py → Compare baseline vs current → Log drift_log.csv
    └── pipeline.py → Orchestrate: monitor → drift → retrain (if needed)

[4] AUTOMATED RETRAINING
    └── retrain.py → Train model_v{n+1}.pkl → Update metrics

[5] VISUALIZATION & CONTROL
    └── dashboard/app.py (Streamlit)
        ├── View metrics history
        ├── View drift events
        ├── List model versions
        └── Trigger pipeline/retrain manually

WORKFLOW:
train.py → model_v1.pkl → predict.py (API)
                ↓
         monitor.py → metrics.csv
                ↓
         drift.py → drift_log.csv
                ↓
    (if drift detected) → retrain.py → model_v2.pkl
                ↓
         dashboard/app.py (visualization)
```

## Project Structure
```
adaptive-ml-health-monitor/
│
├── data/
│   ├── churn.csv          # Synthetic dataset (or your custom data)
│   ├── metrics.csv        # Performance metrics log
│   └── drift_log.csv      # Drift detection results
│
├── models/
│   ├── model_v1.pkl       # Baseline model
│   ├── model_v2.pkl       # Retrained version 2
│   └── model_v3.pkl       # Retrained version 3 (auto-generated)
│
├── src/
│   ├── train.py           # Initial model training
│   ├── predict.py         # FastAPI prediction service
│   ├── monitor.py         # Performance monitoring
│   ├── drift.py           # Drift detection
│   ├── retrain.py         # Automated retraining with versioning
│   └── pipeline.py        # End-to-end orchestration
│
├── dashboard/
│   └── app.py             # Streamlit dashboard
│
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/adaptive-ml-health-monitor.git
cd adaptive-ml-health-monitor
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
scikit-learn
pandas
numpy
joblib
fastapi
uvicorn
pydantic
streamlit
```

## Usage

### Step 1: Train the Initial Model
```bash
python src/train.py
```

This creates `models/model_v1.pkl` and generates/uses `data/churn.csv`.

### Step 2: Start the Prediction API (Optional)
```bash
python src/predict.py
```

The API will be available at `http://localhost:8000`. Access the docs at `http://localhost:8000/docs`.

### Step 3: Run Performance Monitoring
```bash
python src/monitor.py
```

Evaluates the model and logs metrics to `data/metrics.csv`.

### Step 4: Check for Drift
```bash
python src/drift.py
```

Compares baseline vs. current accuracy and logs results to `data/drift_log.csv`.

### Step 5: Run the Complete Pipeline
```bash
python src/pipeline.py
```

Executes: monitor → drift detection → retrain (if drift detected).

### Step 6: Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

Opens an interactive dashboard at `http://localhost:8501` where you can:
- View metrics history and charts
- Check drift detection status
- See all model versions
- Manually trigger pipeline or retraining

## Key Scripts Explained

### `train.py`
Trains the initial RandomForest model on the churn dataset and saves it as `model_v1.pkl`. Creates synthetic data if `churn.csv` doesn't exist.

### `monitor.py`
Loads the current model, evaluates it on test data, computes metrics (accuracy, precision, recall, F1), and appends results to `metrics.csv` with timestamps.

### `drift.py`
Reads `metrics.csv`, compares the baseline accuracy (first row) with the latest accuracy (last row), and determines if performance has dropped beyond a threshold (default: 5%). Logs drift events to `drift_log.csv`.

### `retrain.py`
Automatically trains a new model version by:
1. Loading the dataset
2. Training a RandomForest classifier
3. Auto-incrementing the version number (e.g., `model_v2.pkl`, `model_v3.pkl`)
4. Logging new metrics to `metrics.csv`

### `pipeline.py`
Orchestrates the full workflow:
1. Runs `monitor.py` to evaluate current performance
2. Runs `drift.py` to detect performance degradation
3. Reads `drift_log.csv` to check if drift was detected
4. If drift is detected, automatically calls `retrain.py` to create a new model version

### `predict.py`
FastAPI service that:
- Loads the trained model on startup
- Provides `/health` endpoint for status checks
- Provides `/predict` endpoint for single predictions
- Provides `/predict/batch` endpoint for batch predictions

### `dashboard/app.py`
Streamlit dashboard that displays:
- **Metrics View**: Historical performance charts and data tables
- **Drift View**: Drift detection status and accuracy trends
- **Models View**: List of all model versions with metadata
- **Pipeline Control**: Buttons to trigger health check pipeline or manual retraining

## Example API Request

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "detail": "Model loaded and ready for predictions"
}
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -1.2, 0.8, 1.5, -0.3, 0.2, 1.1, -0.5, 0.7, -0.9,
                 0.4, 1.3, -0.6, 0.9, -1.1, 0.6, 0.3, -0.7, 1.0, -0.4]
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.7823
}
```

## Configuration

### Drift Detection Threshold

Edit `src/drift.py` to adjust the drift threshold:
```python
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop triggers drift alert
```

### Model Parameters

Edit `src/train.py` or `src/retrain.py` to modify RandomForest parameters:
```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
```

## Future Enhancements

- **Dockerization**: Containerize the entire system for easy deployment
- **Alert System**: Send email/Slack notifications when drift is detected
- **Cloud Deployment**: Deploy to AWS/GCP/Azure with managed ML services
- **Database Integration**: Replace CSV logging with PostgreSQL or MongoDB
- **Advanced Drift Detection**: Implement statistical tests (KS test, PSI) for feature drift
- **A/B Testing**: Support multiple model versions in production
- **Automated Scheduling**: Use Airflow or cron jobs for periodic monitoring
- **Model Registry**: Integrate with MLflow for experiment tracking
- **Real-time Monitoring**: Stream predictions and metrics to a monitoring dashboard
- **Multi-model Support**: Extend beyond RandomForest to support other algorithms

## How to Describe This Project in an Interview

"I built an end-to-end MLOps monitoring system that tracks machine learning model performance in production and automatically retrains models when performance degrades. The system includes five core components: first, a training pipeline that creates versioned models; second, a FastAPI prediction service for inference; third, a monitoring module that continuously evaluates model accuracy and logs metrics; fourth, a drift detection system that compares baseline performance against current performance and flags degradation; and fifth, an automated retraining pipeline that creates new model versions when drift is detected. I also built a Streamlit dashboard for visualizing metrics, drift events, and model versions, with controls to manually trigger retraining. The entire system is designed to minimize manual intervention and maintain model accuracy over time without human oversight. This demonstrates my ability to build production-ready ML systems with automated monitoring, versioning, and maintenance workflows."

## License

MIT License - feel free to use this project for learning or production purposes.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or feedback, please open an issue on GitHub.