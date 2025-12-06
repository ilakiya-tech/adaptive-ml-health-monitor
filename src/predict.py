"""
FastAPI prediction service for the adaptive ML health monitoring system.
Provides REST API endpoints for model inference and health checks.
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Health Monitor - Prediction API",
    description="REST API for making predictions using trained ML model",
    version="1.0.0"
)

# Global variable to store loaded model
model = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "model_v1.pkl"


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    
    Attributes:
        features: List of feature values for a single data point.
    """
    features: List[float] = Field(
        ...,
        description="List of feature values for prediction",
        example=[0.5, -1.2, 0.8, 1.5, -0.3, 0.2, 1.1, -0.5, 0.7, -0.9,
                 0.4, 1.3, -0.6, 0.9, -1.1, 0.6, 0.3, -0.7, 1.0, -0.4]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features list is not empty."""
        if not v:
            raise ValueError("Features list cannot be empty")
        if len(v) != 20:
            raise ValueError(f"Expected 20 features, got {len(v)}")
        return v


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    
    Attributes:
        prediction: Predicted class (0 or 1).
        probability: Prediction probability (optional).
    """
    prediction: int = Field(..., description="Predicted class label")
    probability: Optional[float] = Field(None, description="Prediction probability for positive class")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Health status of the service.
        detail: Additional details about service status.
    """
    status: str = Field(..., description="Service status")
    detail: str = Field(..., description="Detailed status message")


def load_model_on_startup():
    """
    Load the trained model from disk on application startup.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
        Exception: If model loading fails.
    """
    global model
    
    logger.info(f"Loading model from {MODEL_PATH}")
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        error_msg = f"Model file not found at {MODEL_PATH}. Please train the model first using src/train.py"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Load model using joblib
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model features: {model.n_features_in_}")
        
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    Loads the model when the application starts.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION API SERVICE")
    logger.info("=" * 60)
    
    try:
        load_model_on_startup()
        logger.info("API service started successfully")
    except Exception as e:
        logger.error(f"Failed to start API service: {e}")
        # Continue running but model won't be available
        logger.warning("API will run but predictions will fail until model is loaded")


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dictionary with API information and available endpoints.
    """
    return {
        "message": "ML Health Monitor - Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        HealthResponse with service status and details.
    
    Raises:
        HTTPException: If model is not loaded.
    """
    if model is None:
        logger.warning("Health check failed: Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    return HealthResponse(
        status="ok",
        detail="Model loaded and ready for predictions"
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint that accepts features and returns model prediction.
    
    Args:
        request: PredictionRequest containing feature values.
    
    Returns:
        PredictionResponse with prediction and probability.
    
    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    # Check if model is loaded
    if model is None:
        logger.error("Prediction failed: Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert features list to 2D numpy array (model expects 2D input)
        features_array = np.array(request.features).reshape(1, -1)
        
        logger.info(f"Received prediction request with {len(request.features)} features")
        logger.debug(f"Features: {request.features}")
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability if model supports it
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_array)[0]
            probability = float(proba[1])  # Probability of positive class
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        else:
            logger.info(f"Prediction: {prediction}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
        
    except ValueError as e:
        logger.error(f"Validation error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint for multiple data points.
    
    Args:
        requests: List of PredictionRequest objects.
    
    Returns:
        List of PredictionResponse objects.
    
    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    # Check if model is loaded
    if model is None:
        logger.error("Batch prediction failed: Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert all feature lists to 2D array
        features_array = np.array([req.features for req in requests])
        
        logger.info(f"Received batch prediction request with {len(requests)} samples")
        
        # Make batch predictions
        predictions = model.predict(features_array)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba_array = model.predict_proba(features_array)
            probabilities = proba_array[:, 1]  # Probabilities of positive class
        
        # Create response list
        responses = []
        for i, pred in enumerate(predictions):
            prob = float(probabilities[i]) if probabilities is not None else None
            responses.append(PredictionResponse(
                prediction=int(pred),
                probability=prob
            ))
        
        logger.info(f"Batch prediction completed for {len(responses)} samples")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "predict:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )