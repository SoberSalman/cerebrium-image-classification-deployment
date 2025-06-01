"""
FastAPI Application for Image Classification on Cerebrium

This module provides the main FastAPI application for serving
the ONNX image classification model on Cerebrium platform.
"""

import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.model import ImageClassificationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="ImageNet classification using ONNX model on Cerebrium",
    version="1.0.0"
)

# Global variables
model_pipeline: Optional[ImageClassificationPipeline] = None


class PredictionRequest(BaseModel):
    """Request model for image prediction."""
    image_data: str  # Base64 or hex encoded image data
    image_path: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for image prediction."""
    class_id: int
    confidence: float
    top_predictions: List[Dict[str, float]]
    processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    timestamp: float


def load_model() -> ImageClassificationPipeline:
    """
    Load the ONNX model pipeline.
    
    Returns:
        Initialized ImageClassificationPipeline
    """
    try:
        # Use different paths for local vs container
        if os.path.exists("/app/models/model.onnx"):
            model_path = "/app/models/model.onnx"  # Container path
        elif os.path.exists("models/model.onnx"):
            model_path = "models/model.onnx"  # Local path
        else:
            raise FileNotFoundError("Model file not found in any expected location")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Initialize pipeline with appropriate providers
        providers = ['CPUExecutionProvider']
        
        # Try to use GPU if available
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("GPU acceleration enabled")
        except Exception:
            logger.info("Using CPU-only inference")
        
        pipeline = ImageClassificationPipeline(model_path, providers)
        logger.info("Model pipeline loaded successfully")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_pipeline
    
    try:
        logger.info("Loading model pipeline...")
        model_pipeline = load_model()
        logger.info("✅ Application startup completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy" if model_pipeline is not None else "unhealthy",
        model_loaded=model_pipeline is not None,
        timestamp=time.time()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """
    Predict image class using the loaded model.
    
    Args:
        request: Prediction request with image data
        
    Returns:
        Prediction results with class ID and probabilities
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        start_time = time.time()
        
        # Decode image data
        try:
            # Try hex decoding first
            image_bytes = bytes.fromhex(request.image_data)
        except ValueError:
            # If hex fails, try base64
            import base64
            try:
                image_bytes = base64.b64decode(request.image_data)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image data format. Use hex or base64 encoding."
                )
        
        # Create BytesIO buffer
        image_buffer = io.BytesIO(image_bytes)
        
        # Get predictions
        top_predictions = model_pipeline.classify_image(image_buffer, top_k=5)
        
        # Get top class
        predicted_class = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        processing_time = time.time() - start_time
        
        # Format top predictions
        formatted_predictions = [
            {"class_id": int(class_id), "confidence": float(prob)}
            for class_id, prob in top_predictions
        ]
        
        logger.info(f"Prediction completed: class={predicted_class}, "
                   f"confidence={confidence:.4f}, time={processing_time:.3f}s")
        
        return PredictionResponse(
            class_id=predicted_class,
            confidence=confidence,
            top_predictions=formatted_predictions,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/upload")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Predict image class from uploaded file.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        start_time = time.time()
        
        # Read file data
        image_data = await file.read()
        image_buffer = io.BytesIO(image_data)
        
        # Get predictions
        top_predictions = model_pipeline.classify_image(image_buffer, top_k=5)
        
        # Get top class
        predicted_class = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        processing_time = time.time() - start_time
        
        # Format response
        formatted_predictions = [
            {"class_id": int(class_id), "confidence": float(prob)}
            for class_id, prob in top_predictions
        ]
        
        logger.info(f"File prediction completed: class={predicted_class}, "
                   f"confidence={confidence:.4f}, time={processing_time:.3f}s")
        
        return PredictionResponse(
            class_id=predicted_class,
            confidence=confidence,
            top_predictions=formatted_predictions,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_upload": "/predict/upload",
            "docs": "/docs"
        },
        "model_status": "loaded" if model_pipeline is not None else "not_loaded"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    
    Args:
        request: FastAPI request object
        exc: Exception that occurred
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check server logs."
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )