"""
ONNX Model Inference and Image Preprocessing Module

This module provides classes for loading ONNX models and preprocessing images
for the ImageNet classification task.
"""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for ImageNet classification."""
    
    def __init__(self):
        """Initialize the image preprocessor with ImageNet constants."""
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
        
    def load_image(self, image_path: Union[str, BytesIO]) -> np.ndarray:
        """
        Load image from file path or BytesIO buffer.
        
        Args:
            image_path: Path to image file or BytesIO buffer
            
        Returns:
            Loaded image as numpy array
        """
        try:
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            else:
                image = Image.open(image_path)
                
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image using bilinear interpolation.
        
        Args:
            image: Input image array
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
            
        try:
            # OpenCV expects (width, height) for resize
            resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Input image array (0-255 range)
            
        Returns:
            Normalized image array
        """
        try:
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            image = (image - self.mean) / self.std
            
            return image
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            raise
    
    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for numpy array.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model inference
        """
        try:
            # Resize image
            image = self.resize_image(image)
            
            # Normalize image
            image = self.normalize_image(image)
            
            # Convert from HWC to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise
    
    def preprocess_from_path(self, image_path: Union[str, BytesIO]) -> np.ndarray:
        """
        Complete preprocessing pipeline from image path.
        
        Args:
            image_path: Path to image file or BytesIO buffer
            
        Returns:
            Preprocessed image ready for model inference
        """
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Apply preprocessing
            return self.preprocess_numpy(image)
            
        except Exception as e:
            logger.error(f"Error preprocessing image from path: {e}")
            raise


class ONNXModelLoader:
    """Handles ONNX model loading and inference."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize ONNX model loader.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        
        # Set default providers if not specified
        if providers is None:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        self.providers = providers
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and initialize session."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=self.providers
            )
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Successfully loaded ONNX model from {self.model_path}")
            logger.info(f"Available providers: {self.session.get_providers()}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output name: {self.output_name}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed input array
            
        Returns:
            Model predictions
        """
        try:
            if self.session is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Ensure input is float32
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def predict_probabilities(self, input_data: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using softmax.
        
        Args:
            input_data: Preprocessed input array
            
        Returns:
            Prediction probabilities
        """
        try:
            # Get raw predictions
            logits = self.predict(input_data)
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error computing probabilities: {e}")
            raise
    
    def get_top_predictions(
        self,
        input_data: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top-k predictions with class indices and probabilities.
        
        Args:
            input_data: Preprocessed input array
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_index, probability) tuples
        """
        try:
            # Get probabilities
            probabilities = self.predict_probabilities(input_data)
            
            # Get top-k indices
            top_indices = np.argsort(probabilities[0])[::-1][:top_k]
            
            # Return class indices and probabilities
            results = [
                (int(idx), float(probabilities[0][idx]))
                for idx in top_indices
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            raise


class ImageClassificationPipeline:
    """Complete image classification pipeline combining preprocessing and inference."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize the classification pipeline.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers
        """
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModelLoader(model_path, providers)
    
    def classify_image(
        self,
        image_path: Union[str, BytesIO],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Classify an image and return top-k predictions.
        
        Args:
            image_path: Path to image file or BytesIO buffer
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_index, probability) tuples
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocessor.preprocess_from_path(image_path)
            
            # Get predictions
            predictions = self.model.get_top_predictions(preprocessed_image, top_k)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            raise
    
    def get_class_prediction(self, image_path: Union[str, BytesIO]) -> int:
        """
        Get the predicted class index for an image.
        
        Args:
            image_path: Path to image file or BytesIO buffer
            
        Returns:
            Predicted class index
        """
        try:
            predictions = self.classify_image(image_path, top_k=1)
            return predictions[0][0]
            
        except Exception as e:
            logger.error(f"Error getting class prediction: {e}")
            raise