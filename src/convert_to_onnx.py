"""
PyTorch to ONNX Model Conversion Module

This module handles the conversion of PyTorch models to ONNX format
for optimized inference deployment on Cerebrium.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import onnx
import torch
import torch.nn as nn
from onnx import checker

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.pytorch_model import Classifier, BasicBlock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """Handles conversion from PyTorch to ONNX format."""
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        """
        Initialize the ONNX converter.
        
        Args:
            input_shape: Input tensor shape (batch_size, channels, height, width)
        """
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pytorch_model(self, weights_path: str) -> nn.Module:
        """
        Load PyTorch model and weights.
        
        Args:
            weights_path: Path to the model weights file
            
        Returns:
            Loaded PyTorch model
        """
        try:
            # Initialize model with correct architecture
            model = Classifier(BasicBlock, [2, 2, 2, 2])
            
            # Load weights
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            else:
                logger.warning(f"Weights file not found at {weights_path}")
                
            model.eval()
            model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise
    
    def convert_to_onnx(
        self,
        pytorch_model: nn.Module,
        output_path: str,
        include_preprocessing: bool = True,
        opset_version: int = 11
    ) -> bool:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            output_path: Path to save ONNX model
            include_preprocessing: Whether to include preprocessing in ONNX model
            opset_version: ONNX opset version
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(self.input_shape).to(self.device)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify the converted model
            if self.verify_onnx_model(output_path):
                logger.info(f"Successfully converted model to ONNX: {output_path}")
                return True
            else:
                logger.error("ONNX model verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
            return False
    
    def verify_onnx_model(self, onnx_path: str) -> bool:
        """
        Verify the ONNX model is valid.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Load and check the model
            onnx_model = onnx.load(onnx_path)
            checker.check_model(onnx_model)
            
            # Print model info
            logger.info(f"ONNX model verification successful")
            logger.info(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
            logger.info(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def compare_outputs(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        test_input: Optional[torch.Tensor] = None,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Compare outputs between PyTorch and ONNX models.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            test_input: Test input tensor
            tolerance: Tolerance for output comparison
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        try:
            import onnxruntime as ort
            
            # Create test input if not provided
            if test_input is None:
                test_input = torch.randn(self.input_shape).to(self.device)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input).cpu().numpy()
            
            # Get ONNX output
            ort_session = ort.InferenceSession(onnx_path)
            onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]
            
            # Compare outputs
            diff = abs(pytorch_output - onnx_output).max()
            
            if diff < tolerance:
                logger.info(f"Output comparison successful. Max difference: {diff}")
                return True
            else:
                logger.warning(f"Output difference too large: {diff} > {tolerance}")
                return False
                
        except Exception as e:
            logger.error(f"Error comparing outputs: {e}")
            return False


def main():
    """Main function to convert PyTorch model to ONNX."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    weights_path = project_root / "models" / "pytorch_model_weights.pth"
    output_path = project_root / "models" / "model.onnx"
    
    # Initialize converter
    converter = ONNXConverter()
    
    try:
        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        pytorch_model = converter.load_pytorch_model(str(weights_path))
        
        # Convert to ONNX
        logger.info("Converting to ONNX...")
        success = converter.convert_to_onnx(pytorch_model, str(output_path))
        
        if success:
            # Compare outputs
            logger.info("Comparing outputs...")
            converter.compare_outputs(pytorch_model, str(output_path))
            
            logger.info("Conversion completed successfully!")
        else:
            logger.error("Conversion failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Conversion process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()