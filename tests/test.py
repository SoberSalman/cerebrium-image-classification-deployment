"""
Comprehensive Test Suite for ML Model Deployment

This module contains tests for all components of the ML deployment pipeline
including preprocessing, model loading, inference, and ONNX conversion.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model import ImagePreprocessor, ONNXModelLoader, ImageClassificationPipeline


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for image preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self):
        """Create a temporary test image."""
        # Create a random RGB image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.jpg', delete=False
        )
        test_image.save(temp_file.name)
        temp_file.close()
        
        return temp_file.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_load_image(self):
        """Test image loading functionality."""
        image = self.preprocessor.load_image(self.test_image_path)
        
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(image.shape), 3)  # Height, Width, Channels
        self.assertEqual(image.shape[2], 3)    # RGB channels
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        # Create test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Resize image
        resized = self.preprocessor.resize_image(test_image)
        
        self.assertEqual(resized.shape[:2], (224, 224))
        self.assertEqual(resized.shape[2], 3)
    
    def test_normalize_image(self):
        """Test image normalization functionality."""
        # Create test image with known values
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Normalize image
        normalized = self.preprocessor.normalize_image(test_image)
        
        self.assertEqual(normalized.shape, (224, 224, 3))
        self.assertTrue(normalized.dtype == np.float32)
        
        # Check if normalization is applied correctly
        expected = (128.0 / 255.0 - self.preprocessor.mean) / self.preprocessor.std
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
    
    def test_preprocess_numpy(self):
        """Test complete preprocessing pipeline for numpy array."""
        # Create test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Preprocess image
        preprocessed = self.preprocessor.preprocess_numpy(test_image)
        
        # Check output shape and type
        self.assertEqual(preprocessed.shape, (1, 3, 224, 224))  # NCHW format
        self.assertTrue(preprocessed.dtype == np.float32)
    
    def test_preprocess_from_path(self):
        """Test preprocessing from image path."""
        preprocessed = self.preprocessor.preprocess_from_path(self.test_image_path)
        
        # Check output shape and type
        self.assertEqual(preprocessed.shape, (1, 3, 224, 224))
        self.assertTrue(preprocessed.dtype == np.float32)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        with self.assertRaises(Exception):
            self.preprocessor.load_image("nonexistent_image.jpg")


class TestONNXModelLoader(unittest.TestCase):
    """Test cases for ONNX model loading and inference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = self._create_dummy_onnx_model()
    
    def _create_dummy_onnx_model(self):
        """Create a dummy ONNX model for testing."""
        # This is a mock test - in real scenario, we'd need an actual ONNX model
        return "dummy_model.onnx"
    
    def tearDown(self):
        """Clean up test fixtures."""
        pass
    
    @patch('onnxruntime.InferenceSession')
    def test_model_loading(self, mock_session):
        """Test ONNX model loading with mocked session."""
        # Mock the session and its methods
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_session.return_value = mock_instance
        
        # Test model loading
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader(self.model_path)
            
            self.assertIsNotNone(loader.session)
            self.assertEqual(loader.input_name, 'input')
            self.assertEqual(loader.output_name, 'output')
    
    @patch('onnxruntime.InferenceSession')
    def test_predict(self, mock_session):
        """Test model prediction functionality."""
        # Mock the session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        # Test prediction
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader(self.model_path)
            
            # Create test input
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Run prediction
            result = loader.predict(test_input)
            
            self.assertIsInstance(result, np.ndarray)
            mock_instance.run.assert_called_once()
    
    @patch('onnxruntime.InferenceSession')
    def test_predict_probabilities(self, mock_session):
        """Test probability prediction functionality."""
        # Mock the session with logits output
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader(self.model_path)
            
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            probabilities = loader.predict_probabilities(test_input)
            
            # Check if probabilities sum to 1
            self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
            self.assertTrue(np.all(probabilities >= 0))
    
    @patch('onnxruntime.InferenceSession')
    def test_top_predictions(self, mock_session):
        """Test top-k predictions functionality."""
        # Mock the session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        
        # Create mock logits with known top classes
        logits = np.random.randn(1, 1000)
        logits[0, 0] = 10  # Highest
        logits[0, 1] = 9   # Second highest
        logits[0, 2] = 8   # Third highest
        mock_instance.run.return_value = [logits]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader(self.model_path)
            
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            top_predictions = loader.get_top_predictions(test_input, top_k=3)
            
            self.assertEqual(len(top_predictions), 3)
            self.assertEqual(top_predictions[0][0], 0)  # Top class should be 0
            self.assertEqual(top_predictions[1][0], 1)  # Second class should be 1
            self.assertEqual(top_predictions[2][0], 2)  # Third class should be 2


class TestONNXConversion(unittest.TestCase):
    """Test cases for ONNX conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
    
    @patch('torch.onnx.export')
    @patch('onnx.load')
    @patch('onnx.checker.check_model')
    def test_onnx_conversion_process(self, mock_checker, mock_load, mock_export):
        """Test ONNX conversion process with mocked dependencies."""
        from src.convert_to_onnx import ONNXConverter
        
        # Mock successful export
        mock_export.return_value = None
        mock_load.return_value = Mock()
        mock_checker.return_value = None
        
        converter = ONNXConverter()
        
        # Create a mock PyTorch model
        mock_model = Mock()
        
        # Test conversion
        with patch('os.makedirs'), patch('os.path.dirname'):
            result = converter.convert_to_onnx(
                mock_model, 
                "test_model.onnx"
            )
            
            # Verify export was called
            mock_export.assert_called_once()
            self.assertTrue(result)
    
    def test_converter_initialization(self):
        """Test ONNX converter initialization."""
        from src.convert_to_onnx import ONNXConverter
        
        converter = ONNXConverter()
        
        self.assertEqual(converter.input_shape, (1, 3, 224, 224))
        self.assertIsNotNone(converter.device)


class TestImageClassificationPipeline(unittest.TestCase):
    """Test cases for the complete classification pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image_path = self._create_test_image()
        self.model_path = "dummy_model.onnx"
    
    def _create_test_image(self):
        """Create a temporary test image."""
        test_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.jpg', delete=False
        )
        test_image.save(temp_file.name)
        temp_file.close()
        
        return temp_file.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    @patch('onnxruntime.InferenceSession')
    def test_classify_image(self, mock_session):
        """Test end-to-end image classification."""
        # Mock the ONNX session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            pipeline = ImageClassificationPipeline(self.model_path)
            
            # Test classification
            predictions = pipeline.classify_image(self.test_image_path, top_k=5)
            
            self.assertEqual(len(predictions), 5)
            for class_idx, prob in predictions:
                self.assertIsInstance(class_idx, int)
                self.assertIsInstance(prob, float)
                self.assertTrue(0 <= prob <= 1)
    
    @patch('onnxruntime.InferenceSession')
    def test_get_class_prediction(self, mock_session):
        """Test getting single class prediction."""
        # Mock the ONNX session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        
        # Set up mock to return class 42 as top prediction
        logits = np.random.randn(1, 1000)
        logits[0, 42] = 10  # Make class 42 have highest logit
        mock_instance.run.return_value = [logits]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            pipeline = ImageClassificationPipeline(self.model_path)
            
            # Test single class prediction
            predicted_class = pipeline.get_class_prediction(self.test_image_path)
            
            self.assertEqual(predicted_class, 42)
            self.assertIsInstance(predicted_class, int)
    
    @patch('onnxruntime.InferenceSession')
    def test_pipeline_with_bytesio(self, mock_session):
        """Test pipeline with BytesIO input."""
        from io import BytesIO
        
        # Mock the ONNX session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            pipeline = ImageClassificationPipeline(self.model_path)
            
            # Create BytesIO buffer with image data
            test_image = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            buffer = BytesIO()
            test_image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # Test classification
            predictions = pipeline.classify_image(buffer, top_k=3)
            
            self.assertEqual(len(predictions), 3)
            for class_idx, prob in predictions:
                self.assertIsInstance(class_idx, int)
                self.assertIsInstance(prob, float)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation and edge cases."""
    
    def test_preprocessing_edge_cases(self):
        """Test preprocessing with edge cases."""
        preprocessor = ImagePreprocessor()
        
        # Test with very small image
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = preprocessor.preprocess_numpy(small_image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        
        # Test with very large image
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        result = preprocessor.preprocess_numpy(large_image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        
        # Test with grayscale image (should be converted to RGB)
        gray_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode='L'
        )
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        gray_image.save(temp_file.name)
        temp_file.close()
        
        try:
            result = preprocessor.preprocess_from_path(temp_file.name)
            self.assertEqual(result.shape, (1, 3, 224, 224))
        finally:
            os.unlink(temp_file.name)
    
    def test_normalization_values(self):
        """Test that normalization produces expected value ranges."""
        preprocessor = ImagePreprocessor()
        
        # Test with all-white image
        white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
        normalized = preprocessor.normalize_image(white_image)
        
        # Calculate expected values for white image
        expected = (1.0 - preprocessor.mean) / preprocessor.std
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
        
        # Test with all-black image
        black_image = np.zeros((224, 224, 3), dtype=np.uint8)
        normalized = preprocessor.normalize_image(black_image)
        
        # Calculate expected values for black image
        expected = (0.0 - preprocessor.mean) / preprocessor.std
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
    
    def test_input_data_types(self):
        """Test handling of different input data types."""
        preprocessor = ImagePreprocessor()
        
        # Test with uint8 input
        uint8_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = preprocessor.preprocess_numpy(uint8_image)
        self.assertEqual(result.dtype, np.float32)
        
        # Test with float32 input (already normalized)
        float32_image = np.random.random((224, 224, 3)).astype(np.float32) * 255
        result = preprocessor.preprocess_numpy(float32_image)
        self.assertEqual(result.dtype, np.float32)
    
    def test_channel_order_validation(self):
        """Test that RGB channel order is maintained."""
        preprocessor = ImagePreprocessor()
        
        # Create image with distinct RGB values
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel
        test_image[:, :, 1] = 128  # Green channel
        test_image[:, :, 2] = 64   # Blue channel
        
        result = preprocessor.preprocess_numpy(test_image)
        
        # Check that channels are in correct order (CHW format)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        
        # Red channel should have highest values after normalization
        red_mean = np.mean(result[0, 0, :, :])
        green_mean = np.mean(result[0, 1, :, :])
        blue_mean = np.mean(result[0, 2, :, :])
        
        self.assertGreater(red_mean, green_mean)
        self.assertGreater(green_mean, blue_mean)


class TestKnownTestCases(unittest.TestCase):
    """Test cases for known test images."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.project_root = Path(__file__).parent.parent
    
    def test_tench_image_preprocessing(self):
        """Test preprocessing of the tench test image."""
        tench_path = self.project_root / "assets" / "n01440764_tench.jpg"
        
        if tench_path.exists():
            preprocessed = self.preprocessor.preprocess_from_path(str(tench_path))
            
            # Verify shape and type
            self.assertEqual(preprocessed.shape, (1, 3, 224, 224))
            self.assertTrue(preprocessed.dtype == np.float32)
            
            # Verify value ranges (should be roughly in [-2, 2] after normalization)
            self.assertTrue(preprocessed.min() >= -3)
            self.assertTrue(preprocessed.max() <= 3)
    
    def test_turtle_image_preprocessing(self):
        """Test preprocessing of the turtle test image."""
        turtle_path = self.project_root / "assets" / "n01667114_mud_turtle.jpg"
        
        if turtle_path.exists():
            preprocessed = self.preprocessor.preprocess_from_path(str(turtle_path))
            
            # Verify shape and type
            self.assertEqual(preprocessed.shape, (1, 3, 224, 224))
            self.assertTrue(preprocessed.dtype == np.float32)
            
            # Verify value ranges
            self.assertTrue(preprocessed.min() >= -3)
            self.assertTrue(preprocessed.max() <= 3)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and robustness."""
    
    def test_model_file_not_found(self):
        """Test handling of missing model file."""
        with self.assertRaises(FileNotFoundError):
            ONNXModelLoader("nonexistent_model.onnx")
    
    def test_invalid_image_formats(self):
        """Test handling of invalid image formats."""
        preprocessor = ImagePreprocessor()
        
        # Test with text file
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w')
        temp_file.write("This is not an image")
        temp_file.close()
        
        try:
            with self.assertRaises(Exception):
                preprocessor.preprocess_from_path(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_corrupted_image_data(self):
        """Test handling of corrupted image data."""
        preprocessor = ImagePreprocessor()
        
        # Create file with invalid image data
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, mode='wb')
        temp_file.write(b"corrupted image data")
        temp_file.close()
        
        try:
            with self.assertRaises(Exception):
                preprocessor.preprocess_from_path(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    @patch('onnxruntime.InferenceSession')
    def test_inference_error_handling(self, mock_session):
        """Test error handling during inference."""
        # Mock session that raises an exception
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.side_effect = RuntimeError("Inference failed")
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader("dummy_model.onnx")
            
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            with self.assertRaises(RuntimeError):
                loader.predict(test_input)


class TestMemoryAndResourceManagement(unittest.TestCase):
    """Test cases for memory usage and resource management."""
    
    def test_memory_cleanup(self):
        """Test that temporary objects are cleaned up properly."""
        preprocessor = ImagePreprocessor()
        
        # Process multiple images and check memory doesn't grow excessively
        initial_objects = len([obj for obj in globals() if isinstance(obj, np.ndarray)])
        
        for i in range(10):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            result = preprocessor.preprocess_numpy(test_image)
            del result  # Explicit cleanup
        
        # Memory should not have grown significantly
        final_objects = len([obj for obj in globals() if isinstance(obj, np.ndarray)])
        self.assertLessEqual(final_objects - initial_objects, 5)
    
    def test_large_batch_processing(self):
        """Test processing of larger batches doesn't cause memory issues."""
        preprocessor = ImagePreprocessor()
        
        # Test with larger image that might cause memory issues
        large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        # Should not raise memory error
        result = preprocessor.preprocess_numpy(large_image)
        self.assertEqual(result.shape, (1, 3, 224, 224))


class TestConcurrencyAndThreadSafety(unittest.TestCase):
    """Test cases for concurrent usage and thread safety."""
    
    def test_concurrent_preprocessing(self):
        """Test that preprocessing can handle concurrent calls."""
        import threading
        import queue
        
        preprocessor = ImagePreprocessor()
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def process_image(thread_id):
            try:
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                result = preprocessor.preprocess_numpy(test_image)
                results_queue.put((thread_id, result.shape))
            except Exception as e:
                errors_queue.put((thread_id, str(e)))
        
        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_image, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(results_queue.qsize(), 5)
        self.assertEqual(errors_queue.qsize(), 0)
        
        # Verify all results are correct
        while not results_queue.empty():
            thread_id, shape = results_queue.get()
            self.assertEqual(shape, (1, 3, 224, 224))


class TestIntegrationWithRealFiles(unittest.TestCase):
    """Integration tests with real file operations."""
    
    def setUp(self):
        """Set up test fixtures with real image files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = []
        
        # Create various test images
        formats = ['JPEG', 'PNG', 'BMP']
        sizes = [(100, 100), (224, 224), (512, 512)]
        
        for fmt in formats:
            for size in sizes:
                image = Image.fromarray(
                    np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
                )
                
                ext = fmt.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                filename = f"test_{fmt}_{size[0]}x{size[1]}.{ext}"
                filepath = os.path.join(self.temp_dir, filename)
                
                image.save(filepath, format=fmt)
                self.test_images.append(filepath)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_real_file_processing(self):
        """Test processing of real image files in various formats."""
        preprocessor = ImagePreprocessor()
        
        for image_path in self.test_images:
            with self.subTest(image=os.path.basename(image_path)):
                try:
                    result = preprocessor.preprocess_from_path(image_path)
                    self.assertEqual(result.shape, (1, 3, 224, 224))
                    self.assertEqual(result.dtype, np.float32)
                except Exception as e:
                    self.fail(f"Failed to process {image_path}: {e}")
    
    @patch('onnxruntime.InferenceSession')
    def test_end_to_end_with_real_files(self, mock_session):
        """Test complete pipeline with real image files."""
        # Mock ONNX session
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            pipeline = ImageClassificationPipeline("dummy_model.onnx")
            
            for image_path in self.test_images[:3]:  # Test subset to save time
                with self.subTest(image=os.path.basename(image_path)):
                    try:
                        predictions = pipeline.classify_image(image_path, top_k=3)
                        self.assertEqual(len(predictions), 3)
                        
                        for class_id, confidence in predictions:
                            self.assertIsInstance(class_id, int)
                            self.assertIsInstance(confidence, float)
                            self.assertTrue(0 <= confidence <= 1)
                    except Exception as e:
                        self.fail(f"Pipeline failed for {image_path}: {e}")


class TestAPICompatibility(unittest.TestCase):
    """Test cases for API compatibility and interface contracts."""
    
    def test_preprocessor_interface(self):
        """Test that ImagePreprocessor maintains expected interface."""
        preprocessor = ImagePreprocessor()
        
        # Check required attributes
        self.assertTrue(hasattr(preprocessor, 'mean'))
        self.assertTrue(hasattr(preprocessor, 'std'))
        self.assertTrue(hasattr(preprocessor, 'target_size'))
        
        # Check required methods
        self.assertTrue(hasattr(preprocessor, 'load_image'))
        self.assertTrue(hasattr(preprocessor, 'resize_image'))
        self.assertTrue(hasattr(preprocessor, 'normalize_image'))
        self.assertTrue(hasattr(preprocessor, 'preprocess_numpy'))
        self.assertTrue(hasattr(preprocessor, 'preprocess_from_path'))
        
        # Check method signatures
        import inspect
        
        # load_image should accept path parameter
        sig = inspect.signature(preprocessor.load_image)
        self.assertIn('image_path', sig.parameters)
        
        # preprocess_numpy should accept image parameter
        sig = inspect.signature(preprocessor.preprocess_numpy)
        self.assertIn('image', sig.parameters)
    
    @patch('onnxruntime.InferenceSession')
    def test_model_loader_interface(self, mock_session):
        """Test that ONNXModelLoader maintains expected interface."""
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            loader = ONNXModelLoader("dummy_model.onnx")
            
            # Check required attributes
            self.assertTrue(hasattr(loader, 'model_path'))
            self.assertTrue(hasattr(loader, 'session'))
            self.assertTrue(hasattr(loader, 'input_name'))
            self.assertTrue(hasattr(loader, 'output_name'))
            
            # Check required methods
            self.assertTrue(hasattr(loader, 'predict'))
            self.assertTrue(hasattr(loader, 'predict_probabilities'))
            self.assertTrue(hasattr(loader, 'get_top_predictions'))
    
    @patch('onnxruntime.InferenceSession')
    def test_pipeline_interface(self, mock_session):
        """Test that ImageClassificationPipeline maintains expected interface."""
        mock_instance = Mock()
        mock_instance.get_inputs.return_value = [Mock(name='input')]
        mock_instance.get_outputs.return_value = [Mock(name='output')]
        mock_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_instance.run.return_value = [np.random.randn(1, 1000)]
        mock_session.return_value = mock_instance
        
        with patch('os.path.exists', return_value=True):
            pipeline = ImageClassificationPipeline("dummy_model.onnx")
            
            # Check required attributes
            self.assertTrue(hasattr(pipeline, 'preprocessor'))
            self.assertTrue(hasattr(pipeline, 'model'))
            
            # Check required methods
            self.assertTrue(hasattr(pipeline, 'classify_image'))
            self.assertTrue(hasattr(pipeline, 'get_class_prediction'))


class TestRegressionPrevention(unittest.TestCase):
    """Test cases to prevent regression of known issues."""
    
    def test_image_channel_order_regression(self):
        """Prevent regression of RGB/BGR channel order issues."""
        preprocessor = ImagePreprocessor()
        
        # Create image with known RGB pattern
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Pure red
        
        result = preprocessor.preprocess_numpy(test_image)
        
        # After preprocessing, red channel should have highest normalized values
        # (accounting for ImageNet normalization)
        red_channel = result[0, 0, :, :]  # First channel in CHW format
        green_channel = result[0, 1, :, :]
        blue_channel = result[0, 2, :, :]
        
        # Red should be significantly different from green and blue
        self.assertGreater(
            abs(red_channel.mean() - green_channel.mean()), 0.5,
            "RGB channel order may be incorrect"
        )
    
    def test_normalization_consistency_regression(self):
        """Prevent regression of normalization inconsistencies."""
        preprocessor = ImagePreprocessor()
        
        # Test that same image produces same result
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result1 = preprocessor.preprocess_numpy(test_image.copy())
        result2 = preprocessor.preprocess_numpy(test_image.copy())
        
        np.testing.assert_array_equal(result1, result2,
                                     "Preprocessing should be deterministic")
    
    def test_shape_consistency_regression(self):
        """Prevent regression of shape inconsistencies."""
        preprocessor = ImagePreprocessor()
        
        # Test various input shapes all produce consistent output
        test_shapes = [(50, 50, 3), (100, 200, 3), (300, 300, 3), (1000, 500, 3)]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                test_image = np.random.randint(0, 255, shape, dtype=np.uint8)
                result = preprocessor.preprocess_numpy(test_image)
                
                self.assertEqual(result.shape, (1, 3, 224, 224),
                               f"Output shape inconsistent for input {shape}")


# Add comprehensive test runner with detailed reporting
class DetailedTestResult(unittest.TextTestResult):
    """Enhanced test result class with detailed reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append(('PASS', test._testMethodName, test.__class__.__name__))
    
    def addError(self, test, err):
        super().addError(test, err)
        self.test_results.append(('ERROR', test._testMethodName, test.__class__.__name__, str(err[1])))
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_results.append(('FAIL', test._testMethodName, test.__class__.__name__, str(err[1])))


def run_all_tests():
    """Run all test suites with enhanced reporting."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes in logical order
    test_classes = [
        TestImagePreprocessor,
        TestONNXModelLoader,
        TestONNXConversion,
        TestImageClassificationPipeline,
        TestDataValidation,
        TestKnownTestCases,
        TestErrorHandling,
        TestMemoryAndResourceManagement,
        TestConcurrencyAndThreadSafety,
        TestIntegrationWithRealFiles,
        TestAPICompatibility,
        TestRegressionPrevention
        # Removed TestPerformance since it's not defined
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with simple runner
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running comprehensive test suite...")
    print("=" * 50)
    
    success = run_all_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)