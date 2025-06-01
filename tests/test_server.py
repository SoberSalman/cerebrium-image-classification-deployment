"""
Cerebrium Deployment Testing Module

This module tests the deployed model on Cerebrium platform.
It includes functionality to test individual images and run comprehensive test suites.

Required functionality:
1. Accept image path and return/print class ID
2. Accept flag to run preset tests using deployed model
3. Monitor deployed model on Cerebrium platform
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default API configuration - These are the actual deployment credentials
DEFAULT_API_ENDPOINT = "https://api.cortex.cerebrium.ai/v4/p-9a3ad118/image-classification-deploy"
DEFAULT_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTlhM2FkMTE4IiwiaWF0IjoxNzQ4Nzg2ODY2LCJleHAiOjIwNjQzNjI4NjZ9.4vrWnBKxfvk8gk3XxuO5DF8B3v88qkcRW3CyG4UUWnpzG_WRNjH5DxoI44SZvkgl8k7oY06yDhoY_eN0NwMIoOZMVOM8uK85larx2vtCIZwC-EXRAQ4IAIgvlG4TAMS1DM-jsMimtaGQSmzp763Vvh5Sp3ncpMG5w-3T84xp8bqu51t2ocuXXscq9AOReoNNdJHrPs7xKCI-rxwpl2wCK8Jsmm1hG4lsgs-tKZZLl7zby57nHo3eAMAdhO2iyY3AgBVStNV0sbWKA2XWh-mSFAPdQ9gKrFZOsvizK9f6-AVJk_X04pVI4SFVX-ljLvfjH8JqBm7gj8ml5rAb5c23WA"


class CerebriumTester:
    """Handles testing of deployed model on Cerebrium platform."""
    
    def __init__(self, api_endpoint: str, api_key: str, timeout: int = 30):
        """
        Initialize Cerebrium tester.
        
        Args:
            api_endpoint: Cerebrium API endpoint URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def test_health_check(self) -> bool:
        """
        Test if the deployed model is healthy and responding.
        
        Returns:
            True if health check passes, False otherwise
        """
        try:
            logger.info("Performing health check...")
            
            # Try to make a simple request to verify endpoint is alive
            response = self.session.get(
                f"{self.api_endpoint}",
                timeout=self.timeout
            )
            
            # Any response (even 404) means the endpoint is reachable
            if response.status_code in [200, 404, 405]:
                logger.info("✅ Health check passed - endpoint is reachable")
                return True
            else:
                logger.warning(f"❌ Health check failed with status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check failed with error: {e}")
            return False
    
    def predict_image(self, image_path: str) -> Optional[Dict]:
        """
        Send image to deployed model for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction response or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            logger.info(f"Sending prediction request for: {image_path}")
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Prepare request payload
            payload = {
                'image_data': image_data.hex(),  # Convert to hex string
                'image_path': os.path.basename(image_path)
            }
            
            # Send request
            start_time = time.time()
            response = self.session.post(
                f"{self.api_endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            logger.info(f"Response time: {response_time:.3f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Prediction successful")
                return {
                    'prediction': result,
                    'response_time': response_time,
                    'status_code': response.status_code
                }
            else:
                logger.error(f"❌ Prediction failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def get_class_id(self, image_path: str) -> Optional[int]:
        """
        Get the predicted class ID for an image.
        This is the main function required by the assignment.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted class ID or None if failed
        """
        try:
            result = self.predict_image(image_path)
            
            if result and 'prediction' in result:
                prediction = result['prediction']
                
                # Extract class ID from different possible response formats
                if isinstance(prediction, dict):
                    if 'class_id' in prediction:
                        return prediction['class_id']
                    elif 'predicted_class' in prediction:
                        return prediction['predicted_class']
                    elif 'top_predictions' in prediction and len(prediction['top_predictions']) > 0:
                        return prediction['top_predictions'][0]['class_id']
                elif isinstance(prediction, (int, float)):
                    return int(prediction)
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting class prediction: {e}")
            return None
    
    def test_response_time(self, image_path: str, max_time: float = 3.0) -> bool:
        """
        Test if response time meets production requirements (2-3 seconds).
        
        Args:
            image_path: Path to test image
            max_time: Maximum allowed response time in seconds
            
        Returns:
            True if response time is acceptable, False otherwise
        """
        try:
            result = self.predict_image(image_path)
            
            if result and 'response_time' in result:
                response_time = result['response_time']
                
                if response_time <= max_time:
                    logger.info(f"✅ Response time acceptable: {response_time:.3f}s <= {max_time}s")
                    return True
                else:
                    logger.warning(f"❌ Response time too slow: {response_time:.3f}s > {max_time}s")
                    return False
            else:
                logger.error("❌ Failed to get response time")
                return False
                
        except Exception as e:
            logger.error(f"Error testing response time: {e}")
            return False
    
    def test_known_images(self) -> Dict[str, bool]:
        """
        Test with known ImageNet test cases.
        
        Returns:
            Dictionary of test results for known images
        """
        logger.info("Testing with known ImageNet images...")
        results = {}
        
        # Find project root and test images
        project_root = Path(__file__).parent.parent
        
        # Helper function to find image with multiple possible extensions
        def find_image_file(base_name):
            extensions = ['.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG']
            for ext in extensions:
                path = project_root / "assets" / f"{base_name}{ext}"
                if path.exists():
                    return path
            return None
        
        test_cases = [
            {
                'path': find_image_file("n01440764_tench"),
                'expected_class': 0,
                'name': 'tench'
            },
            {
                'path': find_image_file("n01667114_mud_turtle"),
                'expected_class': 35,
                'name': 'mud_turtle'
            }
        ]
        
        for test_case in test_cases:
            if test_case['path'] and test_case['path'].exists():
                # Test prediction accuracy
                predicted_class = self.get_class_id(str(test_case['path']))
                accuracy_result = (predicted_class == test_case['expected_class'])
                results[f"{test_case['name']}_accuracy"] = accuracy_result
                
                # Test response time
                speed_result = self.test_response_time(str(test_case['path']))
                results[f"{test_case['name']}_speed"] = speed_result
                
                logger.info(f"Test {test_case['name']}: Predicted={predicted_class}, "
                          f"Expected={test_case['expected_class']}, "
                          f"Correct={accuracy_result}")
            else:
                # Try to find any image file with similar name pattern
                available_files = list((project_root / "assets").glob("*tench*")) if test_case['name'] == 'tench' else list((project_root / "assets").glob("*turtle*"))
                if available_files:
                    logger.info(f"Found alternative {test_case['name']} image: {available_files[0]}")
                    # Test with the found file
                    predicted_class = self.get_class_id(str(available_files[0]))
                    accuracy_result = (predicted_class == test_case['expected_class'])
                    results[f"{test_case['name']}_accuracy"] = accuracy_result
                    
                    speed_result = self.test_response_time(str(available_files[0]))
                    results[f"{test_case['name']}_speed"] = speed_result
                    
                    logger.info(f"Test {test_case['name']}: Predicted={predicted_class}, "
                              f"Expected={test_case['expected_class']}, "
                              f"Correct={accuracy_result}")
                else:
                    logger.warning(f"Test image not found for {test_case['name']}")
                    results[f"{test_case['name']}_accuracy"] = False
                    results[f"{test_case['name']}_speed"] = False
        
        return results
    
    def test_error_handling(self) -> bool:
        """
        Test error handling capabilities of the deployed model.
        
        Returns:
            True if error handling works correctly, False otherwise
        """
        try:
            logger.info("Testing error handling...")
            
            # Test with invalid payload
            response = self.session.post(
                f"{self.api_endpoint}/predict",
                json={'invalid': 'payload'},
                timeout=self.timeout
            )
            
            # Should return an error status for invalid input
            if response.status_code >= 400:
                logger.info("✅ Error handling test passed - returns error for invalid input")
                return True
            else:
                logger.warning("❌ Error handling test failed - no error for invalid input")
                return False
                
        except Exception as e:
            logger.error(f"Error in error handling test: {e}")
            return False
    
    def test_load_handling(self) -> bool:
        """
        Test basic load handling with multiple concurrent requests.
        
        Returns:
            True if load test passes, False otherwise
        """
        try:
            logger.info("Testing load handling...")
            
            # Use any available test image
            project_root = Path(__file__).parent.parent
            
            # Try to find any test image
            test_image_candidates = [
                project_root / "assets" / "n01667114_mud_turtle.JPEG",
                project_root / "assets" / "n01440764_tench.jpeg",
                project_root / "assets" / "n01440764_tench.jpg",
            ]
            
            test_image = None
            for candidate in test_image_candidates:
                if candidate.exists():
                    test_image = candidate
                    break
            
            # If no specific test images, try any image in assets
            if test_image is None:
                assets_dir = project_root / "assets"
                if assets_dir.exists():
                    image_files = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.jpeg")) + list(assets_dir.glob("*.JPEG")) + list(assets_dir.glob("*.JPG"))
                    if image_files:
                        test_image = image_files[0]
            
            if test_image is None:
                logger.warning("No test image found for load testing")
                return False
            
            # Send multiple requests sequentially
            success_count = 0
            total_requests = 5
            
            for i in range(total_requests):
                logger.info(f"Load test request {i+1}/{total_requests}")
                result = self.predict_image(str(test_image))
                if result:
                    success_count += 1
                time.sleep(0.5)  # Small delay between requests
            
            success_rate = success_count / total_requests
            
            if success_rate >= 0.8:  # 80% success rate
                logger.info(f"✅ Load test passed: {success_count}/{total_requests} successful")
                return True
            else:
                logger.warning(f"❌ Load test failed: {success_count}/{total_requests} successful")
                return False
                
        except Exception as e:
            logger.error(f"Error in load test: {e}")
            return False
    
    def test_model_accuracy(self) -> bool:
        """
        Test model accuracy on ImageNet classes.
        
        Returns:
            True if accuracy tests pass, False otherwise
        """
        logger.info("Testing model accuracy on ImageNet classes...")
        
        known_results = self.test_known_images()
        accuracy_tests = [k for k in known_results.keys() if 'accuracy' in k]
        
        if not accuracy_tests:
            logger.warning("No accuracy tests found")
            return False
        
        passed_accuracy_tests = sum([known_results[k] for k in accuracy_tests])
        accuracy_rate = passed_accuracy_tests / len(accuracy_tests)
        
        if accuracy_rate >= 1.0:  # All accuracy tests should pass
            logger.info(f"✅ Model accuracy test passed: {passed_accuracy_tests}/{len(accuracy_tests)}")
            return True
        else:
            logger.warning(f"❌ Model accuracy test failed: {passed_accuracy_tests}/{len(accuracy_tests)}")
            return False
    
    def run_preset_tests(self) -> Dict[str, bool]:
        """
        Run comprehensive preset test suite on deployed model.
        This function implements the required preset tests functionality.
        
        Returns:
            Dictionary of test results
        """
        logger.info("=" * 60)
        logger.info("RUNNING PRESET TESTS ON DEPLOYED MODEL")
        logger.info("=" * 60)
        
        results = {}
        
        # Test 1: Health check
        logger.info("\n1. Health Check Test")
        results['health_check'] = self.test_health_check()
        
        # Test 2: Known ImageNet images accuracy
        logger.info("\n2. ImageNet Accuracy Tests")
        known_results = self.test_known_images()
        results.update(known_results)
        
        # Test 3: Response time requirements
        logger.info("\n3. Performance Tests")
        results['performance'] = all([v for k, v in known_results.items() if 'speed' in k])
        
        # Test 4: Error handling
        logger.info("\n4. Error Handling Tests")
        results['error_handling'] = self.test_error_handling()
        
        # Test 5: Load testing
        logger.info("\n5. Load Handling Tests")
        results['load_test'] = self.test_load_handling()
        
        # Test 6: Model accuracy validation
        logger.info("\n6. Model Accuracy Validation")
        results['model_accuracy'] = self.test_model_accuracy()
        
        return results
    
    def generate_test_report(self, results: Dict[str, bool]) -> str:
        """
        Generate a formatted test report for preset tests.
        
        Args:
            results: Test results dictionary
            
        Returns:
            Formatted test report string
        """
        report = []
        report.append("=" * 60)
        report.append("CEREBRIUM DEPLOYMENT TEST REPORT")
        report.append("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        report.append("")
        
        report.append("DETAILED RESULTS:")
        report.append("-" * 30)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report.append(f"{test_name:<25} | {status}")
        
        report.append("=" * 60)
        
        # Add deployment monitoring info
        report.append("")
        report.append("DEPLOYMENT MONITORING:")
        report.append("-" * 30)
        report.append(f"API Endpoint: {self.api_endpoint}")
        report.append(f"Timeout Setting: {self.timeout}s")
        report.append(f"Retry Strategy: 3 attempts with backoff")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Test deployed image classification model on Cerebrium platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image and print class ID
  python test_server.py assets/n01440764_tench.jpg
  
  # Run comprehensive preset tests
  python test_server.py --preset-tests
  
  # Use custom API configuration
  python test_server.py assets/image.jpg --endpoint "your-endpoint" --api-key "your-key"
        """
    )
    
    parser.add_argument(
        'image_path',
        nargs='?',
        help='Path to image for classification (returns class ID)'
    )
    
    parser.add_argument(
        '--preset-tests',
        action='store_true',
        help='Run comprehensive preset test suite on deployed model'
    )
    
    parser.add_argument(
        '--endpoint',
        default=DEFAULT_API_ENDPOINT,
        help=f'Cerebrium API endpoint URL (default: embedded endpoint)'
    )
    
    parser.add_argument(
        '--api-key',
        default=DEFAULT_API_KEY,
        help='Cerebrium API key (default: embedded key)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CerebriumTester(args.endpoint, args.api_key, args.timeout)
    
    if args.preset_tests:
        # Run preset test suite as required by assignment
        logger.info("Running preset test suite on deployed model...")
        results = tester.run_preset_tests()
        
        # Generate and print report
        report = tester.generate_test_report(results)
        print("\n" + report)
        
        # Exit with appropriate code for CI/CD
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All preset tests passed! Deployment is working correctly.")
            sys.exit(0)
        else:
            print("\n⚠️  Some preset tests failed. Please check the deployment.")
            sys.exit(1)
            
    elif args.image_path:
        # Test single image and return class ID as required by assignment
        if not os.path.exists(args.image_path):
            logger.error(f"Image file not found: {args.image_path}")
            print(f"Error: Image file not found: {args.image_path}")
            sys.exit(1)
        
        # Get class ID
        class_id = tester.get_class_id(args.image_path)
        
        if class_id is not None:
            # Print class ID as required by assignment
            print(f"{class_id}")
            logger.info(f"✅ Successfully predicted class {class_id} for {args.image_path}")
            sys.exit(0)
        else:
            logger.error("❌ Failed to get prediction")
            print("Error: Failed to get prediction from deployed model")
            sys.exit(1)
    
    else:
        # No arguments provided, show help and run a quick demonstration
        parser.print_help()
        print("\n" + "="*50)
        print("QUICK DEMONSTRATION")
        print("="*50)
        
        # Try to run a quick test with default image
        project_root = Path(__file__).parent.parent
        
        # Try to find any available test image
        test_image_candidates = [
            project_root / "assets" / "n01667114_mud_turtle.JPEG",
            project_root / "assets" / "n01440764_tench.jpeg", 
            project_root / "assets" / "n01440764_tench.jpg",
        ]
        
        default_image = None
        for candidate in test_image_candidates:
            if candidate.exists():
                default_image = candidate
                break
        
        # If no specific images, try any image in assets
        if default_image is None:
            assets_dir = project_root / "assets"
            if assets_dir.exists():
                image_files = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.jpeg")) + list(assets_dir.glob("*.JPEG")) + list(assets_dir.glob("*.JPG"))
                if image_files:
                    default_image = image_files[0]
        
        if default_image.exists():
            print(f"Running quick test with: {default_image}")
            class_id = tester.get_class_id(str(default_image))
            
            if class_id is not None:
                print(f"Quick Test Result - Predicted class ID: {class_id}")
                print("✅ Deployment is working!")
                print("\nTo test your own image:")
                print(f"python {sys.argv[0]} path/to/your/image.jpg")
                print("\nTo run full test suite:")
                print(f"python {sys.argv[0]} --preset-tests")
            else:
                print("❌ Quick test failed")
                sys.exit(1)
        else:
            print("No default test image found.")
            print("Usage examples:")
            print(f"  python {sys.argv[0]} path/to/image.jpg")
            print(f"  python {sys.argv[0]} --preset-tests")


if __name__ == "__main__":
    main()