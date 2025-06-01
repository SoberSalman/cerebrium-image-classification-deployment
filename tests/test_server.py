"""
Cerebrium Deployment Testing Module

This module tests the deployed model on Cerebrium platform.
It includes functionality to test individual images and run comprehensive test suites.
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
            
            # Try to ping the endpoint
            response = self.session.get(
                f"{self.api_endpoint}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("✅ Health check passed")
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
    
    def get_predicted_class(self, image_path: str) -> Optional[int]:
        """
        Get the predicted class ID for an image.
        
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
                    elif 'top_predictions' in prediction:
                        return prediction['top_predictions'][0]['class_id']
                elif isinstance(prediction, (int, float)):
                    return int(prediction)
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting class prediction: {e}")
            return None
    
    def test_response_time(self, image_path: str, max_time: float = 3.0) -> bool:
        """
        Test if response time meets requirements.
        
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
    
    def run_preset_tests(self) -> Dict[str, bool]:
        """
        Run comprehensive test suite on deployed model.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Running preset test suite...")
        results = {}
        
        # Test 1: Health check
        results['health_check'] = self.test_health_check()
        
        # Test 2: Known test images
        project_root = Path(__file__).parent.parent
        test_images = [
            {
                'path': project_root / "assets" / "n01440764_tench.jpg",
                'expected_class': 0,
                'name': 'tench'
            },
            {
                'path': project_root / "assets" / "n01667114_mud_turtle.jpg",
                'expected_class': 35,
                'name': 'mud_turtle'
            }
        ]
        
        for test_case in test_images:
            if test_case['path'].exists():
                # Test prediction accuracy
                predicted_class = self.get_predicted_class(str(test_case['path']))
                results[f"{test_case['name']}_accuracy"] = (
                    predicted_class == test_case['expected_class']
                )
                
                # Test response time
                results[f"{test_case['name']}_speed"] = self.test_response_time(
                    str(test_case['path'])
                )
                
                logger.info(f"Test {test_case['name']}: Predicted={predicted_class}, "
                          f"Expected={test_case['expected_class']}")
            else:
                logger.warning(f"Test image not found: {test_case['path']}")
                results[f"{test_case['name']}_accuracy"] = False
                results[f"{test_case['name']}_speed"] = False
        
        # Test 3: Error handling
        results['error_handling'] = self._test_error_handling()
        
        # Test 4: Load testing
        results['load_test'] = self._test_load_handling()
        
        return results
    
    def _test_error_handling(self) -> bool:
        """
        Test error handling capabilities.
        
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
            
            # Should return an error status
            if response.status_code >= 400:
                logger.info("✅ Error handling test passed")
                return True
            else:
                logger.warning("❌ Error handling test failed - no error for invalid input")
                return False
                
        except Exception as e:
            logger.error(f"Error in error handling test: {e}")
            return False
    
    def _test_load_handling(self) -> bool:
        """
        Test basic load handling with multiple requests.
        
        Returns:
            True if load test passes, False otherwise
        """
        try:
            logger.info("Testing load handling...")
            
            # Create a simple test image path
            project_root = Path(__file__).parent.parent
            test_image = project_root / "assets" / "n01440764_tench.jpg"
            
            if not test_image.exists():
                logger.warning("Test image not found for load testing")
                return False
            
            # Send multiple concurrent requests
            success_count = 0
            total_requests = 5
            
            for i in range(total_requests):
                result = self.predict_image(str(test_image))
                if result:
                    success_count += 1
                time.sleep(0.1)  # Small delay between requests
            
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
    
    def generate_test_report(self, results: Dict[str, bool]) -> str:
        """
        Generate a formatted test report.
        
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
        
        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Test deployed model on Cerebrium platform'
    )
    
    parser.add_argument(
        '--endpoint',
        required=True,
        help='Cerebrium API endpoint URL'
    )
    
    parser.add_argument(
        '--api-key',
        required=True,
        help='Cerebrium API key'
    )
    
    parser.add_argument(
        '--image-path',
        help='Path to image for single prediction test'
    )
    
    parser.add_argument(
        '--preset-tests',
        action='store_true',
        help='Run comprehensive preset test suite'
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
        # Run preset test suite
        logger.info("Running preset test suite...")
        results = tester.run_preset_tests()
        
        # Generate and print report
        report = tester.generate_test_report(results)
        print(report)
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    elif args.image_path:
        # Test single image
        if not os.path.exists(args.image_path):
            logger.error(f"Image file not found: {args.image_path}")
            sys.exit(1)
        
        # Get prediction
        predicted_class = tester.get_predicted_class(args.image_path)
        
        if predicted_class is not None:
            print(f"Predicted class ID: {predicted_class}")
            logger.info(f"✅ Successfully predicted class {predicted_class} for {args.image_path}")
        else:
            logger.error("❌ Failed to get prediction")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()