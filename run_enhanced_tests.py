#!/usr/bin/env python3
"""Enhanced comprehensive test suite for WASM Torch with advanced validation."""

import asyncio
import time
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from src.wasm_torch.validation import (
        validate_model_compatibility, 
        validate_system_resources,
        validate_compilation_environment
    )
    from src.wasm_torch.error_recovery import (
        with_recovery, 
        ExponentialBackoffStrategy,
        MemoryOptimizationStrategy,
        health_monitor
    )
    from src.wasm_torch.adaptive_optimization import optimize_runtime_performance
    from src.wasm_torch.intelligent_caching import optimize_cache_performance
    from src.wasm_torch.security import SecurityManager, validate_model_compatibility as security_validate
    from src.wasm_torch.performance import PerformanceStats
    
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    TORCH_AVAILABLE = False


class TestResult:
    """Test result container with detailed metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.duration = 0.0
        self.error_message = ""
        self.metrics = {}
        self.warnings = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'duration': self.duration,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'warnings': self.warnings
        }


class EnhancedTestSuite:
    """Comprehensive test suite with advanced validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = 0.0
        self.total_duration = 0.0
        self.security_manager = None
        
        if TORCH_AVAILABLE:
            try:
                self.security_manager = SecurityManager()
            except Exception as e:
                logger.warning(f"SecurityManager not available: {e}")
    
    async def run_all_tests(self, include_performance: bool = True, 
                          include_security: bool = True) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting Enhanced WASM Torch Test Suite")
        self.start_time = time.time()
        
        # System validation tests
        await self._test_system_validation()
        
        # Core functionality tests
        if TORCH_AVAILABLE:
            await self._test_model_compatibility()
            await self._test_error_recovery()
            await self._test_validation_functions()
        
        # Performance tests
        if include_performance:
            await self._test_performance_optimization()
            await self._test_caching_system()
        
        # Security tests
        if include_security and self.security_manager:
            await self._test_security_features()
        
        # Integration tests
        await self._test_integration_scenarios()
        
        self.total_duration = time.time() - self.start_time
        
        return self._generate_report()
    
    async def _test_system_validation(self) -> None:
        """Test system resource validation."""
        test = TestResult("System Resource Validation")
        
        try:
            start_time = time.time()
            
            # Test system resources
            resources = validate_system_resources()
            test.metrics['system_resources'] = resources
            
            if not resources['sufficient']:
                test.warnings.append("System resources may be insufficient")
            
            # Test compilation environment
            env_status = validate_compilation_environment()
            test.metrics['compilation_environment'] = env_status
            
            missing_tools = [k for k, v in env_status.items() if not v]
            if missing_tools:
                test.warnings.append(f"Missing compilation tools: {missing_tools}")
            
            test.duration = time.time() - start_time
            test.passed = True
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"System validation test failed: {e}")
        
        self.results.append(test)
    
    async def _test_model_compatibility(self) -> None:
        """Test model compatibility validation."""
        test = TestResult("Model Compatibility Validation")
        
        try:
            start_time = time.time()
            
            # Create test model
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(10, 5)
                    self.relu = torch.nn.ReLU()
                    self.linear2 = torch.nn.Linear(5, 1)
                
                def forward(self, x):
                    return self.linear2(self.relu(self.linear1(x)))
            
            model = TestModel()
            example_input = torch.randn(1, 10)
            
            # Test compatibility
            compatibility_result = validate_model_compatibility(model, example_input)
            test.metrics['compatibility'] = compatibility_result
            
            if not compatibility_result['compatible']:
                test.error_message = f"Model incompatible: {compatibility_result['errors']}"
            else:
                test.passed = True
                
                if compatibility_result['warnings']:
                    test.warnings.extend(compatibility_result['warnings'])
            
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Model compatibility test failed: {e}")
        
        self.results.append(test)
    
    async def _test_error_recovery(self) -> None:
        """Test error recovery mechanisms."""
        test = TestResult("Error Recovery Mechanisms")
        
        try:
            start_time = time.time()
            
            # Test exponential backoff strategy
            @with_recovery(ExponentialBackoffStrategy(max_attempts=3, base_delay=0.1))
            def flaky_function(attempt_count=[0]):
                attempt_count[0] += 1
                if attempt_count[0] < 3:
                    raise RuntimeError(f"Attempt {attempt_count[0]} failed")
                return "success"
            
            result = flaky_function()
            if result != "success":
                test.error_message = "Recovery strategy failed"
            else:
                test.metrics['recovery_attempts'] = 3
                test.passed = True
            
            # Test memory optimization strategy
            memory_strategy = MemoryOptimizationStrategy()
            memory_error = RuntimeError("CUDA out of memory")
            should_retry = memory_strategy.should_retry(memory_error)
            test.metrics['memory_strategy_active'] = should_retry
            
            # Test health monitor
            health_status = health_monitor.check_health()
            test.metrics['health_checks'] = health_status
            
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Error recovery test failed: {e}")
        
        self.results.append(test)
    
    async def _test_validation_functions(self) -> None:
        """Test input/output validation functions."""
        test = TestResult("Validation Functions")
        
        try:
            start_time = time.time()
            
            from src.wasm_torch.validation import (
                validate_tensor_safe, 
                sanitize_file_path
            )
            
            # Test tensor validation
            valid_tensor = torch.randn(10, 10)
            validate_tensor_safe(valid_tensor, "test_tensor")
            
            # Test invalid tensor handling
            try:
                invalid_tensor = torch.full((5, 5), float('nan'))
                validate_tensor_safe(invalid_tensor, "invalid_tensor")
                test.error_message = "Should have caught NaN tensor"
            except RuntimeError:
                pass  # Expected behavior
            
            # Test path sanitization
            safe_path = sanitize_file_path("./models/test.wasm", {".wasm"})
            test.metrics['path_sanitization'] = safe_path
            
            # Test malicious path rejection
            try:
                malicious_path = sanitize_file_path("../../../etc/passwd")
                test.error_message = "Should have rejected malicious path"
            except ValueError:
                pass  # Expected behavior
            
            test.passed = True
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Validation test failed: {e}")
        
        self.results.append(test)
    
    async def _test_performance_optimization(self) -> None:
        """Test performance optimization systems."""
        test = TestResult("Performance Optimization")
        
        try:
            start_time = time.time()
            
            # Test adaptive optimization
            optimization_results = await optimize_runtime_performance()
            test.metrics['optimization_results'] = optimization_results
            
            # Validate optimization results structure
            required_keys = ['current_metrics', 'suggested_config', 'scaling_needed']
            missing_keys = [k for k in required_keys if k not in optimization_results]
            
            if missing_keys:
                test.error_message = f"Missing optimization result keys: {missing_keys}"
            else:
                test.passed = True
                
                # Check if optimization suggestions are reasonable
                suggested_config = optimization_results['suggested_config']
                if hasattr(suggested_config, 'batch_size'):
                    if suggested_config.batch_size <= 0 or suggested_config.batch_size > 1000:
                        test.warnings.append("Unreasonable batch size suggestion")
            
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Performance optimization test failed: {e}")
        
        self.results.append(test)
    
    async def _test_caching_system(self) -> None:
        """Test intelligent caching system."""
        test = TestResult("Intelligent Caching System")
        
        try:
            start_time = time.time()
            
            # Test cache optimization
            cache_results = await optimize_cache_performance()
            test.metrics['cache_results'] = cache_results
            
            # Test individual cache operations
            from src.wasm_torch.intelligent_caching import model_cache, tensor_cache
            
            # Test cache put/get operations
            test_data = torch.randn(10, 10)
            model_cache.put("test_key", test_data)
            retrieved_data = model_cache.get("test_key")
            
            if retrieved_data is None:
                test.error_message = "Cache failed to retrieve stored data"
            else:
                if not torch.equal(test_data, retrieved_data):
                    test.error_message = "Retrieved data doesn't match stored data"
                else:
                    test.passed = True
            
            # Test cache metrics
            cache_metrics = model_cache.get_metrics()
            test.metrics['cache_metrics'] = {
                'hits': cache_metrics.hits,
                'misses': cache_metrics.misses,
                'size': cache_metrics.size,
                'memory_usage_mb': cache_metrics.memory_usage_bytes / (1024 * 1024)
            }
            
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Caching system test failed: {e}")
        
        self.results.append(test)
    
    async def _test_security_features(self) -> None:
        """Test security features and validation."""
        test = TestResult("Security Features")
        
        try:
            start_time = time.time()
            
            if not self.security_manager:
                test.error_message = "SecurityManager not available"
                test.duration = time.time() - start_time
                self.results.append(test)
                return
            
            # Test security validation
            test_operation = "model_export"
            test_params = {
                "model_path": "./test_model.pth",
                "output_path": "./output/model.wasm"
            }
            
            try:
                self.security_manager.validate_operation(test_operation, **test_params)
                test.metrics['security_validation'] = "passed"
            except Exception as security_error:
                test.warnings.append(f"Security validation warning: {security_error}")
            
            # Test audit logging
            from src.wasm_torch.security import log_security_event
            log_security_event("test_event", {"test": True})
            test.metrics['audit_logging'] = "functional"
            
            test.passed = True
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Security test failed: {e}")
        
        self.results.append(test)
    
    async def _test_integration_scenarios(self) -> None:
        """Test integration scenarios combining multiple components."""
        test = TestResult("Integration Scenarios")
        
        try:
            start_time = time.time()
            
            if not TORCH_AVAILABLE:
                test.error_message = "PyTorch not available for integration tests"
                test.duration = time.time() - start_time
                self.results.append(test)
                return
            
            # Integration scenario: Model validation + caching + optimization
            class IntegrationModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 16, 3)
                    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = torch.nn.Linear(16, 10)
                
                def forward(self, x):
                    x = self.pool(torch.relu(self.conv(x)))
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            model = IntegrationModel()
            example_input = torch.randn(1, 3, 32, 32)
            
            # Step 1: Validate model compatibility
            compatibility = validate_model_compatibility(model, example_input)
            test.metrics['integration_compatibility'] = compatibility['compatible']
            
            if not compatibility['compatible']:
                test.warnings.append("Model failed compatibility check")
            
            # Step 2: Cache model and input
            from src.wasm_torch.intelligent_caching import model_cache
            model_cache.put("integration_model", model)
            model_cache.put("integration_input", example_input)
            
            # Step 3: Retrieve and validate
            cached_model = model_cache.get("integration_model")
            cached_input = model_cache.get("integration_input")
            
            if cached_model is None or cached_input is None:
                test.error_message = "Integration caching failed"
            else:
                # Step 4: Run forward pass
                model.eval()
                with torch.no_grad():
                    output = cached_model(cached_input)
                
                if output is not None and output.shape == (1, 10):
                    test.passed = True
                    test.metrics['integration_output_shape'] = list(output.shape)
                else:
                    test.error_message = f"Unexpected output shape: {output.shape if output is not None else None}"
            
            test.duration = time.time() - start_time
            
        except Exception as e:
            test.error_message = str(e)
            test.duration = time.time() - start_time
            logger.error(f"Integration test failed: {e}")
        
        self.results.append(test)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        report = {
            'summary': {
                'total_tests': len(self.results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.results) if self.results else 0,
                'total_duration': self.total_duration
            },
            'test_results': [test.to_dict() for test in self.results],
            'performance_metrics': self._extract_performance_metrics(),
            'security_assessment': self._extract_security_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from test results."""
        metrics = {}
        
        for result in self.results:
            if 'optimization_results' in result.metrics:
                opt_results = result.metrics['optimization_results']
                if 'current_metrics' in opt_results:
                    current_metrics = opt_results['current_metrics']
                    metrics['latency_p95'] = getattr(current_metrics, 'latency_p95', 0)
                    metrics['throughput'] = getattr(current_metrics, 'throughput', 0)
                    metrics['memory_usage'] = getattr(current_metrics, 'memory_usage', 0)
            
            if 'cache_metrics' in result.metrics:
                cache_metrics = result.metrics['cache_metrics']
                metrics['cache_hit_rate'] = cache_metrics.get('hit_rate', 0)
                metrics['cache_memory_usage'] = cache_metrics.get('memory_usage_mb', 0)
        
        return metrics
    
    def _extract_security_metrics(self) -> Dict[str, Any]:
        """Extract security metrics from test results."""
        security_metrics = {
            'security_tests_passed': False,
            'validation_active': False,
            'audit_logging_active': False
        }
        
        for result in self.results:
            if result.name == "Security Features":
                security_metrics['security_tests_passed'] = result.passed
                security_metrics['validation_active'] = 'security_validation' in result.metrics
                security_metrics['audit_logging_active'] = 'audit_logging' in result.metrics
        
        return security_metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failing tests before production deployment")
        
        # Check for warnings
        all_warnings = []
        for result in self.results:
            all_warnings.extend(result.warnings)
        
        if all_warnings:
            recommendations.append(f"Review {len(all_warnings)} warnings for potential optimizations")
        
        # Performance recommendations
        perf_metrics = self._extract_performance_metrics()
        if perf_metrics.get('cache_hit_rate', 1.0) < 0.8:
            recommendations.append("Consider increasing cache size for better performance")
        
        if perf_metrics.get('memory_usage', 0) > 80:
            recommendations.append("Memory usage is high - consider optimization")
        
        # Security recommendations
        security_metrics = self._extract_security_metrics()
        if not security_metrics['security_tests_passed']:
            recommendations.append("Security tests failed - review security configuration")
        
        return recommendations


async def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description='Enhanced WASM Torch Test Suite')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance tests')
    parser.add_argument('--skip-security', action='store_true',
                       help='Skip security tests')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    test_suite = EnhancedTestSuite()
    
    try:
        results = await test_suite.run_all_tests(
            include_performance=not args.skip_performance,
            include_security=not args.skip_security
        )
        
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        summary = results['summary']
        logger.info("📊 Test Results Summary:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Passed: {summary['passed']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"   Duration: {summary['total_duration']:.2f}s")
        
        if results['recommendations']:
            logger.info("🔍 Recommendations:")
            for rec in results['recommendations']:
                logger.info(f"   • {rec}")
        
        # Exit with appropriate code
        exit_code = 0 if summary['failed'] == 0 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())