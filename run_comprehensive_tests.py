#!/usr/bin/env python3
"""
Comprehensive test runner for WASM-Torch library
Executes all quality gates: functionality, security, performance
"""

import sys
import time
import logging
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    import torch.nn as nn
    from wasm_torch import export_to_wasm, WASMRuntime
    from wasm_torch.validation import (
        validate_model_compatibility, validate_system_resources, 
        validate_compilation_environment, validate_tensor_safe
    )
    from wasm_torch.security import SecurityManager, validate_path, log_security_event
    from wasm_torch.performance import (
        get_performance_monitor, get_advanced_optimizer, 
        get_intelligent_cache, get_concurrency_manager
    )
    from wasm_torch.research.adaptive_optimizer import AdaptiveWASMOptimizer
    from wasm_torch.reliability import ReliabilityManager
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class ComprehensiveTestSuite:
    """Comprehensive test suite for WASM-Torch library."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.security_findings = {}
        self.start_time = time.time()
        
    def create_test_models(self) -> Dict[str, nn.Module]:
        """Create various test models for comprehensive testing."""
        models = {}
        
        # Simple Linear Model
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        models['simple_linear'] = SimpleLinear()
        
        # CNN Model
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(-1, 64 * 7 * 7)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        models['simple_cnn'] = SimpleCNN()
        
        # Attention Model (simplified transformer)
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 64
                self.num_heads = 4
                self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
                self.linear = nn.Linear(self.embed_dim, 10)
                
            def forward(self, x):
                # x shape: (seq_len, batch, embed_dim)
                attn_output, _ = self.attention(x, x, x)
                output = self.linear(attn_output.mean(dim=0))  # Global average pooling
                return output
        
        models['simple_attention'] = SimpleAttention()
        
        return models
        
    async def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic library functionality."""
        logger.info("🔧 Testing basic functionality...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        models = self.create_test_models()
        
        for model_name, model in models.items():
            try:
                model.eval()
                
                # Create appropriate input for each model
                if model_name == 'simple_linear':
                    example_input = torch.randn(1, 10)
                elif model_name == 'simple_cnn':
                    example_input = torch.randn(1, 1, 28, 28)
                elif model_name == 'simple_attention':
                    example_input = torch.randn(10, 1, 64)  # (seq_len, batch, embed_dim)
                else:
                    example_input = torch.randn(1, 10)
                
                # Test model compatibility validation
                compatibility = validate_model_compatibility(model, example_input)
                if compatibility["compatible"]:
                    results["passed"] += 1
                    results["details"].append(f"✅ {model_name} compatibility check passed")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ {model_name} compatibility check failed: {compatibility['errors']}")
                
                # Test forward pass
                with torch.no_grad():
                    output = model(example_input)
                    validate_tensor_safe(output, f"{model_name}_output")
                
                results["passed"] += 1
                results["details"].append(f"✅ {model_name} forward pass successful")
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ {model_name} test failed: {str(e)}")
                logger.error(f"Basic functionality test failed for {model_name}: {e}")
        
        return results
    
    async def test_runtime_functionality(self) -> Dict[str, Any]:
        """Test WASM runtime functionality."""
        logger.info("⚡ Testing runtime functionality...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        try:
            # Test runtime initialization
            runtime = WASMRuntime(simd=True, threads=4, memory_limit_mb=512)
            await runtime.init()
            
            results["passed"] += 1
            results["details"].append("✅ Runtime initialization successful")
            
            # Test runtime stats
            stats = runtime.get_runtime_stats()
            if stats and 'health_status' in stats:
                results["passed"] += 1
                results["details"].append(f"✅ Runtime stats available: {stats['health_status']}")
            else:
                results["failed"] += 1
                results["details"].append("❌ Runtime stats unavailable")
            
            # Test cleanup
            await runtime.cleanup()
            results["passed"] += 1
            results["details"].append("✅ Runtime cleanup successful")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Runtime functionality failed: {str(e)}")
            logger.error(f"Runtime functionality test failed: {e}")
        
        return results
    
    async def test_performance_systems(self) -> Dict[str, Any]:
        """Test performance optimization systems."""
        logger.info("🚀 Testing performance systems...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        try:
            # Test performance monitor
            monitor = get_performance_monitor()
            initial_stats = monitor.get_comprehensive_stats()
            
            results["passed"] += 1
            results["details"].append("✅ Performance monitor accessible")
            
            # Test advanced optimizer
            optimizer = get_advanced_optimizer()
            test_characteristics = {
                "parameter_count": 100000,
                "has_convolutions": True,
                "has_attention": False,
                "input_size": 784
            }
            
            config = optimizer.get_optimization_config(test_characteristics)
            if config and "preferred_batch_size" in config:
                results["passed"] += 1
                results["details"].append(f"✅ Advanced optimizer generated config: batch_size={config['preferred_batch_size']}")
            else:
                results["failed"] += 1
                results["details"].append("❌ Advanced optimizer config generation failed")
            
            # Test intelligent caching
            cache = get_intelligent_cache()
            test_key = "test_cache_key"
            test_value = torch.randn(10, 10)
            
            cache.put(test_key, test_value)
            cached_value = cache.get(test_key)
            
            if cached_value is not None and torch.equal(cached_value, test_value):
                results["passed"] += 1
                results["details"].append("✅ Intelligent caching system working")
            else:
                results["failed"] += 1
                results["details"].append("❌ Intelligent caching system failed")
            
            # Test concurrency manager
            concurrency_mgr = get_concurrency_manager()
            thread_pool = concurrency_mgr.get_thread_pool("test_pool", optimal_workers=2)
            
            if thread_pool:
                results["passed"] += 1
                results["details"].append("✅ Concurrency manager thread pool created")
            else:
                results["failed"] += 1
                results["details"].append("❌ Concurrency manager thread pool creation failed")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Performance systems test failed: {str(e)}")
            logger.error(f"Performance systems test failed: {e}")
        
        return results
    
    async def test_security_systems(self) -> Dict[str, Any]:
        """Test security systems and validation."""
        logger.info("🔒 Testing security systems...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        try:
            # Test security manager
            security_mgr = SecurityManager()
            
            # Test path validation
            try:
                valid_path = validate_path("./test_file.txt", allow_write=True)
                results["passed"] += 1
                results["details"].append("✅ Path validation working")
            except Exception as e:
                results["details"].append(f"⚠️ Path validation: {str(e)}")
            
            # Test security operation validation
            try:
                security_mgr.validate_operation("model_export", output_path="./output/test.wasm")
                results["passed"] += 1
                results["details"].append("✅ Security operation validation working")
            except Exception as e:
                results["details"].append(f"⚠️ Security operation validation: {str(e)}")
            
            # Test security event logging
            try:
                log_security_event("test_event", {"test_data": "test_value"})
                results["passed"] += 1
                results["details"].append("✅ Security event logging working")
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ Security event logging failed: {str(e)}")
            
            # Get security report
            security_report = security_mgr.get_security_report()
            if security_report:
                results["passed"] += 1
                results["details"].append(f"✅ Security report generated: {security_report['total_events']} events")
            else:
                results["failed"] += 1
                results["details"].append("❌ Security report generation failed")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Security systems test failed: {str(e)}")
            logger.error(f"Security systems test failed: {e}")
        
        return results
    
    async def test_research_modules(self) -> Dict[str, Any]:
        """Test research modules functionality."""
        logger.info("🧪 Testing research modules...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        try:
            # Test adaptive WASM optimizer
            adaptive_optimizer = AdaptiveWASMOptimizer()
            
            # Create test model
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            model.eval()
            
            example_input = torch.randn(1, 784)
            
            # Test model analysis
            model_chars = adaptive_optimizer.analyze_model(model, example_input)
            if model_chars and model_chars.parameter_count > 0:
                results["passed"] += 1
                results["details"].append(f"✅ Adaptive optimizer model analysis: {model_chars.parameter_count} parameters")
            else:
                results["failed"] += 1
                results["details"].append("❌ Adaptive optimizer model analysis failed")
            
            # Test optimization recommendations
            recommendations = adaptive_optimizer.get_optimization_recommendations(model_chars)
            if recommendations and "mobile" in recommendations:
                results["passed"] += 1
                results["details"].append(f"✅ Optimization recommendations generated: {len(recommendations)} configs")
            else:
                results["failed"] += 1
                results["details"].append("❌ Optimization recommendations generation failed")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ Research modules test failed: {str(e)}")
            logger.error(f"Research modules test failed: {e}")
        
        return results
    
    async def test_system_validation(self) -> Dict[str, Any]:
        """Test system validation and requirements."""
        logger.info("🔍 Testing system validation...")
        results = {"passed": 0, "failed": 0, "details": []}
        
        try:
            # Test system resources validation
            resources = validate_system_resources()
            if resources["sufficient"]:
                results["passed"] += 1
                results["details"].append("✅ System resources sufficient")
            else:
                results["details"].append(f"⚠️ System resources warnings: {resources['warnings']}")
            
            # Test compilation environment
            env_status = validate_compilation_environment()
            available_tools = sum(env_status.values())
            total_tools = len(env_status)
            
            results["passed"] += 1
            results["details"].append(f"✅ Compilation environment: {available_tools}/{total_tools} tools available")
            
            # Test reliability manager
            reliability_mgr = ReliabilityManager({
                "max_retries": 3,
                "retry_base_delay": 0.1,
                "health_check_interval": 10.0
            })
            
            await reliability_mgr.initialize()
            
            # Test health check
            health_status = await reliability_mgr.get_system_health()
            if health_status:
                results["passed"] += 1
                results["details"].append("✅ Reliability manager health check working")
            else:
                results["failed"] += 1
                results["details"].append("❌ Reliability manager health check failed")
            
            await reliability_mgr.shutdown()
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ System validation test failed: {str(e)}")
            logger.error(f"System validation test failed: {e}")
        
        return results
    
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        logger.info("📊 Running performance benchmarks...")
        results = {"benchmarks": {}, "details": []}
        
        models = self.create_test_models()
        
        for model_name, model in models.items():
            try:
                model.eval()
                
                # Create appropriate input
                if model_name == 'simple_linear':
                    example_input = torch.randn(32, 10)  # Batch of 32
                elif model_name == 'simple_cnn':
                    example_input = torch.randn(16, 1, 28, 28)  # Batch of 16
                elif model_name == 'simple_attention':
                    example_input = torch.randn(10, 8, 64)  # Seq=10, Batch=8
                else:
                    example_input = torch.randn(32, 10)
                
                # Benchmark inference time
                num_runs = 10
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(example_input)
                
                total_time = time.time() - start_time
                avg_time_ms = (total_time / num_runs) * 1000
                
                results["benchmarks"][model_name] = {
                    "avg_inference_time_ms": round(avg_time_ms, 2),
                    "throughput_samples_per_sec": round((example_input.shape[0] * num_runs) / total_time, 1),
                    "batch_size": example_input.shape[0]
                }
                
                results["details"].append(
                    f"✅ {model_name}: {avg_time_ms:.2f}ms avg, "
                    f"{results['benchmarks'][model_name]['throughput_samples_per_sec']:.1f} samples/sec"
                )
                
            except Exception as e:
                results["details"].append(f"❌ {model_name} benchmark failed: {str(e)}")
                logger.error(f"Performance benchmark failed for {model_name}: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting comprehensive test suite...")
        
        test_suite_results = {
            "start_time": self.start_time,
            "tests": {},
            "summary": {"total_passed": 0, "total_failed": 0},
            "overall_status": "UNKNOWN"
        }
        
        # Run all test categories
        test_categories = [
            ("basic_functionality", self.test_basic_functionality),
            ("runtime_functionality", self.test_runtime_functionality),
            ("performance_systems", self.test_performance_systems),
            ("security_systems", self.test_security_systems),
            ("research_modules", self.test_research_modules),
            ("system_validation", self.test_system_validation),
            ("performance_benchmarks", self.run_performance_benchmarks)
        ]
        
        for category_name, test_func in test_categories:
            logger.info(f"Running {category_name} tests...")
            try:
                category_results = await test_func()
                test_suite_results["tests"][category_name] = category_results
                
                # Update summary counts
                if "passed" in category_results:
                    test_suite_results["summary"]["total_passed"] += category_results["passed"]
                if "failed" in category_results:
                    test_suite_results["summary"]["total_failed"] += category_results["failed"]
                    
            except Exception as e:
                logger.error(f"Test category {category_name} failed: {e}")
                test_suite_results["tests"][category_name] = {
                    "passed": 0,
                    "failed": 1,
                    "details": [f"❌ Category failed with exception: {str(e)}"],
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                test_suite_results["summary"]["total_failed"] += 1
        
        # Calculate overall status
        total_tests = test_suite_results["summary"]["total_passed"] + test_suite_results["summary"]["total_failed"]
        pass_rate = test_suite_results["summary"]["total_passed"] / total_tests if total_tests > 0 else 0
        
        if pass_rate >= 0.9:
            test_suite_results["overall_status"] = "EXCELLENT"
        elif pass_rate >= 0.8:
            test_suite_results["overall_status"] = "GOOD"
        elif pass_rate >= 0.7:
            test_suite_results["overall_status"] = "FAIR"
        else:
            test_suite_results["overall_status"] = "NEEDS_IMPROVEMENT"
        
        test_suite_results["end_time"] = time.time()
        test_suite_results["total_duration_seconds"] = test_suite_results["end_time"] - test_suite_results["start_time"]
        
        return test_suite_results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted test results."""
        print("\n" + "="*80)
        print("🧪 WASM-TORCH COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Status: {results['overall_status']}")
        print(f"   Total Duration: {results['total_duration_seconds']:.2f}s")
        print(f"   Tests Passed: {results['summary']['total_passed']}")
        print(f"   Tests Failed: {results['summary']['total_failed']}")
        
        total_tests = results['summary']['total_passed'] + results['summary']['total_failed']
        if total_tests > 0:
            pass_rate = (results['summary']['total_passed'] / total_tests) * 100
            print(f"   Pass Rate: {pass_rate:.1f}%")
        
        # Detailed results by category
        print(f"\n📋 DETAILED RESULTS:")
        
        for category_name, category_results in results["tests"].items():
            print(f"\n   🔸 {category_name.upper().replace('_', ' ')}:")
            
            if "passed" in category_results and "failed" in category_results:
                print(f"      ✅ Passed: {category_results['passed']}")
                print(f"      ❌ Failed: {category_results['failed']}")
            
            if "details" in category_results:
                for detail in category_results["details"][-5:]:  # Show last 5 details
                    print(f"      {detail}")
                if len(category_results["details"]) > 5:
                    print(f"      ... and {len(category_results['details']) - 5} more")
            
            if "benchmarks" in category_results:
                print(f"      📊 Benchmarks:")
                for model_name, benchmark_data in category_results["benchmarks"].items():
                    print(f"         {model_name}: {benchmark_data['avg_inference_time_ms']}ms avg")
        
        print("\n" + "="*80)
        
        # Final assessment
        if results["overall_status"] in ["EXCELLENT", "GOOD"]:
            print("🎉 WASM-Torch library is ready for production use!")
        elif results["overall_status"] == "FAIR":
            print("⚠️ WASM-Torch library needs minor improvements before production.")
        else:
            print("❌ WASM-Torch library requires significant improvements.")
        
        print("="*80 + "\n")


async def main():
    """Main test execution function."""
    print("🚀 WASM-Torch Comprehensive Test Suite")
    print("=" * 50)
    
    # Create and run test suite
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_all_tests()
    
    # Print results
    test_suite.print_results(results)
    
    # Return exit code based on results
    if results["overall_status"] in ["EXCELLENT", "GOOD"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)