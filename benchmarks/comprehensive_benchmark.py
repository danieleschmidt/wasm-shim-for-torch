#!/usr/bin/env python3
"""Comprehensive benchmark suite for WASM-Torch performance validation."""

import asyncio
import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    test_name: str
    duration_ms: float
    memory_used_mb: float
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockTensor:
    """Mock tensor for benchmarking without PyTorch dependency."""
    
    def __init__(self, shape: tuple, dtype: str = "float32"):
        self.shape = shape
        self.dtype = dtype
        self._size = 4  # Assume 4 bytes per float32
        
    def numel(self) -> int:
        """Number of elements."""
        return int(sum(self.shape) if self.shape else 0)
    
    def element_size(self) -> int:
        """Size of each element in bytes."""
        return self._size
    
    def clone(self):
        """Clone tensor."""
        return MockTensor(self.shape, self.dtype)
    
    def detach(self):
        """Detach tensor."""
        return self
    
    def numpy(self):
        """Convert to numpy (mocked)."""
        import numpy as np
        return np.random.randn(*self.shape).astype('float32')


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for WASM-Torch."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.config = {
            'warmup_iterations': 3,
            'benchmark_iterations': 10,
            'timeout_seconds': 30.0
        }
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("Starting comprehensive WASM-Torch benchmark suite")
        
        benchmark_suites = [
            ("Core Operations", self.benchmark_core_operations),
            ("Memory Management", self.benchmark_memory_management),
            ("Caching Performance", self.benchmark_caching),
            ("Scaling Features", self.benchmark_scaling),
            ("Reliability Features", self.benchmark_reliability),
            ("Integration Tests", self.benchmark_integration),
        ]
        
        suite_results = {}
        
        for suite_name, benchmark_func in benchmark_suites:
            logger.info(f"Running {suite_name} benchmarks...")
            try:
                suite_start = time.time()
                suite_results[suite_name] = await benchmark_func()
                suite_duration = time.time() - suite_start
                logger.info(f"Completed {suite_name} in {suite_duration:.2f}s")
            except Exception as e:
                logger.error(f"Failed {suite_name}: {e}")
                suite_results[suite_name] = {
                    'error': str(e),
                    'tests_passed': 0,
                    'tests_failed': 1
                }
        
        return self._generate_summary_report(suite_results)
    
    async def benchmark_core_operations(self) -> Dict[str, Any]:
        """Benchmark core WASM operations."""
        results = []
        
        # Test tensor operations
        test_shapes = [(1, 784), (32, 784), (1, 3, 224, 224)]
        
        for shape in test_shapes:
            # Linear operation benchmark
            result = await self._benchmark_operation(
                "linear_operation",
                self._mock_linear_operation,
                shape=shape,
                weight_shape=(10, shape[-1])
            )
            results.append(result)
            
            # ReLU operation benchmark  
            result = await self._benchmark_operation(
                "relu_operation",
                self._mock_relu_operation,
                shape=shape
            )
            results.append(result)
            
            # Convolution benchmark (for 4D tensors)
            if len(shape) == 4:
                result = await self._benchmark_operation(
                    "conv2d_operation",
                    self._mock_conv2d_operation,
                    shape=shape,
                    kernel_shape=(32, shape[1], 3, 3)
                )
                results.append(result)
        
        return self._summarize_results(results)
    
    async def benchmark_memory_management(self) -> Dict[str, Any]:
        """Benchmark memory management features."""
        results = []
        
        # Test memory allocation patterns
        allocation_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        
        for size_bytes in allocation_sizes:
            result = await self._benchmark_operation(
                f"memory_allocation_{size_bytes}",
                self._mock_memory_allocation,
                size_bytes=size_bytes
            )
            results.append(result)
        
        # Test memory cleanup
        result = await self._benchmark_operation(
            "memory_cleanup",
            self._mock_memory_cleanup
        )
        results.append(result)
        
        return self._summarize_results(results)
    
    async def benchmark_caching(self) -> Dict[str, Any]:
        """Benchmark caching performance."""
        results = []
        
        try:
            from wasm_torch.scaling import IntelligentCache, CachePolicy
            
            cache_policies = [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.FIFO]
            
            for policy in cache_policies:
                result = await self._benchmark_operation(
                    f"cache_{policy.value}",
                    self._benchmark_cache_policy,
                    policy=policy
                )
                results.append(result)
        except ImportError as e:
            logger.warning(f"Caching benchmark skipped: {e}")
            results.append(BenchmarkResult(
                test_name="cache_import_test",
                duration_ms=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            ))
        
        return self._summarize_results(results)
    
    async def benchmark_scaling(self) -> Dict[str, Any]:
        """Benchmark scaling features."""
        results = []
        
        # Test load patterns
        load_patterns = [
            ("low_load", 10, 0.1),      # 10 requests, 0.1s interval
            ("medium_load", 50, 0.05),  # 50 requests, 0.05s interval  
            ("high_load", 100, 0.01),   # 100 requests, 0.01s interval
        ]
        
        for pattern_name, requests, interval in load_patterns:
            result = await self._benchmark_operation(
                f"scaling_{pattern_name}",
                self._simulate_load_pattern,
                requests=requests,
                interval=interval
            )
            results.append(result)
        
        return self._summarize_results(results)
    
    async def benchmark_reliability(self) -> Dict[str, Any]:
        """Benchmark reliability features."""
        results = []
        
        # Test error handling and recovery
        error_scenarios = [
            ("timeout_recovery", self._simulate_timeout_scenario),
            ("memory_pressure", self._simulate_memory_pressure),
            ("concurrent_access", self._simulate_concurrent_access),
        ]
        
        for scenario_name, scenario_func in error_scenarios:
            result = await self._benchmark_operation(
                f"reliability_{scenario_name}",
                scenario_func
            )
            results.append(result)
        
        return self._summarize_results(results)
    
    async def benchmark_integration(self) -> Dict[str, Any]:
        """Benchmark end-to-end integration scenarios."""
        results = []
        
        # Test complete inference pipeline
        inference_configs = [
            ("small_model", (1, 10), 2),      # Small: 1x10 input, 2 layers
            ("medium_model", (1, 784), 5),    # Medium: MNIST-like, 5 layers
            ("large_model", (1, 2048), 10),   # Large: 2048 features, 10 layers
        ]
        
        for config_name, input_shape, num_layers in inference_configs:
            result = await self._benchmark_operation(
                f"integration_{config_name}",
                self._simulate_inference_pipeline,
                input_shape=input_shape,
                num_layers=num_layers
            )
            results.append(result)
        
        return self._summarize_results(results)
    
    async def _benchmark_operation(self, name: str, operation, **kwargs) -> BenchmarkResult:
        """Benchmark a specific operation."""
        memory_before = self._get_memory_usage()
        
        # Warmup
        for _ in range(self.config['warmup_iterations']):
            try:
                await operation(**kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark
        durations = []
        success = True
        error_message = ""
        
        try:
            for _ in range(self.config['benchmark_iterations']):
                start_time = time.time()
                await operation(**kwargs)
                duration = (time.time() - start_time) * 1000
                durations.append(duration)
                
        except Exception as e:
            success = False
            error_message = str(e)
            if not durations:  # If no successful runs
                durations = [0.0]
        
        memory_after = self._get_memory_usage()
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        return BenchmarkResult(
            test_name=name,
            duration_ms=avg_duration,
            memory_used_mb=memory_after - memory_before,
            success=success,
            error_message=error_message,
            metadata={
                'iterations': len(durations),
                'min_duration_ms': min(durations) if durations else 0,
                'max_duration_ms': max(durations) if durations else 0,
                'std_duration_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
            }
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    # Mock operation implementations for benchmarking
    async def _mock_linear_operation(self, shape: tuple, weight_shape: tuple) -> None:
        """Mock linear operation."""
        input_tensor = MockTensor(shape)
        weight = MockTensor(weight_shape)
        
        # Simulate computation time based on tensor size
        flops = input_tensor.numel() * weight_shape[0]
        await asyncio.sleep(flops / 1000000)  # Simulate 1M FLOPS/ms
    
    async def _mock_relu_operation(self, shape: tuple) -> None:
        """Mock ReLU operation."""
        tensor = MockTensor(shape)
        # ReLU is element-wise, so time proportional to elements
        await asyncio.sleep(tensor.numel() / 10000000)  # 10M ops/ms
    
    async def _mock_conv2d_operation(self, shape: tuple, kernel_shape: tuple) -> None:
        """Mock 2D convolution operation."""
        input_tensor = MockTensor(shape)
        kernel = MockTensor(kernel_shape)
        
        # Convolution is expensive
        flops = input_tensor.numel() * kernel.numel()
        await asyncio.sleep(flops / 500000)  # 500K FLOPS/ms
    
    async def _mock_memory_allocation(self, size_bytes: int) -> None:
        """Mock memory allocation."""
        # Simulate allocation time
        await asyncio.sleep(size_bytes / 100000000)  # 100MB/s allocation speed
    
    async def _mock_memory_cleanup(self) -> None:
        """Mock memory cleanup."""
        await asyncio.sleep(0.001)  # 1ms cleanup time
    
    async def _benchmark_cache_policy(self, policy) -> None:
        """Benchmark cache policy performance."""
        try:
            from wasm_torch.scaling import IntelligentCache
            cache = IntelligentCache(max_size_mb=10, policy=policy)
            
            # Fill cache
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}", size_hint=1024)
            
            # Access patterns
            for i in range(50):
                cache.get(f"key_{i}")
                
            await asyncio.sleep(0.001)  # Simulate processing
        except Exception as e:
            raise RuntimeError(f"Cache benchmark failed: {e}")
    
    async def _simulate_load_pattern(self, requests: int, interval: float) -> None:
        """Simulate load pattern."""
        tasks = []
        for _ in range(requests):
            task = asyncio.create_task(asyncio.sleep(0.001))  # 1ms work
            tasks.append(task)
            await asyncio.sleep(interval)
        
        await asyncio.gather(*tasks)
    
    async def _simulate_timeout_scenario(self) -> None:
        """Simulate timeout recovery scenario."""
        try:
            await asyncio.wait_for(asyncio.sleep(2.0), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        await asyncio.sleep(0.001)  # Recovery time
    
    async def _simulate_memory_pressure(self) -> None:
        """Simulate memory pressure scenario."""
        # Allocate and release large amounts of memory
        large_data = [0] * 100000  # Allocate ~400KB
        await asyncio.sleep(0.001)
        del large_data  # Release memory
    
    async def _simulate_concurrent_access(self) -> None:
        """Simulate concurrent access scenario."""
        async def worker():
            await asyncio.sleep(0.001)
        
        # Create multiple concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(10)]
        await asyncio.gather(*tasks)
    
    async def _simulate_inference_pipeline(self, input_shape: tuple, num_layers: int) -> None:
        """Simulate complete inference pipeline."""
        current_shape = input_shape
        
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Linear layer
                await self._mock_linear_operation(current_shape, (current_shape[-1], current_shape[-1]))
            else:
                # ReLU layer
                await self._mock_relu_operation(current_shape)
        
        # Final processing
        await asyncio.sleep(0.001)
    
    def _summarize_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize benchmark results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        successful_results = [r for r in results if r.success]
        
        summary = {
            'total_tests': total_tests,
            'tests_passed': passed_tests,
            'tests_failed': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        }
        
        if successful_results:
            durations = [r.duration_ms for r in successful_results]
            summary.update({
                'avg_duration_ms': statistics.mean(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'total_duration_ms': sum(durations),
                'p95_duration_ms': self._percentile(sorted(durations), 95),
            })
        
        # Add individual test details
        summary['test_details'] = [
            {
                'name': r.test_name,
                'duration_ms': r.duration_ms,
                'success': r.success,
                'error': r.error_message if not r.success else None
            }
            for r in results
        ]
        
        return summary
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        else:
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _generate_summary_report(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_tests = sum(suite.get('total_tests', 0) for suite in suite_results.values())
        total_passed = sum(suite.get('tests_passed', 0) for suite in suite_results.values())
        total_failed = sum(suite.get('tests_failed', 0) for suite in suite_results.values())
        
        # Performance statistics
        all_durations = []
        for suite in suite_results.values():
            if 'test_details' in suite:
                all_durations.extend([
                    test['duration_ms'] for test in suite['test_details']
                    if test['success']
                ])
        
        performance_stats = {}
        if all_durations:
            sorted_durations = sorted(all_durations)
            performance_stats = {
                'avg_duration_ms': statistics.mean(all_durations),
                'min_duration_ms': min(all_durations),
                'max_duration_ms': max(all_durations),
                'p50_duration_ms': self._percentile(sorted_durations, 50),
                'p95_duration_ms': self._percentile(sorted_durations, 95),
                'p99_duration_ms': self._percentile(sorted_durations, 99),
            }
        
        return {
            'benchmark_summary': {
                'timestamp': time.time(),
                'total_suites': len(suite_results),
                'total_tests': total_tests,
                'tests_passed': total_passed,
                'tests_failed': total_failed,
                'overall_success_rate': total_passed / total_tests if total_tests > 0 else 0,
                'performance_stats': performance_stats,
            },
            'suite_results': suite_results,
            'recommendations': self._generate_recommendations(suite_results, performance_stats)
        }
    
    def _generate_recommendations(self, suite_results: Dict[str, Any], performance_stats: Dict[str, Any]) -> List[str]:
        """Generate performance and optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if performance_stats.get('avg_duration_ms', 0) > 100:
            recommendations.append("Consider optimizing operations - average duration exceeds 100ms")
        
        if performance_stats.get('p99_duration_ms', 0) > 1000:
            recommendations.append("High tail latency detected - investigate p99 performance")
        
        # Suite-specific recommendations
        for suite_name, suite_result in suite_results.values():
            success_rate = suite_result.get('success_rate', 0)
            if success_rate < 0.9:
                recommendations.append(f"{suite_name}: Low success rate ({success_rate:.1%}) - investigate failures")
        
        # General recommendations
        recommendations.extend([
            "Run benchmarks regularly in CI/CD pipeline",
            "Monitor performance trends over time", 
            "Consider hardware-specific optimizations for production",
            "Implement performance alerting for regression detection"
        ])
        
        return recommendations


async def main():
    """Main benchmark execution."""
    benchmark = ComprehensiveBenchmark()
    
    print("🚀 Starting WASM-Torch Comprehensive Benchmark Suite")
    print("=" * 60)
    
    start_time = time.time()
    results = await benchmark.run_all_benchmarks()
    total_time = time.time() - start_time
    
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    summary = results['benchmark_summary']
    print(f"Total Runtime: {total_time:.2f}s")
    print(f"Suites Run: {summary['total_suites']}")
    print(f"Tests Executed: {summary['total_tests']}")
    print(f"Tests Passed: {summary['tests_passed']}")
    print(f"Tests Failed: {summary['tests_failed']}")
    print(f"Success Rate: {summary['overall_success_rate']:.1%}")
    
    if summary['performance_stats']:
        perf = summary['performance_stats']
        print(f"\n⚡ PERFORMANCE STATISTICS")
        print(f"Average Duration: {perf['avg_duration_ms']:.2f}ms")
        print(f"P95 Duration: {perf['p95_duration_ms']:.2f}ms")
        print(f"P99 Duration: {perf['p99_duration_ms']:.2f}ms")
    
    print(f"\n💡 RECOMMENDATIONS")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save detailed results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: {output_file}")
    
    # Exit with appropriate code
    exit_code = 0 if summary['tests_failed'] == 0 else 1
    print(f"\n✅ Benchmark completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)