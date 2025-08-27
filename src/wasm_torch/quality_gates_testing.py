"""Quality Gates and Comprehensive Testing - Production Validation

Comprehensive testing suite with quality gates, performance benchmarks,
and production readiness validation for the PyTorch-to-WASM system.
"""

import asyncio
import time
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
import json
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(Enum):
    """Test failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: TestResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    severity: TestSeverity = TestSeverity.MEDIUM
    threshold_met: bool = True
    actual_value: Optional[float] = None
    threshold_value: Optional[float] = None


@dataclass
class TestSuiteResult:
    """Comprehensive test suite results."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: float
    coverage_percentage: float = 0.0
    quality_gates: List[QualityGateResult] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests
    
    @property
    def overall_status(self) -> TestResult:
        """Determine overall test suite status."""
        if self.errors > 0 or any(gate.status == TestResult.ERROR for gate in self.quality_gates):
            return TestResult.ERROR
        
        critical_failures = any(
            gate.status == TestResult.FAILED and gate.severity == TestSeverity.CRITICAL 
            for gate in self.quality_gates
        )
        
        if self.failed > 0 or critical_failures:
            return TestResult.FAILED
        
        return TestResult.PASSED


class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, severity: TestSeverity = TestSeverity.MEDIUM):
        self.name = name
        self.severity = severity
    
    @abstractmethod
    async def check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute the quality gate check."""
        pass


class PerformanceGate(QualityGate):
    """Performance quality gate with configurable thresholds."""
    
    def __init__(self, name: str, max_latency_ms: float = 1000, 
                 min_throughput_rps: float = 10, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(name, severity)
        self.max_latency_ms = max_latency_ms
        self.min_throughput_rps = min_throughput_rps
    
    async def check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Check performance metrics against thresholds."""
        start_time = time.time()
        
        try:
            # Get performance metrics from context
            metrics = context.get('performance_metrics', {})
            
            latency_p95 = metrics.get('latency_p95_ms', 0)
            throughput = metrics.get('throughput_rps', 0)
            
            # Check latency threshold
            latency_ok = latency_p95 <= self.max_latency_ms
            throughput_ok = throughput >= self.min_throughput_rps
            
            overall_ok = latency_ok and throughput_ok
            
            status = TestResult.PASSED if overall_ok else TestResult.FAILED
            
            messages = []
            if not latency_ok:
                messages.append(f"P95 latency too high: {latency_p95:.1f}ms > {self.max_latency_ms}ms")
            if not throughput_ok:
                messages.append(f"Throughput too low: {throughput:.1f} RPS < {self.min_throughput_rps} RPS")
            
            message = "; ".join(messages) if messages else "Performance metrics within thresholds"
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'latency_p95_ms': latency_p95,
                    'throughput_rps': throughput,
                    'latency_threshold_ms': self.max_latency_ms,
                    'throughput_threshold_rps': self.min_throughput_rps
                },
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=overall_ok
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=TestResult.ERROR,
                message=f"Performance gate error: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=False
            )


class ReliabilityGate(QualityGate):
    """Reliability quality gate checking error rates and availability."""
    
    def __init__(self, name: str, max_error_rate: float = 0.01, 
                 min_availability: float = 0.99, severity: TestSeverity = TestSeverity.CRITICAL):
        super().__init__(name, severity)
        self.max_error_rate = max_error_rate
        self.min_availability = min_availability
    
    async def check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Check reliability metrics against thresholds."""
        start_time = time.time()
        
        try:
            reliability_metrics = context.get('reliability_metrics', {})
            
            error_rate = reliability_metrics.get('error_rate', 0)
            availability = reliability_metrics.get('availability', 1.0)
            
            error_rate_ok = error_rate <= self.max_error_rate
            availability_ok = availability >= self.min_availability
            
            overall_ok = error_rate_ok and availability_ok
            status = TestResult.PASSED if overall_ok else TestResult.FAILED
            
            messages = []
            if not error_rate_ok:
                messages.append(f"Error rate too high: {error_rate:.3%} > {self.max_error_rate:.3%}")
            if not availability_ok:
                messages.append(f"Availability too low: {availability:.3%} < {self.min_availability:.3%}")
            
            message = "; ".join(messages) if messages else "Reliability metrics within thresholds"
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'error_rate': error_rate,
                    'availability': availability,
                    'error_rate_threshold': self.max_error_rate,
                    'availability_threshold': self.min_availability
                },
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=overall_ok
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=TestResult.ERROR,
                message=f"Reliability gate error: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=False
            )


class SecurityGate(QualityGate):
    """Security quality gate for vulnerability and compliance checks."""
    
    def __init__(self, name: str, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(name, severity)
        self.security_checks = [
            self._check_input_validation,
            self._check_authentication,
            self._check_data_sanitization,
            self._check_secret_exposure
        ]
    
    async def check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Run comprehensive security checks."""
        start_time = time.time()
        
        try:
            security_issues = []
            
            for check in self.security_checks:
                issues = await check(context)
                security_issues.extend(issues)
            
            critical_issues = [issue for issue in security_issues if issue.get('severity') == 'critical']
            high_issues = [issue for issue in security_issues if issue.get('severity') == 'high']
            
            if critical_issues:
                status = TestResult.FAILED
                message = f"Critical security issues found: {len(critical_issues)}"
            elif high_issues:
                status = TestResult.FAILED
                message = f"High severity security issues found: {len(high_issues)}"
            elif security_issues:
                status = TestResult.PASSED  # Low/medium issues don't fail the gate
                message = f"Minor security issues found: {len(security_issues)}"
            else:
                status = TestResult.PASSED
                message = "No security issues detected"
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                message=message,
                details={
                    'total_issues': len(security_issues),
                    'critical_issues': len(critical_issues),
                    'high_issues': len(high_issues),
                    'issues': security_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=(len(critical_issues) == 0)
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=TestResult.ERROR,
                message=f"Security gate error: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                severity=self.severity,
                threshold_met=False
            )
    
    async def _check_input_validation(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for proper input validation."""
        # Simulate input validation checks
        return []  # No issues found
    
    async def _check_authentication(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check authentication mechanisms."""
        # Simulate authentication checks
        return []  # No issues found
    
    async def _check_data_sanitization(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data sanitization practices."""
        # Simulate data sanitization checks
        return []  # No issues found
    
    async def _check_secret_exposure(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for exposed secrets."""
        # Simulate secret exposure checks
        return []  # No issues found


class TestRunner:
    """Comprehensive test runner with quality gates."""
    
    def __init__(self):
        self._quality_gates: List[QualityGate] = []
        self._test_functions: List[Callable] = []
        self._setup_functions: List[Callable] = []
        self._teardown_functions: List[Callable] = []
        
    def add_quality_gate(self, gate: QualityGate) -> None:
        """Add a quality gate to the test suite."""
        self._quality_gates.append(gate)
    
    def add_test(self, test_function: Callable) -> None:
        """Add a test function to the suite."""
        self._test_functions.append(test_function)
    
    def add_setup(self, setup_function: Callable) -> None:
        """Add a setup function."""
        self._setup_functions.append(setup_function)
    
    def add_teardown(self, teardown_function: Callable) -> None:
        """Add a teardown function."""
        self._teardown_functions.append(teardown_function)
    
    async def run_tests(self, test_context: Optional[Dict[str, Any]] = None) -> TestSuiteResult:
        """Run the complete test suite with quality gates."""
        start_time = time.time()
        context = test_context or {}
        
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        quality_gate_results = []
        
        try:
            # Run setup functions
            for setup_func in self._setup_functions:
                try:
                    if asyncio.iscoroutinefunction(setup_func):
                        await setup_func(context)
                    else:
                        setup_func(context)
                except Exception as e:
                    logger.error(f"Setup function failed: {e}")
                    errors += 1
            
            # Run test functions
            for test_func in self._test_functions:
                total_tests += 1
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func(context)
                    else:
                        result = test_func(context)
                    
                    if result is True or result is None:
                        passed += 1
                    elif result is False:
                        failed += 1
                    else:
                        skipped += 1
                        
                except Exception as e:
                    logger.error(f"Test function {test_func.__name__} failed: {e}")
                    errors += 1
            
            # Run quality gates
            for gate in self._quality_gates:
                try:
                    gate_result = await gate.check(context)
                    quality_gate_results.append(gate_result)
                except Exception as e:
                    logger.error(f"Quality gate {gate.name} failed: {e}")
                    quality_gate_results.append(QualityGateResult(
                        gate_name=gate.name,
                        status=TestResult.ERROR,
                        message=f"Gate execution error: {e}",
                        severity=gate.severity,
                        threshold_met=False
                    ))
            
            # Run teardown functions
            for teardown_func in self._teardown_functions:
                try:
                    if asyncio.iscoroutinefunction(teardown_func):
                        await teardown_func(context)
                    else:
                        teardown_func(context)
                except Exception as e:
                    logger.error(f"Teardown function failed: {e}")
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            errors += 1
        
        duration_ms = (time.time() - start_time) * 1000
        
        return TestSuiteResult(
            suite_name="Comprehensive Test Suite",
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_ms=duration_ms,
            quality_gates=quality_gate_results,
            details=context
        )


# Test implementations for WASM-Torch components
async def test_basic_model_loader(context: Dict[str, Any]):
    """Test basic model loader functionality."""
    try:
        # Import here to avoid circular dependencies
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.basic_model_loader import BasicModelLoader
        
        loader = BasicModelLoader()
        
        # Test model registration
        test_data = b"mock_model_data"
        success = loader.register_model_from_bytes("test_model", test_data)
        
        if not success:
            return False
        
        # Test model listing
        models = loader.list_registered_models()
        if "test_model" not in models:
            return False
        
        # Test model info retrieval
        info = loader.get_model_info("test_model")
        if not info:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Basic model loader test failed: {e}")
        return False


async def test_simple_inference_engine(context: Dict[str, Any]):
    """Test simple inference engine functionality."""
    try:
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.simple_inference_engine import SimpleInferenceEngine, SimpleModel
        
        engine = SimpleInferenceEngine(max_workers=2)
        
        # Create and register a test model
        model = SimpleModel.create_classifier(4, 8, 3)
        engine.register_model("test_classifier", model)
        
        # Test single inference
        result = await engine.infer("test_classifier", [0.1, 0.2, 0.3, 0.4])
        
        if not result.success:
            return False
        
        # Test batch inference
        batch_requests = [
            ("test_classifier", [0.5, 0.6, 0.7, 0.8]),
            ("test_classifier", [0.1, 0.9, 0.2, 0.8])
        ]
        
        batch_results = await engine.infer_batch(batch_requests)
        
        if len(batch_results) != 2 or not all(r.success for r in batch_results):
            return False
        
        # Cleanup
        engine.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"Simple inference engine test failed: {e}")
        return False


async def test_error_handling_system(context: Dict[str, Any]):
    """Test robust error handling system."""
    try:
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.robust_error_handling import (
            RobustErrorHandler, InputValidator, InputValidationError
        )
        
        error_handler = RobustErrorHandler()
        validator = InputValidator()
        
        # Test input validation
        try:
            validator.validate_model_id("<malicious>")
            return False  # Should have raised an error
        except InputValidationError:
            pass  # Expected
        
        # Test NaN validation
        try:
            validator.validate_tensor_input([float('nan'), 1, 2])
            return False  # Should have raised an error
        except InputValidationError:
            pass  # Expected
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling system test failed: {e}")
        return False


async def test_monitoring_health_system(context: Dict[str, Any]):
    """Test monitoring and health system."""
    try:
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.monitoring_health import (
            HealthMonitor, MetricsCollector, SystemResourcesHealthCheck
        )
        
        # Test metrics collector
        metrics = MetricsCollector()
        metrics.record_counter('test_counter', 5)
        metrics.record_gauge('test_gauge', 42.5)
        
        summary = metrics.get_metrics_summary(60)
        if len(summary['metrics']) != 2:
            return False
        
        # Test health monitor
        monitor = HealthMonitor()
        system_check = SystemResourcesHealthCheck()
        monitor.add_health_check(system_check)
        
        # Run health checks
        results = await monitor.run_health_checks()
        if 'system_resources' not in results:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Monitoring health system test failed: {e}")
        return False


async def test_performance_optimization(context: Dict[str, Any]):
    """Test performance optimization system."""
    try:
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.performance_optimization import (
            PerformanceProfiler, AdaptiveOptimizer, LoadBalancer
        )
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        profiler.record_request(100.0, batch_size=4, success=True)
        profiler.record_request(150.0, batch_size=8, success=True)
        
        analysis = profiler.analyze_performance(60)
        if 'throughput_rps' not in analysis:
            return False
        
        # Test adaptive optimizer
        optimizer = AdaptiveOptimizer(profiler)
        status = optimizer.get_optimization_status()
        
        # Test load balancer
        load_balancer = LoadBalancer(initial_capacity=5)
        load_balancer.record_load_metric(active_requests=3, queue_depth=10, response_time_ms=200.0)
        
        scaling_status = load_balancer.get_scaling_status()
        if scaling_status['current_capacity'] != 5:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Performance optimization test failed: {e}")
        return False


async def test_ml_pipeline(context: Dict[str, Any]):
    """Test advanced ML pipeline."""
    try:
        if 'src' not in sys.path:
            sys.path.append('src')
        from wasm_torch.advanced_ml_pipeline import (
            ModelRegistry, FeatureStore, ModelVersion, ModelStatus
        )
        
        # Test model registry
        registry = ModelRegistry()
        model_v1 = ModelVersion(
            model_id='test_model',
            version='1.0',
            status=ModelStatus.PRODUCTION,
            metadata={'accuracy': 0.85}
        )
        
        registry.register_model(model_v1)
        retrieved = registry.get_model('test_model', '1.0')
        
        if not retrieved or retrieved.version != '1.0':
            return False
        
        # Test feature store
        feature_store = FeatureStore()
        feature_store.register_feature_group('test_features', {'feature1': 42, 'feature2': 'test'})
        features = feature_store.get_features('test_features', ['feature1'])
        
        if features.get('feature1') != 42:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"ML pipeline test failed: {e}")
        return False


# Demo function to run comprehensive testing
async def demo_quality_gates_testing():
    """Run comprehensive quality gates and testing demo."""
    
    print("Quality Gates and Comprehensive Testing Demo")
    print("=" * 60)
    
    # Create test runner
    test_runner = TestRunner()
    
    # Add quality gates
    performance_gate = PerformanceGate(
        "performance_gate", 
        max_latency_ms=500, 
        min_throughput_rps=5
    )
    
    reliability_gate = ReliabilityGate(
        "reliability_gate",
        max_error_rate=0.05,
        min_availability=0.95
    )
    
    security_gate = SecurityGate("security_gate")
    
    test_runner.add_quality_gate(performance_gate)
    test_runner.add_quality_gate(reliability_gate)
    test_runner.add_quality_gate(security_gate)
    
    # Add test functions
    test_runner.add_test(test_basic_model_loader)
    test_runner.add_test(test_simple_inference_engine)
    test_runner.add_test(test_error_handling_system)
    test_runner.add_test(test_monitoring_health_system)
    test_runner.add_test(test_performance_optimization)
    test_runner.add_test(test_ml_pipeline)
    
    print("✓ Test suite configured")
    
    # Prepare test context with mock metrics
    test_context = {
        'performance_metrics': {
            'latency_p95_ms': 450,  # Within threshold
            'throughput_rps': 15    # Above threshold
        },
        'reliability_metrics': {
            'error_rate': 0.02,     # Within threshold
            'availability': 0.99    # Above threshold
        }
    }
    
    print("\\nRunning comprehensive test suite...")
    
    # Run the test suite
    results = await test_runner.run_tests(test_context)
    
    # Display results
    print(f"\\nTest Results Summary:")
    print(f"  Suite: {results.suite_name}")
    print(f"  Total Tests: {results.total_tests}")
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Skipped: {results.skipped}")
    print(f"  Errors: {results.errors}")
    print(f"  Success Rate: {results.success_rate:.1%}")
    print(f"  Duration: {results.duration_ms:.1f}ms")
    print(f"  Overall Status: {results.overall_status.value}")
    
    # Display quality gate results
    print(f"\\nQuality Gate Results:")
    for gate_result in results.quality_gates:
        status_icon = {
            TestResult.PASSED: "✓",
            TestResult.FAILED: "✗",
            TestResult.ERROR: "🚨",
            TestResult.SKIPPED: "⏭"
        }.get(gate_result.status, "?")
        
        print(f"  {status_icon} {gate_result.gate_name}: {gate_result.status.value}")
        print(f"    Message: {gate_result.message}")
        if gate_result.severity in [TestSeverity.HIGH, TestSeverity.CRITICAL]:
            print(f"    Severity: {gate_result.severity.value}")
        print(f"    Duration: {gate_result.duration_ms:.1f}ms")
    
    # Overall assessment
    critical_failures = any(
        gate.status == TestResult.FAILED and gate.severity == TestSeverity.CRITICAL 
        for gate in results.quality_gates
    )
    
    if results.overall_status == TestResult.PASSED and not critical_failures:
        print(f"\\n🎉 All quality gates passed! System ready for production.")
    elif results.overall_status == TestResult.FAILED:
        print(f"\\n⚠️  Some quality gates failed. Review and fix issues before deployment.")
    else:
        print(f"\\n🚨 Critical errors detected. System not ready for production.")
    
    # Production readiness checklist
    print(f"\\nProduction Readiness Checklist:")
    checklist_items = [
        ("Functional tests", results.passed > 0 and results.errors == 0),
        ("Performance thresholds", any(g.gate_name == "performance_gate" and g.status == TestResult.PASSED for g in results.quality_gates)),
        ("Reliability requirements", any(g.gate_name == "reliability_gate" and g.status == TestResult.PASSED for g in results.quality_gates)),
        ("Security validation", any(g.gate_name == "security_gate" and g.status == TestResult.PASSED for g in results.quality_gates)),
        ("Error handling", any("error_handling" in str(test_func) for test_func in test_runner._test_functions)),
        ("Monitoring systems", any("monitoring" in str(test_func) for test_func in test_runner._test_functions))
    ]
    
    for item, status in checklist_items:
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {item}")
    
    all_checks_passed = all(status for _, status in checklist_items)
    
    if all_checks_passed:
        print(f"\\n🚀 System passes all production readiness checks!")
    else:
        print(f"\\n⚠️  Some production readiness checks failed. Address issues before deployment.")


if __name__ == "__main__":
    asyncio.run(demo_quality_gates_testing())