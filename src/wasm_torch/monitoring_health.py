"""Monitoring and Health Systems - Generation 2: Production Reliability

Comprehensive monitoring, health checking, and observability systems
for production PyTorch-to-WASM inference services.
"""

import asyncio
import time
import logging
import threading
import os
from typing import Dict, List, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import hashlib

# Optional psutil import for system monitoring
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    # Mock psutil for demo purposes
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 25.5  # Mock CPU usage
        
        class VirtualMemory:
            percent = 45.2
            available = 8 * 1024**3  # 8GB available
        
        class DiskUsage:
            percent = 60.1
            free = 100 * 1024**3  # 100GB free
        
        @staticmethod
        def virtual_memory():
            return MockPsutil.VirtualMemory()
        
        @staticmethod
        def disk_usage(path):
            return MockPsutil.DiskUsage()
    
    psutil = MockPsutil()

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    critical: bool = False


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, critical: bool = False, timeout: float = 10.0):
        self.name = name
        self.critical = critical
        self.timeout = timeout
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)."""
    
    def __init__(self, 
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0,
                 critical: bool = True):
        super().__init__("system_resources", critical)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_percent = memory.percent
            disk_percent = disk.percent
            
            # Determine overall status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED if cpu_percent < 90 else HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                status = HealthStatus.DEGRADED if memory_percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.disk_threshold:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )


class InferenceEngineHealthCheck(HealthCheck):
    """Health check for inference engine status."""
    
    def __init__(self, inference_engine, critical: bool = True):
        super().__init__("inference_engine", critical)
        self.inference_engine = inference_engine
    
    async def check(self) -> HealthCheckResult:
        """Check inference engine health."""
        start_time = time.time()
        
        try:
            if not hasattr(self.inference_engine, 'get_engine_stats'):
                return HealthCheckResult(
                    check_name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Inference engine not available",
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=self.critical
                )
            
            stats = self.inference_engine.get_engine_stats()
            
            # Analyze engine health
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check success rate
            success_rate = stats.get('success_rate', 0)
            if success_rate < 0.95:
                status = HealthStatus.DEGRADED
                issues.append(f"Low success rate: {success_rate:.2%}")
            if success_rate < 0.8:
                status = HealthStatus.UNHEALTHY
            
            # Check queue depth
            queue_depth = stats.get('queue_depth', 0)
            if queue_depth > 100:
                status = HealthStatus.DEGRADED
                issues.append(f"High queue depth: {queue_depth}")
            if queue_depth > 500:
                status = HealthStatus.UNHEALTHY
            
            # Check response times
            avg_time = stats.get('average_inference_time', 0)
            if avg_time > 1.0:  # 1 second
                status = HealthStatus.DEGRADED
                issues.append(f"Slow inference: {avg_time:.3f}s")
            if avg_time > 5.0:
                status = HealthStatus.UNHEALTHY
            
            message = "Inference engine healthy" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details=stats,
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check inference engine: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )


class ModelAvailabilityHealthCheck(HealthCheck):
    """Health check for model availability."""
    
    def __init__(self, inference_engine, required_models: List[str], critical: bool = True):
        super().__init__("model_availability", critical)
        self.inference_engine = inference_engine
        self.required_models = required_models
    
    async def check(self) -> HealthCheckResult:
        """Check model availability."""
        start_time = time.time()
        
        try:
            if not hasattr(self.inference_engine, 'list_models'):
                return HealthCheckResult(
                    check_name=self.name,
                    status=HealthStatus.CRITICAL,
                    message="Cannot check model availability",
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=self.critical
                )
            
            available_models = self.inference_engine.list_models()
            missing_models = [model for model in self.required_models if model not in available_models]
            
            if missing_models:
                status = HealthStatus.CRITICAL
                message = f"Missing required models: {missing_models}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {len(self.required_models)} required models available"
            
            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details={
                    "required_models": self.required_models,
                    "available_models": available_models,
                    "missing_models": missing_models
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check model availability: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=self.critical
            )


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_history))
        self._aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric."""
        with self._lock:
            self._metrics[metric.name].append(metric)
            self._update_aggregated_metric(metric)
    
    def record_counter(self, name: str, value: Union[int, float] = 1, 
                      tags: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: Union[int, float],
                    tags: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)
    
    def record_timing(self, name: str, duration_ms: float,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        metric = Metric(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMING,
            tags=tags or {},
            unit="ms"
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, value: Union[int, float],
                        tags: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a histogram metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)
    
    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[Metric]:
        """Get metric history for a specific metric."""
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            if limit:
                metrics = metrics[-limit:]
            return metrics
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics summary."""
        with self._lock:
            return dict(self._aggregated_metrics)
    
    def get_metrics_summary(self, time_window_seconds: float = 300) -> Dict[str, Any]:
        """Get metrics summary for a time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds
        
        summary = {
            'time_window_seconds': time_window_seconds,
            'metrics': {}
        }
        
        with self._lock:
            for metric_name, metric_history in self._metrics.items():
                recent_metrics = [
                    m for m in metric_history 
                    if m.timestamp >= cutoff_time
                ]
                
                if not recent_metrics:
                    continue
                
                values = [m.value for m in recent_metrics]
                metric_type = recent_metrics[0].metric_type
                
                metric_summary = {
                    'type': metric_type.value,
                    'count': len(recent_metrics),
                    'latest_value': values[-1] if values else 0,
                    'unit': recent_metrics[0].unit
                }
                
                if metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM, MetricType.TIMING]:
                    metric_summary.update({
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'sum': sum(values)
                    })
                
                if metric_type == MetricType.COUNTER:
                    metric_summary['total'] = sum(values)
                    metric_summary['rate_per_second'] = sum(values) / time_window_seconds
                
                summary['metrics'][metric_name] = metric_summary
        
        return summary
    
    def _update_aggregated_metric(self, metric: Metric) -> None:
        """Update aggregated metrics for a metric."""
        if metric.name not in self._aggregated_metrics:
            self._aggregated_metrics[metric.name] = {
                'type': metric.metric_type.value,
                'total_count': 0,
                'latest_value': 0,
                'latest_timestamp': 0,
                'unit': metric.unit
            }
        
        agg = self._aggregated_metrics[metric.name]
        agg['total_count'] += 1
        agg['latest_value'] = metric.value
        agg['latest_timestamp'] = metric.timestamp
    
    def clear_metrics(self, older_than_seconds: Optional[float] = None) -> int:
        """Clear old metrics to free memory."""
        if older_than_seconds is None:
            # Clear all metrics
            with self._lock:
                total_cleared = sum(len(metrics) for metrics in self._metrics.values())
                self._metrics.clear()
                self._aggregated_metrics.clear()
                return total_cleared
        
        # Clear metrics older than specified time
        current_time = time.time()
        cutoff_time = current_time - older_than_seconds
        total_cleared = 0
        
        with self._lock:
            for metric_name, metric_history in self._metrics.items():
                original_count = len(metric_history)
                
                # Remove old metrics
                while metric_history and metric_history[0].timestamp < cutoff_time:
                    metric_history.popleft()
                
                total_cleared += original_count - len(metric_history)
        
        return total_cleared


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._health_checks: Dict[str, HealthCheck] = {}
        self._health_history: deque = deque(maxlen=100)
        self._current_health: Dict[str, HealthCheckResult] = {}
        self._overall_status = HealthStatus.HEALTHY
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        self._metrics_collector = MetricsCollector()
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check."""
        with self._lock:
            self._health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check."""
        with self._lock:
            if name in self._health_checks:
                del self._health_checks[name]
                self._current_health.pop(name, None)
                logger.info(f"Removed health check: {name}")
                return True
            return False
    
    async def start_monitoring(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            logger.warning("Health monitor already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self) -> None:
        """Stop the health monitoring loop."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        results = {}
        
        with self._lock:
            health_checks = list(self._health_checks.values())
        
        # Run health checks concurrently
        tasks = []
        for health_check in health_checks:
            task = asyncio.create_task(self._run_single_health_check(health_check))
            tasks.append((health_check.name, task))
        
        # Collect results
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {e}",
                    critical=self._health_checks[check_name].critical
                )
        
        # Update current health and overall status
        with self._lock:
            self._current_health.update(results)
            self._update_overall_status()
            
            # Add to history
            health_snapshot = {
                'timestamp': time.time(),
                'overall_status': self._overall_status.value,
                'results': {name: result for name, result in results.items()}
            }
            self._health_history.append(health_snapshot)
        
        # Record metrics
        for result in results.values():
            self._metrics_collector.record_timing(
                f"health_check.{result.check_name}.duration",
                result.duration_ms
            )
            
            status_code = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.DEGRADED: 1,
                HealthStatus.UNHEALTHY: 2,
                HealthStatus.CRITICAL: 3
            }[result.status]
            
            self._metrics_collector.record_gauge(
                f"health_check.{result.check_name}.status",
                status_code
            )
        
        return results
    
    async def _run_single_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Run a single health check with timeout."""
        try:
            result = await asyncio.wait_for(health_check.check(), timeout=health_check.timeout)
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {health_check.timeout}s",
                critical=health_check.critical
            )
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check error: {e}",
                critical=health_check.critical
            )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.debug("Started health monitoring loop")
        
        while self._running:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.debug("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(min(self.check_interval, 10))  # Don't spam on errors
        
        logger.debug("Stopped health monitoring loop")
    
    def _update_overall_status(self) -> None:
        """Update overall health status based on individual checks."""
        if not self._current_health:
            self._overall_status = HealthStatus.HEALTHY
            return
        
        # Find the worst status among critical checks
        critical_statuses = [
            result.status for result in self._current_health.values()
            if result.critical
        ]
        
        # Find the worst status among all checks
        all_statuses = [result.status for result in self._current_health.values()]
        
        # Priority: Critical checks determine overall status
        worst_critical = max(critical_statuses, default=HealthStatus.HEALTHY,
                           key=lambda s: list(HealthStatus).index(s))
        worst_overall = max(all_statuses, default=HealthStatus.HEALTHY,
                          key=lambda s: list(HealthStatus).index(s))
        
        # Overall status is the worst of critical checks, but degraded if non-critical checks fail
        if worst_critical in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self._overall_status = worst_critical
        elif worst_overall == HealthStatus.UNHEALTHY and worst_critical == HealthStatus.HEALTHY:
            self._overall_status = HealthStatus.DEGRADED
        else:
            self._overall_status = worst_critical
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                'overall_status': self._overall_status.value,
                'timestamp': time.time(),
                'checks': {
                    name: {
                        'status': result.status.value,
                        'message': result.message,
                        'details': result.details,
                        'duration_ms': result.duration_ms,
                        'critical': result.critical,
                        'timestamp': result.timestamp
                    }
                    for name, result in self._current_health.items()
                }
            }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health check history."""
        with self._lock:
            history = list(self._health_history)
            return history[-limit:] if limit else history
    
    def get_metrics_collector(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics_collector


# Demo function
async def demo_monitoring_health():
    """Demonstrate monitoring and health checking capabilities."""
    
    print("Monitoring and Health Systems Demo - Generation 2")
    print("=" * 55)
    
    # Create health monitor
    monitor = HealthMonitor(check_interval=5.0)  # Fast interval for demo
    
    # Add health checks
    monitor.add_health_check(SystemResourcesHealthCheck())
    
    # Mock inference engine for demo
    class MockInferenceEngine:
        def get_engine_stats(self):
            return {
                'success_rate': 0.98,
                'queue_depth': 5,
                'average_inference_time': 0.05,
                'total_requests': 1000
            }
        
        def list_models(self):
            return ['model1', 'model2', 'model3']
    
    mock_engine = MockInferenceEngine()
    monitor.add_health_check(InferenceEngineHealthCheck(mock_engine))
    monitor.add_health_check(ModelAvailabilityHealthCheck(mock_engine, ['model1', 'model2']))
    
    print("✓ Added health checks")
    
    # Test metrics collection
    metrics = monitor.get_metrics_collector()
    
    # Record some sample metrics
    for i in range(10):
        metrics.record_counter("requests_total", tags={"endpoint": "/predict"})
        metrics.record_gauge("queue_size", i * 2)
        metrics.record_timing("inference_duration", 50 + i * 10)
    
    print("✓ Recorded sample metrics")
    
    # Run health checks once
    print("\\nRunning health checks...")
    results = await monitor.run_health_checks()
    
    for name, result in results.items():
        status_icon = {
            HealthStatus.HEALTHY: "✓",
            HealthStatus.DEGRADED: "⚠",
            HealthStatus.UNHEALTHY: "✗",
            HealthStatus.CRITICAL: "🚨"
        }.get(result.status, "?")
        
        print(f"{status_icon} {name}: {result.status.value} - {result.message}")
    
    # Show current health
    health = monitor.get_current_health()
    print(f"\\nOverall Health: {health['overall_status']}")
    
    # Show metrics summary
    summary = metrics.get_metrics_summary(time_window_seconds=60)
    print(f"\\nMetrics Summary (last 60s):")
    for metric_name, metric_data in summary['metrics'].items():
        print(f"  {metric_name}: {metric_data['latest_value']} {metric_data.get('unit', '')}")
    
    print("\\n📊 Monitoring and Health Systems Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_monitoring_health())