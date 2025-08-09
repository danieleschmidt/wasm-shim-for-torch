"""Advanced monitoring and observability for WASM-Torch production systems."""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import torch
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSample:
    """A single metric sample with timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric samples."""
    name: str
    metric_type: MetricType
    description: str
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_sample(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new sample to the series."""
        sample = MetricSample(
            value=value,
            timestamp=time.time(),
            labels={**self.labels, **(labels or {})}
        )
        self.samples.append(sample)


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, retention_seconds: int = 3600):  # 1 hour default
        self.metrics: Dict[str, MetricSeries] = {}
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()
        
    def counter(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        """Create or increment a counter metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    description=description,
                    labels=labels or {}
                )
            
            # For counters, always increment by 1
            current_value = 0
            if self.metrics[name].samples:
                current_value = self.metrics[name].samples[-1].value
            
            self.metrics[name].add_sample(current_value + 1, labels)
    
    def gauge(self, name: str, value: float, description: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric to a specific value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    description=description,
                    labels=labels or {}
                )
            
            self.metrics[name].add_sample(value, labels)
    
    def histogram(self, name: str, value: float, description: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=MetricType.HISTOGRAM,
                    description=description,
                    labels=labels or {}
                )
            
            self.metrics[name].add_sample(value, labels)
    
    def summary(self, name: str, value: float, description: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a summary metric."""
        self.histogram(name, value, description, labels)  # Same implementation as histogram
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric by name."""
        with self._lock:
            return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def cleanup_old_samples(self) -> None:
        """Remove samples older than retention period."""
        cutoff_time = time.time() - self.retention_seconds
        
        with self._lock:
            for metric in self.metrics.values():
                while metric.samples and metric.samples[0].timestamp < cutoff_time:
                    metric.samples.popleft()
    
    def get_metric_statistics(self, name: str, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric within a time window."""
        metric = self.get_metric(name)
        if not metric or not metric.samples:
            return {}
        
        # Filter by time window if specified
        cutoff_time = time.time() - (window_seconds or self.retention_seconds)
        values = [
            sample.value for sample in metric.samples 
            if sample.timestamp >= cutoff_time
        ]
        
        if not values:
            return {}
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "latest": values[-1] if values else 0,
        }
        
        if len(values) > 1:
            stats["stddev"] = statistics.stdev(values)
            stats["median"] = statistics.median(values)
            
            # Percentiles for histogram-like metrics
            if metric.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                sorted_values = sorted(values)
                stats["p50"] = statistics.median(sorted_values)
                stats["p90"] = self._percentile(sorted_values, 90)
                stats["p95"] = self._percentile(sorted_values, 95)
                stats["p99"] = self._percentile(sorted_values, 99)
        
        return stats
    
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


class AlertRule:
    """Defines alerting rules based on metrics."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: str,  # e.g., "greater_than", "less_than", "equals"
        threshold: float,
        duration_seconds: int = 60,
        description: str = ""
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.duration_seconds = duration_seconds
        self.description = description
        self.active_since: Optional[float] = None
        self.last_triggered: Optional[float] = None
        self.trigger_count = 0
        
    def evaluate(self, metric_value: float) -> bool:
        """Evaluate if the alert condition is met."""
        if self.condition == "greater_than":
            return metric_value > self.threshold
        elif self.condition == "less_than":
            return metric_value < self.threshold
        elif self.condition == "equals":
            return abs(metric_value - self.threshold) < 0.001
        elif self.condition == "not_equals":
            return abs(metric_value - self.threshold) >= 0.001
        else:
            logger.warning(f"Unknown alert condition: {self.condition}")
            return False
    
    def check_duration(self) -> bool:
        """Check if alert has been active long enough to trigger."""
        if self.active_since is None:
            return False
        
        return time.time() - self.active_since >= self.duration_seconds
    
    def reset(self) -> None:
        """Reset alert state."""
        self.active_since = None


class AlertManager:
    """Manages alerting based on metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_handlers: List[Callable[[AlertRule, float], None]] = []
        self.active_alerts: Dict[str, AlertRule] = {}
        
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def add_alert_handler(self, handler: Callable[[AlertRule, float], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def evaluate_alerts(self) -> List[AlertRule]:
        """Evaluate all alert rules and trigger alerts if needed."""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            metric = self.metrics_collector.get_metric(rule.metric_name)
            if not metric or not metric.samples:
                continue
            
            # Get the latest metric value
            current_value = metric.samples[-1].value
            
            # Check if condition is met
            condition_met = rule.evaluate(current_value)
            
            if condition_met:
                # Start tracking if not already active
                if rule.active_since is None:
                    rule.active_since = time.time()
                
                # Check if duration threshold is met
                if rule.check_duration():
                    # Trigger alert if not already active
                    if rule_name not in self.active_alerts:
                        rule.last_triggered = time.time()
                        rule.trigger_count += 1
                        self.active_alerts[rule_name] = rule
                        triggered_alerts.append(rule)
                        
                        # Call alert handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(rule, current_value)
                            except Exception as e:
                                logger.error(f"Alert handler failed: {e}")
            else:
                # Reset alert if condition is no longer met
                if rule_name in self.active_alerts:
                    del self.active_alerts[rule_name]
                rule.reset()
        
        return triggered_alerts


class PerformanceTracker:
    """Tracks detailed performance metrics for operations."""
    
    def __init__(self):
        self.operation_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_errors: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_operation(self, operation_name: str, duration_ms: float, success: bool = True) -> None:
        """Record an operation execution."""
        with self._lock:
            self.operation_timings[operation_name].append({
                'duration_ms': duration_ms,
                'timestamp': time.time(),
                'success': success
            })
            self.operation_counts[operation_name] += 1
            if not success:
                self.operation_errors[operation_name] += 1
    
    def get_operation_stats(self, operation_name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        with self._lock:
            if operation_name not in self.operation_timings:
                return {}
            
            cutoff_time = time.time() - window_seconds
            recent_operations = [
                op for op in self.operation_timings[operation_name]
                if op['timestamp'] >= cutoff_time
            ]
            
            if not recent_operations:
                return {}
            
            durations = [op['duration_ms'] for op in recent_operations]
            successful_ops = [op for op in recent_operations if op['success']]
            
            return {
                'total_operations': len(recent_operations),
                'successful_operations': len(successful_ops),
                'error_rate': 1.0 - (len(successful_ops) / len(recent_operations)),
                'avg_duration_ms': statistics.mean(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'p50_duration_ms': statistics.median(durations),
                'p95_duration_ms': self._percentile(sorted(durations), 95),
                'p99_duration_ms': self._percentile(sorted(durations), 99),
                'throughput_ops_per_sec': len(recent_operations) / window_seconds
            }
    
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


class MonitoringSystem:
    """Comprehensive monitoring system for WASM-Torch."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.metrics_collector = MetricsCollector(
            retention_seconds=self.config.get('retention_seconds', 3600)
        )
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_tracker = PerformanceTracker()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default alerts
        self._setup_default_alerts()
        self._setup_default_alert_handlers()
    
    def _setup_default_alerts(self) -> None:
        """Setup default alerting rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric_name="wasm_torch_error_rate",
                condition="greater_than",
                threshold=0.05,  # 5% error rate
                duration_seconds=60,
                description="Error rate is above 5%"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="wasm_torch_memory_usage_mb",
                condition="greater_than",
                threshold=900,  # 900MB
                duration_seconds=120,
                description="Memory usage is above 900MB"
            ),
            AlertRule(
                name="slow_inference",
                metric_name="wasm_torch_inference_time_ms",
                condition="greater_than",
                threshold=5000,  # 5 seconds
                duration_seconds=30,
                description="Inference time is above 5 seconds"
            ),
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def _setup_default_alert_handlers(self) -> None:
        """Setup default alert handlers."""
        
        def log_alert_handler(rule: AlertRule, value: float) -> None:
            """Log alerts to the standard logger."""
            logger.warning(f"ALERT: {rule.name} - {rule.description} "
                          f"(current value: {value:.2f}, threshold: {rule.threshold})")
        
        self.alert_manager.add_alert_handler(log_alert_handler)
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Evaluate alerts
                triggered_alerts = self.alert_manager.evaluate_alerts()
                
                if triggered_alerts:
                    logger.info(f"Triggered {len(triggered_alerts)} alerts")
                
                # Sleep before next evaluation
                await asyncio.sleep(self.config.get('evaluation_interval', 10))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of old metrics."""
        while self._running:
            try:
                # Clean up old samples
                self.metrics_collector.cleanup_old_samples()
                
                # Sleep for cleanup interval
                await asyncio.sleep(self.config.get('cleanup_interval', 300))  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(30)
    
    def record_inference_metrics(self, model_id: str, inference_time_ms: float, success: bool, input_shape: tuple) -> None:
        """Record metrics for a model inference operation."""
        # Record basic metrics
        self.metrics_collector.counter("wasm_torch_inferences_total", 
                                      labels={"model_id": model_id, "status": "success" if success else "error"})
        
        if success:
            self.metrics_collector.histogram("wasm_torch_inference_time_ms", inference_time_ms,
                                            labels={"model_id": model_id})
        else:
            self.metrics_collector.counter("wasm_torch_errors_total", 
                                          labels={"model_id": model_id, "operation": "inference"})
        
        # Record performance tracking
        self.performance_tracker.record_operation(f"inference_{model_id}", inference_time_ms, success)
        
        # Record tensor size metric
        tensor_size = 1
        for dim in input_shape:
            tensor_size *= dim
        self.metrics_collector.histogram("wasm_torch_input_tensor_size", tensor_size,
                                        labels={"model_id": model_id})
    
    def record_memory_metrics(self, allocated_mb: float, peak_mb: float, limit_mb: float) -> None:
        """Record memory usage metrics."""
        self.metrics_collector.gauge("wasm_torch_memory_usage_mb", allocated_mb)
        self.metrics_collector.gauge("wasm_torch_memory_peak_mb", peak_mb)
        self.metrics_collector.gauge("wasm_torch_memory_usage_ratio", allocated_mb / limit_mb)
    
    def record_operation_metrics(self, operation_name: str, duration_ms: float, success: bool) -> None:
        """Record metrics for a specific operation."""
        self.metrics_collector.counter(f"wasm_torch_operation_total",
                                      labels={"operation": operation_name, "status": "success" if success else "error"})
        
        if success:
            self.metrics_collector.histogram(f"wasm_torch_operation_duration_ms", duration_ms,
                                            labels={"operation": operation_name})
        
        self.performance_tracker.record_operation(operation_name, duration_ms, success)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboards."""
        # Get all metrics with statistics
        metrics_data = {}
        for name, metric in self.metrics_collector.get_all_metrics().items():
            metrics_data[name] = {
                "type": metric.metric_type.value,
                "description": metric.description,
                "statistics": self.metrics_collector.get_metric_statistics(name, window_seconds=300)
            }
        
        # Get active alerts
        active_alerts = [
            {
                "name": alert.name,
                "description": alert.description,
                "threshold": alert.threshold,
                "active_since": alert.active_since,
                "trigger_count": alert.trigger_count
            }
            for alert in self.alert_manager.active_alerts.values()
        ]
        
        # Get performance data for top operations
        operation_stats = {}
        for op_name in list(self.performance_tracker.operation_counts.keys())[:10]:  # Top 10
            operation_stats[op_name] = self.performance_tracker.get_operation_stats(op_name)
        
        return {
            "timestamp": time.time(),
            "metrics": metrics_data,
            "active_alerts": active_alerts,
            "operation_performance": operation_stats,
            "system_health": {
                "monitoring_active": self._running,
                "total_metrics": len(self.metrics_collector.metrics),
                "total_alert_rules": len(self.alert_manager.alert_rules),
                "active_alert_count": len(self.alert_manager.active_alerts)
            }
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, metric in self.metrics_collector.get_all_metrics().items():
            # Add help text
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")
            
            # Add type
            lines.append(f"# TYPE {name} {metric.metric_type.value}")
            
            # Add samples
            if metric.samples:
                latest_sample = metric.samples[-1]
                label_str = ""
                if latest_sample.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in latest_sample.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{name}{label_str} {latest_sample.value} {int(latest_sample.timestamp * 1000)}")
        
        return "\n".join(lines)
    
    def add_custom_alert(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.alert_manager.add_alert_rule(rule)
    
    def add_custom_alert_handler(self, handler: Callable[[AlertRule, float], None]) -> None:
        """Add a custom alert handler."""
        self.alert_manager.add_alert_handler(handler)