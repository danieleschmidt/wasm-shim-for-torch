"""
Robust Monitoring System - Generation 2: Make It Robust
Comprehensive monitoring, alerting, and observability for production systems.
"""

import asyncio
import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import weakref
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    
    COUNTER = auto()      # Monotonic counter (requests, errors)
    GAUGE = auto()        # Point-in-time value (CPU, memory)
    HISTOGRAM = auto()    # Distribution of values (latency)
    SUMMARY = auto()      # Summary statistics
    TIMER = auto()        # Timing measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = auto()         # Informational
    WARNING = auto()      # Warning condition
    CRITICAL = auto()     # Critical condition
    EMERGENCY = auto()    # System emergency


@dataclass
class Metric:
    """Individual metric data point."""
    
    name: str
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'type': self.metric_type.name,
            'metadata': self.metadata
        }


@dataclass
class Alert:
    """Alert data structure."""
    
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'severity': self.severity.name,
            'message': self.message,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'metadata': self.metadata
        }


class MetricCollector:
    """
    Thread-safe metric collection system with efficient storage.
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Performance counters
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric with thread safety."""
        with self._lock:
            metric_key = self._get_metric_key(metric.name, metric.labels)
            self._metrics[metric_key].append(metric)
            
            # Update typed storage for faster access
            if metric.metric_type == MetricType.COUNTER:
                self._counters[metric_key] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self._gauges[metric_key] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self._histograms[metric_key].append(metric.value)
                # Keep only recent values for performance
                if len(self._histograms[metric_key]) > 1000:
                    self._histograms[metric_key] = self._histograms[metric_key][-1000:]
            elif metric.metric_type == MetricType.TIMER:
                self._timers[metric_key].append(metric.value)
                if len(self._timers[metric_key]) > 1000:
                    self._timers[metric_key] = self._timers[metric_key][-1000:]
            
            # Store metadata
            if metric_key not in self._metric_metadata:
                self._metric_metadata[metric_key] = {
                    'first_seen': metric.timestamp,
                    'metric_type': metric.metric_type,
                    'labels': metric.labels.copy()
                }
            self._metric_metadata[metric_key]['last_seen'] = metric.timestamp
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value of a metric."""
        metric_key = self._get_metric_key(name, labels or {})
        
        with self._lock:
            if metric_key in self._counters:
                return self._counters[metric_key]
            elif metric_key in self._gauges:
                return self._gauges[metric_key]
            elif metric_key in self._metrics and self._metrics[metric_key]:
                return self._metrics[metric_key][-1].value
        
        return None
    
    def get_metric_history(
        self, 
        name: str, 
        labels: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None
    ) -> List[Metric]:
        """Get historical values for a metric."""
        metric_key = self._get_metric_key(name, labels or {})
        
        with self._lock:
            if metric_key not in self._metrics:
                return []
            
            history = list(self._metrics[metric_key])
            if limit:
                history = history[-limit:]
            
            return history
    
    def get_histogram_stats(
        self, 
        name: str, 
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, float]]:
        """Get statistical summary of histogram data."""
        metric_key = self._get_metric_key(name, labels or {})
        
        with self._lock:
            if metric_key not in self._histograms or not self._histograms[metric_key]:
                return None
            
            values = self._histograms[metric_key]
            
            try:
                return {
                    'count': len(values),
                    'sum': sum(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99)
                }
            except Exception as e:
                logger.error(f"Error calculating histogram stats: {e}")
                return None
    
    def list_metrics(self) -> List[str]:
        """List all metric keys."""
        with self._lock:
            return list(self._metric_metadata.keys())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                'total_metrics': len(self._metric_metadata),
                'counters': len(self._counters),
                'gauges': len(self._gauges),
                'histograms': len(self._histograms),
                'timers': len(self._timers),
                'metrics_by_type': defaultdict(int)
            }
            
            for metadata in self._metric_metadata.values():
                metric_type = metadata['metric_type'].name
                summary['metrics_by_type'][metric_type] += 1
            
            return summary
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate metric key from name and labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class AlertManager:
    """
    Alert management system with deduplication and notification.
    """
    
    def __init__(self):
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        self._notification_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'alerts_by_severity': {sev.name: 0 for sev in AlertSeverity}
        }
    
    def register_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]) -> None:
        """Register an alert rule function."""
        with self._lock:
            self._alert_rules.append(rule)
    
    def register_notification_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a notification callback."""
        with self._lock:
            self._notification_callbacks.append(callback)
    
    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger an alert with deduplication."""
        with self._lock:
            # Check for existing alert with same ID
            if alert.id in self._active_alerts:
                existing_alert = self._active_alerts[alert.id]
                if not existing_alert.resolved:
                    # Update existing alert
                    existing_alert.timestamp = alert.timestamp
                    existing_alert.metadata.update(alert.metadata)
                    return False  # Not a new alert
            
            # Add new alert
            self._active_alerts[alert.id] = alert
            self._alert_history.append(alert)
            
            # Update statistics
            self._stats['total_alerts'] += 1
            self._stats['active_alerts'] += 1
            self._stats['alerts_by_severity'][alert.severity.name] += 1
            
            # Trigger notifications
            for callback in self._notification_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")
            
            logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
            return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                if not alert.resolved:
                    alert.resolve()
                    self._stats['active_alerts'] -= 1
                    self._stats['resolved_alerts'] += 1
                    
                    logger.info(f"Alert resolved: {alert.name}")
                    return True
            
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [alert for alert in self._active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            history = list(self._alert_history)
            return history[-limit:] if limit else history
    
    def check_alert_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check all registered alert rules against metrics."""
        triggered_alerts = []
        
        with self._lock:
            for rule in self._alert_rules:
                try:
                    alert = rule(metrics)
                    if alert and self.trigger_alert(alert):
                        triggered_alerts.append(alert)
                except Exception as e:
                    logger.error(f"Alert rule check failed: {e}")
        
        return triggered_alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['active_alerts'] = len(self.get_active_alerts())
            return stats


class RobustMonitoringSystem:
    """
    Comprehensive monitoring system combining metrics and alerts.
    Generation 2: Make It Robust - Enhanced reliability and performance.
    """
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # System metrics
        self._system_start_time = time.time()
        self._last_health_check = time.time()
        
        # Performance tracking
        self._operation_timers: Dict[str, float] = {}
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    async def start(self) -> bool:
        """Start the monitoring system."""
        try:
            if self._running:
                return True
            
            logger.info("Starting robust monitoring system")
            self._running = True
            
            # Start background monitoring
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            # Record startup metric
            self.record_counter("system_starts", 1, {"component": "monitoring"})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        try:
            logger.info("Stopping robust monitoring system")
            self._running = False
            
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Record shutdown metric
            self.record_counter("system_stops", 1, {"component": "monitoring"})
            
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")
    
    def record_counter(
        self, 
        name: str, 
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=MetricType.COUNTER
        )
        self.metric_collector.record_metric(metric)
    
    def record_gauge(
        self, 
        name: str, 
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self.metric_collector.record_metric(metric)
    
    def record_histogram(
        self, 
        name: str, 
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric."""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=MetricType.HISTOGRAM
        )
        self.metric_collector.record_metric(metric)
    
    def record_timer(
        self, 
        name: str, 
        duration: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timer metric."""
        metric = Metric(
            name=name,
            value=duration,
            labels=labels or {},
            metric_type=MetricType.TIMER
        )
        self.metric_collector.record_metric(metric)
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(f"operation_duration", duration, 
                            {**(labels or {}), "operation": operation_name})
            self.record_histogram(f"operation_latency", duration,
                                {**(labels or {}), "operation": operation_name})
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        current_time = time.time()
        uptime = current_time - self._system_start_time
        
        # Collect basic metrics
        basic_metrics = {
            'uptime_seconds': uptime,
            'last_health_check': self._last_health_check,
            'time_since_health_check': current_time - self._last_health_check,
            'monitoring_running': self._running
        }
        
        # Get metric summary
        metric_summary = self.metric_collector.get_metrics_summary()
        
        # Get alert status
        alert_stats = self.alert_manager.get_alert_statistics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Calculate health score
        health_score = self._calculate_health_score(alert_stats, metric_summary)
        
        return {
            'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy',
            'health_score': health_score,
            'uptime_seconds': uptime,
            'metrics': metric_summary,
            'alerts': {
                'active_count': len(active_alerts),
                'total_alerts': alert_stats['total_alerts'],
                'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'emergency_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.EMERGENCY])
            },
            'last_updated': current_time
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics."""
        metrics = {}
        
        # Operation latencies
        latency_stats = self.metric_collector.get_histogram_stats("operation_latency")
        if latency_stats:
            metrics['operation_latency'] = latency_stats
        
        # System counters
        for counter_name in ['requests_total', 'errors_total', 'system_starts']:
            value = self.metric_collector.get_metric_value(counter_name)
            if value is not None:
                metrics[counter_name] = value
        
        # Recent error rate
        error_count = self.metric_collector.get_metric_value('errors_total') or 0
        request_count = self.metric_collector.get_metric_value('requests_total') or 0
        if request_count > 0:
            metrics['error_rate'] = error_count / request_count
        else:
            metrics['error_rate'] = 0.0
        
        return metrics
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                current_time = time.time()
                
                # Update health check timestamp
                self._last_health_check = current_time
                self.record_gauge("last_health_check", current_time)
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check alert rules
                metrics = self.get_performance_metrics()
                self.alert_manager.check_alert_rules(metrics)
                
                # Sleep for monitoring interval
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _collect_system_metrics(self) -> None:
        """Collect basic system metrics."""
        try:
            current_time = time.time()
            
            # Record uptime
            uptime = current_time - self._system_start_time
            self.record_gauge("system_uptime_seconds", uptime)
            
            # Record active tasks
            try:
                active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
                self.record_gauge("active_async_tasks", active_tasks)
            except Exception:
                pass  # Ignore task counting errors
            
            # Record memory usage (basic approximation)
            import sys
            self.record_gauge("python_objects_total", len(gc.get_objects()) if 'gc' in sys.modules else 0)
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        
        def high_error_rate_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
            """Alert on high error rate."""
            error_rate = metrics.get('error_rate', 0.0)
            if error_rate > 0.1:  # 10% error rate
                return Alert(
                    id="high_error_rate",
                    name="High Error Rate",
                    severity=AlertSeverity.WARNING if error_rate < 0.2 else AlertSeverity.CRITICAL,
                    message=f"Error rate is {error_rate:.2%}",
                    labels={"rule": "error_rate"}
                )
            return None
        
        def system_health_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
            """Alert on system health issues."""
            # Check for very old health check
            last_check = self.metric_collector.get_metric_value("last_health_check")
            if last_check and time.time() - last_check > 60:  # 1 minute
                return Alert(
                    id="stale_health_check",
                    name="Stale Health Check",
                    severity=AlertSeverity.WARNING,
                    message="Health check is stale",
                    labels={"rule": "health_check"}
                )
            return None
        
        self.alert_manager.register_alert_rule(high_error_rate_rule)
        self.alert_manager.register_alert_rule(system_health_rule)
    
    def _calculate_health_score(
        self, 
        alert_stats: Dict[str, Any], 
        metric_summary: Dict[str, Any]
    ) -> float:
        """Calculate overall system health score."""
        score = 1.0
        
        # Reduce score based on active alerts
        active_alerts = alert_stats.get('active_alerts', 0)
        if active_alerts > 0:
            score -= min(0.1 * active_alerts, 0.3)  # Max 30% reduction
        
        # Reduce score based on critical alerts
        critical_alerts = alert_stats.get('alerts_by_severity', {}).get('CRITICAL', 0)
        if critical_alerts > 0:
            score -= min(0.2 * critical_alerts, 0.4)  # Max 40% reduction
        
        # Reduce score based on emergency alerts
        emergency_alerts = alert_stats.get('alerts_by_severity', {}).get('EMERGENCY', 0)
        if emergency_alerts > 0:
            score -= 0.5  # 50% reduction for any emergency
        
        return max(0.0, score)


# Global monitoring instance
_global_monitoring_system: Optional[RobustMonitoringSystem] = None


def get_global_monitoring_system() -> RobustMonitoringSystem:
    """Get the global monitoring system instance."""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = RobustMonitoringSystem()
    return _global_monitoring_system


# Import gc for memory metrics
try:
    import gc
except ImportError:
    gc = None


# Example usage and testing
async def demo_robust_monitoring():
    """Demonstration of robust monitoring system."""
    monitoring = get_global_monitoring_system()
    
    try:
        # Start monitoring
        await monitoring.start()
        print("Monitoring system started")
        
        # Record some sample metrics
        monitoring.record_counter("requests_total", 100)
        monitoring.record_counter("errors_total", 5)
        monitoring.record_gauge("active_connections", 25)
        
        # Record some latency measurements
        for i in range(10):
            latency = 0.1 + (i * 0.01)  # Simulate increasing latency
            monitoring.record_histogram("request_duration", latency)
        
        # Use timing context manager
        async with monitoring.time_operation("test_operation"):
            await asyncio.sleep(0.1)  # Simulate work
        
        # Wait a bit for monitoring to collect data
        await asyncio.sleep(2)
        
        # Get system health
        health = monitoring.get_system_health()
        print(f"\nSystem Health:")
        print(json.dumps(health, indent=2))
        
        # Get performance metrics
        perf_metrics = monitoring.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(json.dumps(perf_metrics, indent=2))
        
        # Show active alerts
        active_alerts = monitoring.alert_manager.get_active_alerts()
        print(f"\nActive Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  - {alert.name}: {alert.message}")
        
        # Show metric summary
        metric_summary = monitoring.metric_collector.get_metrics_summary()
        print(f"\nMetric Summary:")
        print(json.dumps(metric_summary, indent=2))
        
    finally:
        # Stop monitoring
        await monitoring.stop()
        print("\nMonitoring system stopped")


if __name__ == "__main__":
    asyncio.run(demo_robust_monitoring())