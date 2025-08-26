"""Advanced monitoring system with ML-powered anomaly detection and predictive analytics."""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
import statistics
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert with context."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Component health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: float
    timeout_seconds: float
    healthy_threshold: int = 3
    unhealthy_threshold: int = 2
    enabled: bool = True


class AnomalyDetector:
    """ML-based anomaly detection for metrics."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def add_value(self, metric_name: str, value: float, timestamp: float) -> bool:
        """Add metric value and detect anomaly."""
        with self._lock:
            window = self.metric_windows[metric_name]
            window.append({'value': value, 'timestamp': timestamp})
            
            if len(window) < 10:  # Need minimum samples
                return False
            
            # Calculate baseline statistics
            values = [item['value'] for item in window]
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            
            self.baseline_stats[metric_name] = {
                'mean': mean,
                'stdev': stdev,
                'min': min(values),
                'max': max(values),
                'last_updated': timestamp
            }
            
            # Detect anomaly using z-score
            if stdev > 0:
                z_score = abs((value - mean) / stdev)
                return z_score > self.sensitivity
            
            return False
    
    def get_prediction(self, metric_name: str, steps_ahead: int = 5) -> List[float]:
        """Simple trend-based prediction."""
        with self._lock:
            window = self.metric_windows.get(metric_name)
            if not window or len(window) < 10:
                return []
            
            values = [item['value'] for item in window]
            timestamps = [item['timestamp'] for item in window]
            
            # Simple linear regression for trend
            n = len(values)
            x = np.array(range(n))
            y = np.array(values)
            
            # Calculate slope (trend)
            slope = np.polyfit(x, y, 1)[0] if n > 1 else 0
            
            # Predict future values
            last_value = values[-1]
            predictions = []
            for i in range(1, steps_ahead + 1):
                predicted_value = last_value + (slope * i)
                predictions.append(predicted_value)
            
            return predictions
    
    def get_metric_insights(self, metric_name: str) -> Dict[str, Any]:
        """Get insights about a metric."""
        with self._lock:
            if metric_name not in self.baseline_stats:
                return {'error': 'Metric not found'}
            
            stats = self.baseline_stats[metric_name]
            window = self.metric_windows[metric_name]
            
            recent_values = [item['value'] for item in list(window)[-10:]]
            recent_trend = 'stable'
            
            if len(recent_values) >= 3:
                first_third = statistics.mean(recent_values[:3])
                last_third = statistics.mean(recent_values[-3:])
                change_pct = ((last_third - first_third) / first_third * 100) if first_third != 0 else 0
                
                if change_pct > 10:
                    recent_trend = 'increasing'
                elif change_pct < -10:
                    recent_trend = 'decreasing'
            
            return {
                'baseline_stats': stats,
                'recent_trend': recent_trend,
                'sample_count': len(window),
                'predictions': self.get_prediction(metric_name)
            }


class MetricCollector:
    """Collects and stores metrics with efficient aggregation."""
    
    def __init__(self, max_retention_hours: int = 24):
        self.max_retention_hours = max_retention_hours
        self.metrics_storage: Dict[str, List[MetricValue]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        self._lock = threading.Lock()
        
        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def start(self) -> None:
        """Start the metric collector."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        with self._lock:
            self.metrics_storage[metric.name].append(metric)
            
            # Update aggregated metrics for faster queries
            hour_key = int(metric.timestamp // 3600)
            if hour_key not in self.aggregated_metrics[metric.name]:
                self.aggregated_metrics[metric.name][hour_key] = []
            
            self.aggregated_metrics[metric.name][hour_key].append(metric.value)
    
    def get_metric_values(
        self, 
        metric_name: str, 
        start_time: float, 
        end_time: float,
        aggregation: Optional[str] = None
    ) -> List[Union[MetricValue, float]]:
        """Get metric values for time range with optional aggregation."""
        with self._lock:
            if metric_name not in self.metrics_storage:
                return []
            
            # Filter by time range
            filtered_metrics = [
                m for m in self.metrics_storage[metric_name]
                if start_time <= m.timestamp <= end_time
            ]
            
            if not aggregation:
                return filtered_metrics
            
            # Apply aggregation
            values = [m.value for m in filtered_metrics]
            if not values:
                return []
            
            if aggregation == 'avg':
                return [statistics.mean(values)]
            elif aggregation == 'sum':
                return [sum(values)]
            elif aggregation == 'max':
                return [max(values)]
            elif aggregation == 'min':
                return [min(values)]
            elif aggregation == 'count':
                return [len(values)]
            else:
                return values
    
    def get_latest_value(self, metric_name: str) -> Optional[MetricValue]:
        """Get the latest value for a metric."""
        with self._lock:
            if metric_name in self.metrics_storage and self.metrics_storage[metric_name]:
                return self.metrics_storage[metric_name][-1]
            return None
    
    def get_metric_summary(self, metric_name: str, hours_back: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        end_time = time.time()
        start_time = end_time - (hours_back * 3600)
        
        values = self.get_metric_values(metric_name, start_time, end_time)
        if not values:
            return {'error': 'No data available'}
        
        metric_values = [v.value if isinstance(v, MetricValue) else v for v in values]
        
        return {
            'count': len(metric_values),
            'avg': statistics.mean(metric_values),
            'min': min(metric_values),
            'max': max(metric_values),
            'sum': sum(metric_values),
            'latest': metric_values[-1],
            'stdev': statistics.stdev(metric_values) if len(metric_values) > 1 else 0
        }
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old metrics."""
        while not self._shutdown:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Metric cleanup error: {e}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = time.time() - (self.max_retention_hours * 3600)
        
        with self._lock:
            for metric_name in list(self.metrics_storage.keys()):
                # Clean raw metrics
                self.metrics_storage[metric_name] = [
                    m for m in self.metrics_storage[metric_name]
                    if m.timestamp > cutoff_time
                ]
                
                # Clean aggregated metrics
                cutoff_hour = int(cutoff_time // 3600)
                for hour_key in list(self.aggregated_metrics[metric_name].keys()):
                    if hour_key < cutoff_hour:
                        del self.aggregated_metrics[metric_name][hour_key]
        
        logger.info("Completed metric cleanup")
    
    async def shutdown(self) -> None:
        """Shutdown the metric collector."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self._lock = threading.Lock()
    
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        component: str = "system"
    ) -> None:
        """Add alert rule for metric monitoring."""
        self.alert_rules[name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'severity': severity,
            'component': component,
            'enabled': True
        }
        logger.info(f"Added alert rule: {name}")
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def evaluate_metric(self, metric: MetricValue) -> List[Alert]:
        """Evaluate metric against alert rules."""
        new_alerts = []
        
        with self._lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule['enabled'] or rule['metric_name'] != metric.name:
                    continue
                
                should_alert = False
                condition = rule['condition']
                threshold = rule['threshold']
                
                if condition == 'gt' and metric.value > threshold:
                    should_alert = True
                elif condition == 'lt' and metric.value < threshold:
                    should_alert = True
                elif condition == 'eq' and abs(metric.value - threshold) < 0.001:
                    should_alert = True
                
                if should_alert:
                    alert_id = f"{rule_name}:{metric.timestamp}"
                    
                    if alert_id not in self.active_alerts:
                        alert = Alert(
                            id=alert_id,
                            severity=rule['severity'],
                            title=f"Alert: {rule_name}",
                            description=f"Metric {metric.name} value {metric.value} {condition} threshold {threshold}",
                            component=rule['component'],
                            metric_name=metric.name,
                            threshold_value=threshold,
                            current_value=metric.value,
                            timestamp=metric.timestamp
                        )
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        new_alerts.append(alert)
        
        # Send notifications for new alerts
        for alert in new_alerts:
            self._send_notification(alert)
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = time.time()
                del self.active_alerts[alert_id]
                logger.info(f"Resolved alert: {alert_id}")
                return True
        return False
    
    def _send_notification(self, alert: Alert) -> None:
        """Send notification for alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary."""
        with self._lock:
            return {
                'active_alerts': len(self.active_alerts),
                'total_rules': len(self.alert_rules),
                'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule['enabled']),
                'alert_history_count': len(self.alert_history),
                'severity_breakdown': {
                    severity.value: sum(1 for alert in self.active_alerts.values() 
                                       if alert.severity == severity)
                    for severity in AlertSeverity
                }
            }


class HealthMonitor:
    """Monitors component health with configurable checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = {
            'status': 'unknown',
            'last_check': None,
            'consecutive_successes': 0,
            'consecutive_failures': 0,
            'total_checks': 0,
            'last_error': None
        }
        logger.info(f"Registered health check: {health_check.name}")
    
    async def start_monitoring(self) -> None:
        """Start all health check monitoring."""
        for name, check in self.health_checks.items():
            if check.enabled:
                self._check_tasks[name] = asyncio.create_task(
                    self._health_check_loop(name, check)
                )
        logger.info("Health monitoring started")
    
    async def _health_check_loop(self, check_name: str, health_check: HealthCheck) -> None:
        """Run health check in a loop."""
        while not self._shutdown:
            try:
                # Execute health check with timeout
                check_start = time.time()
                try:
                    result = await asyncio.wait_for(
                        self._execute_health_check(health_check),
                        timeout=health_check.timeout_seconds
                    )
                    self._record_health_result(check_name, True, None, time.time() - check_start)
                
                except Exception as e:
                    self._record_health_result(check_name, False, str(e), time.time() - check_start)
                
                # Wait for next check
                await asyncio.sleep(health_check.interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error for {check_name}: {e}")
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _execute_health_check(self, health_check: HealthCheck) -> bool:
        """Execute individual health check."""
        if asyncio.iscoroutinefunction(health_check.check_function):
            return await health_check.check_function()
        else:
            return health_check.check_function()
    
    def _record_health_result(self, check_name: str, success: bool, error: Optional[str], duration: float) -> None:
        """Record health check result."""
        status = self.health_status[check_name]
        status['last_check'] = time.time()
        status['total_checks'] += 1
        status['check_duration'] = duration
        
        if success:
            status['consecutive_successes'] += 1
            status['consecutive_failures'] = 0
            status['last_error'] = None
            
            # Update status based on thresholds
            if status['consecutive_successes'] >= self.health_checks[check_name].healthy_threshold:
                status['status'] = 'healthy'
        else:
            status['consecutive_failures'] += 1
            status['consecutive_successes'] = 0
            status['last_error'] = error
            
            # Update status based on thresholds
            if status['consecutive_failures'] >= self.health_checks[check_name].unhealthy_threshold:
                status['status'] = 'unhealthy'
    
    def get_health_status(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for specific check or all checks."""
        if check_name:
            return self.health_status.get(check_name, {'error': 'Check not found'})
        return self.health_status.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        healthy_checks = sum(1 for status in self.health_status.values() 
                            if status['status'] == 'healthy')
        total_checks = len(self.health_status)
        
        overall_status = 'healthy'
        if healthy_checks == 0:
            overall_status = 'critical'
        elif healthy_checks < total_checks * 0.8:
            overall_status = 'degraded'
        
        return {
            'overall_status': overall_status,
            'healthy_checks': healthy_checks,
            'total_checks': total_checks,
            'health_percentage': (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            'unhealthy_components': [
                name for name, status in self.health_status.items()
                if status['status'] != 'healthy'
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown health monitoring."""
        self._shutdown = True
        for task in self._check_tasks.values():
            task.cancel()
        await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)


class AdvancedMonitoringSystem:
    """Advanced monitoring system with ML-powered analytics."""
    
    def __init__(self, enable_anomaly_detection: bool = True, enable_predictions: bool = True):
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictions = enable_predictions
        
        # Core components
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        
        # Performance tracking
        self.system_metrics: Dict[str, Any] = {
            'monitoring_start_time': time.time(),
            'metrics_processed': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0
        }
        
        self._initialized = False
        logger.info("Advanced Monitoring System initialized")
    
    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        await self.metric_collector.start()
        await self.health_monitor.start_monitoring()
        
        # Register default notification handler
        self.alert_manager.register_notification_handler(self._default_notification_handler)
        
        # Add default health checks
        await self._setup_default_health_checks()
        
        # Add default alert rules
        self._setup_default_alert_rules()
        
        self._initialized = True
        logger.info("Advanced Monitoring System started")
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ) -> None:
        """Record a metric value with optional anomaly detection."""
        if not self._initialized:
            return
        
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        # Store metric
        self.metric_collector.record_metric(metric)
        self.system_metrics['metrics_processed'] += 1
        
        # Anomaly detection
        if self.anomaly_detector:
            is_anomaly = self.anomaly_detector.add_value(name, value, metric.timestamp)
            if is_anomaly:
                self.system_metrics['anomalies_detected'] += 1
                logger.warning(f"Anomaly detected in {name}: {value}")
        
        # Evaluate against alert rules
        alerts = self.alert_manager.evaluate_metric(metric)
        self.system_metrics['alerts_generated'] += len(alerts)
    
    def get_metric_insights(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive insights about a metric."""
        insights = {'metric_name': metric_name}
        
        # Basic statistics
        summary = self.metric_collector.get_metric_summary(metric_name)
        insights['summary'] = summary
        
        # Anomaly detection insights
        if self.anomaly_detector:
            anomaly_insights = self.anomaly_detector.get_metric_insights(metric_name)
            insights['anomaly_analysis'] = anomaly_insights
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alert_manager.alert_history
            if alert.metric_name == metric_name and 
               time.time() - alert.timestamp < 3600
        ]
        insights['recent_alerts'] = len(recent_alerts)
        
        return insights
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data."""
        return {
            'system_metrics': self.system_metrics.copy(),
            'overall_health': self.health_monitor.get_overall_health(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'top_metrics': self._get_top_metrics(),
            'performance_trends': self._get_performance_trends(),
            'system_status': self._get_system_status()
        }
    
    def _get_top_metrics(self) -> List[Dict[str, Any]]:
        """Get top metrics by activity."""
        # This would analyze metric activity and return top metrics
        # Simplified implementation
        return [
            {'name': 'inference_latency', 'activity_score': 95},
            {'name': 'memory_usage', 'activity_score': 87},
            {'name': 'request_rate', 'activity_score': 82}
        ]
    
    def _get_performance_trends(self) -> Dict[str, str]:
        """Get performance trend analysis."""
        return {
            'inference_latency': 'improving',
            'throughput': 'stable',
            'error_rate': 'decreasing',
            'memory_usage': 'stable'
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        health = self.health_monitor.get_overall_health()
        active_alerts = len(self.alert_manager.get_active_alerts())
        
        status = 'healthy'
        if health['overall_status'] == 'critical' or active_alerts > 5:
            status = 'critical'
        elif health['overall_status'] == 'degraded' or active_alerts > 2:
            status = 'degraded'
        
        return {
            'status': status,
            'uptime_seconds': time.time() - self.system_metrics['monitoring_start_time'],
            'components_healthy': health['health_percentage'],
            'active_alerts': active_alerts
        }
    
    async def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        # Memory usage check
        async def check_memory() -> bool:
            # Simplified check - in production, use psutil or similar
            return True
        
        # System responsiveness check
        async def check_responsiveness() -> bool:
            # Simplified check
            return True
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="memory_usage",
                check_function=check_memory,
                interval_seconds=30.0,
                timeout_seconds=5.0
            )
        )
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="system_responsiveness",
                check_function=check_responsiveness,
                interval_seconds=60.0,
                timeout_seconds=10.0
            )
        )
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            "high_inference_latency",
            "inference_latency",
            "gt",
            500.0,  # 500ms
            AlertSeverity.WARNING
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "memory_usage_percent",
            "gt",
            85.0,  # 85%
            AlertSeverity.ERROR
        )
        
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            "error_rate",
            "gt",
            5.0,  # 5%
            AlertSeverity.CRITICAL
        )
    
    def _default_notification_handler(self, alert: Alert) -> None:
        """Default alert notification handler."""
        log_level = logging.WARNING
        if alert.severity == AlertSeverity.ERROR:
            log_level = logging.ERROR
        elif alert.severity == AlertSeverity.CRITICAL:
            log_level = logging.CRITICAL
        
        logger.log(log_level, f"ALERT: {alert.title} - {alert.description}")
    
    async def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append(f"# HELP wasm_torch_metrics_processed_total Total metrics processed")
        lines.append(f"# TYPE wasm_torch_metrics_processed_total counter")
        lines.append(f"wasm_torch_metrics_processed_total {self.system_metrics['metrics_processed']}")
        
        lines.append(f"# HELP wasm_torch_anomalies_detected_total Total anomalies detected")
        lines.append(f"# TYPE wasm_torch_anomalies_detected_total counter")
        lines.append(f"wasm_torch_anomalies_detected_total {self.system_metrics['anomalies_detected']}")
        
        return "\n".join(lines)
    
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': time.time(),
            'system_metrics': self.system_metrics,
            'health_status': self.health_monitor.get_overall_health(),
            'alert_summary': self.alert_manager.get_alert_summary()
        }
        return json.dumps(export_data, indent=2)
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring system."""
        logger.info("Shutting down Advanced Monitoring System")
        await self.metric_collector.shutdown()
        await self.health_monitor.shutdown()
        logger.info("Advanced Monitoring System shutdown complete")
