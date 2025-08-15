"""Autonomous Scaling System with AI-powered resource management for WASM-Torch."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import math
import random

# Optional dependencies - gracefully handle missing imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil functions
    class MockPsutil:
        @staticmethod
        def cpu_count():
            return 4  # Default to 4 cores
        
        @staticmethod
        def cpu_percent(interval=None):
            return random.uniform(20, 80)  # Mock CPU usage
        
        class VirtualMemory:
            def __init__(self):
                self.total = 8 * 1024**3  # 8GB
                self.available = 4 * 1024**3  # 4GB available
                self.percent = random.uniform(40, 80)
        
        @staticmethod
        def virtual_memory():
            return MockPsutil.VirtualMemory()
        
        class DiskIO:
            def __init__(self):
                self.read_bytes = random.randint(1000000, 10000000)
                self.write_bytes = random.randint(1000000, 10000000)
        
        @staticmethod
        def disk_io_counters():
            return MockPsutil.DiskIO()
        
        class NetIO:
            def __init__(self):
                self.bytes_sent = random.randint(1000000, 10000000)
                self.bytes_recv = random.randint(1000000, 10000000)
        
        @staticmethod
        def net_io_counters():
            return MockPsutil.NetIO()
    
    psutil = MockPsutil()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy functions
    class MockNumpy:
        @staticmethod
        def random():
            return random
        
        @staticmethod
        def normal(mean=0, std=1, size=None):
            if size is None:
                return random.gauss(mean, std)
            return [random.gauss(mean, std) for _ in range(size)]
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data)
        
        @staticmethod
        def array(data):
            return data
    
    np = MockNumpy()


logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_CORES = "cpu_cores"
    MEMORY = "memory"
    THREAD_POOL = "thread_pool"
    CACHE_SIZE = "cache_size"
    BATCH_SIZE = "batch_size"
    INFERENCE_WORKERS = "inference_workers"
    PREPROCESSING_WORKERS = "preprocessing_workers"
    IO_WORKERS = "io_workers"


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Different scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    ML_POWERED = "ml_powered"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io_rate: float
    network_io_rate: float
    active_requests: int
    queue_length: int
    response_time: float
    error_rate: float
    throughput: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    cooldown_period: float
    min_value: Any
    max_value: Any
    step_size: Any
    enabled: bool = True
    priority: int = 5


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: Any
    new_value: Any
    trigger_metric: str
    trigger_value: float
    success: bool
    execution_time: float
    reason: str


class PredictiveModel:
    """Simple predictive model for resource scaling."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.predictions: Dict[str, float] = {}
        self.model_accuracy: Dict[str, float] = {}
    
    def add_measurement(self, metric_name: str, value: float, timestamp: float) -> None:
        """Add a new measurement to the model."""
        self.metrics_history[metric_name].append((timestamp, value))
    
    def predict_next_value(self, metric_name: str, time_horizon: float = 60.0) -> Optional[float]:
        """Predict the next value for a metric using simple time series analysis."""
        if metric_name not in self.metrics_history:
            return None
        
        history = list(self.metrics_history[metric_name])
        if len(history) < 5:  # Need minimum data points
            return None
        
        # Extract timestamps and values
        timestamps = np.array([point[0] for point in history])
        values = np.array([point[1] for point in history])
        
        # Simple linear trend analysis
        if len(timestamps) > 1:
            try:
                # Calculate trend using least squares
                x = timestamps - timestamps[0]  # Normalize timestamps
                coeffs = np.polyfit(x, values, 1)  # Linear fit
                
                # Predict future value
                future_x = (timestamps[-1] + time_horizon) - timestamps[0]
                predicted_value = np.polyval(coeffs, future_x)
                
                # Apply bounds checking (prevent unrealistic predictions)
                current_value = values[-1]
                max_change = abs(current_value) * 2.0  # Maximum 200% change
                predicted_value = max(
                    current_value - max_change,
                    min(current_value + max_change, predicted_value)
                )
                
                self.predictions[metric_name] = predicted_value
                
                # Simple accuracy tracking
                if len(history) > 10:
                    # Compare previous predictions with actual values
                    recent_errors = []
                    for i in range(-10, -1):
                        if i + 1 < 0:
                            predicted = values[i] + coeffs[0] * (timestamps[i+1] - timestamps[i])
                            actual = values[i+1]
                            error = abs(predicted - actual) / max(abs(actual), 1e-6)
                            recent_errors.append(error)
                    
                    if recent_errors:
                        self.model_accuracy[metric_name] = 1.0 - np.mean(recent_errors)
                
                return predicted_value
                
            except Exception as e:
                logger.warning(f"Error predicting {metric_name}: {e}")
                return None
        
        return None
    
    def get_trend_direction(self, metric_name: str) -> Optional[str]:
        """Get trend direction for a metric."""
        if metric_name not in self.metrics_history:
            return None
        
        history = list(self.metrics_history[metric_name])
        if len(history) < 3:
            return None
        
        recent_values = [point[1] for point in history[-3:]]
        
        if recent_values[2] > recent_values[1] > recent_values[0]:
            return "increasing"
        elif recent_values[2] < recent_values[1] < recent_values[0]:
            return "decreasing"
        else:
            return "stable"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get predictive model statistics."""
        return {
            "tracked_metrics": list(self.metrics_history.keys()),
            "data_points": {
                metric: len(history) 
                for metric, history in self.metrics_history.items()
            },
            "model_accuracy": self.model_accuracy.copy(),
            "recent_predictions": self.predictions.copy()
        }


class ResourceController:
    """Controls and manages system resources."""
    
    def __init__(self):
        self.current_resources: Dict[ResourceType, Any] = {}
        self.resource_limits: Dict[ResourceType, Tuple[Any, Any]] = {}
        self.resource_controllers: Dict[ResourceType, Callable] = {}
        self.lock = threading.RLock()
        self._initialize_default_resources()
    
    def _initialize_default_resources(self) -> None:
        """Initialize default resource configurations."""
        try:
            # Get system information
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            
            # Set initial resources
            self.current_resources = {
                ResourceType.CPU_CORES: min(4, cpu_count),
                ResourceType.MEMORY: min(1024, memory_info.total // (1024**2) // 2),  # Half of available memory in MB
                ResourceType.THREAD_POOL: 4,
                ResourceType.CACHE_SIZE: 256,  # MB
                ResourceType.BATCH_SIZE: 16,
                ResourceType.INFERENCE_WORKERS: 2,
                ResourceType.PREPROCESSING_WORKERS: 2,
                ResourceType.IO_WORKERS: 2,
            }
            
            # Set resource limits
            self.resource_limits = {
                ResourceType.CPU_CORES: (1, cpu_count),
                ResourceType.MEMORY: (128, memory_info.total // (1024**2) * 0.8),  # 80% of total memory
                ResourceType.THREAD_POOL: (2, 32),
                ResourceType.CACHE_SIZE: (64, 2048),  # 64MB to 2GB
                ResourceType.BATCH_SIZE: (1, 128),
                ResourceType.INFERENCE_WORKERS: (1, 16),
                ResourceType.PREPROCESSING_WORKERS: (1, 8),
                ResourceType.IO_WORKERS: (1, 8),
            }
            
        except Exception as e:
            logger.error(f"Error initializing default resources: {e}")
            # Fallback to conservative defaults
            self.current_resources = {rt: 2 for rt in ResourceType}
            self.resource_limits = {rt: (1, 8) for rt in ResourceType}
    
    def register_resource_controller(self, resource_type: ResourceType, controller: Callable) -> None:
        """Register a controller function for a resource type."""
        self.resource_controllers[resource_type] = controller
        logger.info(f"Registered controller for {resource_type.value}")
    
    async def scale_resource(
        self, 
        resource_type: ResourceType, 
        target_value: Any,
        reason: str = ""
    ) -> bool:
        """Scale a resource to target value."""
        with self.lock:
            current_value = self.current_resources.get(resource_type)
            if current_value is None:
                logger.error(f"Resource type {resource_type.value} not found")
                return False
            
            # Check limits
            min_val, max_val = self.resource_limits.get(resource_type, (None, None))
            if min_val is not None and target_value < min_val:
                logger.warning(f"Target value {target_value} below minimum {min_val} for {resource_type.value}")
                target_value = min_val
            if max_val is not None and target_value > max_val:
                logger.warning(f"Target value {target_value} above maximum {max_val} for {resource_type.value}")
                target_value = max_val
            
            if target_value == current_value:
                return True  # No change needed
            
            # Apply resource scaling
            try:
                controller = self.resource_controllers.get(resource_type)
                if controller:
                    success = await controller(current_value, target_value, reason)
                    if success:
                        self.current_resources[resource_type] = target_value
                        logger.info(f"Scaled {resource_type.value} from {current_value} to {target_value}. Reason: {reason}")
                        return True
                    else:
                        logger.error(f"Failed to scale {resource_type.value}")
                        return False
                else:
                    # Default scaling (just update the value)
                    self.current_resources[resource_type] = target_value
                    logger.info(f"Scaled {resource_type.value} from {current_value} to {target_value} (no controller)")
                    return True
                    
            except Exception as e:
                logger.error(f"Error scaling {resource_type.value}: {e}")
                return False
    
    def get_current_resource(self, resource_type: ResourceType) -> Any:
        """Get current value of a resource."""
        return self.current_resources.get(resource_type)
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get utilization percentage for each resource."""
        utilization = {}
        
        for resource_type, current_value in self.current_resources.items():
            min_val, max_val = self.resource_limits.get(resource_type, (0, current_value))
            if max_val > min_val:
                util_pct = (current_value - min_val) / (max_val - min_val)
                utilization[resource_type] = min(1.0, max(0.0, util_pct))
            else:
                utilization[resource_type] = 0.0
        
        return utilization


class AutonomousScalingSystem:
    """Main autonomous scaling system with AI-powered decision making."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.resource_controller = ResourceController()
        self.predictive_model = PredictiveModel()
        self.scaling_rules: Dict[ResourceType, List[ScalingRule]] = defaultdict(list)
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_events: List[ScalingEvent] = []
        self.cooldown_timers: Dict[ResourceType, float] = {}
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.lock = threading.RLock()
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.scaling_effectiveness: Dict[ResourceType, float] = {}
        
        self._initialize_default_rules()
        logger.info(f"Autonomous Scaling System initialized with strategy: {strategy.value}")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default scaling rules."""
        # CPU scaling rules
        self.add_scaling_rule(ScalingRule(
            resource_type=ResourceType.CPU_CORES,
            metric_name="cpu_usage",
            threshold_up=0.8,
            threshold_down=0.3,
            cooldown_period=60.0,
            min_value=1,
            max_value=psutil.cpu_count(),
            step_size=1
        ))
        
        # Memory scaling rules
        self.add_scaling_rule(ScalingRule(
            resource_type=ResourceType.MEMORY,
            metric_name="memory_usage",
            threshold_up=0.85,
            threshold_down=0.4,
            cooldown_period=30.0,
            min_value=128,
            max_value=psutil.virtual_memory().total // (1024**2) * 0.8,
            step_size=128
        ))
        
        # Thread pool scaling
        self.add_scaling_rule(ScalingRule(
            resource_type=ResourceType.THREAD_POOL,
            metric_name="queue_length",
            threshold_up=10,
            threshold_down=2,
            cooldown_period=30.0,
            min_value=2,
            max_value=32,
            step_size=2
        ))
        
        # Batch size optimization
        self.add_scaling_rule(ScalingRule(
            resource_type=ResourceType.BATCH_SIZE,
            metric_name="response_time",
            threshold_up=100,  # ms
            threshold_down=20,   # ms
            cooldown_period=45.0,
            min_value=1,
            max_value=128,
            step_size=2
        ))
        
        # Worker scaling
        self.add_scaling_rule(ScalingRule(
            resource_type=ResourceType.INFERENCE_WORKERS,
            metric_name="active_requests",
            threshold_up=8,
            threshold_down=2,
            cooldown_period=30.0,
            min_value=1,
            max_value=16,
            step_size=1
        ))
    
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a custom scaling rule."""
        with self.lock:
            self.scaling_rules[rule.resource_type].append(rule)
        logger.info(f"Added scaling rule for {rule.resource_type.value}")
    
    def remove_scaling_rule(self, resource_type: ResourceType, metric_name: str) -> bool:
        """Remove a scaling rule."""
        with self.lock:
            rules = self.scaling_rules[resource_type]
            initial_count = len(rules)
            self.scaling_rules[resource_type] = [
                rule for rule in rules if rule.metric_name != metric_name
            ]
            removed = len(rules) - len(self.scaling_rules[resource_type])
            
        if removed > 0:
            logger.info(f"Removed {removed} scaling rule(s) for {resource_type.value}.{metric_name}")
            return True
        return False
    
    async def start_autonomous_scaling(self) -> None:
        """Start the autonomous scaling system."""
        if self.is_running:
            logger.warning("Autonomous scaling is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Autonomous scaling system started")
    
    async def stop_autonomous_scaling(self) -> None:
        """Stop the autonomous scaling system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous scaling system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = await self._collect_system_metrics()
                
                # Store metrics for analysis
                with self.lock:
                    self.metrics_history.append(current_metrics)
                
                # Update predictive model
                self._update_predictive_model(current_metrics)
                
                # Make scaling decisions
                await self._make_scaling_decisions(current_metrics)
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Longer sleep on error
    
    async def _collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Get application-specific metrics (would be implemented by the application)
            active_requests = await self._get_active_request_count()
            queue_length = await self._get_queue_length()
            response_time = await self._get_average_response_time()
            error_rate = await self._get_error_rate()
            throughput = await self._get_throughput()
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_percent / 100.0,
                memory_usage=memory.percent / 100.0,
                memory_available=memory.available,
                disk_io_rate=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
                network_io_rate=network_io.bytes_sent + network_io.bytes_recv if network_io else 0,
                active_requests=active_requests,
                queue_length=queue_length,
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=0.5,
                memory_usage=0.5,
                memory_available=1024*1024*1024,  # 1GB default
                disk_io_rate=0,
                network_io_rate=0,
                active_requests=0,
                queue_length=0,
                response_time=50.0,
                error_rate=0.0,
                throughput=10.0
            )
    
    def _update_predictive_model(self, metrics: ResourceMetrics) -> None:
        """Update the predictive model with new metrics."""
        # Add measurements to predictive model
        self.predictive_model.add_measurement("cpu_usage", metrics.cpu_usage, metrics.timestamp)
        self.predictive_model.add_measurement("memory_usage", metrics.memory_usage, metrics.timestamp)
        self.predictive_model.add_measurement("queue_length", float(metrics.queue_length), metrics.timestamp)
        self.predictive_model.add_measurement("response_time", metrics.response_time, metrics.timestamp)
        self.predictive_model.add_measurement("active_requests", float(metrics.active_requests), metrics.timestamp)
        self.predictive_model.add_measurement("throughput", metrics.throughput, metrics.timestamp)
        self.predictive_model.add_measurement("error_rate", metrics.error_rate, metrics.timestamp)
    
    async def _make_scaling_decisions(self, current_metrics: ResourceMetrics) -> None:
        """Make scaling decisions based on current metrics and strategy."""
        scaling_decisions = []
        
        # Evaluate each resource type
        for resource_type in ResourceType:
            if resource_type not in self.scaling_rules:
                continue
            
            # Check cooldown period
            last_scaled = self.cooldown_timers.get(resource_type, 0)
            if time.time() - last_scaled < 30:  # Minimum 30 second cooldown
                continue
            
            # Evaluate scaling rules for this resource
            for rule in self.scaling_rules[resource_type]:
                if not rule.enabled:
                    continue
                
                # Get current metric value
                metric_value = getattr(current_metrics, rule.metric_name, None)
                if metric_value is None:
                    continue
                
                # Determine scaling direction
                direction = None
                if metric_value > rule.threshold_up:
                    direction = ScalingDirection.UP
                elif metric_value < rule.threshold_down:
                    direction = ScalingDirection.DOWN
                
                if direction:
                    # Calculate new resource value
                    current_value = self.resource_controller.get_current_resource(resource_type)
                    if current_value is None:
                        continue
                    
                    if direction == ScalingDirection.UP:
                        new_value = current_value + rule.step_size
                        # Apply max limit
                        if rule.max_value is not None:
                            new_value = min(new_value, rule.max_value)
                    else:
                        new_value = current_value - rule.step_size
                        # Apply min limit
                        if rule.min_value is not None:
                            new_value = max(new_value, rule.min_value)
                    
                    if new_value != current_value:
                        scaling_decisions.append((
                            resource_type, direction, current_value, new_value,
                            rule.metric_name, metric_value, f"Rule-based scaling: {rule.metric_name} = {metric_value}"
                        ))
        
        # Apply predictive scaling if enabled
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID, ScalingStrategy.ML_POWERED]:
            predictive_decisions = await self._make_predictive_decisions(current_metrics)
            scaling_decisions.extend(predictive_decisions)
        
        # Execute scaling decisions
        for decision in scaling_decisions:
            resource_type, direction, old_value, new_value, trigger_metric, trigger_value, reason = decision
            await self._execute_scaling_decision(
                resource_type, direction, old_value, new_value, trigger_metric, trigger_value, reason
            )
    
    async def _make_predictive_decisions(self, current_metrics: ResourceMetrics) -> List[Tuple]:
        """Make predictive scaling decisions."""
        decisions = []
        
        # Predict future metrics
        predictions = {}
        for metric_name in ["cpu_usage", "memory_usage", "queue_length", "response_time"]:
            predicted = self.predictive_model.predict_next_value(metric_name, 60.0)  # 1 minute ahead
            if predicted is not None:
                predictions[metric_name] = predicted
        
        # Make proactive scaling decisions based on predictions
        for metric_name, predicted_value in predictions.items():
            # Find relevant scaling rules
            for resource_type in ResourceType:
                for rule in self.scaling_rules[resource_type]:
                    if rule.metric_name == metric_name and rule.enabled:
                        # Check if predicted value will trigger scaling
                        if predicted_value > rule.threshold_up * 1.1:  # 10% buffer for predictions
                            current_value = self.resource_controller.get_current_resource(resource_type)
                            if current_value is not None:
                                new_value = current_value + rule.step_size
                                if rule.max_value is None or new_value <= rule.max_value:
                                    decisions.append((
                                        resource_type, ScalingDirection.UP, current_value, new_value,
                                        metric_name, predicted_value,
                                        f"Predictive scaling: {metric_name} predicted to reach {predicted_value:.2f}"
                                    ))
        
        return decisions
    
    async def _execute_scaling_decision(
        self,
        resource_type: ResourceType,
        direction: ScalingDirection,
        old_value: Any,
        new_value: Any,
        trigger_metric: str,
        trigger_value: float,
        reason: str
    ) -> None:
        """Execute a scaling decision."""
        start_time = time.time()
        
        try:
            # Execute the scaling
            success = await self.resource_controller.scale_resource(resource_type, new_value, reason)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=start_time,
                resource_type=resource_type,
                direction=direction,
                old_value=old_value,
                new_value=new_value if success else old_value,
                trigger_metric=trigger_metric,
                trigger_value=trigger_value,
                success=success,
                execution_time=time.time() - start_time,
                reason=reason
            )
            
            with self.lock:
                self.scaling_events.append(event)
                self.cooldown_timers[resource_type] = time.time()
            
            if success:
                logger.info(f"Scaling successful: {resource_type.value} {old_value} -> {new_value}")
            else:
                logger.error(f"Scaling failed: {resource_type.value} {old_value} -> {new_value}")
                
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    # Mock methods for application-specific metrics (to be implemented by the application)
    
    async def _get_active_request_count(self) -> int:
        """Get current number of active requests."""
        # This would be implemented by the application
        return max(0, int(np.random.normal(5, 2)))
    
    async def _get_queue_length(self) -> int:
        """Get current queue length."""
        # This would be implemented by the application
        return max(0, int(np.random.normal(3, 1)))
    
    async def _get_average_response_time(self) -> float:
        """Get average response time in milliseconds."""
        # This would be implemented by the application
        return max(10, np.random.normal(50, 15))
    
    async def _get_error_rate(self) -> float:
        """Get current error rate (0.0 to 1.0)."""
        # This would be implemented by the application
        return max(0, min(1, np.random.normal(0.02, 0.01)))
    
    async def _get_throughput(self) -> float:
        """Get current throughput (requests per second)."""
        # This would be implemented by the application
        return max(1, np.random.normal(20, 5))
    
    def register_metric_collector(self, metric_name: str, collector: Callable) -> None:
        """Register a custom metric collector function."""
        # This would store custom metric collectors
        logger.info(f"Registered metric collector for: {metric_name}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        with self.lock:
            total_events = len(self.scaling_events)
            successful_events = sum(1 for event in self.scaling_events if event.success)
            
            # Calculate scaling frequency by resource type
            scaling_frequency = defaultdict(int)
            for event in self.scaling_events[-100:]:  # Last 100 events
                scaling_frequency[event.resource_type.value] += 1
            
            # Calculate average response time for scaling
            scaling_times = [event.execution_time for event in self.scaling_events if event.success]
            avg_scaling_time = np.mean(scaling_times) if scaling_times else 0
            
            # Get current resource utilization
            resource_utilization = self.resource_controller.get_resource_utilization()
            
            # Get predictive model statistics
            model_stats = self.predictive_model.get_model_statistics()
            
            return {
                "strategy": self.strategy.value,
                "total_scaling_events": total_events,
                "successful_scaling_events": successful_events,
                "success_rate": successful_events / max(total_events, 1),
                "average_scaling_time": avg_scaling_time,
                "scaling_frequency": dict(scaling_frequency),
                "resource_utilization": {rt.value: util for rt, util in resource_utilization.items()},
                "current_resources": {
                    rt.value: self.resource_controller.get_current_resource(rt)
                    for rt in ResourceType
                },
                "predictive_model": model_stats,
                "active_rules": sum(len(rules) for rules in self.scaling_rules.values()),
                "is_running": self.is_running
            }
    
    def export_scaling_report(self, file_path: str) -> None:
        """Export detailed scaling report."""
        report_data = {
            "timestamp": time.time(),
            "statistics": self.get_scaling_statistics(),
            "scaling_events": [
                {
                    "timestamp": event.timestamp,
                    "resource_type": event.resource_type.value,
                    "direction": event.direction.value,
                    "old_value": event.old_value,
                    "new_value": event.new_value,
                    "trigger_metric": event.trigger_metric,
                    "trigger_value": event.trigger_value,
                    "success": event.success,
                    "execution_time": event.execution_time,
                    "reason": event.reason
                }
                for event in self.scaling_events
            ],
            "scaling_rules": [
                {
                    "resource_type": rule.resource_type.value,
                    "metric_name": rule.metric_name,
                    "threshold_up": rule.threshold_up,
                    "threshold_down": rule.threshold_down,
                    "cooldown_period": rule.cooldown_period,
                    "min_value": rule.min_value,
                    "max_value": rule.max_value,
                    "step_size": rule.step_size,
                    "enabled": rule.enabled,
                    "priority": rule.priority
                }
                for rules in self.scaling_rules.values()
                for rule in rules
            ],
            "metrics_history": [
                {
                    "timestamp": metrics.timestamp,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "active_requests": metrics.active_requests,
                    "queue_length": metrics.queue_length,
                    "response_time": metrics.response_time,
                    "error_rate": metrics.error_rate,
                    "throughput": metrics.throughput
                }
                for metrics in list(self.metrics_history)[-100:]  # Last 100 measurements
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Scaling report exported to {file_path}")


# Global scaling system instance
_global_scaling_system = None


def get_global_scaling_system() -> AutonomousScalingSystem:
    """Get global autonomous scaling system instance."""
    global _global_scaling_system
    if _global_scaling_system is None:
        _global_scaling_system = AutonomousScalingSystem()
    return _global_scaling_system


# Scaling decorators
def auto_scale_resources(resources: List[ResourceType] = None):
    """Decorator for automatic resource scaling based on function performance."""
    if resources is None:
        resources = [ResourceType.THREAD_POOL, ResourceType.BATCH_SIZE]
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            scaling_system = get_global_scaling_system()
            
            # Start monitoring if not already running
            if not scaling_system.is_running:
                await scaling_system.start_autonomous_scaling()
            
            # Execute function with performance monitoring
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                
                # Trigger optimization if execution is slow
                if execution_time > 1.0:  # More than 1 second
                    for resource_type in resources:
                        current_value = scaling_system.resource_controller.get_current_resource(resource_type)
                        if current_value is not None and resource_type == ResourceType.THREAD_POOL:
                            # Suggest scaling up thread pool for slow operations
                            new_value = min(current_value + 2, 16)
                            await scaling_system.resource_controller.scale_resource(
                                resource_type, new_value, f"Slow execution detected: {execution_time:.2f}s"
                            )
                
                return result
                
            except Exception as e:
                # Record error and potentially scale resources
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    
    return decorator


# Resource controller implementations (to be registered with the ResourceController)

async def cpu_cores_controller(current_value: int, target_value: int, reason: str) -> bool:
    """Controller for CPU core allocation."""
    # This would implement actual CPU affinity or process scaling
    logger.info(f"CPU cores scaling: {current_value} -> {target_value}. Reason: {reason}")
    return True  # Mock success


async def memory_controller(current_value: int, target_value: int, reason: str) -> bool:
    """Controller for memory allocation."""
    # This would implement memory limit adjustment
    logger.info(f"Memory scaling: {current_value}MB -> {target_value}MB. Reason: {reason}")
    return True  # Mock success


async def thread_pool_controller(current_value: int, target_value: int, reason: str) -> bool:
    """Controller for thread pool size."""
    # This would implement actual thread pool scaling
    logger.info(f"Thread pool scaling: {current_value} -> {target_value}. Reason: {reason}")
    return True  # Mock success


def register_default_controllers(scaling_system: AutonomousScalingSystem) -> None:
    """Register default resource controllers."""
    scaling_system.resource_controller.register_resource_controller(
        ResourceType.CPU_CORES, cpu_cores_controller
    )
    scaling_system.resource_controller.register_resource_controller(
        ResourceType.MEMORY, memory_controller
    )
    scaling_system.resource_controller.register_resource_controller(
        ResourceType.THREAD_POOL, thread_pool_controller
    )