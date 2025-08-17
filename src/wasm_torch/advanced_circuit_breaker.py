"""Advanced Circuit Breaker System for WASM-Torch

Intelligent circuit breaker with adaptive thresholds, predictive failure detection,
and autonomous recovery mechanisms for maximum system reliability.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics
from collections import deque, defaultdict
import json
import numpy as np

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, rejecting requests
    HALF_OPEN = "half_open"    # Testing recovery
    LEARNING = "learning"      # Adaptive learning mode

class FailureType(Enum):
    """Types of failures the circuit breaker can detect"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    SLOW_RESPONSE = "slow_response"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class FailureMetrics:
    """Metrics for tracking failures"""
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    last_failure_time: Optional[float] = None
    failure_types: Dict[str, int] = field(default_factory=dict)
    
    def calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout_duration: float = 30.0
    slow_response_threshold: float = 5.0
    error_rate_threshold: float = 0.5
    monitoring_window: float = 300.0  # 5 minutes
    adaptive_thresholds: bool = True
    predictive_mode: bool = True

class AdaptiveThresholdCalculator:
    """Calculates adaptive thresholds based on historical data"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self.load_levels: deque = deque(maxlen=window_size)
        
    def update_metrics(self, response_time: float, error_rate: float, load_level: float):
        """Update historical metrics"""
        self.response_times.append(response_time)
        self.error_rates.append(error_rate)
        self.load_levels.append(load_level)
    
    def calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout threshold"""
        if len(self.response_times) < 10:
            return 5.0  # Default
        
        # Use 95th percentile + buffer
        p95 = np.percentile(list(self.response_times), 95)
        buffer = np.std(list(self.response_times)) * 2
        return max(1.0, p95 + buffer)
    
    def calculate_adaptive_error_threshold(self) -> float:
        """Calculate adaptive error rate threshold"""
        if len(self.error_rates) < 10:
            return 0.5  # Default
        
        # Use mean + 3 standard deviations
        mean_error_rate = np.mean(list(self.error_rates))
        std_error_rate = np.std(list(self.error_rates))
        return min(0.8, max(0.1, mean_error_rate + 3 * std_error_rate))
    
    def predict_failure_probability(self) -> float:
        """Predict probability of system failure based on trends"""
        if len(self.response_times) < 20:
            return 0.0
        
        # Analyze trends in recent data
        recent_response_times = list(self.response_times)[-20:]
        recent_error_rates = list(self.error_rates)[-20:]
        
        # Calculate trend slopes
        x = np.arange(len(recent_response_times))
        response_time_slope = np.polyfit(x, recent_response_times, 1)[0]
        error_rate_slope = np.polyfit(x, recent_error_rates, 1)[0]
        
        # Simple failure probability based on trends
        failure_prob = 0.0
        
        # Increasing response times
        if response_time_slope > 0.1:
            failure_prob += 0.3
        
        # Increasing error rates
        if error_rate_slope > 0.01:
            failure_prob += 0.4
        
        # High current values
        if recent_response_times[-1] > self.calculate_adaptive_timeout():
            failure_prob += 0.2
        
        if recent_error_rates[-1] > self.calculate_adaptive_error_threshold():
            failure_prob += 0.3
        
        return min(1.0, failure_prob)

class PredictiveFailureDetector:
    """Detects potential failures before they occur"""
    
    def __init__(self):
        self.anomaly_threshold = 2.0  # Standard deviations
        self.pattern_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def detect_anomalies(self, 
                        response_time: float, 
                        error_rate: float, 
                        memory_usage: float) -> List[str]:
        """Detect anomalies in system metrics"""
        anomalies = []
        
        # Check response time anomaly
        if self._is_anomaly("response_time", response_time):
            anomalies.append("response_time_anomaly")
        
        # Check error rate anomaly
        if self._is_anomaly("error_rate", error_rate):
            anomalies.append("error_rate_anomaly")
        
        # Check memory usage anomaly
        if self._is_anomaly("memory_usage", memory_usage):
            anomalies.append("memory_usage_anomaly")
        
        # Update history
        self.pattern_history["response_time"].append(response_time)
        self.pattern_history["error_rate"].append(error_rate)
        self.pattern_history["memory_usage"].append(memory_usage)
        
        return anomalies
    
    def _is_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if value is anomalous for given metric"""
        history = self.pattern_history[metric_name]
        
        if len(history) < 10:
            return False
        
        mean_val = statistics.mean(history)
        std_val = statistics.stdev(history) if len(history) > 1 else 0
        
        if std_val == 0:
            return False
        
        z_score = abs(value - mean_val) / std_val
        return z_score > self.anomaly_threshold

class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive and predictive capabilities"""
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = FailureMetrics()
        
        # Adaptive components
        self.threshold_calculator = AdaptiveThresholdCalculator()
        self.failure_detector = PredictiveFailureDetector()
        
        # State management
        self.state_change_time = time.time()
        self.half_open_success_count = 0
        self.recent_requests: deque = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.RLock()
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
    
    async def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if request should be allowed
            if not self._should_allow_request():
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        request_start = time.time()
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self._get_current_timeout()
            )
            
            response_time = time.time() - request_start
            
            # Record success
            await self._record_success(response_time)
            
            return result
            
        except asyncio.TimeoutError:
            response_time = time.time() - request_start
            await self._record_failure(FailureType.TIMEOUT, response_time)
            raise CircuitBreakerTimeoutError(f"Request timed out after {response_time:.2f}s")
            
        except Exception as e:
            response_time = time.time() - request_start
            await self._record_failure(FailureType.EXCEPTION, response_time, str(e))
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with proper async handling"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on current state"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.state_change_time >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        elif self.state == CircuitState.LEARNING:
            return True
        
        return False
    
    def _get_current_timeout(self) -> float:
        """Get current timeout threshold (adaptive if enabled)"""
        if self.config.adaptive_thresholds:
            return self.threshold_calculator.calculate_adaptive_timeout()
        return self.config.timeout_duration
    
    async def _record_success(self, response_time: float):
        """Record successful request"""
        with self._lock:
            self.metrics.success_count += 1
            self.metrics.total_requests += 1
            
            # Update running average
            total_responses = self.metrics.success_count + self.metrics.failure_count
            if total_responses > 0:
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * (total_responses - 1) + response_time) 
                    / total_responses
                )
            
            # Update adaptive thresholds
            if self.config.adaptive_thresholds:
                current_error_rate = self.metrics.calculate_error_rate()
                load_level = len(self.recent_requests) / 1000.0  # Simplified load measure
                self.threshold_calculator.update_metrics(response_time, current_error_rate, load_level)
            
            # Record request timing
            self.recent_requests.append({
                "timestamp": time.time(),
                "response_time": response_time,
                "success": True
            })
            
            # Check for state transitions
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_success_count += 1
                if self.half_open_success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            # Check for slow responses
            if response_time > self.config.slow_response_threshold:
                await self._record_failure(FailureType.SLOW_RESPONSE, response_time)
    
    async def _record_failure(self, 
                             failure_type: FailureType, 
                             response_time: float,
                             error_details: str = ""):
        """Record failed request"""
        with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_requests += 1
            self.metrics.last_failure_time = time.time()
            
            # Update failure type counts
            if failure_type.value in self.metrics.failure_types:
                self.metrics.failure_types[failure_type.value] += 1
            else:
                self.metrics.failure_types[failure_type.value] = 1
            
            # Record request
            self.recent_requests.append({
                "timestamp": time.time(),
                "response_time": response_time,
                "success": False,
                "failure_type": failure_type.value,
                "error_details": error_details
            })
            
            # Check if circuit should open
            if self._should_trip_circuit():
                self._transition_to_open()
            
            self.logger.warning(f"Failure recorded: {failure_type.value}, response_time={response_time:.2f}s")
    
    def _should_trip_circuit(self) -> bool:
        """Determine if circuit should trip to OPEN state"""
        if self.state == CircuitState.OPEN:
            return False
        
        # Check failure count threshold
        if self.metrics.failure_count >= self.config.failure_threshold:
            return True
        
        # Check error rate threshold
        error_rate = self.metrics.calculate_error_rate()
        threshold = (self.threshold_calculator.calculate_adaptive_error_threshold() 
                   if self.config.adaptive_thresholds 
                   else self.config.error_rate_threshold)
        
        if error_rate >= threshold and self.metrics.total_requests >= 10:
            return True
        
        # Predictive failure detection
        if self.config.predictive_mode:
            failure_probability = self.threshold_calculator.predict_failure_probability()
            if failure_probability > 0.8:  # 80% failure probability
                self.logger.warning(f"Predictive failure detected: {failure_probability:.2f} probability")
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.logger.error(f"Circuit breaker {self.name} transitioned to OPEN state")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.half_open_success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN state")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        # Reset failure metrics
        self.metrics.failure_count = 0
        self.metrics.failure_types.clear()
        self.logger.info(f"Circuit breaker {self.name} transitioned to CLOSED state")
    
    async def start_monitoring(self):
        """Start background monitoring for predictive failure detection"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"Started monitoring for circuit breaker {self.name}")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info(f"Stopped monitoring for circuit breaker {self.name}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Perform health checks
                await self._perform_health_check()
                
                # Clean old metrics
                self._clean_old_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        current_time = time.time()
        
        # Analyze recent requests for patterns
        recent_window = [
            req for req in self.recent_requests
            if current_time - req["timestamp"] <= 60  # Last minute
        ]
        
        if len(recent_window) >= 10:
            error_rate = sum(1 for req in recent_window if not req["success"]) / len(recent_window)
            avg_response_time = statistics.mean([req["response_time"] for req in recent_window])
            memory_usage = 0.5  # Placeholder - would get actual memory usage
            
            # Detect anomalies
            anomalies = self.failure_detector.detect_anomalies(
                avg_response_time, error_rate, memory_usage
            )
            
            if anomalies and self.state == CircuitState.CLOSED:
                self.logger.warning(f"Anomalies detected: {anomalies}")
                # Could preemptively transition to learning mode
    
    def _clean_old_metrics(self):
        """Clean old metrics outside monitoring window"""
        current_time = time.time()
        cutoff_time = current_time - self.config.monitoring_window
        
        # Remove old requests
        while self.recent_requests and self.recent_requests[0]["timestamp"] < cutoff_time:
            self.recent_requests.popleft()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status"""
        with self._lock:
            current_time = time.time()
            
            # Calculate recent metrics
            recent_requests = [
                req for req in self.recent_requests
                if current_time - req["timestamp"] <= 300  # Last 5 minutes
            ]
            
            recent_error_rate = 0.0
            recent_avg_response_time = 0.0
            
            if recent_requests:
                recent_failures = sum(1 for req in recent_requests if not req["success"])
                recent_error_rate = recent_failures / len(recent_requests)
                recent_avg_response_time = statistics.mean([req["response_time"] for req in recent_requests])
            
            return {
                "name": self.name,
                "state": self.state.value,
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "success_count": self.metrics.success_count,
                    "failure_count": self.metrics.failure_count,
                    "error_rate": self.metrics.calculate_error_rate(),
                    "avg_response_time": self.metrics.avg_response_time,
                    "failure_types": self.metrics.failure_types.copy()
                },
                "recent_metrics": {
                    "requests_last_5min": len(recent_requests),
                    "error_rate_last_5min": recent_error_rate,
                    "avg_response_time_last_5min": recent_avg_response_time
                },
                "adaptive_thresholds": {
                    "timeout": self.threshold_calculator.calculate_adaptive_timeout(),
                    "error_rate": self.threshold_calculator.calculate_adaptive_error_threshold(),
                    "failure_probability": self.threshold_calculator.predict_failure_probability()
                } if self.config.adaptive_thresholds else None,
                "state_info": {
                    "time_in_current_state": current_time - self.state_change_time,
                    "half_open_success_count": self.half_open_success_count if self.state == CircuitState.HALF_OPEN else None
                },
                "configuration": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "adaptive_thresholds": self.config.adaptive_thresholds,
                    "predictive_mode": self.config.predictive_mode
                }
            }

class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self._lock = threading.RLock()
        
        # Configure logging
        self.logger = logging.getLogger("CircuitBreakerManager")
    
    def create_circuit_breaker(self, 
                              name: str, 
                              config: Optional[CircuitBreakerConfig] = None) -> AdvancedCircuitBreaker:
        """Create or get circuit breaker"""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = AdvancedCircuitBreaker(name, config)
                self.logger.info(f"Created circuit breaker: {name}")
            
            return self.circuit_breakers[name]
    
    async def start_all_monitoring(self):
        """Start monitoring for all circuit breakers"""
        for cb in self.circuit_breakers.values():
            await cb.start_monitoring()
    
    async def stop_all_monitoring(self):
        """Stop monitoring for all circuit breakers"""
        for cb in self.circuit_breakers.values():
            await cb.stop_monitoring()
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                "circuit_breakers": {
                    name: cb.get_status() 
                    for name, cb in self.circuit_breakers.items()
                },
                "summary": {
                    "total_breakers": len(self.circuit_breakers),
                    "open_breakers": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN),
                    "half_open_breakers": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN),
                    "closed_breakers": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.CLOSED)
                }
            }

# Exceptions
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when request times out"""
    pass

# Global circuit breaker manager
_global_cb_manager: Optional[CircuitBreakerManager] = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    global _global_cb_manager
    if _global_cb_manager is None:
        _global_cb_manager = CircuitBreakerManager()
    return _global_cb_manager

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection"""
    def decorator(func: Callable):
        cb_manager = get_circuit_breaker_manager()
        cb = cb_manager.create_circuit_breaker(name, config)
        
        async def async_wrapper(*args, **kwargs):
            return await cb(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create an async wrapper
            async def async_func():
                return func(*args, **kwargs)
            
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(cb(async_func))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator