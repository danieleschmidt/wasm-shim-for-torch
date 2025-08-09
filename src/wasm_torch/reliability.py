"""Reliability and resilience features for WASM-Torch production systems."""

import asyncio
import logging
import time
import threading
import json
import hashlib
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthStatus:
    """System health status information."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: float
    error_rate: float
    latency_p99: float
    details: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0


@dataclass 
class FailurePoint:
    """Tracks a specific failure point in the system."""
    component: str
    failure_type: str
    timestamp: float
    error_message: str
    recovery_action: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for fault tolerance and fail-fast behavior."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        timeout_seconds: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
        
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (self.state == CircuitBreakerState.OPEN and
                time.time() - self.last_failure_time > self.recovery_timeout)
    
    @asynccontextmanager
    async def protect(self, operation_name: str = "operation"):
        """Context manager that protects operations with circuit breaker."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {operation_name} entering HALF_OPEN state")
                else:
                    raise RuntimeError(f"Circuit breaker {operation_name} is OPEN - operation blocked")
        
        start_time = time.time()
        try:
            # Execute with timeout
            yield
            
            # Success handling
            with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                        logger.info(f"Circuit breaker {operation_name} reset to CLOSED state")
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0  # Reset failure count on success
                    
        except Exception as e:
            # Failure handling
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker {operation_name} tripped to OPEN state after {self.failure_count} failures")
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker {operation_name} failed during recovery, returning to OPEN state")
            
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "time_until_retry": max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
            }


class RetryManager:
    """Intelligent retry manager with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
    async def retry_async(
        self,
        operation: Callable,
        *args,
        retry_on: Tuple[Exception, ...] = (Exception,),
        **kwargs
    ) -> Any:
        """Retry an async operation with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except retry_on as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Operation failed after {self.max_retries} retries: {e}")
                    break
                
                delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                
                if self.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                             f"retrying in {delay:.2f}s: {e}")
                
                await asyncio.sleep(delay)
        
        raise last_exception


class HealthMonitor:
    """Comprehensive health monitoring for WASM-Torch components."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.failure_history: List[FailurePoint] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
        logger.info(f"Health check registered for component: {component}")
        
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return
            
        self._running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
        
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(min(self.check_interval, 10.0))
                
    async def run_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        
        for component, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await self._run_single_check(component, check_func)
                check_duration = time.time() - start_time
                
                # Update health status
                status = HealthStatus(
                    component=component,
                    status=result.get("status", "unknown"),
                    last_check=time.time(),
                    error_rate=result.get("error_rate", 0.0),
                    latency_p99=result.get("latency_p99", check_duration * 1000),
                    details=result.get("details", {}),
                    uptime_seconds=result.get("uptime_seconds", 0.0)
                )
                
                self.health_status[component] = status
                results[component] = status
                
                # Log unhealthy components
                if status.status in ["degraded", "unhealthy"]:
                    logger.warning(f"Component {component} status: {status.status} "
                                 f"(error_rate: {status.error_rate:.2%}, "
                                 f"latency_p99: {status.latency_p99:.1f}ms)")
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                
                # Record failure
                failure = FailurePoint(
                    component=component,
                    failure_type="health_check_failure",
                    timestamp=time.time(),
                    error_message=str(e),
                    context={"check_type": "health_monitor"}
                )
                self.failure_history.append(failure)
                
                # Mark as unhealthy
                status = HealthStatus(
                    component=component,
                    status="unhealthy",
                    last_check=time.time(),
                    error_rate=1.0,
                    latency_p99=float('inf'),
                    details={"error": str(e)}
                )
                self.health_status[component] = status
                results[component] = status
        
        return results
    
    async def _run_single_check(self, component: str, check_func: Callable) -> Dict[str, Any]:
        """Run a single health check with timeout protection."""
        try:
            if asyncio.iscoroutinefunction(check_func):
                return await asyncio.wait_for(check_func(), timeout=10.0)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, check_func)
        except asyncio.TimeoutError:
            return {"status": "degraded", "error": "Health check timeout"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_status:
            return {
                "overall_status": "unknown",
                "components": {},
                "summary": {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
            }
        
        summary = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
        
        for status in self.health_status.values():
            summary[status.status] = summary.get(status.status, 0) + 1
        
        # Determine overall status
        if summary["unhealthy"] > 0:
            overall_status = "unhealthy"
        elif summary["degraded"] > 0:
            overall_status = "degraded"
        elif summary["healthy"] > 0:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        return {
            "overall_status": overall_status,
            "components": {name: status.__dict__ for name, status in self.health_status.items()},
            "summary": summary,
            "last_updated": max((s.last_check for s in self.health_status.values()), default=0),
            "recent_failures": len([f for f in self.failure_history if time.time() - f.timestamp < 3600])  # Last hour
        }


class GracefulDegradation:
    """Implements graceful degradation strategies for service resilience."""
    
    def __init__(self):
        self.degradation_strategies: Dict[str, Callable] = {}
        self.feature_flags: Dict[str, bool] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
    def register_degradation_strategy(self, feature: str, strategy: Callable) -> None:
        """Register a degradation strategy for a feature."""
        self.degradation_strategies[feature] = strategy
        self.feature_flags[feature] = True
        logger.info(f"Degradation strategy registered for feature: {feature}")
        
    def register_fallback_handler(self, operation: str, handler: Callable) -> None:
        """Register a fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
        logger.info(f"Fallback handler registered for operation: {operation}")
        
    def disable_feature(self, feature: str, reason: str = "degradation") -> None:
        """Disable a feature due to issues."""
        self.feature_flags[feature] = False
        logger.warning(f"Feature {feature} disabled: {reason}")
        
    def enable_feature(self, feature: str) -> None:
        """Re-enable a previously disabled feature."""
        self.feature_flags[feature] = True
        logger.info(f"Feature {feature} re-enabled")
        
    async def execute_with_fallback(
        self,
        operation_name: str,
        primary_operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with fallback support."""
        try:
            return await primary_operation(*args, **kwargs)
        except Exception as primary_error:
            logger.warning(f"Primary operation {operation_name} failed: {primary_error}")
            
            if operation_name in self.fallback_handlers:
                try:
                    fallback_handler = self.fallback_handlers[operation_name]
                    logger.info(f"Executing fallback for {operation_name}")
                    
                    if asyncio.iscoroutinefunction(fallback_handler):
                        return await fallback_handler(*args, **kwargs)
                    else:
                        return fallback_handler(*args, **kwargs)
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback for {operation_name} also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Both primary and fallback operations failed. "
                        f"Primary: {primary_error}, Fallback: {fallback_error}"
                    ) from primary_error
            else:
                logger.error(f"No fallback handler available for {operation_name}")
                raise primary_error
    
    def get_feature_status(self) -> Dict[str, bool]:
        """Get current status of all features."""
        return self.feature_flags.copy()


class ReliabilityManager:
    """Central manager for all reliability features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager(
            max_retries=self.config.get("max_retries", 3),
            base_delay=self.config.get("retry_base_delay", 1.0),
            max_delay=self.config.get("retry_max_delay", 60.0)
        )
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get("health_check_interval", 30.0)
        )
        self.graceful_degradation = GracefulDegradation()
        
        # Metrics
        self.metrics = {
            "operations_total": 0,
            "operations_failed": 0,
            "operations_retried": 0,
            "circuit_breaker_trips": 0,
            "fallback_executions": 0,
            "start_time": time.time()
        }
        
    def get_or_create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker for a named operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
            logger.info(f"Circuit breaker created for: {name}")
        return self.circuit_breakers[name]
    
    async def execute_with_reliability(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        use_fallback: bool = True,
        **kwargs
    ) -> Any:
        """Execute an operation with full reliability features."""
        self.metrics["operations_total"] += 1
        
        circuit_breaker = None
        if use_circuit_breaker:
            circuit_breaker = self.get_or_create_circuit_breaker(operation_name)
        
        async def protected_operation():
            if use_circuit_breaker and circuit_breaker:
                async with circuit_breaker.protect(operation_name):
                    if asyncio.iscoroutinefunction(operation):
                        return await operation(*args, **kwargs)
                    else:
                        return operation(*args, **kwargs)
            else:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)
        
        try:
            if use_retry:
                result = await self.retry_manager.retry_async(protected_operation)
                if self.metrics["operations_retried"] > 0:  # If we had to retry
                    self.metrics["operations_retried"] += 1
                return result
            else:
                return await protected_operation()
                
        except Exception as e:
            self.metrics["operations_failed"] += 1
            
            if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
                self.metrics["circuit_breaker_trips"] += 1
            
            if use_fallback:
                self.metrics["fallback_executions"] += 1
                return await self.graceful_degradation.execute_with_fallback(
                    operation_name, protected_operation, *args, **kwargs
                )
            else:
                raise
    
    async def initialize(self) -> None:
        """Initialize reliability manager and start monitoring."""
        # Register default health checks
        self._register_default_health_checks()
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        logger.info("Reliability manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown reliability manager."""
        await self.health_monitor.stop_monitoring()
        logger.info("Reliability manager shut down")
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks for common components."""
        
        async def memory_health_check():
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "status": "healthy" if memory.percent < 85 else "degraded" if memory.percent < 95 else "unhealthy",
                    "memory_percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "details": {
                        "total_gb": memory.total / (1024**3),
                        "used_gb": memory.used / (1024**3)
                    }
                }
            except ImportError:
                # Fallback if psutil not available
                return {
                    "status": "unknown",
                    "details": {"error": "psutil not available for memory monitoring"}
                }
        
        async def disk_health_check():
            """Check disk space."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                usage_percent = (disk.used / disk.total) * 100
                return {
                    "status": "healthy" if usage_percent < 85 else "degraded" if usage_percent < 95 else "unhealthy",
                    "disk_percent": usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "details": {
                        "total_gb": disk.total / (1024**3),
                        "used_gb": disk.used / (1024**3)
                    }
                }
            except ImportError:
                return {
                    "status": "unknown",
                    "details": {"error": "psutil not available for disk monitoring"}
                }
        
        def thread_pool_health_check():
            """Check thread pool health."""
            try:
                # This would check actual thread pool status in real implementation
                return {
                    "status": "healthy",
                    "active_threads": threading.active_count(),
                    "details": {"thread_count": threading.active_count()}
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "details": {"error": str(e)}
                }
        
        # Register health checks
        self.health_monitor.register_health_check("memory", memory_health_check)
        self.health_monitor.register_health_check("disk", disk_health_check)
        self.health_monitor.register_health_check("threads", thread_pool_health_check)
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive reliability metrics."""
        uptime = time.time() - self.metrics["start_time"]
        
        metrics = {
            "uptime_seconds": uptime,
            "operations": self.metrics.copy(),
            "success_rate": 1.0 - (self.metrics["operations_failed"] / max(self.metrics["operations_total"], 1)),
            "circuit_breakers": {
                name: breaker.get_status() 
                for name, breaker in self.circuit_breakers.items()
            },
            "health_status": self.health_monitor.get_overall_health(),
            "feature_flags": self.graceful_degradation.get_feature_status()
        }
        
        return metrics