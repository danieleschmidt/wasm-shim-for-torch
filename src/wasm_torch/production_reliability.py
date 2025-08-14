"""Production-grade reliability and resilience framework."""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import random
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for system monitoring."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    error_rate: float = 0.0
    request_rate: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    availability: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReliabilityConfig:
    """Configuration for production reliability features."""
    health_check_interval_seconds: int = 30
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_success_threshold: int = 3
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    bulkhead_max_concurrent_requests: int = 100
    timeout_request_seconds: float = 30.0
    degraded_mode_threshold: float = 0.8
    critical_threshold: float = 0.5
    auto_recovery_enabled: bool = True
    failover_enabled: bool = True
    chaos_testing_enabled: bool = False


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and self-healing."""
    
    def __init__(self, 
                 name: str, 
                 config: ReliabilityConfig,
                 failure_threshold: Optional[int] = None,
                 timeout_seconds: Optional[int] = None,
                 success_threshold: Optional[int] = None):
        self.name = name
        self.config = config
        self.failure_threshold = failure_threshold or config.circuit_breaker_failure_threshold
        self.timeout_seconds = timeout_seconds or config.circuit_breaker_timeout_seconds
        self.success_threshold = success_threshold or config.circuit_breaker_success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.request_history: deque = deque(maxlen=1000)
        self.adaptive_threshold = self.failure_threshold
        self._lock = threading.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function call through circuit breaker."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"🔄 Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._record_success(time.time() - start_time)
            return result
        except Exception as e:
            await self._record_failure(time.time() - start_time, e)
            raise
    
    async def _record_success(self, duration: float) -> None:
        """Record successful operation."""
        with self._lock:
            self.request_history.append({"success": True, "duration": duration, "timestamp": time.time()})
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"✅ Circuit breaker {self.name} closed after successful recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
        
        # Adaptive threshold adjustment
        await self._adjust_adaptive_threshold()
    
    async def _record_failure(self, duration: float, error: Exception) -> None:
        """Record failed operation."""
        with self._lock:
            self.request_history.append({"success": False, "duration": duration, "timestamp": time.time(), "error": str(error)})
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.adaptive_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"⚠️ Circuit breaker {self.name} opened after {self.failure_count} failures")
        
        # Adaptive threshold adjustment
        await self._adjust_adaptive_threshold()
    
    async def _adjust_adaptive_threshold(self) -> None:
        """Adjust failure threshold based on recent performance."""
        if len(self.request_history) < 10:
            return
        
        recent_requests = list(self.request_history)[-100:]  # Last 100 requests
        error_rate = sum(1 for req in recent_requests if not req["success"]) / len(recent_requests)
        
        # Adjust threshold based on error rate
        if error_rate > 0.5:  # High error rate - be more sensitive
            self.adaptive_threshold = max(2, self.failure_threshold - 2)
        elif error_rate < 0.1:  # Low error rate - be more tolerant
            self.adaptive_threshold = min(self.failure_threshold + 3, self.failure_threshold * 2)
        else:
            self.adaptive_threshold = self.failure_threshold
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            recent_requests = list(self.request_history)[-100:]
            if recent_requests:
                success_rate = sum(1 for req in recent_requests if req["success"]) / len(recent_requests)
                avg_duration = statistics.mean([req["duration"] for req in recent_requests])
            else:
                success_rate = 1.0
                avg_duration = 0.0
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "adaptive_threshold": self.adaptive_threshold,
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "total_requests": len(self.request_history)
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ExponentialBackoffRetry:
    """Exponential backoff retry mechanism with jitter."""
    
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        self.max_attempts = config.retry_max_attempts
        self.base_delay = config.retry_base_delay_seconds
        self.max_delay = config.retry_max_delay_seconds
    
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, 
                               retryable_exceptions: Tuple = (Exception,),
                               **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    logger.error(f"All {self.max_attempts} retry attempts failed for {func.__name__}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                jitter = random.uniform(0, 0.1 * delay)  # Add up to 10% jitter
                total_delay = delay + jitter
                
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {total_delay:.2f}s")
                await asyncio.sleep(total_delay)
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func.__name__}: {e}")
                raise
        
        # This should never be reached, but just in case
        raise last_exception


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """Acquire resource with bulkhead isolation."""
        async with self._lock:
            self.total_requests += 1
        
        try:
            if timeout:
                await asyncio.wait_for(self.semaphore.acquire(), timeout=timeout)
            else:
                await self.semaphore.acquire()
            
            async with self._lock:
                self.active_requests += 1
            
            try:
                yield
            finally:
                async with self._lock:
                    self.active_requests -= 1
                self.semaphore.release()
        
        except asyncio.TimeoutError:
            async with self._lock:
                self.rejected_requests += 1
            raise BulkheadRejectionError(f"Bulkhead {self.name} rejected request due to timeout")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejected_requests / max(self.total_requests, 1),
            "utilization": self.active_requests / self.max_concurrent
        }


class BulkheadRejectionError(Exception):
    """Exception raised when bulkhead rejects request."""
    pass


class ComprehensiveHealthChecker:
    """Comprehensive health checking system with detailed diagnostics."""
    
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        self.health_history: deque = deque(maxlen=1000)
        self.health_checks: Dict[str, Callable] = {}
        self.critical_dependencies: List[str] = []
        self.last_health_check = 0.0
        self._register_default_health_checks()
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        self.health_checks.update({
            "system_resources": self._check_system_resources,
            "memory_usage": self._check_memory_usage,
            "disk_space": self._check_disk_space,
            "network_connectivity": self._check_network_connectivity,
            "dependencies": self._check_dependencies,
            "performance_metrics": self._check_performance_metrics
        })
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_result = {
            "overall_status": HealthStatus.UNKNOWN,
            "timestamp": time.time(),
            "checks": {},
            "metrics": HealthMetrics(),
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Perform individual health checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = await check_func()
                health_result["checks"][check_name] = check_result
                
                if not check_result["healthy"]:
                    health_result["errors"].extend(check_result.get("errors", []))
                    health_result["warnings"].extend(check_result.get("warnings", []))
                
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_result["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_result["errors"].append(f"Health check {check_name} failed: {e}")
        
        # Calculate overall health status
        health_result["overall_status"] = self._calculate_overall_health(health_result["checks"])
        
        # Generate recommendations
        health_result["recommendations"] = self._generate_health_recommendations(health_result)
        
        # Store in history
        self.health_history.append(health_result)
        self.last_health_check = time.time()
        
        return health_result
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        # Simulate system resource check
        cpu_usage = random.uniform(0.1, 0.9)
        memory_usage = random.uniform(0.2, 0.8)
        
        result = {
            "healthy": cpu_usage < 0.8 and memory_usage < 0.9,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "warnings": [],
            "errors": []
        }
        
        if cpu_usage > 0.8:
            result["warnings"].append(f"High CPU usage: {cpu_usage:.1%}")
        
        if memory_usage > 0.9:
            result["errors"].append(f"Critical memory usage: {memory_usage:.1%}")
        
        return result
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage patterns."""
        # Simulate memory usage check
        heap_usage = random.uniform(0.3, 0.7)
        gc_frequency = random.uniform(0.1, 2.0)  # GC per second
        
        result = {
            "healthy": heap_usage < 0.8 and gc_frequency < 1.0,
            "heap_usage": heap_usage,
            "gc_frequency": gc_frequency,
            "warnings": [],
            "errors": []
        }
        
        if heap_usage > 0.8:
            result["warnings"].append(f"High heap usage: {heap_usage:.1%}")
        
        if gc_frequency > 1.0:
            result["warnings"].append(f"High GC frequency: {gc_frequency:.1f}/s")
        
        return result
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        # Simulate disk space check
        disk_usage = random.uniform(0.2, 0.9)
        
        result = {
            "healthy": disk_usage < 0.9,
            "disk_usage": disk_usage,
            "warnings": [],
            "errors": []
        }
        
        if disk_usage > 0.9:
            result["errors"].append(f"Critical disk usage: {disk_usage:.1%}")
        elif disk_usage > 0.8:
            result["warnings"].append(f"High disk usage: {disk_usage:.1%}")
        
        return result
    
    async def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity and latency."""
        # Simulate network connectivity check
        latency = random.uniform(10, 200)  # ms
        packet_loss = random.uniform(0, 0.05)  # %
        
        result = {
            "healthy": latency < 100 and packet_loss < 0.01,
            "latency_ms": latency,
            "packet_loss_rate": packet_loss,
            "warnings": [],
            "errors": []
        }
        
        if latency > 100:
            result["warnings"].append(f"High network latency: {latency:.0f}ms")
        
        if packet_loss > 0.01:
            result["warnings"].append(f"Network packet loss: {packet_loss:.1%}")
        
        return result
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies availability."""
        # Simulate dependency checks
        dependencies_status = {}
        
        for dep in ["database", "cache", "external_api"]:
            # Simulate dependency check
            available = random.random() > 0.05  # 95% availability
            response_time = random.uniform(10, 500) if available else float('inf')
            
            dependencies_status[dep] = {
                "available": available,
                "response_time_ms": response_time if available else None
            }
        
        all_healthy = all(dep["available"] for dep in dependencies_status.values())
        
        result = {
            "healthy": all_healthy,
            "dependencies": dependencies_status,
            "warnings": [],
            "errors": []
        }
        
        for dep_name, dep_status in dependencies_status.items():
            if not dep_status["available"]:
                result["errors"].append(f"Dependency {dep_name} is unavailable")
            elif dep_status["response_time_ms"] > 1000:
                result["warnings"].append(f"Dependency {dep_name} is slow: {dep_status['response_time_ms']:.0f}ms")
        
        return result
    
    async def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics and trends."""
        # Simulate performance metrics check
        throughput = random.uniform(100, 1000)  # requests/second
        error_rate = random.uniform(0, 0.1)  # %
        p95_latency = random.uniform(50, 500)  # ms
        
        result = {
            "healthy": throughput > 200 and error_rate < 0.05 and p95_latency < 200,
            "throughput_rps": throughput,
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency,
            "warnings": [],
            "errors": []
        }
        
        if throughput < 200:
            result["warnings"].append(f"Low throughput: {throughput:.0f} RPS")
        
        if error_rate > 0.05:
            result["errors"].append(f"High error rate: {error_rate:.1%}")
        
        if p95_latency > 200:
            result["warnings"].append(f"High P95 latency: {p95_latency:.0f}ms")
        
        return result
    
    def _calculate_overall_health(self, checks: Dict[str, Any]) -> HealthStatus:
        """Calculate overall health status from individual checks."""
        healthy_checks = sum(1 for check in checks.values() if check.get("healthy", False))
        total_checks = len(checks)
        
        if total_checks == 0:
            return HealthStatus.UNKNOWN
        
        health_ratio = healthy_checks / total_checks
        
        if health_ratio >= 1.0:
            return HealthStatus.HEALTHY
        elif health_ratio >= 0.8:
            return HealthStatus.DEGRADED
        elif health_ratio >= 0.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _generate_health_recommendations(self, health_result: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on check results."""
        recommendations = []
        
        # Check for patterns in errors
        errors = health_result.get("errors", [])
        warnings = health_result.get("warnings", [])
        
        if "High CPU usage" in str(errors) or "High CPU usage" in str(warnings):
            recommendations.append("Consider scaling up compute resources or optimizing CPU-intensive operations")
        
        if "High memory usage" in str(errors) or "High memory usage" in str(warnings):
            recommendations.append("Investigate memory leaks or consider increasing memory allocation")
        
        if "High network latency" in str(warnings):
            recommendations.append("Investigate network connectivity issues or consider CDN optimization")
        
        if "Dependency" in str(errors):
            recommendations.append("Implement fallback mechanisms for critical dependencies")
        
        if "High error rate" in str(errors):
            recommendations.append("Investigate error patterns and implement better error handling")
        
        # General recommendations based on overall health
        if health_result["overall_status"] in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            recommendations.extend([
                "Enable degraded mode operation",
                "Activate emergency response procedures",
                "Consider system restart or failover"
            ])
        
        return recommendations
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_health = [h for h in self.health_history if h["timestamp"] > cutoff_time]
        
        if not recent_health:
            return {"trend": "insufficient_data"}
        
        # Calculate trends
        status_counts = defaultdict(int)
        for health in recent_health:
            status_counts[health["overall_status"].value] += 1
        
        total_checks = len(recent_health)
        availability = status_counts["healthy"] / total_checks if total_checks > 0 else 0
        
        return {
            "trend": "improving" if availability > 0.9 else "degrading" if availability < 0.7 else "stable",
            "availability": availability,
            "total_health_checks": total_checks,
            "status_distribution": dict(status_counts)
        }


class ProductionReliabilityFramework:
    """Comprehensive production reliability framework."""
    
    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.retry_handler = ExponentialBackoffRetry(self.config)
        self.health_checker = ComprehensiveHealthChecker(self.config)
        self.reliability_metrics: Dict[str, Any] = {}
        self._initialize_reliability_framework()
    
    def _initialize_reliability_framework(self) -> None:
        """Initialize reliability framework components."""
        logger.info("🔧 Initializing Production Reliability Framework")
        
        # Initialize default circuit breakers
        self.circuit_breakers.update({
            "model_inference": AdvancedCircuitBreaker("model_inference", self.config),
            "data_processing": AdvancedCircuitBreaker("data_processing", self.config),
            "external_api": AdvancedCircuitBreaker("external_api", self.config, failure_threshold=3),
            "database": AdvancedCircuitBreaker("database", self.config, failure_threshold=2)
        })
        
        # Initialize default bulkheads
        self.bulkheads.update({
            "inference_requests": BulkheadIsolation("inference_requests", self.config.bulkhead_max_concurrent_requests),
            "batch_processing": BulkheadIsolation("batch_processing", 20),
            "admin_operations": BulkheadIsolation("admin_operations", 5)
        })
        
        # Initialize metrics
        self.reliability_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_activations": 0,
            "bulkhead_rejections": 0,
            "retry_attempts": 0,
            "recovery_events": 0,
            "mean_time_to_recovery": 0.0,
            "availability_sla": 0.999
        }
        
        logger.info("✅ Production Reliability Framework initialized")
    
    async def execute_with_reliability(self, 
                                     operation_name: str, 
                                     func: Callable, 
                                     *args, 
                                     circuit_breaker: Optional[str] = None,
                                     bulkhead: Optional[str] = None,
                                     enable_retry: bool = True,
                                     timeout: Optional[float] = None,
                                     **kwargs) -> Any:
        """Execute operation with comprehensive reliability patterns."""
        start_time = time.time()
        self.reliability_metrics["total_requests"] += 1
        
        try:
            # Apply bulkhead isolation if specified
            if bulkhead and bulkhead in self.bulkheads:
                async with self.bulkheads[bulkhead].acquire(timeout=timeout):
                    return await self._execute_with_patterns(
                        operation_name, func, args, kwargs, circuit_breaker, enable_retry, timeout
                    )
            else:
                return await self._execute_with_patterns(
                    operation_name, func, args, kwargs, circuit_breaker, enable_retry, timeout
                )
        
        except Exception as e:
            self.reliability_metrics["failed_requests"] += 1
            execution_time = time.time() - start_time
            
            # Log reliability event
            logger.error(f"Operation {operation_name} failed after {execution_time:.2f}s: {e}")
            
            raise
        
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Operation {operation_name} completed in {execution_time:.2f}s")
    
    async def _execute_with_patterns(self, 
                                   operation_name: str, 
                                   func: Callable, 
                                   args: tuple, 
                                   kwargs: dict,
                                   circuit_breaker: Optional[str],
                                   enable_retry: bool,
                                   timeout: Optional[float]) -> Any:
        """Execute operation with reliability patterns."""
        # Apply circuit breaker if specified
        if circuit_breaker and circuit_breaker in self.circuit_breakers:
            breaker = self.circuit_breakers[circuit_breaker]
            
            if enable_retry:
                # Combine circuit breaker with retry
                async def breaker_wrapped():
                    return await breaker.call(func, *args, **kwargs)
                
                result = await self.retry_handler.execute_with_retry(
                    breaker_wrapped,
                    retryable_exceptions=(Exception,)
                )
            else:
                # Circuit breaker only
                result = await breaker.call(func, *args, **kwargs)
        else:
            # Retry only or direct execution
            if enable_retry:
                result = await self.retry_handler.execute_with_retry(
                    func, *args, retryable_exceptions=(Exception,), **kwargs
                )
            else:
                # Direct execution with timeout
                if timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        self.reliability_metrics["successful_requests"] += 1
        return result
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_result = await self.health_checker.perform_health_check()
        
        # Add reliability-specific metrics
        health_result["reliability_metrics"] = self.get_reliability_metrics()
        health_result["circuit_breaker_status"] = self.get_circuit_breaker_status()
        health_result["bulkhead_status"] = self.get_bulkhead_status()
        
        return health_result
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive reliability metrics."""
        total_requests = self.reliability_metrics["total_requests"]
        successful_requests = self.reliability_metrics["successful_requests"]
        
        metrics = self.reliability_metrics.copy()
        
        if total_requests > 0:
            metrics["success_rate"] = successful_requests / total_requests
            metrics["error_rate"] = (total_requests - successful_requests) / total_requests
        else:
            metrics["success_rate"] = 1.0
            metrics["error_rate"] = 0.0
        
        return metrics
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {}
        
        for name, breaker in self.circuit_breakers.items():
            status[name] = breaker.get_metrics()
        
        return status
    
    def get_bulkhead_status(self) -> Dict[str, Any]:
        """Get status of all bulkheads."""
        status = {}
        
        for name, bulkhead in self.bulkheads.items():
            status[name] = bulkhead.get_metrics()
        
        return status
    
    async def enable_degraded_mode(self, reason: str) -> Dict[str, Any]:
        """Enable degraded mode operation."""
        logger.warning(f"⚠️ Enabling degraded mode: {reason}")
        
        # Adjust circuit breaker thresholds for degraded mode
        for breaker in self.circuit_breakers.values():
            breaker.adaptive_threshold = max(1, breaker.adaptive_threshold // 2)
        
        # Reduce bulkhead limits
        for bulkhead in self.bulkheads.values():
            bulkhead.max_concurrent = max(1, bulkhead.max_concurrent // 2)
            bulkhead.semaphore = asyncio.Semaphore(bulkhead.max_concurrent)
        
        degraded_config = {
            "mode": "degraded",
            "reason": reason,
            "timestamp": time.time(),
            "circuit_breaker_adjustments": len(self.circuit_breakers),
            "bulkhead_adjustments": len(self.bulkheads)
        }
        
        return degraded_config
    
    async def auto_recovery_check(self) -> Dict[str, Any]:
        """Perform automatic recovery check and actions."""
        if not self.config.auto_recovery_enabled:
            return {"auto_recovery": "disabled"}
        
        recovery_actions = []
        
        # Check circuit breakers for recovery opportunities
        for name, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitState.OPEN:
                # Check if we should attempt recovery
                time_since_failure = time.time() - breaker.last_failure_time
                if time_since_failure > breaker.timeout_seconds * 2:  # Extended timeout
                    breaker.state = CircuitState.HALF_OPEN
                    breaker.success_count = 0
                    recovery_actions.append(f"Reset circuit breaker: {name}")
                    self.reliability_metrics["recovery_events"] += 1
        
        # Check system health for recovery
        health_result = await self.health_checker.perform_health_check()
        if health_result["overall_status"] == HealthStatus.HEALTHY:
            # System is healthy, reset any temporary restrictions
            recovery_actions.append("System health restored")
        
        return {
            "auto_recovery_enabled": True,
            "recovery_actions": recovery_actions,
            "recovery_timestamp": time.time()
        }
    
    async def chaos_testing(self) -> Dict[str, Any]:
        """Perform chaos testing to validate reliability."""
        if not self.config.chaos_testing_enabled:
            return {"chaos_testing": "disabled"}
        
        chaos_experiments = []
        
        # Randomly trip a circuit breaker
        if random.random() < 0.1:  # 10% chance
            breaker_name = random.choice(list(self.circuit_breakers.keys()))
            breaker = self.circuit_breakers[breaker_name]
            breaker.failure_count = breaker.adaptive_threshold
            breaker.state = CircuitState.OPEN
            chaos_experiments.append(f"Chaos: Opened circuit breaker {breaker_name}")
        
        # Simulate temporary resource exhaustion
        if random.random() < 0.05:  # 5% chance
            bulkhead_name = random.choice(list(self.bulkheads.keys()))
            bulkhead = self.bulkheads[bulkhead_name]
            original_limit = bulkhead.max_concurrent
            bulkhead.max_concurrent = 1  # Severely limit resources
            bulkhead.semaphore = asyncio.Semaphore(1)
            chaos_experiments.append(f"Chaos: Limited bulkhead {bulkhead_name} to 1 concurrent")
            
            # Schedule recovery after 30 seconds
            async def restore_bulkhead():
                await asyncio.sleep(30)
                bulkhead.max_concurrent = original_limit
                bulkhead.semaphore = asyncio.Semaphore(original_limit)
            
            asyncio.create_task(restore_bulkhead())
        
        return {
            "chaos_testing_enabled": True,
            "experiments_conducted": chaos_experiments,
            "chaos_timestamp": time.time()
        }


async def main():
    """Main function demonstrating production reliability features."""
    print("🔧 PRODUCTION RELIABILITY FRAMEWORK DEMONSTRATION")
    
    # Initialize reliability framework
    config = ReliabilityConfig(
        health_check_interval_seconds=10,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_timeout_seconds=30,
        retry_max_attempts=3,
        bulkhead_max_concurrent_requests=50,
        auto_recovery_enabled=True,
        chaos_testing_enabled=True
    )
    
    reliability_framework = ProductionReliabilityFramework(config)
    
    # Simulate reliable operation execution
    async def sample_operation(value: int) -> int:
        # Simulate potential failure
        if random.random() < 0.2:  # 20% failure rate
            raise Exception(f"Simulated failure for value {value}")
        
        await asyncio.sleep(0.1)  # Simulate work
        return value * 2
    
    # Execute operation with reliability patterns
    try:
        result = await reliability_framework.execute_with_reliability(
            "sample_operation",
            sample_operation,
            42,
            circuit_breaker="model_inference",
            bulkhead="inference_requests",
            enable_retry=True,
            timeout=5.0
        )
        print(f"✅ Operation succeeded: {result}")
    except Exception as e:
        print(f"❌ Operation failed: {e}")
    
    # Perform health check
    health_result = await reliability_framework.perform_health_check()
    print(f"🏥 System Health: {health_result['overall_status'].value}")
    print(f"📏 Success Rate: {reliability_framework.get_reliability_metrics()['success_rate']:.3f}")
    
    # Show circuit breaker status
    cb_status = reliability_framework.get_circuit_breaker_status()
    for name, status in cb_status.items():
        print(f"⚡ Circuit Breaker {name}: {status['state']} (success rate: {status['success_rate']:.3f})")
    
    # Perform auto-recovery check
    recovery_result = await reliability_framework.auto_recovery_check()
    if recovery_result.get("recovery_actions"):
        print(f"🔄 Recovery Actions: {recovery_result['recovery_actions']}")
    
    # Perform chaos testing
    chaos_result = await reliability_framework.chaos_testing()
    if chaos_result.get("experiments_conducted"):
        print(f"🌀 Chaos Experiments: {chaos_result['experiments_conducted']}")
    
    return {
        "health_result": health_result,
        "reliability_metrics": reliability_framework.get_reliability_metrics(),
        "circuit_breaker_status": cb_status,
        "recovery_result": recovery_result,
        "chaos_result": chaos_result
    }


if __name__ == "__main__":
    asyncio.run(main())
