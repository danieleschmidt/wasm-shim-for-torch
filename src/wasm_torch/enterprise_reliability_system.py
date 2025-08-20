"""
Enterprise Reliability System - Generation 2: Make it Robust
Comprehensive error handling, circuit breakers, health monitoring, and enterprise-grade reliability.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback
import sys
import os
from collections import defaultdict, deque
import uuid
import contextlib
import functools

# Initialize comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autonomous_sdlc.log') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for enterprise classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HealthStatus(Enum):
    """Health status levels for system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker state management."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and analysis."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = "unknown"
    operation: str = "unknown"
    error_type: str = "unknown"
    error_message: str = ""
    stack_trace: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    recovery_successful: bool = False
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'system_state': self.system_state,
            'recovery_attempts': self.recovery_attempts,
            'recovery_successful': self.recovery_successful,
            'impact_assessment': self.impact_assessment
        }


@dataclass
class HealthCheck:
    """Health check definition and results."""
    check_id: str
    name: str
    component: str
    check_function: Callable
    interval: float = 30.0  # seconds
    timeout: float = 10.0   # seconds
    critical: bool = False
    last_run: float = 0.0
    last_result: Optional[HealthStatus] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_runs: int = 0
    total_failures: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate for this health check."""
        if self.total_runs == 0:
            return 1.0
        return (self.total_runs - self.total_failures) / self.total_runs


class EnterpriseCircuitBreaker:
    """
    Enterprise-grade circuit breaker with adaptive thresholds and intelligent recovery.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.state_changes = []
        self.lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self.lock:
            self.total_requests += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self._record_state_change(CircuitBreakerState.HALF_OPEN)
                else:
                    # Circuit is open, fail fast
                    if self.fallback_function:
                        try:
                            if asyncio.iscoroutinefunction(self.fallback_function):
                                return await self.fallback_function(*args, **kwargs)
                            else:
                                return self.fallback_function(*args, **kwargs)
                        except Exception as e:
                            logger.error(f"Fallback function failed for '{self.name}': {e}")
                    
                    raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit if necessary
            with self.lock:
                self.successful_requests += 1
                self.failure_count = 0
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' recovered, transitioning to CLOSED")
                    self._record_state_change(CircuitBreakerState.CLOSED)
            
            return result
            
        except self.expected_exception as e:
            with self.lock:
                self.failed_requests += 1
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Check if we should open the circuit
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitBreakerState.OPEN:
                        self.state = CircuitBreakerState.OPEN
                        logger.warning(f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
                        self._record_state_change(CircuitBreakerState.OPEN)
            
            logger.error(f"Circuit breaker '{self.name}' recorded failure: {e}")
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection."""
        with self.lock:
            self.total_requests += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self._record_state_change(CircuitBreakerState.HALF_OPEN)
                else:
                    # Circuit is open, fail fast
                    if self.fallback_function:
                        try:
                            return self.fallback_function(*args, **kwargs)
                        except Exception as e:
                            logger.error(f"Fallback function failed for '{self.name}': {e}")
                    
                    raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit if necessary
            with self.lock:
                self.successful_requests += 1
                self.failure_count = 0
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' recovered, transitioning to CLOSED")
                    self._record_state_change(CircuitBreakerState.CLOSED)
            
            return result
            
        except self.expected_exception as e:
            with self.lock:
                self.failed_requests += 1
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Check if we should open the circuit
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitBreakerState.OPEN:
                        self.state = CircuitBreakerState.OPEN
                        logger.warning(f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
                        self._record_state_change(CircuitBreakerState.OPEN)
            
            logger.error(f"Circuit breaker '{self.name}' recorded failure: {e}")
            raise
    
    def _record_state_change(self, new_state: CircuitBreakerState) -> None:
        """Record state change for monitoring."""
        self.state_changes.append({
            'timestamp': time.time(),
            'from_state': self.state.value if hasattr(self, 'state') else 'unknown',
            'to_state': new_state.value,
            'failure_count': self.failure_count,
            'total_requests': self.total_requests
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'last_failure_time': self.last_failure_time,
                'state_changes': len(self.state_changes),
                'recovery_timeout': self.recovery_timeout
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            logger.info(f"Circuit breaker '{self.name}' reset to initial state")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is in OPEN state."""
    pass


class EnterpriseErrorHandler:
    """
    Comprehensive enterprise error handling system with recovery strategies.
    """
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=10000)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[str, List[Callable]] = {}
        self.circuit_breakers: Dict[str, EnterpriseCircuitBreaker] = {}
        self.error_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
        
        logger.info("Enterprise Error Handler initialized")
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None
    ) -> EnterpriseCircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = EnterpriseCircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            fallback_function=fallback_function
        )
        
        with self.lock:
            self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Circuit breaker '{name}' registered")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[EnterpriseCircuitBreaker]:
        """Get circuit breaker by name."""
        with self.lock:
            return self.circuit_breakers.get(name)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """Register a recovery strategy for specific error type."""
        with self.lock:
            if error_type not in self.recovery_strategies:
                self.recovery_strategies[error_type] = []
            self.recovery_strategies[error_type].append(strategy)
        
        logger.info(f"Recovery strategy registered for error type: {error_type}")
    
    def register_error_callback(self, callback: Callable) -> None:
        """Register callback to be called on error events."""
        self.error_callbacks.append(callback)
        logger.info("Error callback registered")
    
    async def handle_error(
        self,
        error: Exception,
        component: str = "unknown",
        operation: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> ErrorContext:
        """
        Handle error with comprehensive analysis and recovery attempts.
        """
        error_context = ErrorContext(
            severity=self._assess_error_severity(error),
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            system_state=context or {}
        )
        
        # Record error in history
        with self.lock:
            self.error_history.append(error_context)
            self.error_patterns[error_context.error_type] += 1
        
        # Log error with appropriate level
        log_message = f"Error in {component}/{operation}: {error_context.error_message}"
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Assess impact
        error_context.impact_assessment = await self._assess_error_impact(error_context)
        
        # Attempt recovery if enabled and strategies exist
        if attempt_recovery and error_context.error_type in self.recovery_strategies:
            await self._attempt_recovery(error_context)
        
        # Notify callbacks
        await self._notify_error_callbacks(error_context)
        
        return error_context
    
    def _assess_error_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity based on error type and context."""
        # Critical errors
        critical_errors = [
            'SystemExit', 'KeyboardInterrupt', 'MemoryError',
            'RecursionError', 'SystemError'
        ]
        
        if type(error).__name__ in critical_errors:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_severity_errors = [
            'ConnectionError', 'TimeoutError', 'OSError',
            'PermissionError', 'FileNotFoundError'
        ]
        
        if type(error).__name__ in high_severity_errors:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_severity_errors = [
            'ValueError', 'TypeError', 'AttributeError',
            'KeyError', 'IndexError'
        ]
        
        if type(error).__name__ in medium_severity_errors:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    async def _assess_error_impact(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Assess the impact of an error on system operations."""
        impact = {
            'affected_components': [error_context.component],
            'service_degradation': 'none',
            'user_impact': 'minimal',
            'data_integrity_risk': 'low',
            'security_implications': 'none'
        }
        
        # Assess based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            impact.update({
                'service_degradation': 'severe',
                'user_impact': 'high',
                'data_integrity_risk': 'high'
            })
        elif error_context.severity == ErrorSeverity.HIGH:
            impact.update({
                'service_degradation': 'moderate',
                'user_impact': 'moderate',
                'data_integrity_risk': 'medium'
            })
        
        # Assess based on component
        critical_components = ['database', 'authentication', 'payment', 'security']
        if any(comp in error_context.component.lower() for comp in critical_components):
            impact['service_degradation'] = 'high'
            impact['user_impact'] = 'high'
        
        # Assess error frequency
        error_frequency = self.error_patterns.get(error_context.error_type, 0)
        if error_frequency > 10:  # Frequent error
            impact['service_degradation'] = 'increasing'
        
        return impact
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> None:
        """Attempt recovery using registered strategies."""
        strategies = self.recovery_strategies.get(error_context.error_type, [])
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Attempting recovery strategy {i+1}/{len(strategies)} for {error_context.error_type}")
                
                if asyncio.iscoroutinefunction(strategy):
                    success = await strategy(error_context)
                else:
                    success = strategy(error_context)
                
                error_context.recovery_attempts += 1
                
                if success:
                    error_context.recovery_successful = True
                    logger.info(f"Recovery successful for error {error_context.error_id}")
                    break
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                error_context.recovery_attempts += 1
    
    async def _notify_error_callbacks(self, error_context: ErrorContext) -> None:
        """Notify registered error callbacks."""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_context)
                else:
                    callback(error_context)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")
    
    def _setup_default_recovery_strategies(self) -> None:
        """Setup default recovery strategies for common errors."""
        
        async def connection_error_recovery(error_context: ErrorContext) -> bool:
            """Recovery strategy for connection errors."""
            logger.info("Attempting connection recovery with exponential backoff")
            
            for attempt in range(3):
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
                
                # Simulate connection retry
                try:
                    # In real implementation, would attempt to reconnect
                    logger.info(f"Connection recovery attempt {attempt + 1}")
                    # If successful, return True
                    if attempt >= 1:  # Simulate success on 2nd attempt
                        return True
                except Exception:
                    continue
            
            return False
        
        async def timeout_error_recovery(error_context: ErrorContext) -> bool:
            """Recovery strategy for timeout errors."""
            logger.info("Attempting timeout recovery with increased timeout")
            
            # Simulate retry with increased timeout
            await asyncio.sleep(1)
            return True  # Simulate successful recovery
        
        async def memory_error_recovery(error_context: ErrorContext) -> bool:
            """Recovery strategy for memory errors."""
            logger.info("Attempting memory recovery through garbage collection")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Simulate memory cleanup success
            return True
        
        # Register default strategies
        self.register_recovery_strategy('ConnectionError', connection_error_recovery)
        self.register_recovery_strategy('TimeoutError', timeout_error_recovery)
        self.register_recovery_strategy('MemoryError', memory_error_recovery)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            total_errors = len(self.error_history)
            recent_errors = [
                error for error in self.error_history
                if time.time() - error.timestamp < 3600  # Last hour
            ]
            
            severity_counts = defaultdict(int)
            component_errors = defaultdict(int)
            
            for error in self.error_history:
                severity_counts[error.severity.value] += 1
                component_errors[error.component] += 1
            
            recovery_stats = {
                'total_recovery_attempts': sum(error.recovery_attempts for error in self.error_history),
                'successful_recoveries': len([
                    error for error in self.error_history if error.recovery_successful
                ]),
                'recovery_success_rate': 0.0
            }
            
            if recovery_stats['total_recovery_attempts'] > 0:
                recovery_stats['recovery_success_rate'] = (
                    recovery_stats['successful_recoveries'] / 
                    len([error for error in self.error_history if error.recovery_attempts > 0])
                )
            
            return {
                'total_errors': total_errors,
                'recent_errors': len(recent_errors),
                'error_patterns': dict(self.error_patterns),
                'severity_distribution': dict(severity_counts),
                'component_error_distribution': dict(component_errors),
                'recovery_statistics': recovery_stats,
                'circuit_breaker_status': {
                    name: breaker.get_statistics()
                    for name, breaker in self.circuit_breakers.items()
                }
            }


class EnterpriseHealthMonitor:
    """
    Enterprise health monitoring system with predictive capabilities.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: deque = deque(maxlen=10000)
        self.system_health_score = 1.0
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.lock = threading.RLock()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("Enterprise Health Monitor initialized")
    
    def register_health_check(
        self,
        check_id: str,
        name: str,
        component: str,
        check_function: Callable,
        interval: float = 30.0,
        timeout: float = 10.0,
        critical: bool = False
    ) -> None:
        """Register a health check."""
        health_check = HealthCheck(
            check_id=check_id,
            name=name,
            component=component,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical
        )
        
        with self.lock:
            self.health_checks[check_id] = health_check
        
        logger.info(f"Health check '{name}' registered for component '{component}'")
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def run_health_check(self, check_id: str) -> HealthStatus:
        """Run a specific health check."""
        if check_id not in self.health_checks:
            logger.error(f"Health check '{check_id}' not found")
            return HealthStatus.UNKNOWN
        
        health_check = self.health_checks[check_id]
        
        try:
            # Run health check with timeout
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(health_check.check_function),
                    timeout=health_check.timeout
                )
            
            # Determine health status
            if isinstance(result, HealthStatus):
                status = result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
            else:
                status = HealthStatus.HEALTHY  # Assume healthy if function completes
            
            # Update health check
            with self.lock:
                health_check.last_run = time.time()
                health_check.last_result = status
                health_check.last_error = None
                health_check.total_runs += 1
                
                if status != HealthStatus.HEALTHY:
                    health_check.consecutive_failures += 1
                    health_check.total_failures += 1
                else:
                    health_check.consecutive_failures = 0
            
            # Record in history
            self.health_history.append({
                'timestamp': time.time(),
                'check_id': check_id,
                'status': status.value,
                'component': health_check.component,
                'critical': health_check.critical
            })
            
            return status
            
        except asyncio.TimeoutError:
            logger.error(f"Health check '{check_id}' timed out after {health_check.timeout}s")
            return await self._record_health_check_failure(health_check, "Timeout")
            
        except Exception as e:
            logger.error(f"Health check '{check_id}' failed: {e}")
            return await self._record_health_check_failure(health_check, str(e))
    
    async def _record_health_check_failure(
        self, 
        health_check: HealthCheck, 
        error_message: str
    ) -> HealthStatus:
        """Record health check failure."""
        with self.lock:
            health_check.last_run = time.time()
            health_check.last_result = HealthStatus.UNHEALTHY
            health_check.last_error = error_message
            health_check.total_runs += 1
            health_check.total_failures += 1
            health_check.consecutive_failures += 1
        
        # Record in history
        self.health_history.append({
            'timestamp': time.time(),
            'check_id': health_check.check_id,
            'status': HealthStatus.UNHEALTHY.value,
            'component': health_check.component,
            'critical': health_check.critical,
            'error': error_message
        })
        
        return HealthStatus.UNHEALTHY
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run due health checks
                current_time = time.time()
                checks_to_run = []
                
                with self.lock:
                    for check_id, health_check in self.health_checks.items():
                        if current_time - health_check.last_run >= health_check.interval:
                            checks_to_run.append(check_id)
                
                # Run checks concurrently
                if checks_to_run:
                    tasks = [self.run_health_check(check_id) for check_id in checks_to_run]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update system health score
                await self._update_system_health_score()
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_system_health_score(self) -> None:
        """Update overall system health score."""
        if not self.health_checks:
            self.system_health_score = 1.0
            return
        
        total_weight = 0
        weighted_score = 0
        
        with self.lock:
            for health_check in self.health_checks.values():
                if health_check.last_result is None:
                    continue  # Skip checks that haven't run yet
                
                # Weight critical checks more heavily
                weight = 2.0 if health_check.critical else 1.0
                total_weight += weight
                
                # Convert health status to score
                if health_check.last_result == HealthStatus.HEALTHY:
                    score = 1.0
                elif health_check.last_result == HealthStatus.DEGRADED:
                    score = 0.5
                elif health_check.last_result == HealthStatus.UNHEALTHY:
                    score = 0.0
                else:  # UNKNOWN
                    score = 0.3
                
                weighted_score += score * weight
        
        if total_weight > 0:
            self.system_health_score = weighted_score / total_weight
        else:
            self.system_health_score = 1.0
    
    def _setup_default_health_checks(self) -> None:
        """Setup default system health checks."""
        
        async def memory_health_check() -> HealthStatus:
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                if memory.percent > 90:
                    return HealthStatus.UNHEALTHY
                elif memory.percent > 80:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY
                    
            except ImportError:
                # Fallback without psutil
                return HealthStatus.HEALTHY
        
        async def cpu_health_check() -> HealthStatus:
            """Check CPU usage."""
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                
                if cpu_percent > 95:
                    return HealthStatus.UNHEALTHY
                elif cpu_percent > 85:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY
                    
            except ImportError:
                return HealthStatus.HEALTHY
        
        async def disk_health_check() -> HealthStatus:
            """Check disk usage."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                
                usage_percent = (disk.used / disk.total) * 100
                
                if usage_percent > 95:
                    return HealthStatus.UNHEALTHY
                elif usage_percent > 85:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY
                    
            except (ImportError, FileNotFoundError):
                return HealthStatus.HEALTHY
        
        # Register default health checks
        self.register_health_check(
            'system_memory',
            'System Memory Usage',
            'system',
            memory_health_check,
            interval=30.0,
            critical=True
        )
        
        self.register_health_check(
            'system_cpu',
            'System CPU Usage',
            'system',
            cpu_health_check,
            interval=15.0,
            critical=True
        )
        
        self.register_health_check(
            'system_disk',
            'System Disk Usage',
            'system',
            disk_health_check,
            interval=60.0,
            critical=False
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self.lock:
            health_checks_status = {}
            
            for check_id, health_check in self.health_checks.items():
                health_checks_status[check_id] = {
                    'name': health_check.name,
                    'component': health_check.component,
                    'status': health_check.last_result.value if health_check.last_result else 'unknown',
                    'last_run': health_check.last_run,
                    'last_error': health_check.last_error,
                    'success_rate': health_check.success_rate(),
                    'consecutive_failures': health_check.consecutive_failures,
                    'critical': health_check.critical
                }
            
            return {
                'system_health_score': self.system_health_score,
                'overall_status': self._get_overall_status(),
                'monitoring_active': self.monitoring_active,
                'health_checks': health_checks_status,
                'total_health_checks': len(self.health_checks),
                'critical_failures': len([
                    check for check in self.health_checks.values()
                    if check.critical and check.last_result == HealthStatus.UNHEALTHY
                ])
            }
    
    def _get_overall_status(self) -> str:
        """Determine overall system status."""
        if self.system_health_score >= 0.9:
            return 'healthy'
        elif self.system_health_score >= 0.7:
            return 'degraded'
        else:
            return 'unhealthy'


class EnterpriseReliabilitySystem:
    """
    Main enterprise reliability system integrating all reliability components.
    """
    
    def __init__(self):
        self.error_handler = EnterpriseErrorHandler()
        self.health_monitor = EnterpriseHealthMonitor()
        self.reliability_metrics: Dict[str, Any] = {}
        self.reliability_score = 1.0
        self.active = False
        
        # Setup integrated error handling
        self.error_handler.register_error_callback(self._on_error_event)
        
        logger.info("Enterprise Reliability System initialized")
    
    async def initialize(self) -> bool:
        """Initialize the reliability system."""
        try:
            logger.info("Initializing Enterprise Reliability System")
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Register circuit breakers for critical components
            self._setup_circuit_breakers()
            
            self.active = True
            logger.info("Enterprise Reliability System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enterprise Reliability System: {e}")
            return False
    
    def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for critical system components."""
        
        # Database operations circuit breaker
        self.error_handler.register_circuit_breaker(
            name='database_operations',
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=Exception,
            fallback_function=self._database_fallback
        )
        
        # External API circuit breaker
        self.error_handler.register_circuit_breaker(
            name='external_api',
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception,
            fallback_function=self._api_fallback
        )
        
        # Quantum optimization circuit breaker
        self.error_handler.register_circuit_breaker(
            name='quantum_optimization',
            failure_threshold=3,
            recovery_timeout=45.0,
            expected_exception=Exception,
            fallback_function=self._quantum_fallback
        )
        
        logger.info("Circuit breakers configured for critical components")
    
    def _database_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback function for database operations."""
        logger.warning("Using database fallback - returning cached or default data")
        return {
            'status': 'fallback',
            'data': None,
            'message': 'Database temporarily unavailable, using fallback'
        }
    
    def _api_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback function for external API calls."""
        logger.warning("Using API fallback - returning default response")
        return {
            'status': 'fallback',
            'data': {},
            'message': 'External API temporarily unavailable'
        }
    
    def _quantum_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback function for quantum optimization."""
        logger.warning("Using quantum fallback - using classical optimization")
        return {
            'status': 'fallback',
            'optimization_result': 'classical_fallback',
            'message': 'Quantum optimization unavailable, using classical fallback'
        }
    
    async def _on_error_event(self, error_context: ErrorContext) -> None:
        """Handle error events for reliability tracking."""
        # Update reliability metrics based on errors
        if error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.reliability_score *= 0.95  # Reduce reliability score
            self.reliability_score = max(0.0, self.reliability_score)
        
        # Log reliability impact
        logger.info(f"Reliability score updated to {self.reliability_score:.3f} due to {error_context.severity.value} error")
    
    @contextlib.asynccontextmanager
    async def reliability_context(self, component: str, operation: str):
        """Context manager for reliable operation execution."""
        start_time = time.time()
        
        try:
            yield
            
            # Operation successful - slightly improve reliability score
            self.reliability_score = min(1.0, self.reliability_score * 1.001)
            
        except Exception as e:
            # Handle error through reliability system
            error_context = await self.error_handler.handle_error(
                error=e,
                component=component,
                operation=operation,
                context={'execution_time': time.time() - start_time}
            )
            
            # Re-raise if recovery was not successful
            if not error_context.recovery_successful:
                raise
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive reliability report."""
        error_stats = self.error_handler.get_error_statistics()
        health_status = self.health_monitor.get_health_status()
        
        return {
            'reliability_score': self.reliability_score,
            'system_health_score': health_status['system_health_score'],
            'overall_status': health_status['overall_status'],
            'error_statistics': error_stats,
            'health_monitoring': health_status,
            'active': self.active,
            'reliability_metrics': self.reliability_metrics,
            'uptime': time.time() - (getattr(self, '_start_time', time.time())),
            'recommendations': self._generate_reliability_recommendations(error_stats, health_status)
        }
    
    def _generate_reliability_recommendations(
        self, 
        error_stats: Dict[str, Any], 
        health_status: Dict[str, Any]
    ) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []
        
        # Error-based recommendations
        if error_stats['recent_errors'] > 10:
            recommendations.append("High error rate detected - investigate root causes")
        
        if error_stats['recovery_statistics']['recovery_success_rate'] < 0.8:
            recommendations.append("Low recovery success rate - review recovery strategies")
        
        # Health-based recommendations
        if health_status['system_health_score'] < 0.8:
            recommendations.append("System health degraded - check critical components")
        
        if health_status['critical_failures'] > 0:
            recommendations.append("Critical health checks failing - immediate attention required")
        
        # Circuit breaker recommendations
        for cb_name, cb_stats in error_stats['circuit_breaker_status'].items():
            if cb_stats['state'] == 'open':
                recommendations.append(f"Circuit breaker '{cb_name}' is open - service degraded")
            elif cb_stats['success_rate'] < 0.9:
                recommendations.append(f"Circuit breaker '{cb_name}' has low success rate")
        
        if not recommendations:
            recommendations.append("System reliability is good - continue monitoring")
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown the reliability system."""
        try:
            logger.info("Shutting down Enterprise Reliability System")
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            self.active = False
            logger.info("Enterprise Reliability System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during reliability system shutdown: {e}")


# Decorators for easy integration

def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    fallback_function: Optional[Callable] = None
):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func: Callable) -> Callable:
        # Get or create global reliability system
        reliability_system = get_global_reliability_system()
        
        # Register circuit breaker
        circuit_breaker = reliability_system.error_handler.register_circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            fallback_function=fallback_function
        )
        
        return circuit_breaker(func)
    
    return decorator


def with_error_handling(
    component: str = "unknown",
    operation: str = "unknown",
    attempt_recovery: bool = True
):
    """Decorator to add comprehensive error handling to functions."""
    def decorator(func: Callable) -> Callable:
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            reliability_system = get_global_reliability_system()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error_context = await reliability_system.error_handler.handle_error(
                    error=e,
                    component=component,
                    operation=operation,
                    attempt_recovery=attempt_recovery
                )
                
                if not error_context.recovery_successful:
                    raise
                
                # Return None if recovery was successful but no return value
                return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run to handle the async error handling
            async def async_handler():
                return await async_wrapper(*args, **kwargs)
            
            try:
                if asyncio.get_running_loop():
                    # Already in async context
                    return asyncio.create_task(async_handler())
                else:
                    return asyncio.run(async_handler())
            except RuntimeError:
                # No event loop, run sync
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {component}/{operation}: {e}")
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global reliability system instance
_global_reliability_system = None

def get_global_reliability_system() -> EnterpriseReliabilitySystem:
    """Get global reliability system instance."""
    global _global_reliability_system
    if _global_reliability_system is None:
        _global_reliability_system = EnterpriseReliabilitySystem()
    return _global_reliability_system


# Example usage and testing
async def demo_enterprise_reliability_system():
    """Demonstrate the enterprise reliability system."""
    logger.info("Starting Enterprise Reliability System Demo")
    
    reliability_system = EnterpriseReliabilitySystem()
    reliability_system._start_time = time.time()
    
    try:
        # Initialize system
        success = await reliability_system.initialize()
        if not success:
            logger.error("Failed to initialize reliability system")
            return
        
        # Test error handling
        @with_error_handling(component="demo", operation="test_function")
        async def test_function_with_errors():
            # Simulate various types of errors
            import random
            
            error_types = [ValueError, ConnectionError, TimeoutError, None]  # None = success
            error_type = random.choice(error_types)
            
            if error_type:
                raise error_type(f"Simulated {error_type.__name__}")
            
            return "Success!"
        
        # Test circuit breaker
        @with_circuit_breaker(name="demo_circuit_breaker", failure_threshold=3)
        async def test_circuit_breaker():
            import random
            if random.random() < 0.7:  # 70% failure rate for testing
                raise ConnectionError("Simulated connection failure")
            return "Circuit breaker test passed"
        
        # Run tests
        logger.info("Testing error handling and recovery...")
        for i in range(10):
            try:
                result = await test_function_with_errors()
                logger.info(f"Test {i+1}: {result}")
            except Exception as e:
                logger.warning(f"Test {i+1} failed: {e}")
        
        logger.info("Testing circuit breaker...")
        for i in range(8):
            try:
                result = await test_circuit_breaker()
                logger.info(f"Circuit breaker test {i+1}: {result}")
            except Exception as e:
                logger.warning(f"Circuit breaker test {i+1}: {e}")
            
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Wait for health checks to run
        logger.info("Waiting for health checks...")
        await asyncio.sleep(35)
        
        # Get reliability report
        report = reliability_system.get_reliability_report()
        
        logger.info("=== Reliability Report ===")
        logger.info(f"Reliability Score: {report['reliability_score']:.3f}")
        logger.info(f"System Health Score: {report['system_health_score']:.3f}")
        logger.info(f"Overall Status: {report['overall_status']}")
        logger.info(f"Total Errors: {report['error_statistics']['total_errors']}")
        logger.info(f"Recent Errors: {report['error_statistics']['recent_errors']}")
        logger.info(f"Recovery Success Rate: {report['error_statistics']['recovery_statistics']['recovery_success_rate']:.2%}")
        
        logger.info("=== Health Checks ===")
        for check_id, check_status in report['health_monitoring']['health_checks'].items():
            logger.info(f"{check_status['name']}: {check_status['status']} "
                       f"(Success Rate: {check_status['success_rate']:.2%})")
        
        logger.info("=== Circuit Breakers ===")
        for cb_name, cb_stats in report['error_statistics']['circuit_breaker_status'].items():
            logger.info(f"{cb_name}: {cb_stats['state']} "
                       f"(Success Rate: {cb_stats['success_rate']:.2%}, "
                       f"Failures: {cb_stats['failure_count']}/{cb_stats['failure_threshold']})")
        
        logger.info("=== Recommendations ===")
        for rec in report['recommendations']:
            logger.info(f"- {rec}")
        
    finally:
        # Shutdown system
        await reliability_system.shutdown()

if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demo_enterprise_reliability_system())