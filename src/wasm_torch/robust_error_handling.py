"""
Robust Error Handling - Generation 2: Make It Robust
Comprehensive error handling, recovery, and resilience systems.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import threading
from contextlib import asynccontextmanager, contextmanager
import weakref
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    
    LOW = auto()          # Minor issues, can continue
    MEDIUM = auto()       # Significant issues, degraded performance
    HIGH = auto()         # Major issues, partial system failure
    CRITICAL = auto()     # System-breaking issues, immediate attention


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""
    
    VALIDATION = auto()   # Input/output validation errors
    RESOURCE = auto()     # Resource exhaustion, allocation failures
    NETWORK = auto()      # Network connectivity, timeout issues
    PERMISSION = auto()   # Access control, authorization failures
    CORRUPTION = auto()   # Data corruption, integrity issues
    TIMEOUT = auto()      # Operation timeout errors
    DEPENDENCY = auto()   # Missing or failed dependencies
    CONFIGURATION = auto() # Configuration errors
    RUNTIME = auto()      # General runtime errors


@dataclass
class ErrorContext:
    """Rich context information for errors."""
    
    error_id: str
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.RUNTIME
    component: str = "unknown"
    operation: str = "unknown"
    error_type: str = "Exception"
    error_message: str = ""
    traceback_info: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    last_recovery_attempt: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.name,
            'category': self.category.name,
            'component': self.component,
            'operation': self.operation,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'traceback_info': self.traceback_info,
            'metadata': self.metadata,
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts,
            'last_recovery_attempt': self.last_recovery_attempt
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing, block requests
    HALF_OPEN = auto() # Testing if recovered


class RobustCircuitBreaker:
    """
    Robust circuit breaker implementation with adaptive thresholds.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.blocked_calls = 0
        
        self._lock = threading.RLock()
        
    @asynccontextmanager
    async def protect(self):
        """Async context manager for circuit breaker protection."""
        if not self._should_allow_call():
            self.blocked_calls += 1
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        
        self.total_calls += 1
        start_time = time.time()
        
        try:
            yield
            # Success case
            execution_time = time.time() - start_time
            await self._on_success(execution_time)
            
        except Exception as e:
            # Failure case
            execution_time = time.time() - start_time
            await self._on_failure(e, execution_time)
            raise
    
    def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                # Check if we should transition to half-open
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.recovery_timeout):
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            
            return False
    
    async def _on_success(self, execution_time: float) -> None:
        """Handle successful operation."""
        with self._lock:
            self.successful_calls += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception, execution_time: float) -> None:
        """Handle failed operation."""
        with self._lock:
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after half-open failure")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            success_rate = (
                self.successful_calls / self.total_calls 
                if self.total_calls > 0 else 0.0
            )
            
            return {
                'name': self.name,
                'state': self.state.name,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'blocked_calls': self.blocked_calls,
                'success_rate': success_rate,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RobustRetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or [Exception]
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        # Check if error type is retryable
        return any(isinstance(error, exc_type) for exc_type in self.retry_exceptions)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter: ±20% of the delay
            import random
            jitter_factor = 0.8 + (0.4 * random.random())
            delay *= jitter_factor
        
        return delay


class RobustErrorManager:
    """
    Comprehensive error management system for robust operations.
    """
    
    def __init__(self):
        self._error_history: List[ErrorContext] = []
        self._circuit_breakers: Dict[str, RobustCircuitBreaker] = {}
        self._error_handlers: Dict[str, Callable] = {}
        self._recovery_strategies: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        # Error statistics
        self._stats = {
            'total_errors': 0,
            'errors_by_severity': {sev.name: 0 for sev in ErrorSeverity},
            'errors_by_category': {cat.name: 0 for cat in ErrorCategory},
            'recovery_success_rate': 0.0,
            'circuit_breaker_trips': 0
        }
    
    def register_circuit_breaker(
        self, 
        name: str, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> RobustCircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            circuit_breaker = RobustCircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            self._circuit_breakers[name] = circuit_breaker
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[RobustCircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuit_breakers.get(name)
    
    def register_error_handler(
        self, 
        error_type: str, 
        handler: Callable[[ErrorContext], Any]
    ) -> None:
        """Register a custom error handler."""
        self._error_handlers[error_type] = handler
    
    def register_recovery_strategy(
        self,
        component: str,
        strategy: Callable[[ErrorContext], bool]
    ) -> None:
        """Register a recovery strategy for a component."""
        self._recovery_strategies[component] = strategy
    
    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle an error with full context and recovery."""
        
        # Create error context
        error_context = ErrorContext(
            error_id=f"{component}_{operation}_{int(time.time() * 1000000)}",
            severity=severity,
            category=category,
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_info=traceback.format_exc(),
            metadata=metadata or {}
        )
        
        # Record error
        with self._lock:
            self._error_history.append(error_context)
            self._stats['total_errors'] += 1
            self._stats['errors_by_severity'][severity.name] += 1
            self._stats['errors_by_category'][category.name] += 1
        
        # Log error with appropriate level
        self._log_error(error_context)
        
        # Try custom error handler
        error_type = type(error).__name__
        if error_type in self._error_handlers:
            try:
                await self._error_handlers[error_type](error_context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        
        # Attempt recovery if appropriate
        if severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]:
            recovery_success = await self._attempt_recovery(error_context)
            if recovery_success:
                logger.info(f"Recovery successful for error {error_context.error_id}")
        
        return error_context
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error."""
        if error_context.recovery_attempts >= error_context.max_recovery_attempts:
            return False
        
        component = error_context.component
        if component not in self._recovery_strategies:
            return False
        
        try:
            error_context.recovery_attempts += 1
            error_context.last_recovery_attempt = time.time()
            
            recovery_strategy = self._recovery_strategies[component]
            success = await recovery_strategy(error_context)
            
            if success:
                # Update success rate
                with self._lock:
                    total_recoveries = sum(
                        ec.recovery_attempts for ec in self._error_history
                    )
                    successful_recoveries = len([
                        ec for ec in self._error_history 
                        if ec.recovery_attempts > 0
                    ])
                    if total_recoveries > 0:
                        self._stats['recovery_success_rate'] = (
                            successful_recoveries / total_recoveries
                        )
            
            return success
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity."""
        message = (
            f"Error in {error_context.component}.{error_context.operation}: "
            f"{error_context.error_message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            recent_errors = [
                ec for ec in self._error_history
                if time.time() - ec.timestamp < 3600  # Last hour
            ]
            
            circuit_breaker_stats = {
                name: cb.get_stats() 
                for name, cb in self._circuit_breakers.items()
            }
            
            return {
                'total_errors': self._stats['total_errors'],
                'recent_errors': len(recent_errors),
                'errors_by_severity': self._stats['errors_by_severity'].copy(),
                'errors_by_category': self._stats['errors_by_category'].copy(),
                'recovery_success_rate': self._stats['recovery_success_rate'],
                'circuit_breakers': circuit_breaker_stats,
                'error_history_size': len(self._error_history)
            }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors for analysis."""
        with self._lock:
            recent_errors = sorted(
                self._error_history,
                key=lambda ec: ec.timestamp,
                reverse=True
            )[:limit]
            
            return [ec.to_dict() for ec in recent_errors]


# Global error manager instance
_global_error_manager: Optional[RobustErrorManager] = None


def get_global_error_manager() -> RobustErrorManager:
    """Get the global error manager instance."""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = RobustErrorManager()
    return _global_error_manager


# Decorators for robust error handling
def robust_operation(
    component: str,
    operation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.RUNTIME,
    retry_policy: Optional[RobustRetryPolicy] = None,
    circuit_breaker: Optional[str] = None
):
    """
    Decorator for robust error handling with retries and circuit breakers.
    """
    def decorator(func):
        actual_operation = operation or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_manager = get_global_error_manager()
            retry_pol = retry_policy or RobustRetryPolicy()
            
            # Get circuit breaker if specified
            cb = None
            if circuit_breaker:
                cb = error_manager.get_circuit_breaker(circuit_breaker)
                if not cb:
                    cb = error_manager.register_circuit_breaker(circuit_breaker)
            
            last_error = None
            
            for attempt in range(retry_pol.max_attempts):
                try:
                    if cb:
                        async with cb.protect():
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                        
                except Exception as e:
                    last_error = e
                    
                    # Handle error
                    await error_manager.handle_error(
                        error=e,
                        component=component,
                        operation=actual_operation,
                        severity=severity,
                        category=category,
                        metadata={
                            'attempt': attempt + 1,
                            'max_attempts': retry_pol.max_attempts,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        }
                    )
                    
                    # Check if should retry
                    if not retry_pol.should_retry(e, attempt + 1):
                        break
                    
                    # Wait before retry
                    if attempt < retry_pol.max_attempts - 1:
                        delay = retry_pol.calculate_delay(attempt)
                        await asyncio.sleep(delay)
            
            # All retries failed
            raise last_error
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in async context
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage and testing
async def demo_robust_error_handling():
    """Demonstration of robust error handling."""
    error_manager = get_global_error_manager()
    
    # Register circuit breaker
    cb = error_manager.register_circuit_breaker("demo_service", failure_threshold=3)
    
    # Register recovery strategy
    async def demo_recovery_strategy(error_context: ErrorContext) -> bool:
        print(f"Attempting recovery for {error_context.error_id}")
        # Simulate recovery logic
        return error_context.recovery_attempts == 1  # Succeed on second attempt
    
    error_manager.register_recovery_strategy("demo_component", demo_recovery_strategy)
    
    # Test robust operation
    @robust_operation(
        component="demo_component",
        operation="test_operation",
        circuit_breaker="demo_service",
        retry_policy=RobustRetryPolicy(max_attempts=2)
    )
    async def failing_operation(fail_count: int = 0):
        if fail_count > 0:
            raise ValueError(f"Simulated failure {fail_count}")
        return "Success!"
    
    print("Testing robust error handling...")
    
    # Test successful operation
    try:
        result = await failing_operation(0)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test operation with failures
    try:
        result = await failing_operation(1)
        print(f"Success after retry: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    # Show statistics
    stats = error_manager.get_error_statistics()
    print(f"Error statistics: {json.dumps(stats, indent=2)}")
    
    # Show circuit breaker stats
    cb_stats = cb.get_stats()
    print(f"Circuit breaker stats: {json.dumps(cb_stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(demo_robust_error_handling())