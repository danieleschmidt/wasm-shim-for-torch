"""Robust Error Handling - Generation 2: Reliability Systems

Comprehensive error handling, recovery mechanisms, and validation systems
for production-ready PyTorch-to-WASM inference.
"""

import asyncio
import time
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import json
import hashlib
import sys

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    INPUT_VALIDATION = "input_validation"
    MODEL_EXECUTION = "model_execution"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and recovery."""
    error_id: str = field(default_factory=lambda: hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8])
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    error_type: str = "UnknownError"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    backoff_delay: float = 1.0
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'error_type': self.error_type,
            'message': self.message,
            'details': self.details,
            'stack_trace': self.stack_trace,
            'recovery_suggestions': self.recovery_suggestions,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'backoff_delay': self.backoff_delay,
            'context_data': self.context_data
        }
    
    def should_retry(self) -> bool:
        """Check if error should be retried."""
        return (self.retry_count < self.max_retries and 
                self.category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.TIMEOUT, ErrorCategory.RESOURCE_EXHAUSTION])


class WASMTorchError(Exception):
    """Base exception for WASM-Torch library."""
    
    def __init__(self, message: str, error_context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.error_context = error_context or ErrorContext(message=message)
        self.error_context.message = message
        self.error_context.error_type = self.__class__.__name__


class ModelNotFoundError(WASMTorchError):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_id: str):
        context = ErrorContext(
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL_EXECUTION,
            message=f"Model not found: {model_id}",
            details={"model_id": model_id},
            recovery_suggestions=["Check if model is registered", "Verify model ID spelling"]
        )
        super().__init__(f"Model not found: {model_id}", context)


class InputValidationError(WASMTorchError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, input_data: Any = None):
        context = ErrorContext(
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INPUT_VALIDATION,
            message=message,
            details={"input_type": type(input_data).__name__ if input_data else None},
            recovery_suggestions=["Check input format", "Validate input dimensions", "Ensure input is not None"]
        )
        super().__init__(message, context)


class ResourceExhaustionError(WASMTorchError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource_type: str, current_usage: float, limit: float):
        context = ErrorContext(
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            message=f"{resource_type} exhausted: {current_usage}/{limit}",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "usage_percentage": (current_usage / limit) * 100 if limit > 0 else 0
            },
            recovery_suggestions=[
                "Wait for resources to free up",
                "Increase resource limits",
                "Optimize resource usage"
            ]
        )
        super().__init__(f"{resource_type} exhausted: {current_usage}/{limit}", context)


class InferenceTimeoutError(WASMTorchError):
    """Raised when inference times out."""
    
    def __init__(self, timeout: float, elapsed: float):
        context = ErrorContext(
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TIMEOUT,
            message=f"Inference timeout: {elapsed:.2f}s > {timeout:.2f}s",
            details={
                "timeout": timeout,
                "elapsed": elapsed,
                "overhead": elapsed - timeout
            },
            recovery_suggestions=[
                "Increase timeout value",
                "Optimize model performance",
                "Check system load"
            ]
        )
        super().__init__(f"Inference timeout: {elapsed:.2f}s > {timeout:.2f}s", context)


class CircuitBreakerError(WASMTorchError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, failure_rate: float):
        context = ErrorContext(
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM_ERROR,
            message=f"Circuit breaker open for {service_name} (failure rate: {failure_rate:.2%})",
            details={
                "service_name": service_name,
                "failure_rate": failure_rate
            },
            recovery_suggestions=[
                "Wait for circuit breaker to reset",
                "Check service health",
                "Implement fallback mechanism"
            ]
        )
        super().__init__(f"Circuit breaker open for {service_name}", context)


class ErrorRecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies."""
    
    @abstractmethod
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can recover from the error."""
        pass
    
    @abstractmethod
    async def recover(self, error_context: ErrorContext) -> Any:
        """Attempt to recover from the error."""
        pass


class RetryRecoveryStrategy(ErrorRecoveryStrategy):
    """Retry-based recovery strategy with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if error can be recovered by retrying."""
        return error_context.should_retry()
    
    async def recover(self, error_context: ErrorContext) -> Any:
        """Attempt recovery by waiting and retrying."""
        error_context.retry_count += 1
        
        # Calculate exponential backoff delay
        delay = min(
            self.base_delay * (2 ** (error_context.retry_count - 1)),
            self.max_delay
        )
        
        logger.info(f"Retrying after {delay:.2f}s (attempt {error_context.retry_count}/{error_context.max_retries})")
        await asyncio.sleep(delay)
        
        return None  # Signal that operation should be retried


class FallbackRecoveryStrategy(ErrorRecoveryStrategy):
    """Fallback-based recovery strategy."""
    
    def __init__(self, fallback_function: Callable):
        self.fallback_function = fallback_function
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if fallback is available."""
        return self.fallback_function is not None
    
    async def recover(self, error_context: ErrorContext) -> Any:
        """Attempt recovery using fallback function."""
        logger.warning(f"Using fallback for error: {error_context.message}")
        
        if asyncio.iscoroutinefunction(self.fallback_function):
            return await self.fallback_function(error_context)
        else:
            return self.fallback_function(error_context)


class CircuitBreakerRecoveryStrategy(ErrorRecoveryStrategy):
    """Circuit breaker-based recovery strategy."""
    
    def __init__(self, failure_threshold: float = 0.5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._circuit_states: Dict[str, Dict] = {}
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker allows recovery."""
        service_name = error_context.context_data.get('service_name', 'default')
        state = self._circuit_states.get(service_name, {
            'state': 'closed',
            'failure_count': 0,
            'success_count': 0,
            'last_failure_time': 0
        })
        
        return state['state'] != 'open'
    
    async def recover(self, error_context: ErrorContext) -> Any:
        """Update circuit breaker state."""
        service_name = error_context.context_data.get('service_name', 'default')
        
        if service_name not in self._circuit_states:
            self._circuit_states[service_name] = {
                'state': 'closed',
                'failure_count': 0,
                'success_count': 0,
                'last_failure_time': 0
            }
        
        state = self._circuit_states[service_name]
        current_time = time.time()
        
        # Update failure count
        state['failure_count'] += 1
        state['last_failure_time'] = current_time
        
        # Calculate failure rate
        total_requests = state['failure_count'] + state['success_count']
        if total_requests > 0:
            failure_rate = state['failure_count'] / total_requests
            
            # Open circuit if failure rate exceeds threshold
            if failure_rate > self.failure_threshold and total_requests >= 10:
                state['state'] = 'open'
                logger.warning(f"Circuit breaker opened for {service_name} (failure rate: {failure_rate:.2%})")
                raise CircuitBreakerError(service_name, failure_rate)
        
        return None


class InputValidator:
    """Comprehensive input validation system."""
    
    @staticmethod
    def validate_tensor_input(input_data: Any, expected_shape: Optional[List[int]] = None,
                            expected_dtype: Optional[str] = None) -> None:
        """Validate tensor input data."""
        if input_data is None:
            raise InputValidationError("Input data cannot be None")
        
        # Check for NaN and infinite values
        if hasattr(input_data, '__iter__') and not isinstance(input_data, str):
            try:
                flat_data = list(input_data) if isinstance(input_data, (list, tuple)) else [input_data]
                for value in flat_data:
                    if isinstance(value, (int, float)):
                        if str(value).lower() in ['nan', 'inf', '-inf']:
                            raise InputValidationError(f"Input contains invalid value: {value}")
                        # Check for extremely large values that might cause overflow
                        if abs(value) > 1e10:
                            logger.warning(f"Input contains very large value: {value}")
            except (TypeError, ValueError) as e:
                raise InputValidationError(f"Failed to validate input values: {e}")
        
        # Validate shape if expected
        if expected_shape is not None and hasattr(input_data, 'shape'):
            if list(input_data.shape) != expected_shape:
                raise InputValidationError(
                    f"Input shape mismatch: expected {expected_shape}, got {list(input_data.shape)}"
                )
        
        # Validate data type if expected
        if expected_dtype is not None and hasattr(input_data, 'dtype'):
            if str(input_data.dtype) != expected_dtype:
                raise InputValidationError(
                    f"Input dtype mismatch: expected {expected_dtype}, got {input_data.dtype}"
                )
    
    @staticmethod
    def validate_model_id(model_id: str) -> None:
        """Validate model ID format and content."""
        if not isinstance(model_id, str):
            raise InputValidationError("Model ID must be a string")
        
        if not model_id.strip():
            raise InputValidationError("Model ID cannot be empty")
        
        if len(model_id) > 100:
            raise InputValidationError("Model ID too long (max 100 characters)")
        
        # Check for potentially malicious characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', '..']
        for char in dangerous_chars:
            if char in model_id:
                raise InputValidationError(f"Model ID contains dangerous character: {char}")
    
    @staticmethod
    def validate_timeout(timeout: float) -> None:
        """Validate timeout value."""
        if not isinstance(timeout, (int, float)):
            raise InputValidationError("Timeout must be a number")
        
        if timeout <= 0:
            raise InputValidationError("Timeout must be positive")
        
        if timeout > 3600:  # 1 hour
            logger.warning(f"Very long timeout specified: {timeout}s")
    
    @staticmethod
    def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> None:
        """Validate batch size."""
        if not isinstance(batch_size, int):
            raise InputValidationError("Batch size must be an integer")
        
        if batch_size <= 0:
            raise InputValidationError("Batch size must be positive")
        
        if batch_size > max_batch_size:
            raise InputValidationError(f"Batch size too large: {batch_size} > {max_batch_size}")


class RobustErrorHandler:
    """Comprehensive error handling system with recovery strategies."""
    
    def __init__(self):
        self._recovery_strategies: List[ErrorRecoveryStrategy] = []
        self._error_history: List[ErrorContext] = []
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        # Add default recovery strategies
        self.add_recovery_strategy(RetryRecoveryStrategy())
        self.add_recovery_strategy(CircuitBreakerRecoveryStrategy())
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Add a recovery strategy."""
        self._recovery_strategies.append(strategy)
    
    async def handle_error(self, error: Exception, context_data: Optional[Dict[str, Any]] = None) -> Any:
        """Handle error with recovery strategies."""
        # Create error context
        error_context = self._create_error_context(error, context_data or {})
        
        # Log error
        self._log_error(error_context)
        
        # Record error in history
        with self._lock:
            self._error_history.append(error_context)
            self._error_counts[error_context.error_type] = (
                self._error_counts.get(error_context.error_type, 0) + 1
            )
            
            # Limit history size
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-500:]
        
        # Attempt recovery
        for strategy in self._recovery_strategies:
            if await strategy.can_recover(error_context):
                try:
                    recovery_result = await strategy.recover(error_context)
                    logger.info(f"Error recovered using {strategy.__class__.__name__}")
                    return recovery_result
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
                    continue
        
        # No recovery possible, re-raise with enhanced context
        if isinstance(error, WASMTorchError):
            raise error
        else:
            raise WASMTorchError(str(error), error_context)
    
    def _create_error_context(self, error: Exception, context_data: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        # Extract error details
        error_type = type(error).__name__
        message = str(error)
        stack_trace = traceback.format_exc()
        
        # Categorize error
        category = self._categorize_error(error)
        severity = self._determine_severity(error, category)
        
        # Create context
        error_context = ErrorContext(
            severity=severity,
            category=category,
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            context_data=context_data
        )
        
        # Add recovery suggestions
        error_context.recovery_suggestions = self._generate_recovery_suggestions(error, category)
        
        return error_context
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if isinstance(error, (ValueError, TypeError)) or 'validation' in error_message:
            return ErrorCategory.INPUT_VALIDATION
        elif 'timeout' in error_message or isinstance(error, asyncio.TimeoutError):
            return ErrorCategory.TIMEOUT
        elif 'memory' in error_message or 'resource' in error_message:
            return ErrorCategory.RESOURCE_EXHAUSTION
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK_ERROR
        elif 'permission' in error_message or 'access' in error_message:
            return ErrorCategory.AUTHORIZATION
        elif 'model' in error_message and 'not found' in error_message:
            return ErrorCategory.MODEL_EXECUTION
        else:
            return ErrorCategory.SYSTEM_ERROR
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity."""
        if category in [ErrorCategory.SYSTEM_ERROR, ErrorCategory.RESOURCE_EXHAUSTION]:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.MODEL_EXECUTION, ErrorCategory.TIMEOUT]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.INPUT_VALIDATION, ErrorCategory.NETWORK_ERROR]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_recovery_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate context-specific recovery suggestions."""
        suggestions = []
        
        if category == ErrorCategory.INPUT_VALIDATION:
            suggestions.extend([
                "Validate input format and types",
                "Check for null or empty values",
                "Verify input dimensions match model requirements"
            ])
        elif category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Increase timeout value",
                "Check system performance",
                "Optimize model or reduce input size"
            ])
        elif category == ErrorCategory.RESOURCE_EXHAUSTION:
            suggestions.extend([
                "Free up system resources",
                "Increase memory limits",
                "Process smaller batches"
            ])
        elif category == ErrorCategory.MODEL_EXECUTION:
            suggestions.extend([
                "Verify model is properly loaded",
                "Check model compatibility",
                "Validate model file integrity"
            ])
        
        return suggestions
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity."""
        log_message = f"Error {error_context.error_id}: {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={'error_context': error_context.to_dict()})
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={'error_context': error_context.to_dict()})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'error_context': error_context.to_dict()})
        else:
            logger.info(log_message, extra={'error_context': error_context.to_dict()})
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        with self._lock:
            recent_errors = [
                error for error in self._error_history
                if time.time() - error.timestamp < 3600  # Last hour
            ]
            
            return {
                'total_errors': len(self._error_history),
                'recent_errors': len(recent_errors),
                'error_counts_by_type': dict(self._error_counts),
                'error_counts_by_category': {
                    category.value: sum(1 for error in recent_errors if error.category == category)
                    for category in ErrorCategory
                },
                'error_counts_by_severity': {
                    severity.value: sum(1 for error in recent_errors if error.severity == severity)
                    for severity in ErrorSeverity
                }
            }
    
    @asynccontextmanager
    async def error_context(self, operation_name: str, **context_kwargs):
        """Context manager for handling errors in operations."""
        context_data = {
            'operation': operation_name,
            'timestamp': time.time(),
            **context_kwargs
        }
        
        try:
            yield context_data
        except Exception as e:
            await self.handle_error(e, context_data)
            raise


# Demo function
async def demo_robust_error_handling():
    """Demonstrate robust error handling capabilities."""
    
    print("Robust Error Handling Demo - Generation 2")
    print("=" * 50)
    
    # Create error handler
    error_handler = RobustErrorHandler()
    validator = InputValidator()
    
    # Test input validation
    print("\\nTesting input validation...")
    try:
        validator.validate_model_id("")
    except InputValidationError as e:
        print(f"✓ Caught validation error: {e.error_context.message}")
    
    try:
        validator.validate_tensor_input([float('nan'), 1, 2])
    except InputValidationError as e:
        print(f"✓ Caught NaN validation error: {e.error_context.message}")
    
    # Test error handling with recovery
    print("\\nTesting error handling with recovery...")
    async def failing_function():
        raise ConnectionError("Network temporarily unavailable")
    
    # Add fallback strategy
    def fallback_function(error_context):
        return "fallback_result"
    
    error_handler.add_recovery_strategy(FallbackRecoveryStrategy(fallback_function))
    
    async with error_handler.error_context("test_operation", service="test_service"):
        try:
            result = await failing_function()
        except Exception as e:
            print(f"✓ Handling error: {e}")
            # Error will be handled by context manager
    
    # Show error statistics
    stats = error_handler.get_error_statistics()
    print(f"\\nError Statistics:")
    print(f"Total errors: {stats['total_errors']}")
    print(f"Recent errors: {stats['recent_errors']}")
    print(f"Error types: {stats['error_counts_by_type']}")
    
    print("\\n🛡️ Robust Error Handling Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_robust_error_handling())