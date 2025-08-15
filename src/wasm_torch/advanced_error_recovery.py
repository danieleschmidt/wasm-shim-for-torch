"""Advanced error recovery and resilience system for WASM-Torch."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import json
import traceback

# Optional dependencies - gracefully handle missing imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for systematic handling."""
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    VALIDATION = "validation"
    RESOURCE = "resource"
    NETWORK = "network"
    SECURITY = "security"
    USER_INPUT = "user_input"


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error_id: str
    timestamp: float
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    stack_trace: str
    retry_count: int = 0
    max_retries: int = 3
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: List[str] = field(default_factory=list)


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class AdvancedErrorRecovery:
    """Advanced error recovery system with intelligent retry and fallback strategies."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.recovery_strategies = {
            ErrorCategory.COMPILATION: [
                self._retry_with_lower_optimization,
                self._disable_simd_and_retry,
                self._fallback_to_basic_compilation,
            ],
            ErrorCategory.RUNTIME: [
                self._restart_wasm_runtime,
                self._clear_memory_and_retry,
                self._fallback_to_cpu_mode,
            ],
            ErrorCategory.VALIDATION: [
                self._sanitize_input_and_retry,
                self._use_default_parameters,
                self._skip_validation_temporarily,
            ],
            ErrorCategory.RESOURCE: [
                self._free_unused_memory,
                self._reduce_batch_size,
                self._use_memory_mapped_files,
            ],
            ErrorCategory.NETWORK: [
                self._retry_with_exponential_backoff,
                self._switch_to_backup_endpoint,
                self._use_cached_data,
            ],
            ErrorCategory.SECURITY: [
                self._sanitize_and_retry,
                self._use_restricted_mode,
                self._reject_request,
            ],
            ErrorCategory.USER_INPUT: [
                self._validate_and_correct_input,
                self._use_default_values,
                self._provide_user_guidance,
            ],
        }
    
    async def handle_error(
        self,
        error: Exception,
        operation: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Handle error with intelligent recovery strategies."""
        error_context = self._create_error_context(error, operation, context_data or {})
        
        logger.warning(f"Handling error in {operation}: {error_context.message}")
        
        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(operation)
        if not circuit_breaker.can_execute():
            logger.error(f"Circuit breaker OPEN for {operation}")
            return await self._execute_fallback(operation, error_context)
        
        # Attempt recovery strategies
        recovery_result = await self._attempt_recovery(error_context)
        
        if recovery_result is not None:
            circuit_breaker.record_success()
            logger.info(f"Error recovery successful for {operation}")
            return recovery_result
        else:
            circuit_breaker.record_failure()
            logger.error(f"Error recovery failed for {operation}")
            return await self._execute_fallback(operation, error_context)
    
    def _create_error_context(
        self,
        error: Exception,
        operation: str,
        context_data: Dict[str, Any]
    ) -> ErrorContext:
        """Create error context for systematic handling."""
        error_id = f"{operation}_{int(time.time() * 1000)}"
        severity = self._determine_error_severity(error)
        category = self._categorize_error(error, operation)
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            operation=operation,
            severity=severity,
            category=category,
            message=str(error),
            stack_trace=traceback.format_exc(),
            context_data=context_data
        )
        
        self.error_history.append(error_context)
        self._update_error_patterns(error_context)
        
        return error_context
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, RuntimeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _categorize_error(self, error: Exception, operation: str) -> ErrorCategory:
        """Categorize error for appropriate handling strategy."""
        error_msg = str(error).lower()
        
        if "compilation" in error_msg or "emscripten" in error_msg:
            return ErrorCategory.COMPILATION
        elif "memory" in error_msg or "out of memory" in error_msg:
            return ErrorCategory.RESOURCE
        elif "validation" in error_msg or "invalid" in error_msg:
            return ErrorCategory.VALIDATION
        elif "network" in error_msg or "connection" in error_msg:
            return ErrorCategory.NETWORK
        elif "security" in error_msg or "permission" in error_msg:
            return ErrorCategory.SECURITY
        elif "input" in error_msg or "parameter" in error_msg:
            return ErrorCategory.USER_INPUT
        else:
            return ErrorCategory.RUNTIME
    
    def _get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker()
        return self.circuit_breakers[operation]
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt recovery using appropriate strategies."""
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            if error_context.retry_count >= error_context.max_retries:
                logger.warning(f"Max retries exceeded for {error_context.error_id}")
                break
            
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                error_context.recovery_attempted.append(strategy.__name__)
                error_context.retry_count += 1
                
                # Execute recovery strategy
                result = await self._execute_recovery_strategy(strategy, error_context)
                
                if result is not None:
                    logger.info(f"Recovery successful with {strategy.__name__}")
                    return result
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        return None
    
    async def _execute_recovery_strategy(
        self,
        strategy: Callable,
        error_context: ErrorContext
    ) -> Optional[Any]:
        """Execute a recovery strategy with timeout."""
        try:
            if asyncio.iscoroutinefunction(strategy):
                return await asyncio.wait_for(strategy(error_context), timeout=30.0)
            else:
                return strategy(error_context)
        except asyncio.TimeoutError:
            logger.error(f"Recovery strategy {strategy.__name__} timed out")
            return None
    
    async def _execute_fallback(self, operation: str, error_context: ErrorContext) -> Optional[Any]:
        """Execute fallback handler for operation."""
        fallback_handler = self.fallback_handlers.get(operation)
        
        if fallback_handler:
            try:
                logger.info(f"Executing fallback for {operation}")
                if asyncio.iscoroutinefunction(fallback_handler):
                    return await fallback_handler(error_context)
                else:
                    return fallback_handler(error_context)
            except Exception as fallback_error:
                logger.error(f"Fallback handler failed: {fallback_error}")
        
        # Default fallback: log error and return None
        logger.error(f"No fallback available for {operation}")
        return None
    
    def _update_error_patterns(self, error_context: ErrorContext) -> None:
        """Update error patterns for predictive analysis."""
        pattern_key = f"{error_context.category.value}_{error_context.operation}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "count": 0,
                "last_seen": 0,
                "common_causes": {},
                "successful_recoveries": {},
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_seen"] = error_context.timestamp
        
        # Track common error messages
        error_key = error_context.message[:100]  # Truncate for grouping
        pattern["common_causes"][error_key] = pattern["common_causes"].get(error_key, 0) + 1
    
    # Recovery Strategy Implementations
    
    async def _retry_with_lower_optimization(self, error_context: ErrorContext) -> Optional[Any]:
        """Retry compilation with lower optimization level."""
        if "optimization" in error_context.context_data:
            current_level = error_context.context_data["optimization"]
            if current_level == "O3":
                error_context.context_data["optimization"] = "O2"
            elif current_level == "O2":
                error_context.context_data["optimization"] = "O1"
            elif current_level == "O1":
                error_context.context_data["optimization"] = "O0"
            else:
                return None
            
            logger.info(f"Retrying with optimization level {error_context.context_data['optimization']}")
            return {"retry_with_context": error_context.context_data}
        return None
    
    async def _disable_simd_and_retry(self, error_context: ErrorContext) -> Optional[Any]:
        """Disable SIMD and retry compilation."""
        if error_context.context_data.get("use_simd", False):
            error_context.context_data["use_simd"] = False
            logger.info("Retrying with SIMD disabled")
            return {"retry_with_context": error_context.context_data}
        return None
    
    async def _fallback_to_basic_compilation(self, error_context: ErrorContext) -> Optional[Any]:
        """Fallback to basic compilation settings."""
        error_context.context_data.update({
            "optimization": "O0",
            "use_simd": False,
            "use_threads": False
        })
        logger.info("Falling back to basic compilation settings")
        return {"retry_with_context": error_context.context_data}
    
    async def _restart_wasm_runtime(self, error_context: ErrorContext) -> Optional[Any]:
        """Restart WASM runtime."""
        logger.info("Attempting to restart WASM runtime")
        # Implementation would restart the actual runtime
        return {"action": "restart_runtime"}
    
    async def _clear_memory_and_retry(self, error_context: ErrorContext) -> Optional[Any]:
        """Clear memory and retry operation."""
        logger.info("Clearing memory and retrying")
        # Implementation would clear memory pools and caches
        return {"action": "clear_memory"}
    
    async def _fallback_to_cpu_mode(self, error_context: ErrorContext) -> Optional[Any]:
        """Fallback to CPU-only mode."""
        error_context.context_data["cpu_only"] = True
        logger.info("Falling back to CPU-only mode")
        return {"retry_with_context": error_context.context_data}
    
    async def _sanitize_input_and_retry(self, error_context: ErrorContext) -> Optional[Any]:
        """Sanitize input data and retry."""
        if "input_data" in error_context.context_data:
            # Basic input sanitization
            input_data = error_context.context_data["input_data"]
            if hasattr(input_data, "clamp"):
                # Clamp values to reasonable range
                input_data = input_data.clamp(-1000, 1000)
                error_context.context_data["input_data"] = input_data
                logger.info("Sanitized input data and retrying")
                return {"retry_with_context": error_context.context_data}
        return None
    
    async def _use_default_parameters(self, error_context: ErrorContext) -> Optional[Any]:
        """Use default parameters for operation."""
        defaults = {
            "batch_size": 1,
            "optimization": "O1",
            "use_simd": False,
            "use_threads": False,
            "memory_limit": 256
        }
        
        error_context.context_data.update(defaults)
        logger.info("Using default parameters")
        return {"retry_with_context": error_context.context_data}
    
    async def _skip_validation_temporarily(self, error_context: ErrorContext) -> Optional[Any]:
        """Skip validation temporarily (for non-critical operations)."""
        if error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            error_context.context_data["skip_validation"] = True
            logger.warning("Temporarily skipping validation")
            return {"retry_with_context": error_context.context_data}
        return None
    
    async def _free_unused_memory(self, error_context: ErrorContext) -> Optional[Any]:
        """Free unused memory."""
        logger.info("Freeing unused memory")
        # Implementation would call garbage collection and free caches
        return {"action": "free_memory"}
    
    async def _reduce_batch_size(self, error_context: ErrorContext) -> Optional[Any]:
        """Reduce batch size to use less memory."""
        if "batch_size" in error_context.context_data:
            current_batch = error_context.context_data["batch_size"]
            if current_batch > 1:
                error_context.context_data["batch_size"] = max(1, current_batch // 2)
                logger.info(f"Reducing batch size to {error_context.context_data['batch_size']}")
                return {"retry_with_context": error_context.context_data}
        return None
    
    async def _use_memory_mapped_files(self, error_context: ErrorContext) -> Optional[Any]:
        """Use memory-mapped files for large data."""
        error_context.context_data["use_mmap"] = True
        logger.info("Enabling memory-mapped file access")
        return {"retry_with_context": error_context.context_data}
    
    async def _retry_with_exponential_backoff(self, error_context: ErrorContext) -> Optional[Any]:
        """Retry with exponential backoff."""
        wait_time = min(60, 2 ** error_context.retry_count)
        logger.info(f"Retrying after {wait_time}s backoff")
        await asyncio.sleep(wait_time)
        return {"action": "retry"}
    
    async def _switch_to_backup_endpoint(self, error_context: ErrorContext) -> Optional[Any]:
        """Switch to backup network endpoint."""
        if "primary_endpoint" in error_context.context_data:
            error_context.context_data["use_backup_endpoint"] = True
            logger.info("Switching to backup endpoint")
            return {"retry_with_context": error_context.context_data}
        return None
    
    async def _use_cached_data(self, error_context: ErrorContext) -> Optional[Any]:
        """Use cached data if available."""
        error_context.context_data["use_cache"] = True
        logger.info("Attempting to use cached data")
        return {"retry_with_context": error_context.context_data}
    
    async def _sanitize_and_retry(self, error_context: ErrorContext) -> Optional[Any]:
        """Sanitize security-related input and retry."""
        # Remove potentially dangerous characters/patterns
        if "user_input" in error_context.context_data:
            user_input = str(error_context.context_data["user_input"])
            sanitized = "".join(c for c in user_input if c.isalnum() or c in " ._-")
            error_context.context_data["user_input"] = sanitized
            logger.info("Sanitized user input for security")
            return {"retry_with_context": error_context.context_data}
        return None
    
    async def _use_restricted_mode(self, error_context: ErrorContext) -> Optional[Any]:
        """Enable restricted mode with limited capabilities."""
        error_context.context_data["restricted_mode"] = True
        logger.warning("Enabling restricted mode")
        return {"retry_with_context": error_context.context_data}
    
    async def _reject_request(self, error_context: ErrorContext) -> Optional[Any]:
        """Reject request for security reasons."""
        logger.error("Rejecting request due to security concerns")
        return {"action": "reject", "reason": "security"}
    
    async def _validate_and_correct_input(self, error_context: ErrorContext) -> Optional[Any]:
        """Validate and correct user input."""
        if "input_data" in error_context.context_data:
            # Basic validation and correction
            input_data = error_context.context_data["input_data"]
            
            # Convert to expected type if possible
            if isinstance(input_data, str) and input_data.replace(".", "").replace("-", "").isdigit():
                try:
                    corrected = float(input_data)
                    error_context.context_data["input_data"] = corrected
                    logger.info("Corrected input data type")
                    return {"retry_with_context": error_context.context_data}
                except ValueError:
                    pass
        return None
    
    async def _use_default_values(self, error_context: ErrorContext) -> Optional[Any]:
        """Use default values for invalid input."""
        defaults = {
            "input_shape": [1, 3, 224, 224],
            "batch_size": 1,
            "dtype": "float32"
        }
        
        error_context.context_data.update(defaults)
        logger.info("Using default values for invalid input")
        return {"retry_with_context": error_context.context_data}
    
    async def _provide_user_guidance(self, error_context: ErrorContext) -> Optional[Any]:
        """Provide guidance to user for input correction."""
        guidance = {
            "error_type": error_context.category.value,
            "suggestion": "Please check your input parameters",
            "expected_format": "Numeric values in valid range"
        }
        
        logger.info("Providing user guidance for input correction")
        return {"action": "user_guidance", "guidance": guidance}
    
    def register_fallback_handler(self, operation: str, handler: Callable) -> None:
        """Register a custom fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
        logger.info(f"Registered fallback handler for {operation}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        categories = {}
        severities = {}
        
        for error in self.error_history:
            cat = error.category.value
            sev = error.severity.value
            
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "categories": categories,
            "severities": severities,
            "recent_errors": len([e for e in self.error_history if time.time() - e.timestamp < 3600]),
            "error_patterns": self.error_patterns
        }
    
    def export_error_report(self, file_path: str) -> None:
        """Export detailed error report to file."""
        report = {
            "timestamp": time.time(),
            "statistics": self.get_error_statistics(),
            "error_history": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp,
                    "operation": error.operation,
                    "severity": error.severity.value,
                    "category": error.category.value,
                    "message": error.message,
                    "retry_count": error.retry_count,
                    "recovery_attempted": error.recovery_attempted
                }
                for error in self.error_history
            ],
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Error report exported to {file_path}")


# Decorators for automatic error handling

def with_error_recovery(operation: str, max_retries: int = 3):
    """Decorator for automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            recovery_system = AdvancedErrorRecovery()
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    
                    context_data = {
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200]
                    }
                    
                    result = await recovery_system.handle_error(e, operation, context_data)
                    if result and result.get("action") == "retry":
                        continue
                    elif result and "retry_with_context" in result:
                        # Update function parameters based on recovery context
                        kwargs.update(result["retry_with_context"])
                        continue
                    else:
                        raise
            
            raise RuntimeError(f"Max retries exceeded for {operation}")
        
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@asynccontextmanager
async def resilient_operation(operation_name: str, context_data: Optional[Dict[str, Any]] = None):
    """Context manager for resilient operations."""
    recovery_system = AdvancedErrorRecovery()
    
    try:
        yield recovery_system
    except Exception as e:
        result = await recovery_system.handle_error(e, operation_name, context_data or {})
        if result is None or result.get("action") == "reject":
            raise
        # If recovery provides a result, operation should handle it appropriately


# Global error recovery instance
_global_recovery_system = None


def get_global_recovery_system() -> AdvancedErrorRecovery:
    """Get global error recovery system instance."""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = AdvancedErrorRecovery()
    return _global_recovery_system