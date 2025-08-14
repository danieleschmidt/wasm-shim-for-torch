"""Advanced error recovery mechanisms for WASM Torch operations."""

import logging
import time
import functools
import traceback
from typing import Optional, Callable, Any, Dict, List, Union
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        self.attempt_count = 0
        
    def should_retry(self, exception: Exception) -> bool:
        """Determine if the error should trigger a retry."""
        return self.attempt_count < self.max_attempts
    
    def on_attempt(self, attempt: int) -> None:
        """Called before each retry attempt."""
        self.attempt_count = attempt
        logger.info(f"Recovery strategy '{self.name}' attempt {attempt}/{self.max_attempts}")
    
    def on_success(self, result: Any) -> Any:
        """Called when operation succeeds."""
        logger.info(f"Recovery strategy '{self.name}' succeeded after {self.attempt_count} attempts")
        return result
    
    def on_failure(self, exception: Exception) -> None:
        """Called when all attempts fail."""
        logger.error(f"Recovery strategy '{self.name}' failed after {self.max_attempts} attempts: {exception}")


class ExponentialBackoffStrategy(RecoveryStrategy):
    """Retry with exponential backoff."""
    
    def __init__(self, name: str = "exponential_backoff", max_attempts: int = 3, 
                 base_delay: float = 1.0, max_delay: float = 30.0):
        super().__init__(name, max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def on_attempt(self, attempt: int) -> None:
        super().on_attempt(attempt)
        if attempt > 1:
            delay = min(self.base_delay * (2 ** (attempt - 2)), self.max_delay)
            logger.info(f"Waiting {delay:.1f}s before retry attempt {attempt}")
            time.sleep(delay)


class MemoryOptimizationStrategy(RecoveryStrategy):
    """Recovery strategy for memory-related errors."""
    
    def __init__(self, name: str = "memory_optimization", max_attempts: int = 2):
        super().__init__(name, max_attempts)
        self.original_settings = {}
    
    def should_retry(self, exception: Exception) -> bool:
        """Retry for memory-related errors."""
        error_msg = str(exception).lower()
        memory_indicators = ["out of memory", "cuda out of memory", "allocation failed", 
                           "memory error", "insufficient memory"]
        
        is_memory_error = any(indicator in error_msg for indicator in memory_indicators)
        return is_memory_error and super().should_retry(exception)
    
    def on_attempt(self, attempt: int) -> None:
        super().on_attempt(attempt)
        if attempt > 1:
            self._apply_memory_optimizations()
    
    def _apply_memory_optimizations(self) -> None:
        """Apply memory optimization techniques."""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA memory cache")
            
            # Garbage collection
            import gc
            gc.collect()
            logger.info("Triggered garbage collection")
            
            # Log memory usage if possible
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                logger.info(f"Memory usage: {memory_info.percent}% ({memory_info.used / 1024**3:.1f}GB used)")
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")


class ModelFallbackStrategy(RecoveryStrategy):
    """Fallback strategy for model-related errors."""
    
    def __init__(self, name: str = "model_fallback", max_attempts: int = 2):
        super().__init__(name, max_attempts)
        self.fallback_options = {
            "optimization_level": ["O3", "O2", "O1", "O0"],
            "use_simd": [True, False],
            "use_threads": [True, False],
            "quantization": [False, True]
        }
    
    def should_retry(self, exception: Exception) -> bool:
        """Retry for compilation or model export errors."""
        error_msg = str(exception).lower()
        compilation_indicators = ["compilation failed", "export failed", "unsupported operation",
                                "torch script", "tracing failed", "optimization failed"]
        
        is_compilation_error = any(indicator in error_msg for indicator in compilation_indicators)
        return is_compilation_error and super().should_retry(exception)
    
    def get_fallback_config(self, attempt: int) -> Dict[str, Any]:
        """Get fallback configuration for the given attempt."""
        config = {}
        
        if attempt == 2:
            config.update({
                "optimization_level": "O1",
                "use_simd": False
            })
        elif attempt >= 3:
            config.update({
                "optimization_level": "O0",
                "use_simd": False,
                "use_threads": False
            })
        
        logger.info(f"Using fallback configuration: {config}")
        return config


class ValidationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for validation errors."""
    
    def __init__(self, name: str = "validation_recovery", max_attempts: int = 2):
        super().__init__(name, max_attempts)
    
    def should_retry(self, exception: Exception) -> bool:
        """Retry for certain validation errors."""
        error_msg = str(exception).lower()
        recoverable_errors = ["nan values", "infinite values", "tensor too large",
                            "unsupported dtype", "shape mismatch"]
        
        is_recoverable = any(error in error_msg for error in recoverable_errors)
        return is_recoverable and super().should_retry(exception)
    
    def sanitize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply sanitization to recover from validation errors."""
        try:
            # Handle NaN and infinite values
            if torch.isnan(tensor).any():
                logger.warning("Replacing NaN values with zeros")
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            
            if torch.isinf(tensor).any():
                logger.warning("Clamping infinite values")
                tensor = torch.clamp(tensor, -1e6, 1e6)
            
            # Handle extreme values
            if tensor.dtype.is_floating_point:
                tensor_min, tensor_max = tensor.min(), tensor.max()
                if tensor_min < -1e6 or tensor_max > 1e6:
                    logger.warning(f"Clamping extreme values: min={tensor_min:.2e}, max={tensor_max:.2e}")
                    tensor = torch.clamp(tensor, -1e6, 1e6)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor sanitization failed: {e}")
            raise


def with_recovery(*strategies: RecoveryStrategy):
    """Decorator to add error recovery to functions.
    
    Args:
        *strategies: Recovery strategies to apply
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for strategy in strategies:
                strategy.attempt_count = 0
                
                for attempt in range(1, strategy.max_attempts + 1):
                    try:
                        strategy.on_attempt(attempt)
                        
                        # Apply strategy-specific modifications
                        modified_kwargs = kwargs.copy()
                        if isinstance(strategy, ModelFallbackStrategy) and attempt > 1:
                            fallback_config = strategy.get_fallback_config(attempt)
                            modified_kwargs.update(fallback_config)
                        
                        result = func(*args, **modified_kwargs)
                        return strategy.on_success(result)
                        
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Attempt {attempt} failed with {strategy.name}: {e}")
                        
                        if not strategy.should_retry(e):
                            break
                        
                        # Apply recovery actions
                        if isinstance(strategy, ValidationRecoveryStrategy):
                            # Try to sanitize inputs if they're tensors
                            for i, arg in enumerate(args):
                                if isinstance(arg, torch.Tensor):
                                    args = list(args)
                                    args[i] = strategy.sanitize_input(arg)
                                    args = tuple(args)
                
                strategy.on_failure(last_exception)
            
            # If all strategies failed, raise the last exception
            if last_exception:
                logger.error(f"All recovery strategies failed for {func.__name__}")
                raise last_exception
            
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, name: str = "default"):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        current_time = time.time()
        
        if self.state == "OPEN":
            if current_time - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                raise RuntimeError(f"Circuit breaker {self.name} is OPEN - operation blocked")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
            
            raise


# Global circuit breakers for common operations
export_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=120.0, name="export")
runtime_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0, name="runtime")
validation_circuit_breaker = CircuitBreaker(failure_threshold=10, timeout=30.0, name="validation")


class HealthMonitor:
    """Monitor system health and trigger recovery actions."""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_time = {}
        self.check_interval = 30.0  # seconds
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.last_check_time[name] = 0
    
    def check_health(self, name: str = None) -> Dict[str, bool]:
        """Run health checks and return results."""
        current_time = time.time()
        results = {}
        
        checks_to_run = [name] if name else self.health_checks.keys()
        
        for check_name in checks_to_run:
            if check_name not in self.health_checks:
                continue
                
            last_check = self.last_check_time.get(check_name, 0)
            if current_time - last_check < self.check_interval:
                continue
            
            try:
                result = self.health_checks[check_name]()
                results[check_name] = result
                self.last_check_time[check_name] = current_time
                
                if not result:
                    logger.warning(f"Health check '{check_name}' failed")
                else:
                    logger.debug(f"Health check '{check_name}' passed")
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' error: {e}")
                results[check_name] = False
        
        return results
    
    def is_healthy(self) -> bool:
        """Check if all systems are healthy."""
        results = self.check_health()
        return all(results.values()) if results else True


# Global health monitor
health_monitor = HealthMonitor()


def register_default_health_checks():
    """Register default health checks."""
    
    def memory_health_check() -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            return memory_info.percent < 90.0
        except ImportError:
            return True
    
    def disk_health_check() -> bool:
        """Check if disk space is sufficient."""
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            return free_gb > 1.0
        except Exception:
            return True
    
    def torch_health_check() -> bool:
        """Check if PyTorch is functioning correctly."""
        try:
            test_tensor = torch.randn(10, 10)
            return test_tensor.sum().item() is not None
        except Exception:
            return False
    
    health_monitor.register_health_check("memory", memory_health_check)
    health_monitor.register_health_check("disk", disk_health_check)
    health_monitor.register_health_check("torch", torch_health_check)


# Initialize default health checks
register_default_health_checks()


class ErrorAnalyzer:
    """Analyze errors and suggest recovery actions."""
    
    def __init__(self):
        self.error_patterns = {
            "memory": {
                "patterns": ["out of memory", "allocation failed", "cuda out of memory"],
                "suggestions": [
                    "Reduce batch size or model size",
                    "Enable gradient checkpointing",
                    "Use CPU instead of GPU",
                    "Clear cache and try again"
                ]
            },
            "compilation": {
                "patterns": ["compilation failed", "emscripten", "cmake error"],
                "suggestions": [
                    "Check Emscripten installation",
                    "Verify CMake version compatibility",
                    "Try lower optimization level",
                    "Check for unsupported operations"
                ]
            },
            "validation": {
                "patterns": ["nan values", "infinite values", "tensor", "shape"],
                "suggestions": [
                    "Check input data quality",
                    "Verify tensor shapes and types",
                    "Apply input normalization",
                    "Use different model architecture"
                ]
            },
            "export": {
                "patterns": ["torch script", "tracing failed", "export failed"],
                "suggestions": [
                    "Use torch.jit.script instead of trace",
                    "Simplify model architecture",
                    "Check for dynamic shapes",
                    "Use static input shapes"
                ]
            }
        }
    
    def analyze_error(self, exception: Exception) -> Dict[str, Any]:
        """Analyze an error and suggest recovery actions."""
        error_msg = str(exception).lower()
        error_type = type(exception).__name__
        
        analysis = {
            "error_type": error_type,
            "error_message": str(exception),
            "category": "unknown",
            "suggestions": [],
            "severity": "medium",
            "recoverable": False
        }
        
        # Categorize error
        for category, info in self.error_patterns.items():
            if any(pattern in error_msg for pattern in info["patterns"]):
                analysis["category"] = category
                analysis["suggestions"] = info["suggestions"]
                analysis["recoverable"] = True
                break
        
        # Determine severity
        critical_indicators = ["fatal", "critical", "system", "corruption"]
        if any(indicator in error_msg for indicator in critical_indicators):
            analysis["severity"] = "critical"
        elif "warning" in error_msg or analysis["recoverable"]:
            analysis["severity"] = "low"
        
        logger.info(f"Error analysis: {analysis['category']} error with {analysis['severity']} severity")
        return analysis


# Global error analyzer
error_analyzer = ErrorAnalyzer()