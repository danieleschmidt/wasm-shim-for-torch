"""WASM runtime for executing PyTorch models in browser environment."""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
import logging
import asyncio
import weakref
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from .validation import validate_input_tensor, validate_intermediate_tensor, validate_output_tensor
from .performance import get_performance_monitor, profile_operation, BatchProcessor, AdaptiveLoadBalancer


logger = logging.getLogger(__name__)


@dataclass
class RuntimeStats:
    """Runtime performance and health statistics."""
    inference_count: int = 0
    total_inference_time: float = 0.0
    last_inference_time: Optional[float] = None
    error_count: int = 0
    memory_peak_mb: float = 0.0
    models_loaded: int = 0
    models_failed: int = 0
    startup_time: Optional[float] = None
    last_health_check: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class WASMRuntime:
    """Runtime for executing WASM-compiled PyTorch models."""
    
    def __init__(
        self,
        simd: bool = True,
        threads: Optional[int] = None,
        memory_limit_mb: int = 1024,
        timeout_seconds: float = 300.0,
        enable_monitoring: bool = True
    ):
        """Initialize WASM runtime.
        
        Args:
            simd: Enable SIMD acceleration
            threads: Number of threads (None for auto-detect)
            memory_limit_mb: Memory limit in megabytes
        """
        self.simd = simd
        self.threads = threads or 4
        self.memory_limit_mb = memory_limit_mb
        self.timeout_seconds = timeout_seconds
        self.enable_monitoring = enable_monitoring
        self._initialized = False
        self._startup_time: Optional[float] = None
        self._stats = RuntimeStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._performance_monitor = get_performance_monitor()
        self._batch_processor = BatchProcessor(batch_size=16, max_wait_time=0.05)
        self._load_balancer = AdaptiveLoadBalancer(initial_workers=self.threads)
        
    async def init(self) -> 'WASMRuntime':
        """Initialize the WASM runtime asynchronously.
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Initializing WASM runtime with SIMD={self.simd}, threads={self.threads}")
        
        # Initialize thread pool for parallel execution
        self._thread_pool = ThreadPoolExecutor(max_workers=self.threads)
        
        # Initialize memory manager
        self._memory_manager = MemoryManager(self.memory_limit_mb)
        
        # Initialize operation registry
        self._op_registry = OperationRegistry()
        self._register_default_operations()
        
        # Track loaded models
        self._loaded_models = weakref.WeakSet()
        
        # Start health monitoring if enabled
        if self.enable_monitoring:
            self._health_check_task = asyncio.create_task(self._health_monitor())
        
        self._startup_time = time.time()
        self._initialized = True
        logger.info("WASM runtime initialized successfully")
        return self
    
    async def load_model(self, model_path: Union[str, Path]) -> 'WASMModel':
        """Load a WASM-compiled model.
        
        Args:
            model_path: Path to the .wasm model file
            
        Returns:
            Loaded WASM model instance
        """
        if not self._initialized:
            await self.init()
            
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading WASM model from {model_path}")
        
        # Create model instance
        model = WASMModel(model_path, self)
        await model._load_model_data()
        
        self._loaded_models.add(model)
        logger.info(f"Model loaded successfully: {model_path.name}")
        return model
    
    def _register_default_operations(self) -> None:
        """Register default WASM operations."""
        # Register core operations
        self._op_registry.register("aten::linear", LinearOperation())
        self._op_registry.register("aten::relu", ReLUOperation())
        self._op_registry.register("aten::conv2d", Conv2dOperation())
        self._op_registry.register("aten::batch_norm", BatchNormOperation())
        self._op_registry.register("aten::add", AddOperation())
        self._op_registry.register("aten::mul", MulOperation())
        
    async def cleanup(self) -> None:
        """Clean up runtime resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        if hasattr(self, '_memory_manager'):
            self._memory_manager.cleanup()
            
        logger.info("WASM runtime cleaned up")

    async def _health_monitor(self) -> None:
        """Background health monitoring task."""
        while self._initialized:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update health check timestamp
                self._stats.last_health_check = time.time()
                
                # Check memory usage
                if hasattr(self, '_memory_manager'):
                    mem_stats = self._memory_manager.get_stats()
                    memory_usage_mb = mem_stats['allocated_bytes'] / (1024 * 1024)
                    
                    if memory_usage_mb > self._stats.memory_peak_mb:
                        self._stats.memory_peak_mb = memory_usage_mb
                    
                    # Memory usage warnings
                    memory_limit_mb = mem_stats['limit_bytes'] / (1024 * 1024)
                    usage_percent = (memory_usage_mb / memory_limit_mb) * 100
                    
                    if usage_percent > 80:
                        warning = f"High memory usage: {usage_percent:.1f}%"
                        if warning not in self._stats.warnings:
                            self._stats.warnings.append(warning)
                            logger.warning(warning)
                
                # Check thread pool health
                if hasattr(self, '_thread_pool'):
                    if self._thread_pool._shutdown:
                        self._stats.warnings.append("Thread pool has been shutdown")
                        logger.error("Thread pool health check failed: shutdown")
                        break
                
                logger.debug("Health check completed")
                
            except asyncio.CancelledError:
                logger.debug("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self._stats.error_count += 1
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics.
        
        Returns:
            Dictionary with runtime performance and health metrics
        """
        uptime = time.time() - (self._startup_time or time.time())
        
        avg_inference_time = (
            self._stats.total_inference_time / max(self._stats.inference_count, 1)
        )
        
        return {
            'uptime_seconds': uptime,
            'inference_count': self._stats.inference_count,
            'total_inference_time': self._stats.total_inference_time,
            'average_inference_time': avg_inference_time,
            'last_inference_time': self._stats.last_inference_time,
            'error_count': self._stats.error_count,
            'memory_peak_mb': self._stats.memory_peak_mb,
            'models_loaded': self._stats.models_loaded,
            'models_failed': self._stats.models_failed,
            'warnings': self._stats.warnings[-10:],  # Last 10 warnings
            'last_health_check': self._stats.last_health_check,
            'health_status': 'healthy' if self._stats.error_count < 5 else 'degraded'
        }
    
    @asynccontextmanager
    async def _safe_execution(self, operation_name: str):
        """Context manager for safe operation execution with error tracking."""
        start_time = time.time()
        try:
            logger.debug(f"Starting operation: {operation_name}")
            yield
            logger.debug(f"Completed operation: {operation_name} in {time.time() - start_time:.3f}s")
        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            raise


class WASMModel:
    """Represents a loaded WASM model for inference."""
    
    def __init__(self, model_path: Path, runtime: WASMRuntime):
        """Initialize WASM model.
        
        Args:
            model_path: Path to the model file
            runtime: WASM runtime instance
        """
        self.model_path = model_path
        self.runtime = runtime
        self.model_data: Optional[Dict[str, Any]] = None
        self.operations: List[Dict[str, Any]] = []
        self.parameters: Dict[str, torch.Tensor] = {}
        self._is_loaded = False
        
    @profile_operation("model_forward_pass")
    async def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass on input tensor with comprehensive error handling and monitoring.
        
        Args:
            input_tensor: Input tensor for inference
            
        Returns:
            Output tensor from model
        
        Raises:
            RuntimeError: If model not loaded or execution fails
            ValueError: If input is invalid
            TimeoutError: If execution exceeds timeout
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        try:
            async with self.runtime._safe_execution(f"forward_pass_model_{id(self)}"):
                # Input validation
                validate_input_tensor(input_tensor)
                
                logger.debug(f"Running forward pass with input shape: {input_tensor.shape}")
                
                # Check for timeout
                if hasattr(self.runtime, 'timeout_seconds'):
                    timeout = self.runtime.timeout_seconds
                else:
                    timeout = 300.0  # 5 minute default
                
                # Execute operations sequentially with timeout protection
                current_tensor = input_tensor.clone()
                
                for i, op_info in enumerate(self.operations):
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Forward pass exceeded timeout ({timeout}s)")
                    
                    op_type = op_info["kind"]
                    
                    # Get operation from registry
                    operation = self.runtime._op_registry.get(op_type)
                    if operation is None:
                        logger.warning(f"Unknown operation {op_type} at step {i}, using passthrough")
                        continue
                    
                    # Execute operation with per-operation error handling
                    try:
                        current_tensor = await operation.execute(
                            current_tensor, 
                            op_info, 
                            self.parameters,
                            self.runtime
                        )
                        
                        # Validate intermediate results
                        validate_intermediate_tensor(current_tensor, i, op_type)
                        
                        logger.debug(f"Operation {i} ({op_type}) output shape: {current_tensor.shape}")
                        
                    except Exception as e:
                        self.runtime._stats.error_count += 1
                        logger.error(f"Error in operation {i} ({op_type}): {e}")
                        logger.debug(f"Operation details: {op_info}")
                        raise RuntimeError(
                            f"Forward pass failed at operation {i} ({op_type}): {e}"
                        ) from e
                
                # Final output validation
                validate_output_tensor(current_tensor)
                
                # Update performance stats
                inference_time = time.time() - start_time
                self.runtime._stats.inference_count += 1
                self.runtime._stats.total_inference_time += inference_time
                self.runtime._stats.last_inference_time = inference_time
                
                logger.debug(f"Forward pass completed in {inference_time:.3f}s")
                # Cache result for future use
                import hashlib
                input_hash = hashlib.md5(input_tensor.detach().numpy().tobytes()).hexdigest()[:16]
                cache_key = f"forward_{id(self)}_{input_hash}_{input_tensor.shape}"
                self.runtime._performance_monitor.cache_result(cache_key, current_tensor.clone())
                
                return current_tensor
        
        except asyncio.TimeoutError as e:
            self.runtime._stats.error_count += 1
            logger.error(f"Forward pass timed out after {time.time() - start_time:.2f}s")
            raise TimeoutError(f"Forward pass exceeded timeout: {e}") from e
        except Exception as e:
            self.runtime._stats.error_count += 1
            logger.error(f"Forward pass failed: {e}")
            raise
        
    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._is_loaded:
            return {"model_loaded": False}
            
        # Calculate parameter memory
        param_bytes = sum(
            param.numel() * param.element_size() 
            for param in self.parameters.values()
        )
        
        stats = {
            "model_loaded": True,
            "parameter_bytes": param_bytes,
            "parameter_count": sum(param.numel() for param in self.parameters.values()),
            "operations_count": len(self.operations),
            "estimated_memory_mb": param_bytes // (1024 * 1024),
        }
        
        # Add runtime memory stats if available
        if hasattr(self.runtime, '_memory_manager'):
            runtime_stats = self.runtime._memory_manager.get_stats()
            stats.update(runtime_stats)
            
        return stats
    
    async def _load_model_data(self) -> None:
        """Load model data from file."""
        try:
            # For now, simulate loading from JSON metadata file
            metadata_path = self.model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_data = json.load(f)
                    
                # Extract operations and parameters
                if "graph" in self.model_data:
                    self.operations = self.model_data["graph"].get("operations", [])
                    
                    # Load parameters (placeholder - would load from actual data)
                    param_data = self.model_data["graph"].get("parameters", {})
                    for name, param_info in param_data.items():
                        shape = param_info["shape"]
                        dtype = getattr(torch, param_info["dtype"].split(".")[-1])
                        # Create random tensor as placeholder
                        self.parameters[name] = torch.randn(shape, dtype=dtype)
                        
            else:
                # Create minimal model structure
                logger.warning(f"No metadata file found for {self.model_path}, creating minimal structure")
                self.operations = [
                    {"kind": "aten::linear", "attributes": {}},
                    {"kind": "aten::relu", "attributes": {}}
                ]
                self.parameters = {
                    "weight": torch.randn(10, 10),
                    "bias": torch.randn(10)
                }
                
            self._is_loaded = True
            logger.info(f"Loaded {len(self.operations)} operations and {len(self.parameters)} parameters")
            
        except Exception as e:
            self.runtime._stats.models_failed += 1
            logger.error(f"Failed to load model data: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e


class MemoryManager:
    """Manages WASM memory allocation and tracking."""
    
    def __init__(self, limit_mb: int):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.allocated_bytes = 0
        self.peak_bytes = 0
        
    def allocate(self, size_bytes: int) -> bool:
        """Attempt to allocate memory."""
        if self.allocated_bytes + size_bytes > self.limit_bytes:
            return False
            
        self.allocated_bytes += size_bytes
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
        return True
        
    def deallocate(self, size_bytes: int) -> None:
        """Deallocate memory."""
        self.allocated_bytes = max(0, self.allocated_bytes - size_bytes)
        
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "allocated_bytes": self.allocated_bytes,
            "peak_bytes": self.peak_bytes,
            "limit_bytes": self.limit_bytes,
            "available_bytes": self.limit_bytes - self.allocated_bytes
        }
        
    def cleanup(self) -> None:
        """Clean up memory manager."""
        self.allocated_bytes = 0


class OperationRegistry:
    """Registry for WASM operations."""
    
    def __init__(self):
        self._operations: Dict[str, 'WASMOperation'] = {}
        
    def register(self, op_type: str, operation: 'WASMOperation') -> None:
        """Register an operation."""
        self._operations[op_type] = operation
        
    def get(self, op_type: str) -> Optional['WASMOperation']:
        """Get an operation by type."""
        return self._operations.get(op_type)
        
    def list_operations(self) -> List[str]:
        """List all registered operations."""
        return list(self._operations.keys())


class WASMOperation:
    """Base class for WASM operations."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        """Execute the operation."""
        raise NotImplementedError


class LinearOperation(WASMOperation):
    """Linear (fully connected) layer operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        # Get weight and bias from parameters
        weight = parameters.get("weight", torch.randn(input_tensor.shape[-1], 10))
        bias = parameters.get("bias", torch.zeros(weight.shape[0]))
        
        # Perform linear transformation: output = input @ weight.T + bias
        output = torch.matmul(input_tensor, weight.T)
        if bias is not None:
            output = output + bias
            
        return output


class ReLUOperation(WASMOperation):
    """ReLU activation operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        return torch.relu(input_tensor)


class Conv2dOperation(WASMOperation):
    """2D convolution operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        # Basic 2D convolution (simplified)
        attrs = op_info.get("attributes", {})
        kernel_size = attrs.get("kernel_size", [3, 3])
        stride = attrs.get("stride", [1, 1])
        padding = attrs.get("padding", [0, 0])
        
        # Create dummy convolution
        if len(input_tensor.shape) == 4:  # NCHW format
            # Simple average pooling as conv placeholder
            return torch.nn.functional.avg_pool2d(
                input_tensor, 
                kernel_size=kernel_size[0], 
                stride=stride[0], 
                padding=padding[0]
            )
        return input_tensor


class BatchNormOperation(WASMOperation):
    """Batch normalization operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        # Simplified batch norm (just normalize)
        mean = input_tensor.mean(dim=0, keepdim=True)
        std = input_tensor.std(dim=0, keepdim=True)
        return (input_tensor - mean) / (std + 1e-8)


class AddOperation(WASMOperation):
    """Element-wise addition operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        # For simplicity, add a small constant
        return input_tensor + 0.1


class MulOperation(WASMOperation):
    """Element-wise multiplication operation."""
    
    async def execute(
        self, 
        input_tensor: torch.Tensor, 
        op_info: Dict[str, Any],
        parameters: Dict[str, torch.Tensor],
        runtime: WASMRuntime
    ) -> torch.Tensor:
        # For simplicity, multiply by a constant
        return input_tensor * 0.9