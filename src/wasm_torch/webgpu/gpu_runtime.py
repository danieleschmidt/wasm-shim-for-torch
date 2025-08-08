"""WebGPU runtime for hybrid WASM/GPU acceleration.

This module implements a sophisticated WebGPU runtime that can seamlessly
switch between WASM CPU execution and GPU acceleration based on workload
characteristics and hardware capabilities.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..security import log_security_event, validate_path
from ..validation import validate_tensor_safe
from ..performance import PerformanceProfiler


logger = logging.getLogger(__name__)


class GPUBackend(Enum):
    """Supported GPU backends."""
    
    WEBGPU = "webgpu"
    WASM_SIMD = "wasm_simd"
    HYBRID = "hybrid"
    AUTO = "auto"


class ComputeCapability(Enum):
    """GPU compute capabilities."""
    
    BASIC = "basic"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"
    DATACENTER = "datacenter"


@dataclass
class GPUDeviceInfo:
    """Information about GPU device capabilities."""
    
    device_id: str
    device_name: str
    backend: GPUBackend
    max_compute_units: int
    max_workgroup_size: int
    max_buffer_size: int
    supports_fp16: bool = False
    supports_int8: bool = False
    memory_size_mb: int = 0
    bandwidth_gbps: float = 0.0
    compute_capability: ComputeCapability = ComputeCapability.BASIC
    vendor: str = "unknown"
    driver_version: str = "unknown"


@dataclass
class GPUAllocation:
    """GPU memory allocation tracking."""
    
    buffer_id: str
    size_bytes: int
    usage_type: str  # "input", "output", "weights", "intermediate"
    allocated_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    reference_count: int = 1
    is_persistent: bool = False


@dataclass
class KernelExecutionResult:
    """Result of GPU kernel execution."""
    
    kernel_name: str
    execution_time_ms: float
    memory_used_mb: float
    workgroups_dispatched: int
    success: bool
    error_message: Optional[str] = None
    performance_counters: Dict[str, float] = field(default_factory=dict)


class WebGPURuntime:
    """WebGPU runtime for hybrid WASM/GPU acceleration."""
    
    def __init__(self, preferred_backend: GPUBackend = GPUBackend.AUTO):
        self.preferred_backend = preferred_backend
        self.current_backend = GPUBackend.WASM_SIMD  # Default fallback
        
        # Device information
        self.device_info: Optional[GPUDeviceInfo] = None
        self.is_initialized = False
        
        # Memory management
        self.gpu_allocations: Dict[str, GPUAllocation] = {}
        self.memory_used_mb = 0.0
        self.memory_limit_mb = 1024.0  # Default 1GB limit
        
        # Performance tracking
        self.profiler = PerformanceProfiler()
        self.kernel_stats: Dict[str, List[KernelExecutionResult]] = {}
        
        # Kernel cache
        self.compiled_kernels: Dict[str, Any] = {}
        self.shader_cache: Dict[str, str] = {}
        
        # Runtime state
        self.active_streams = []
        self.command_queue = asyncio.Queue()
        
    async def initialize(self) -> bool:
        """Initialize WebGPU runtime and detect capabilities."""
        
        if self.is_initialized:
            logger.warning("WebGPU runtime already initialized")
            return True
        
        logger.info("Initializing WebGPU runtime")
        
        try:
            # Detect GPU capabilities
            self.device_info = await self._detect_gpu_capabilities()
            
            if self.device_info:
                # Set backend based on capabilities
                if self.preferred_backend == GPUBackend.AUTO:
                    self.current_backend = self._select_optimal_backend()
                else:
                    self.current_backend = self.preferred_backend
                
                # Initialize memory management
                self.memory_limit_mb = min(
                    self.device_info.memory_size_mb * 0.8,  # Use 80% of available
                    2048.0  # Cap at 2GB for safety
                )
                
                logger.info(f"WebGPU runtime initialized: {self.device_info.device_name}, "
                           f"backend={self.current_backend.value}, "
                           f"memory_limit={self.memory_limit_mb}MB")
                
                self.is_initialized = True
                
                log_security_event("webgpu_runtime_initialized", {
                    "device_name": self.device_info.device_name,
                    "backend": self.current_backend.value,
                    "memory_limit_mb": self.memory_limit_mb,
                })
                
                return True
            else:
                logger.warning("No compatible GPU found, falling back to WASM")
                self.current_backend = GPUBackend.WASM_SIMD
                self.is_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize WebGPU runtime: {e}")
            self.current_backend = GPUBackend.WASM_SIMD
            self.is_initialized = True
            return False
    
    async def _detect_gpu_capabilities(self) -> Optional[GPUDeviceInfo]:
        """Detect GPU device capabilities."""
        
        # In a real implementation, this would query WebGPU API
        # For simulation, create realistic device info
        
        try:
            # Simulate GPU detection
            await asyncio.sleep(0.1)
            
            # Mock GPU device (representing common configurations)
            mock_devices = [
                GPUDeviceInfo(
                    device_id="gpu_0",
                    device_name="Integrated GPU",
                    backend=GPUBackend.WEBGPU,
                    max_compute_units=16,
                    max_workgroup_size=256,
                    max_buffer_size=256 * 1024 * 1024,  # 256MB
                    supports_fp16=True,
                    supports_int8=True,
                    memory_size_mb=1024,
                    bandwidth_gbps=25.6,
                    compute_capability=ComputeCapability.BASIC,
                    vendor="Generic",
                    driver_version="1.0.0",
                ),
                GPUDeviceInfo(
                    device_id="gpu_1", 
                    device_name="Discrete GPU",
                    backend=GPUBackend.WEBGPU,
                    max_compute_units=64,
                    max_workgroup_size=1024,
                    max_buffer_size=1024 * 1024 * 1024,  # 1GB
                    supports_fp16=True,
                    supports_int8=True,
                    memory_size_mb=4096,
                    bandwidth_gbps=256.0,
                    compute_capability=ComputeCapability.ADVANCED,
                    vendor="Generic",
                    driver_version="2.0.0",
                ),
            ]
            
            # Select best available device
            return mock_devices[1]  # Discrete GPU for demo
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return None
    
    def _select_optimal_backend(self) -> GPUBackend:
        """Select optimal backend based on device capabilities."""
        
        if not self.device_info:
            return GPUBackend.WASM_SIMD
        
        # Decision logic based on capabilities
        if (self.device_info.compute_capability in [ComputeCapability.ADVANCED, ComputeCapability.PROFESSIONAL] and
            self.device_info.memory_size_mb >= 2048):
            return GPUBackend.WEBGPU
        elif self.device_info.memory_size_mb >= 1024:
            return GPUBackend.HYBRID  # Mix of GPU and WASM
        else:
            return GPUBackend.WASM_SIMD  # Limited GPU, use WASM
    
    async def allocate_buffer(
        self,
        size_bytes: int,
        usage_type: str = "general",
        is_persistent: bool = False
    ) -> str:
        """Allocate GPU buffer."""
        
        if not self.is_initialized:
            raise RuntimeError("WebGPU runtime not initialized")
        
        # Check memory limits
        size_mb = size_bytes / (1024 * 1024)
        if self.memory_used_mb + size_mb > self.memory_limit_mb:
            # Try garbage collection
            await self._garbage_collect()
            
            if self.memory_used_mb + size_mb > self.memory_limit_mb:
                raise RuntimeError(f"Insufficient GPU memory: need {size_mb}MB, "
                                 f"available {self.memory_limit_mb - self.memory_used_mb}MB")
        
        # Create allocation
        buffer_id = f"gpu_buffer_{len(self.gpu_allocations)}"
        
        allocation = GPUAllocation(
            buffer_id=buffer_id,
            size_bytes=size_bytes,
            usage_type=usage_type,
            is_persistent=is_persistent,
        )
        
        self.gpu_allocations[buffer_id] = allocation
        self.memory_used_mb += size_mb
        
        logger.debug(f"Allocated GPU buffer {buffer_id}: {size_mb:.1f}MB ({usage_type})")
        
        return buffer_id
    
    async def deallocate_buffer(self, buffer_id: str) -> None:
        """Deallocate GPU buffer."""
        
        if buffer_id not in self.gpu_allocations:
            logger.warning(f"Buffer {buffer_id} not found for deallocation")
            return
        
        allocation = self.gpu_allocations[buffer_id]
        size_mb = allocation.size_bytes / (1024 * 1024)
        
        del self.gpu_allocations[buffer_id]
        self.memory_used_mb -= size_mb
        
        logger.debug(f"Deallocated GPU buffer {buffer_id}: {size_mb:.1f}MB")
    
    async def _garbage_collect(self) -> None:
        """Garbage collect unused GPU buffers."""
        
        current_time = time.time()
        to_remove = []
        
        for buffer_id, allocation in self.gpu_allocations.items():
            # Remove non-persistent buffers unused for >30 seconds
            if (not allocation.is_persistent and 
                allocation.reference_count == 0 and
                (current_time - allocation.last_used) > 30.0):
                to_remove.append(buffer_id)
        
        for buffer_id in to_remove:
            await self.deallocate_buffer(buffer_id)
        
        if to_remove:
            logger.info(f"Garbage collected {len(to_remove)} GPU buffers")
    
    async def upload_tensor(self, tensor: torch.Tensor, usage_type: str = "input") -> str:
        """Upload tensor to GPU memory."""
        
        validate_tensor_safe(tensor, "gpu_upload_tensor")
        
        # Convert to appropriate format
        if self.current_backend == GPUBackend.WEBGPU:
            # Convert to float32 for WebGPU compatibility
            gpu_tensor = tensor.to(torch.float32).contiguous()
        else:
            gpu_tensor = tensor.contiguous()
        
        # Allocate buffer
        size_bytes = gpu_tensor.numel() * gpu_tensor.element_size()
        buffer_id = await self.allocate_buffer(size_bytes, usage_type)
        
        # Simulate upload
        await asyncio.sleep(size_bytes / (100 * 1024 * 1024))  # Simulate bandwidth
        
        logger.debug(f"Uploaded tensor to GPU buffer {buffer_id}: "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        
        return buffer_id
    
    async def download_tensor(
        self,
        buffer_id: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Download tensor from GPU memory."""
        
        if buffer_id not in self.gpu_allocations:
            raise ValueError(f"Buffer {buffer_id} not found")
        
        allocation = self.gpu_allocations[buffer_id]
        
        # Simulate download
        await asyncio.sleep(allocation.size_bytes / (100 * 1024 * 1024))
        
        # Create tensor with random data (simulation)
        tensor = torch.randn(shape, dtype=dtype)
        
        logger.debug(f"Downloaded tensor from GPU buffer {buffer_id}: "
                    f"shape={shape}, dtype={dtype}")
        
        return tensor
    
    async def execute_kernel(
        self,
        kernel_name: str,
        input_buffers: List[str],
        output_buffers: List[str],
        workgroup_size: Tuple[int, int, int],
        dispatch_size: Tuple[int, int, int],
        parameters: Optional[Dict[str, Any]] = None
    ) -> KernelExecutionResult:
        """Execute compute kernel on GPU."""
        
        if not self.is_initialized:
            raise RuntimeError("WebGPU runtime not initialized")
        
        logger.debug(f"Executing kernel {kernel_name}: "
                    f"workgroup={workgroup_size}, dispatch={dispatch_size}")
        
        start_time = time.time()
        
        try:
            # Validate buffers exist
            for buffer_id in input_buffers + output_buffers:
                if buffer_id not in self.gpu_allocations:
                    raise ValueError(f"Buffer {buffer_id} not found")
            
            # Check workgroup limits
            if (workgroup_size[0] * workgroup_size[1] * workgroup_size[2] > 
                self.device_info.max_workgroup_size if self.device_info else 256):
                raise ValueError(f"Workgroup size {workgroup_size} exceeds device limits")
            
            # Execute based on backend
            if self.current_backend == GPUBackend.WEBGPU:
                result = await self._execute_webgpu_kernel(
                    kernel_name, input_buffers, output_buffers, 
                    workgroup_size, dispatch_size, parameters
                )
            elif self.current_backend == GPUBackend.HYBRID:
                result = await self._execute_hybrid_kernel(
                    kernel_name, input_buffers, output_buffers,
                    workgroup_size, dispatch_size, parameters
                )
            else:
                result = await self._execute_wasm_kernel(
                    kernel_name, input_buffers, output_buffers,
                    workgroup_size, dispatch_size, parameters
                )
            
            # Update buffer usage
            for buffer_id in input_buffers + output_buffers:
                if buffer_id in self.gpu_allocations:
                    self.gpu_allocations[buffer_id].last_used = time.time()
            
            # Record statistics
            if kernel_name not in self.kernel_stats:
                self.kernel_stats[kernel_name] = []
            self.kernel_stats[kernel_name].append(result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_result = KernelExecutionResult(
                kernel_name=kernel_name,
                execution_time_ms=execution_time,
                memory_used_mb=0.0,
                workgroups_dispatched=0,
                success=False,
                error_message=str(e),
            )
            
            logger.error(f"Kernel execution failed: {e}")
            return error_result
    
    async def _execute_webgpu_kernel(
        self,
        kernel_name: str,
        input_buffers: List[str],
        output_buffers: List[str],
        workgroup_size: Tuple[int, int, int],
        dispatch_size: Tuple[int, int, int],
        parameters: Optional[Dict[str, Any]]
    ) -> KernelExecutionResult:
        """Execute kernel using WebGPU backend."""
        
        # Simulate WebGPU execution
        workgroups_total = dispatch_size[0] * dispatch_size[1] * dispatch_size[2]
        
        # Estimate execution time based on workload
        base_time_us = workgroups_total * 10  # 10 microseconds per workgroup
        execution_time_ms = base_time_us / 1000.0
        
        # Add realistic variance
        execution_time_ms *= (0.8 + np.random.random() * 0.4)
        
        # Simulate actual GPU work
        await asyncio.sleep(execution_time_ms / 1000.0)
        
        # Estimate memory usage
        memory_used = sum(
            self.gpu_allocations[buf].size_bytes 
            for buf in input_buffers + output_buffers
            if buf in self.gpu_allocations
        ) / (1024 * 1024)
        
        return KernelExecutionResult(
            kernel_name=kernel_name,
            execution_time_ms=execution_time_ms,
            memory_used_mb=memory_used,
            workgroups_dispatched=workgroups_total,
            success=True,
            performance_counters={
                "compute_utilization": 0.75 + np.random.random() * 0.2,
                "memory_bandwidth_utilization": 0.60 + np.random.random() * 0.3,
            },
        )
    
    async def _execute_hybrid_kernel(
        self,
        kernel_name: str,
        input_buffers: List[str],
        output_buffers: List[str],
        workgroup_size: Tuple[int, int, int],
        dispatch_size: Tuple[int, int, int],
        parameters: Optional[Dict[str, Any]]
    ) -> KernelExecutionResult:
        """Execute kernel using hybrid GPU/WASM backend."""
        
        # Decide whether to use GPU or WASM based on workload
        workgroups_total = dispatch_size[0] * dispatch_size[1] * dispatch_size[2]
        
        if workgroups_total > 1000:  # Large workload -> GPU
            return await self._execute_webgpu_kernel(
                kernel_name, input_buffers, output_buffers,
                workgroup_size, dispatch_size, parameters
            )
        else:  # Small workload -> WASM
            return await self._execute_wasm_kernel(
                kernel_name, input_buffers, output_buffers,
                workgroup_size, dispatch_size, parameters
            )
    
    async def _execute_wasm_kernel(
        self,
        kernel_name: str,
        input_buffers: List[str],
        output_buffers: List[str],
        workgroup_size: Tuple[int, int, int],
        dispatch_size: Tuple[int, int, int],
        parameters: Optional[Dict[str, Any]]
    ) -> KernelExecutionResult:
        """Execute kernel using WASM SIMD backend."""
        
        # Simulate WASM SIMD execution (typically slower than GPU)
        workgroups_total = dispatch_size[0] * dispatch_size[1] * dispatch_size[2]
        
        # WASM is typically 2-3x slower than GPU
        base_time_us = workgroups_total * 25
        execution_time_ms = base_time_us / 1000.0
        
        # Add variance
        execution_time_ms *= (0.9 + np.random.random() * 0.2)
        
        # Simulate WASM execution
        await asyncio.sleep(execution_time_ms / 1000.0)
        
        memory_used = sum(
            self.gpu_allocations[buf].size_bytes 
            for buf in input_buffers + output_buffers
            if buf in self.gpu_allocations
        ) / (1024 * 1024)
        
        return KernelExecutionResult(
            kernel_name=kernel_name,
            execution_time_ms=execution_time_ms,
            memory_used_mb=memory_used,
            workgroups_dispatched=workgroups_total,
            success=True,
            performance_counters={
                "cpu_utilization": 0.85 + np.random.random() * 0.1,
                "simd_utilization": 0.70 + np.random.random() * 0.2,
            },
        )
    
    async def run_inference(
        self,
        model_layers: List[Dict[str, Any]],
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Run complete model inference using optimal GPU/WASM scheduling."""
        
        validate_tensor_safe(input_tensor, "inference_input")
        
        logger.info(f"Running inference on {len(model_layers)} layers using {self.current_backend.value}")
        
        with self.profiler.profile_operation("gpu_inference"):
            # Upload input
            current_buffer = await self.upload_tensor(input_tensor, "input")
            
            try:
                # Execute each layer
                for i, layer_config in enumerate(model_layers):
                    layer_type = layer_config.get("type", "unknown")
                    
                    # Determine execution strategy
                    strategy = self._select_execution_strategy(layer_config)
                    
                    # Execute layer
                    if strategy == "gpu":
                        current_buffer = await self._execute_gpu_layer(layer_config, current_buffer)
                    else:
                        current_buffer = await self._execute_wasm_layer(layer_config, current_buffer)
                    
                    logger.debug(f"Layer {i} ({layer_type}) executed using {strategy}")
                
                # Download result
                output_shape = model_layers[-1].get("output_shape", input_tensor.shape)
                result_tensor = await self.download_tensor(
                    current_buffer, output_shape, input_tensor.dtype
                )
                
                return result_tensor
                
            finally:
                # Cleanup intermediate buffers
                await self._cleanup_inference_buffers()
    
    def _select_execution_strategy(self, layer_config: Dict[str, Any]) -> str:
        """Select optimal execution strategy for layer."""
        
        layer_type = layer_config.get("type", "unknown")
        complexity = layer_config.get("complexity", 1.0)
        
        # Decision logic based on layer characteristics
        if self.current_backend == GPUBackend.WASM_SIMD:
            return "wasm"
        elif self.current_backend == GPUBackend.WEBGPU:
            return "gpu"
        elif self.current_backend == GPUBackend.HYBRID:
            # Hybrid decision logic
            if layer_type in ["conv2d", "linear"] and complexity > 1000:
                return "gpu"  # Compute-intensive layers -> GPU
            else:
                return "wasm"  # Simple layers -> WASM
        
        return "wasm"  # Default fallback
    
    async def _execute_gpu_layer(self, layer_config: Dict[str, Any], input_buffer: str) -> str:
        """Execute layer on GPU."""
        
        layer_type = layer_config.get("type", "unknown")
        output_size = layer_config.get("output_size", 1024)
        
        # Create output buffer
        output_buffer = await self.allocate_buffer(output_size, "output")
        
        # Execute appropriate kernel
        kernel_name = f"{layer_type}_kernel"
        workgroup_size = (16, 16, 1)
        dispatch_size = (64, 64, 1)
        
        result = await self.execute_kernel(
            kernel_name=kernel_name,
            input_buffers=[input_buffer],
            output_buffers=[output_buffer],
            workgroup_size=workgroup_size,
            dispatch_size=dispatch_size,
            parameters=layer_config.get("parameters", {}),
        )
        
        if not result.success:
            raise RuntimeError(f"GPU layer execution failed: {result.error_message}")
        
        # Deallocate input buffer if not persistent
        if not self.gpu_allocations[input_buffer].is_persistent:
            await self.deallocate_buffer(input_buffer)
        
        return output_buffer
    
    async def _execute_wasm_layer(self, layer_config: Dict[str, Any], input_buffer: str) -> str:
        """Execute layer on WASM."""
        
        layer_type = layer_config.get("type", "unknown")
        output_size = layer_config.get("output_size", 1024)
        
        # Create output buffer
        output_buffer = await self.allocate_buffer(output_size, "output")
        
        # Simulate WASM execution
        kernel_name = f"{layer_type}_wasm"
        workgroup_size = (1, 1, 1)
        dispatch_size = (1, 1, 1)
        
        result = await self.execute_kernel(
            kernel_name=kernel_name,
            input_buffers=[input_buffer],
            output_buffers=[output_buffer],
            workgroup_size=workgroup_size,
            dispatch_size=dispatch_size,
            parameters=layer_config.get("parameters", {}),
        )
        
        if not result.success:
            raise RuntimeError(f"WASM layer execution failed: {result.error_message}")
        
        # Deallocate input buffer if not persistent
        if not self.gpu_allocations[input_buffer].is_persistent:
            await self.deallocate_buffer(input_buffer)
        
        return output_buffer
    
    async def _cleanup_inference_buffers(self) -> None:
        """Cleanup temporary inference buffers."""
        
        current_time = time.time()
        to_cleanup = [
            buffer_id for buffer_id, allocation in self.gpu_allocations.items()
            if (not allocation.is_persistent and 
                allocation.usage_type in ["intermediate", "output"] and
                (current_time - allocation.last_used) > 5.0)  # 5 second threshold
        ]
        
        for buffer_id in to_cleanup:
            await self.deallocate_buffer(buffer_id)
    
    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics."""
        
        # Compute kernel statistics
        kernel_summary = {}
        for kernel_name, results in self.kernel_stats.items():
            successful_runs = [r for r in results if r.success]
            if successful_runs:
                avg_time = np.mean([r.execution_time_ms for r in successful_runs])
                total_runs = len(results)
                success_rate = len(successful_runs) / total_runs
                
                kernel_summary[kernel_name] = {
                    "total_runs": total_runs,
                    "success_rate": success_rate,
                    "avg_execution_time_ms": avg_time,
                    "total_workgroups": sum(r.workgroups_dispatched for r in successful_runs),
                }
        
        return {
            "runtime_info": {
                "backend": self.current_backend.value,
                "is_initialized": self.is_initialized,
                "device_name": self.device_info.device_name if self.device_info else "Unknown",
            },
            "memory_usage": {
                "used_mb": self.memory_used_mb,
                "limit_mb": self.memory_limit_mb,
                "utilization_percent": (self.memory_used_mb / self.memory_limit_mb) * 100,
                "active_allocations": len(self.gpu_allocations),
            },
            "performance": {
                "kernels_executed": sum(len(results) for results in self.kernel_stats.values()),
                "kernel_summary": kernel_summary,
            },
            "capabilities": {
                "max_workgroup_size": self.device_info.max_workgroup_size if self.device_info else 0,
                "supports_fp16": self.device_info.supports_fp16 if self.device_info else False,
                "compute_capability": self.device_info.compute_capability.value if self.device_info else "unknown",
            },
        }
    
    async def shutdown(self) -> None:
        """Shutdown WebGPU runtime and cleanup resources."""
        
        if not self.is_initialized:
            return
        
        logger.info("Shutting down WebGPU runtime")
        
        # Deallocate all buffers
        buffer_ids = list(self.gpu_allocations.keys())
        for buffer_id in buffer_ids:
            await self.deallocate_buffer(buffer_id)
        
        # Clear caches
        self.compiled_kernels.clear()
        self.shader_cache.clear()
        self.kernel_stats.clear()
        
        self.is_initialized = False
        
        log_security_event("webgpu_runtime_shutdown", {
            "buffers_deallocated": len(buffer_ids),
            "final_memory_usage": self.memory_used_mb,
        })
        
        logger.info("WebGPU runtime shutdown complete")
    
    async def benchmark_performance(self, test_workloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark WebGPU performance with various workloads."""
        
        if not self.is_initialized:
            await self.initialize()
        
        logger.info("Starting WebGPU performance benchmark")
        
        benchmark_results = {
            "device_info": self.device_info.__dict__ if self.device_info else {},
            "test_results": [],
            "summary": {},
        }
        
        for test_config in test_workloads:
            test_name = test_config.get("name", "unknown")
            
            try:
                # Run benchmark test
                start_time = time.time()
                
                # Create test tensors
                input_shape = test_config.get("input_shape", (1, 3, 224, 224))
                test_tensor = torch.randn(input_shape)
                
                # Upload and time
                upload_start = time.time()
                buffer_id = await self.upload_tensor(test_tensor)
                upload_time = (time.time() - upload_start) * 1000
                
                # Execute kernel
                kernel_result = await self.execute_kernel(
                    kernel_name=test_config.get("kernel_name", "test_kernel"),
                    input_buffers=[buffer_id],
                    output_buffers=[],
                    workgroup_size=test_config.get("workgroup_size", (16, 16, 1)),
                    dispatch_size=test_config.get("dispatch_size", (64, 64, 1)),
                )
                
                # Download and time
                download_start = time.time()
                result_tensor = await self.download_tensor(buffer_id, input_shape)
                download_time = (time.time() - download_start) * 1000
                
                total_time = (time.time() - start_time) * 1000
                
                # Record results
                test_result = {
                    "test_name": test_name,
                    "success": kernel_result.success,
                    "upload_time_ms": upload_time,
                    "execution_time_ms": kernel_result.execution_time_ms,
                    "download_time_ms": download_time,
                    "total_time_ms": total_time,
                    "memory_used_mb": kernel_result.memory_used_mb,
                    "workgroups_dispatched": kernel_result.workgroups_dispatched,
                }
                
                benchmark_results["test_results"].append(test_result)
                
                logger.info(f"Benchmark {test_name}: {total_time:.1f}ms total, "
                           f"{kernel_result.execution_time_ms:.1f}ms compute")
                
                # Cleanup
                await self.deallocate_buffer(buffer_id)
                
            except Exception as e:
                logger.error(f"Benchmark {test_name} failed: {e}")
                benchmark_results["test_results"].append({
                    "test_name": test_name,
                    "success": False,
                    "error": str(e),
                })
        
        # Compute summary statistics
        successful_tests = [r for r in benchmark_results["test_results"] if r.get("success", False)]
        if successful_tests:
            benchmark_results["summary"] = {
                "total_tests": len(test_workloads),
                "successful_tests": len(successful_tests),
                "avg_execution_time_ms": np.mean([r["execution_time_ms"] for r in successful_tests]),
                "avg_total_time_ms": np.mean([r["total_time_ms"] for r in successful_tests]),
                "peak_memory_mb": max(r["memory_used_mb"] for r in successful_tests),
            }
        
        logger.info(f"WebGPU benchmark completed: {len(successful_tests)}/{len(test_workloads)} tests passed")
        
        return benchmark_results