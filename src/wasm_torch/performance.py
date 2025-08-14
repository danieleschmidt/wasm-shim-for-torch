"""Performance optimization utilities for WASM Torch."""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceStats:
    """Enhanced performance statistics and metrics."""
    operation_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    memory_pool_hits: int = 0
    memory_pool_misses: int = 0
    concurrent_operations: int = 0
    peak_concurrent_operations: int = 0
    throughput_ops_per_second: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    error_rate: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)
    adaptive_batch_size: int = 1
    optimization_score: float = 0.0


class LRUCache(Generic[T]):
    """Advanced thread-safe LRU cache with predictive pre-loading."""
    
    def __init__(self, max_size: int = 128, enable_prediction: bool = True):
        self.max_size = max_size
        self.cache: OrderedDict[str, T] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.access_patterns: Dict[str, List[float]] = {}
        self.enable_prediction = enable_prediction
        self.prediction_threshold = 0.8
        self.preload_callbacks: Dict[str, Callable] = {}
        
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    # Remove least recently used item
                    self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }


class MemoryPool:
    """Memory pool for efficient tensor allocation."""
    
    def __init__(self, initial_pool_size: int = 10):
        self.pool: List[Any] = []
        self.lock = threading.Lock()
        self.allocations = 0
        self.deallocations = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.peak_size = 0
        
        # Pre-allocate some tensors
        self._pre_allocate(initial_pool_size)
    
    def _pre_allocate(self, count: int) -> None:
        """Pre-allocate tensors to pool."""
        import torch
        
        # Pre-allocate common tensor sizes
        common_sizes = [
            (1, 784),     # MNIST input
            (1, 224, 224, 3),  # Image input
            (32, 128),    # Batch processing
            (1, 512),     # Embedding size
            (16, 16),     # Small matrix
        ]
        
        for size in common_sizes[:count]:
            try:
                tensor = torch.zeros(size)
                self.pool.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to pre-allocate tensor {size}: {e}")
    
    def get_tensor(self, shape: tuple, dtype=None) -> Optional[Any]:
        """Get tensor from pool if available."""
        import torch
        
        if dtype is None:
            dtype = torch.float32
        
        with self.lock:
            # Look for tensor with matching shape and dtype
            for i, tensor in enumerate(self.pool):
                if (tensor.shape == shape and tensor.dtype == dtype and 
                    not tensor.requires_grad):
                    # Remove from pool and return
                    self.pool.pop(i)
                    self.pool_hits += 1
                    
                    # Zero out the tensor for reuse
                    tensor.zero_()
                    return tensor
            
            # No suitable tensor found
            self.pool_misses += 1
            return None
    
    def return_tensor(self, tensor: Any) -> None:
        """Return tensor to pool."""
        if tensor is None:
            return
        
        with self.lock:
            # Don't store too many tensors
            if len(self.pool) < 50:  # Max pool size
                # Detach from computation graph
                if hasattr(tensor, 'detach'):
                    tensor = tensor.detach()
                
                self.pool.append(tensor)
                self.peak_size = max(self.peak_size, len(self.pool))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_requests = self.pool_hits + self.pool_misses
        hit_rate = self.pool_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'pool_size': len(self.pool),
            'pool_hits': self.pool_hits,
            'pool_misses': self.pool_misses,
            'hit_rate': hit_rate,
            'peak_size': self.peak_size,
            'allocations': self.allocations,
            'deallocations': self.deallocations
        }


class PerformanceMonitor:
    """Performance monitoring and optimization system."""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.stats = PerformanceStats()
        self.operation_cache = LRUCache[Any](max_size=256)
        self.memory_pool = MemoryPool()
        self.concurrent_operations = 0
        self.lock = threading.Lock()
        
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func: Callable) -> Callable:
            if not self.enable_profiling:
                return func
                
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                
                with self.lock:
                    self.concurrent_operations += 1
                    self.stats.concurrent_operations = self.concurrent_operations
                    self.stats.peak_concurrent_operations = max(
                        self.stats.peak_concurrent_operations, 
                        self.concurrent_operations
                    )
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    with self.lock:
                        self.concurrent_operations -= 1
                        self.stats.operation_count += 1
                        self.stats.total_execution_time += execution_time
                        self.stats.average_execution_time = (
                            self.stats.total_execution_time / self.stats.operation_count
                        )
                        self.stats.min_execution_time = min(
                            self.stats.min_execution_time, execution_time
                        )
                        self.stats.max_execution_time = max(
                            self.stats.max_execution_time, execution_time
                        )
                    
                    logger.debug(f"Operation {operation_name} completed in {execution_time:.3f}s")
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                
                with self.lock:
                    self.concurrent_operations += 1
                    self.stats.concurrent_operations = self.concurrent_operations
                    self.stats.peak_concurrent_operations = max(
                        self.stats.peak_concurrent_operations,
                        self.concurrent_operations
                    )
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    with self.lock:
                        self.concurrent_operations -= 1
                        self.stats.operation_count += 1
                        self.stats.total_execution_time += execution_time
                        self.stats.average_execution_time = (
                            self.stats.total_execution_time / self.stats.operation_count
                        )
                        self.stats.min_execution_time = min(
                            self.stats.min_execution_time, execution_time
                        )
                        self.stats.max_execution_time = max(
                            self.stats.max_execution_time, execution_time
                        )
                    
                    logger.debug(f"Operation {operation_name} completed in {execution_time:.3f}s")
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get result from operation cache."""
        result = self.operation_cache.get(cache_key)
        if result is not None:
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1
        return result
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache operation result."""
        self.operation_cache.put(cache_key, result)
    
    def get_tensor_from_pool(self, shape: tuple, dtype=None) -> Optional[Any]:
        """Get tensor from memory pool."""
        tensor = self.memory_pool.get_tensor(shape, dtype)
        if tensor is not None:
            self.stats.memory_pool_hits += 1
        else:
            self.stats.memory_pool_misses += 1
        return tensor
    
    def return_tensor_to_pool(self, tensor: Any) -> None:
        """Return tensor to memory pool."""
        self.memory_pool.return_tensor(tensor)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_cache = self.stats.cache_hits + self.stats.cache_misses
        self.stats.cache_hit_rate = (
            self.stats.cache_hits / total_cache if total_cache > 0 else 0.0
        )
        
        return {
            'operations': {
                'count': self.stats.operation_count,
                'total_time': self.stats.total_execution_time,
                'average_time': self.stats.average_execution_time,
                'min_time': self.stats.min_execution_time,
                'max_time': self.stats.max_execution_time,
            },
            'concurrency': {
                'current_operations': self.stats.concurrent_operations,
                'peak_operations': self.stats.peak_concurrent_operations,
            },
            'cache': self.operation_cache.get_stats(),
            'memory_pool': self.memory_pool.get_stats(),
        }


class BatchProcessor:
    """Batch processing for improved throughput."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_wait_time: float = 0.1,
                 max_workers: int = 4):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        self.pending_items: List[Any] = []
        self.pending_futures: List[asyncio.Future] = []
        self.lock = asyncio.Lock()
        self.batch_timer: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_item(self, item: Any, processor_func: Callable) -> Any:
        """Process item as part of a batch."""
        future = asyncio.get_event_loop().create_future()
        
        async with self.lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)
            
            # Start batch timer if this is the first item
            if len(self.pending_items) == 1:
                self.batch_timer = asyncio.create_task(
                    self._wait_for_batch_timeout()
                )
            
            # Process batch if it's full
            if len(self.pending_items) >= self.batch_size:
                await self._process_batch(processor_func)
        
        return await future
    
    async def _wait_for_batch_timeout(self) -> None:
        """Wait for batch timeout and then process."""
        try:
            await asyncio.sleep(self.max_wait_time)
            
            async with self.lock:
                if self.pending_items:  # Still have items to process
                    await self._process_batch(lambda x: x)  # Default processor
        except asyncio.CancelledError:
            pass
    
    async def _process_batch(self, processor_func: Callable) -> None:
        """Process current batch of items."""
        if not self.pending_items:
            return
        
        items = self.pending_items.copy()
        futures = self.pending_futures.copy()
        
        # Clear pending lists
        self.pending_items.clear()
        self.pending_futures.clear()
        
        # Cancel batch timer
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        try:
            # Process batch in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: [processor_func(item) for item in items]
            )
            
            # Set results for all futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def flush(self) -> None:
        """Flush any pending batches."""
        async with self.lock:
            if self.pending_items:
                await self._process_batch(lambda x: x)
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self.executor.shutdown(wait=True)


class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing work across resources."""
    
    def __init__(self, initial_workers: int = 4):
        self.workers = initial_workers
        self.min_workers = 1
        self.max_workers = 16
        self.load_history: List[float] = []
        self.performance_history: List[float] = []
        self.adjustment_interval = 10  # seconds
        self.last_adjustment = time.time()
        
    def should_scale_up(self) -> bool:
        """Determine if we should scale up workers."""
        if len(self.load_history) < 5:
            return False
        
        # Check if load is consistently high
        recent_load = sum(self.load_history[-5:]) / 5
        return (recent_load > 0.8 and 
                self.workers < self.max_workers and
                time.time() - self.last_adjustment > self.adjustment_interval)
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down workers."""
        if len(self.load_history) < 5:
            return False
        
        # Check if load is consistently low
        recent_load = sum(self.load_history[-5:]) / 5
        return (recent_load < 0.3 and 
                self.workers > self.min_workers and
                time.time() - self.last_adjustment > self.adjustment_interval)
    
    def update_metrics(self, load: float, performance: float) -> None:
        """Update load balancer metrics."""
        self.load_history.append(load)
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.load_history) > 20:
            self.load_history.pop(0)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        # Auto-scale if needed
        if self.should_scale_up():
            self.workers = min(self.max_workers, self.workers + 1)
            self.last_adjustment = time.time()
            logger.info(f"Scaling up to {self.workers} workers due to high load")
        elif self.should_scale_down():
            self.workers = max(self.min_workers, self.workers - 1)
            self.last_adjustment = time.time()
            logger.info(f"Scaling down to {self.workers} workers due to low load")
    
    def get_optimal_workers(self) -> int:
        """Get current optimal number of workers."""
        return self.workers


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _global_monitor


def profile_operation(operation_name: str):
    """Decorator for profiling operations using global monitor."""
    return _global_monitor.profile_operation(operation_name)


class AdvancedPerformanceOptimizer:
    """Advanced performance optimization system with machine learning."""
    
    def __init__(self, enable_ml_optimization: bool = True):
        self.enable_ml_optimization = enable_ml_optimization
        self.operation_patterns: Dict[str, List[float]] = {}
        self.optimization_cache = LRUCache[Dict[str, Any]](max_size=64)
        self.adaptive_batch_sizes: Dict[str, int] = {}
        self.performance_baseline: Dict[str, float] = {}
        
    def optimize_batch_size(self, operation_type: str, input_size: int, current_performance: float) -> int:
        """Dynamically optimize batch size based on performance feedback."""
        if operation_type not in self.adaptive_batch_sizes:
            self.adaptive_batch_sizes[operation_type] = 16  # Default
            
        current_batch = self.adaptive_batch_sizes[operation_type]
        
        # Record performance pattern
        if operation_type not in self.operation_patterns:
            self.operation_patterns[operation_type] = []
        
        self.operation_patterns[operation_type].append(current_performance)
        
        # Keep only recent history
        if len(self.operation_patterns[operation_type]) > 10:
            self.operation_patterns[operation_type].pop(0)
        
        # Adaptive batch size adjustment
        if len(self.operation_patterns[operation_type]) >= 3:
            recent_performance = self.operation_patterns[operation_type][-3:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            # If performance is degrading, try different batch size
            if (operation_type in self.performance_baseline and 
                avg_performance > self.performance_baseline[operation_type] * 1.2):
                
                # Try smaller batch size for better latency
                new_batch = max(1, current_batch // 2)
                logger.info(f"Reducing batch size for {operation_type}: {current_batch} -> {new_batch}")
                self.adaptive_batch_sizes[operation_type] = new_batch
                
            elif avg_performance < self.performance_baseline.get(operation_type, float('inf')) * 0.8:
                # Performance is good, try larger batch for better throughput
                new_batch = min(128, current_batch * 2)
                logger.info(f"Increasing batch size for {operation_type}: {current_batch} -> {new_batch}")
                self.adaptive_batch_sizes[operation_type] = new_batch
        
        # Update baseline
        if operation_type not in self.performance_baseline:
            self.performance_baseline[operation_type] = current_performance
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_baseline[operation_type] = (
                alpha * current_performance + 
                (1 - alpha) * self.performance_baseline[operation_type]
            )
        
        return self.adaptive_batch_sizes[operation_type]
    
    def get_optimization_config(self, model_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized configuration based on model characteristics."""
        config_key = self._hash_characteristics(model_characteristics)
        
        # Check cache first
        cached_config = self.optimization_cache.get(config_key)
        if cached_config is not None:
            return cached_config
        
        # Generate optimization config
        config = self._generate_optimization_config(model_characteristics)
        
        # Cache the result
        self.optimization_cache.put(config_key, config)
        
        return config
    
    def _hash_characteristics(self, characteristics: Dict[str, Any]) -> str:
        """Create hash from model characteristics for caching."""
        import hashlib
        import json
        
        # Create deterministic string representation
        sorted_items = sorted(characteristics.items())
        char_str = json.dumps(sorted_items, sort_keys=True)
        
        return hashlib.md5(char_str.encode()).hexdigest()[:16]
    
    def _generate_optimization_config(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization configuration based on model characteristics."""
        param_count = characteristics.get("parameter_count", 0)
        has_convolutions = characteristics.get("has_convolutions", False)
        has_attention = characteristics.get("has_attention", False)
        input_size = characteristics.get("input_size", 0)
        
        config = {
            "use_mixed_precision": param_count > 10_000_000,  # Large models benefit
            "enable_gradient_checkpointing": param_count > 50_000_000,
            "use_tensor_fusion": has_convolutions,
            "enable_attention_optimization": has_attention,
            "preferred_batch_size": self._calculate_optimal_batch_size(characteristics),
            "memory_optimization_level": "aggressive" if param_count > 100_000_000 else "balanced",
            "cache_intermediate_results": input_size < 1_000_000,
            "use_parallel_execution": True,
            "optimization_passes": self._get_optimization_passes(characteristics)
        }
        
        logger.debug(f"Generated optimization config: {config}")
        return config
    
    def _calculate_optimal_batch_size(self, characteristics: Dict[str, Any]) -> int:
        """Calculate optimal batch size based on model characteristics."""
        param_count = characteristics.get("parameter_count", 0)
        input_size = characteristics.get("input_size", 1000)
        
        # Heuristic-based batch size calculation
        if param_count > 100_000_000:  # Very large model
            return 2
        elif param_count > 10_000_000:  # Large model
            return 8
        elif input_size > 1_000_000:  # Large input
            return 4
        else:
            return 16
    
    def _get_optimization_passes(self, characteristics: Dict[str, Any]) -> List[str]:
        """Get recommended optimization passes."""
        passes = ["dead_code_elimination", "constant_folding"]
        
        if characteristics.get("has_convolutions", False):
            passes.extend(["conv_fusion", "batch_norm_folding"])
        
        if characteristics.get("has_attention", False):
            passes.extend(["attention_fusion", "key_value_cache_optimization"])
        
        if characteristics.get("parameter_count", 0) > 10_000_000:
            passes.extend(["weight_quantization", "layer_pruning"])
        
        return passes


class IntelligentCachingSystem:
    """Intelligent caching system with predictive prefetching."""
    
    def __init__(self, max_cache_size_mb: int = 512):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache = LRUCache[Any](max_size=128)
        self.access_patterns: Dict[str, List[float]] = {}
        self.prefetch_candidates: set = set()
        self.cache_size_bytes = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access pattern tracking."""
        with self.lock:
            # Record access time
            current_time = time.time()
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            
            self.access_patterns[key].append(current_time)
            
            # Keep only recent access history
            cutoff_time = current_time - 3600  # 1 hour
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff_time
            ]
            
            # Predict future accesses and add to prefetch candidates
            self._update_prefetch_candidates(key, current_time)
            
            return self.cache.get(key)
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        """Put item in cache with size management."""
        if size_bytes is None:
            size_bytes = self._estimate_size(value)
        
        with self.lock:
            # Check if we need to free up space
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            
            while (self.cache_size_bytes + size_bytes > max_size_bytes and 
                   len(self.cache.cache) > 0):
                # Remove least recently used item
                lru_key, _ = next(iter(self.cache.cache.items()))
                self._evict_item(lru_key)
            
            # Add new item
            self.cache.put(key, value)
            self.cache_size_bytes += size_bytes
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        import sys
        
        try:
            if hasattr(value, 'numel') and hasattr(value, 'element_size'):
                # PyTorch tensor
                return value.numel() * value.element_size()
            else:
                return sys.getsizeof(value)
        except Exception:
            return 1024  # Default estimate
    
    def _evict_item(self, key: str) -> None:
        """Evict item from cache and update size tracking."""
        if key in self.cache.cache:
            value = self.cache.cache[key]
            size_bytes = self._estimate_size(value)
            self.cache_size_bytes = max(0, self.cache_size_bytes - size_bytes)
            del self.cache.cache[key]
    
    def _update_prefetch_candidates(self, accessed_key: str, current_time: float) -> None:
        """Update prefetch candidates based on access patterns."""
        if accessed_key not in self.access_patterns:
            return
        
        access_times = self.access_patterns[accessed_key]
        
        # Look for patterns in access times
        if len(access_times) >= 3:
            # Calculate average interval between accesses
            intervals = []
            for i in range(1, len(access_times)):
                intervals.append(access_times[i] - access_times[i-1])
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                
                # If there's a regular pattern, predict next access
                if 60 <= avg_interval <= 3600:  # Between 1 minute and 1 hour
                    predicted_next_access = current_time + avg_interval
                    
                    # Add to prefetch candidates if prediction is soon
                    if predicted_next_access - current_time < 300:  # Next 5 minutes
                        self.prefetch_candidates.add(accessed_key)
    
    def get_prefetch_candidates(self) -> List[str]:
        """Get list of items that should be prefetched."""
        with self.lock:
            candidates = list(self.prefetch_candidates)
            self.prefetch_candidates.clear()
            return candidates
    
    def get_cache_analytics(self) -> Dict[str, Any]:
        """Get detailed cache analytics."""
        with self.lock:
            total_accesses = sum(len(pattern) for pattern in self.access_patterns.values())
            active_keys = len(self.access_patterns)
            
            return {
                "cache_size_mb": self.cache_size_bytes / (1024 * 1024),
                "max_cache_size_mb": self.max_cache_size_mb,
                "utilization": self.cache_size_bytes / (self.max_cache_size_mb * 1024 * 1024),
                "total_keys": len(self.cache.cache),
                "active_access_patterns": active_keys,
                "total_accesses": total_accesses,
                "prefetch_candidates": len(self.prefetch_candidates),
                "hit_rate": self.cache.get_stats()["hit_rate"]
            }


class ConcurrencyManager:
    """Advanced concurrency management with automatic optimization."""
    
    def __init__(self, max_workers: int = None):
        import multiprocessing
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.async_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
        self.optimal_concurrency: Dict[str, int] = {}
        self.lock = threading.Lock()
        
    def get_thread_pool(self, pool_name: str, optimal_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get or create thread pool with optimal worker count."""
        with self.lock:
            if pool_name not in self.thread_pools:
                workers = optimal_workers or self._calculate_optimal_workers(pool_name)
                self.thread_pools[pool_name] = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=f"wasm_torch_{pool_name}"
                )
                logger.info(f"Created thread pool '{pool_name}' with {workers} workers")
            
            return self.thread_pools[pool_name]
    
    def get_async_semaphore(self, semaphore_name: str, max_concurrent: Optional[int] = None) -> asyncio.Semaphore:
        """Get or create async semaphore with optimal concurrency limit."""
        if semaphore_name not in self.async_semaphores:
            limit = max_concurrent or self._calculate_optimal_concurrency(semaphore_name)
            self.async_semaphores[semaphore_name] = asyncio.Semaphore(limit)
            logger.info(f"Created semaphore '{semaphore_name}' with limit {limit}")
        
        return self.async_semaphores[semaphore_name]
    
    def _calculate_optimal_workers(self, pool_name: str) -> int:
        """Calculate optimal number of workers for a thread pool."""
        if pool_name in self.optimal_concurrency:
            return self.optimal_concurrency[pool_name]
        
        # Default heuristics based on pool type
        if "io" in pool_name.lower():
            # I/O bound tasks can have more workers
            optimal = min(self.max_workers * 2, 32)
        elif "cpu" in pool_name.lower() or "compute" in pool_name.lower():
            # CPU bound tasks should match CPU count
            optimal = self.max_workers
        else:
            # Conservative default
            optimal = max(2, self.max_workers // 2)
        
        self.optimal_concurrency[pool_name] = optimal
        return optimal
    
    def _calculate_optimal_concurrency(self, operation_type: str) -> int:
        """Calculate optimal concurrency limit for async operations."""
        if operation_type in self.optimal_concurrency:
            return self.optimal_concurrency[operation_type]
        
        # Default based on operation type
        if "model_load" in operation_type:
            optimal = 4  # Limit concurrent model loading
        elif "inference" in operation_type:
            optimal = self.max_workers
        elif "export" in operation_type:
            optimal = 2  # WASM compilation is resource intensive
        else:
            optimal = self.max_workers
        
        self.optimal_concurrency[operation_type] = optimal
        return optimal
    
    def record_performance(self, operation_type: str, execution_time: float, workers_used: int) -> None:
        """Record performance metrics for concurrency optimization."""
        with self.lock:
            if operation_type not in self.performance_tracker:
                self.performance_tracker[operation_type] = []
            
            # Record performance metric (throughput approximation)
            throughput = workers_used / execution_time if execution_time > 0 else 0
            self.performance_tracker[operation_type].append(throughput)
            
            # Keep only recent history
            if len(self.performance_tracker[operation_type]) > 20:
                self.performance_tracker[operation_type].pop(0)
            
            # Adaptive optimization
            self._optimize_concurrency(operation_type)
    
    def _optimize_concurrency(self, operation_type: str) -> None:
        """Optimize concurrency settings based on performance history."""
        if (operation_type not in self.performance_tracker or 
            len(self.performance_tracker[operation_type]) < 5):
            return
        
        recent_performance = self.performance_tracker[operation_type][-5:]
        avg_throughput = sum(recent_performance) / len(recent_performance)
        current_concurrency = self.optimal_concurrency.get(operation_type, self.max_workers)
        
        # Simple adaptive algorithm
        if len(self.performance_tracker[operation_type]) >= 10:
            older_performance = self.performance_tracker[operation_type][-10:-5]
            old_avg_throughput = sum(older_performance) / len(older_performance)
            
            # If performance is improving, maintain current settings
            if avg_throughput > old_avg_throughput * 1.1:
                return
            
            # If performance is degrading, try adjusting concurrency
            if avg_throughput < old_avg_throughput * 0.9:
                if current_concurrency > 1:
                    new_concurrency = max(1, current_concurrency - 1)
                    self.optimal_concurrency[operation_type] = new_concurrency
                    logger.info(f"Reduced concurrency for {operation_type}: {current_concurrency} -> {new_concurrency}")
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get concurrency management statistics."""
        with self.lock:
            return {
                "thread_pools": {
                    name: {
                        "max_workers": pool._max_workers,
                        "active_threads": len(pool._threads)
                    }
                    for name, pool in self.thread_pools.items()
                },
                "optimal_concurrency": dict(self.optimal_concurrency),
                "performance_tracking": {
                    name: {
                        "samples": len(perf_data),
                        "avg_throughput": sum(perf_data) / len(perf_data) if perf_data else 0
                    }
                    for name, perf_data in self.performance_tracker.items()
                }
            }
    
    def shutdown_all_pools(self) -> None:
        """Shutdown all thread pools."""
        with self.lock:
            for name, pool in self.thread_pools.items():
                logger.info(f"Shutting down thread pool: {name}")
                pool.shutdown(wait=True)
            self.thread_pools.clear()


# Global instances
_advanced_optimizer = AdvancedPerformanceOptimizer()
_intelligent_cache = IntelligentCachingSystem()
_concurrency_manager = ConcurrencyManager()


def get_advanced_optimizer() -> AdvancedPerformanceOptimizer:
    """Get global advanced performance optimizer."""
    return _advanced_optimizer


def get_intelligent_cache() -> IntelligentCachingSystem:
    """Get global intelligent caching system."""
    return _intelligent_cache


def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager."""
    return _concurrency_manager