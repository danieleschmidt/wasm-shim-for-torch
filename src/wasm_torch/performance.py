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
    """Performance statistics and metrics."""
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


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with performance monitoring."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: OrderedDict[str, T] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
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