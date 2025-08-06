"""Tests for performance optimization features."""

import pytest
import asyncio
import torch
import time
from wasm_torch.performance import (
    LRUCache, MemoryPool, PerformanceMonitor, BatchProcessor,
    AdaptiveLoadBalancer, get_performance_monitor, profile_operation
)


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache[str](max_size=3)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
        assert stats['max_size'] == 3
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache[int](max_size=2)
        
        cache.put("key1", 1)
        cache.put("key2", 2)
        cache.put("key3", 3)  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == 2     # Still there
        assert cache.get("key3") == 3     # Still there
    
    def test_cache_lru_ordering(self):
        """Test LRU ordering."""
        cache = LRUCache[str](max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should be there


class TestMemoryPool:
    """Test memory pool functionality."""
    
    def test_memory_pool_basic(self):
        """Test basic memory pool operations."""
        pool = MemoryPool(initial_pool_size=3)
        
        # Pool should have some pre-allocated tensors
        stats = pool.get_stats()
        assert stats['pool_size'] >= 0
        
        # Try to get a tensor
        tensor = pool.get_tensor((1, 784))  # Common MNIST input size
        if tensor is not None:
            assert tensor.shape == (1, 784)
            # Return tensor to pool
            pool.return_tensor(tensor)
    
    def test_memory_pool_stats(self):
        """Test memory pool statistics."""
        pool = MemoryPool(initial_pool_size=2)
        
        # Get initial stats
        initial_stats = pool.get_stats()
        initial_size = initial_stats['pool_size']
        
        # Request a tensor
        tensor = pool.get_tensor((10, 10))
        
        stats = pool.get_stats()
        if tensor is not None:
            # Pool should have one less tensor
            assert stats['pool_hits'] == initial_stats['pool_hits'] + 1
            assert stats['pool_size'] == initial_size - 1
            
            # Return tensor
            pool.return_tensor(tensor)
            
            # Pool should be back to original size
            final_stats = pool.get_stats()
            assert final_stats['pool_size'] == initial_size
        else:
            # No suitable tensor found
            assert stats['pool_misses'] == initial_stats['pool_misses'] + 1


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_performance_profiling(self):
        """Test operation profiling."""
        monitor = PerformanceMonitor()
        
        @monitor.profile_operation("test_op")
        async def test_operation():
            await asyncio.sleep(0.01)
            return "result"
        
        # Run operation
        result = await test_operation()
        assert result == "result"
        
        # Check stats
        stats = monitor.get_comprehensive_stats()
        assert stats['operations']['count'] >= 1
        assert stats['operations']['total_time'] > 0
    
    def test_performance_caching(self):
        """Test performance monitor caching."""
        monitor = PerformanceMonitor()
        
        # Cache a result
        monitor.cache_result("test_key", "test_value")
        
        # Retrieve cached result
        cached = monitor.get_cached_result("test_key")
        assert cached == "test_value"
        
        # Check cache stats
        stats = monitor.get_comprehensive_stats()
        assert stats['cache']['hits'] >= 1
    
    def test_memory_pool_integration(self):
        """Test memory pool integration."""
        monitor = PerformanceMonitor()
        
        # Try to get tensor from pool
        tensor = monitor.get_tensor_from_pool((5, 5))
        
        # Check stats
        stats = monitor.get_comprehensive_stats()
        pool_stats = stats['memory_pool']
        
        if tensor is not None:
            assert pool_stats['pool_hits'] >= 1
            # Return tensor
            monitor.return_tensor_to_pool(tensor)
        else:
            assert pool_stats['pool_misses'] >= 1


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test basic batch processing."""
        processor = BatchProcessor(batch_size=3, max_wait_time=0.1)
        
        # Define a simple processor function
        def multiply_by_two(x):
            return x * 2
        
        # Process items
        tasks = []
        for i in range(3):
            task = processor.process_item(i, multiply_by_two)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Check results
        expected = [0, 2, 4]
        assert results == expected
        
        # Cleanup
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self):
        """Test batch processing with timeout."""
        processor = BatchProcessor(batch_size=10, max_wait_time=0.05)
        
        def simple_processor(x):
            return x + 1
        
        # Process single item (should timeout and process)
        start_time = time.time()
        result = await processor.process_item(5, simple_processor)
        end_time = time.time()
        
        assert result == 6
        assert end_time - start_time >= 0.04  # Should wait for timeout
        
        processor.shutdown()


class TestAdaptiveLoadBalancer:
    """Test adaptive load balancing."""
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization."""
        balancer = AdaptiveLoadBalancer(initial_workers=4)
        
        assert balancer.get_optimal_workers() == 4
        assert balancer.min_workers == 1
        assert balancer.max_workers == 16
    
    def test_load_balancer_scaling_up(self):
        """Test scaling up decision."""
        balancer = AdaptiveLoadBalancer(initial_workers=2)
        
        # Add high load history
        for _ in range(10):
            balancer.update_metrics(load=0.9, performance=100.0)
        
        # Should scale up due to high load
        final_workers = balancer.get_optimal_workers()
        assert final_workers > 2
    
    def test_load_balancer_scaling_down(self):
        """Test scaling down decision."""
        balancer = AdaptiveLoadBalancer(initial_workers=8)
        
        # Add low load history
        for _ in range(10):
            balancer.update_metrics(load=0.1, performance=50.0)
        
        # Should scale down due to low load
        final_workers = balancer.get_optimal_workers()
        assert final_workers < 8
    
    def test_load_balancer_limits(self):
        """Test scaling limits."""
        balancer = AdaptiveLoadBalancer(initial_workers=1)
        balancer.min_workers = 1
        balancer.max_workers = 2
        
        # Try to scale down below minimum
        for _ in range(20):
            balancer.update_metrics(load=0.0, performance=10.0)
        
        assert balancer.get_optimal_workers() >= balancer.min_workers
        
        # Try to scale up above maximum
        balancer.workers = 2
        for _ in range(20):
            balancer.update_metrics(load=1.0, performance=200.0)
        
        assert balancer.get_optimal_workers() <= balancer.max_workers


class TestGlobalPerformanceMonitor:
    """Test global performance monitor."""
    
    def test_global_monitor_singleton(self):
        """Test global monitor is singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
    
    @pytest.mark.asyncio
    async def test_global_profiling_decorator(self):
        """Test global profiling decorator."""
        
        @profile_operation("global_test")
        async def test_function():
            await asyncio.sleep(0.001)
            return "success"
        
        result = await test_function()
        assert result == "success"
        
        # Check that it was profiled
        monitor = get_performance_monitor()
        stats = monitor.get_comprehensive_stats()
        assert stats['operations']['count'] >= 1