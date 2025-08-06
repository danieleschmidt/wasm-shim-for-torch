#!/usr/bin/env python3
"""Test performance optimizations."""

import asyncio
import torch
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from wasm_torch.performance import get_performance_monitor, profile_operation, LRUCache, MemoryPool
from wasm_torch.runtime import WASMRuntime


@profile_operation("test_computation")
async def test_computation(x: torch.Tensor) -> torch.Tensor:
    """Test computation function."""
    # Simulate some processing
    await asyncio.sleep(0.01)
    return x * 2 + 1


async def test_performance_features():
    """Test various performance features."""
    print("🔥 Testing WASM Torch Performance Optimizations")
    print("=" * 50)
    
    # Test LRU Cache
    print("1. Testing LRU Cache...")
    cache = LRUCache[str](max_size=3)
    
    # Add items
    cache.put("key1", "value1")
    cache.put("key2", "value2") 
    cache.put("key3", "value3")
    
    # Test hits and misses
    result1 = cache.get("key1")  # Hit
    result2 = cache.get("key4")  # Miss
    
    stats = cache.get_stats()
    print(f"   Cache stats: {stats}")
    print(f"   ✅ Cache hit rate: {stats['hit_rate']:.2f}")
    
    # Test Memory Pool
    print("\n2. Testing Memory Pool...")
    pool = MemoryPool(initial_pool_size=5)
    
    # Get tensor from pool
    tensor = pool.get_tensor((10, 10))
    if tensor is not None:
        print(f"   ✅ Got tensor from pool: {tensor.shape}")
        pool.return_tensor(tensor)
    else:
        print("   ❌ No tensor available from pool")
    
    pool_stats = pool.get_stats()
    print(f"   Pool stats: {pool_stats}")
    
    # Test Performance Monitor
    print("\n3. Testing Performance Monitor...")
    monitor = get_performance_monitor()
    
    # Run some profiled operations
    for i in range(5):
        test_tensor = torch.randn(100, 100)
        result = await test_computation(test_tensor)
        print(f"   Operation {i+1} completed")
    
    comprehensive_stats = monitor.get_comprehensive_stats()
    print(f"   Performance stats: {comprehensive_stats['operations']}")
    print(f"   Cache stats: {comprehensive_stats['cache']}")
    
    # Test Runtime with Performance Features
    print("\n4. Testing Enhanced Runtime...")
    runtime = WASMRuntime(simd=True, threads=2, enable_monitoring=True)
    await runtime.init()
    
    # Get runtime stats with performance metrics
    runtime_stats = runtime.get_runtime_stats()
    print(f"   Runtime uptime: {runtime_stats['uptime_seconds']:.2f}s")
    print(f"   Load balancer workers: {runtime_stats['load_balancing']['current_workers']}")
    print(f"   Performance operations: {runtime_stats['performance']['operations']['count']}")
    
    await runtime.cleanup()
    
    print("\n✅ All performance tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_performance_features())