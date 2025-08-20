"""Advanced scaling and performance optimization for WASM-Torch production systems."""

import asyncio
import logging
import time
import threading
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
from pathlib import Path
try:
    import psutil
except ImportError:
    from .mock_dependencies import psutil
import weakref

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for distributed execution."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LATENCY_AWARE = "latency_aware"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    SIZE_AWARE = "size_aware"  # Evict based on memory usage


@dataclass
class WorkerStats:
    """Statistics for a worker instance."""
    worker_id: str
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    last_request_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    queue_depth: int = 0
    health_status: str = "healthy"  # healthy, degraded, unhealthy


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_time: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_time > self.ttl_seconds


class IntelligentCache:
    """High-performance intelligent caching system with multiple eviction policies."""
    
    def __init__(
        self,
        max_size_mb: int = 512,
        policy: CachePolicy = CachePolicy.LRU,
        default_ttl_seconds: Optional[float] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl_seconds = default_ttl_seconds
        
        self.cache: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self.access_order = deque()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.insertion_order = deque()  # For FIFO
        
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access statistics
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.access_frequency[key] += 1
            
            # Update access order for LRU
            if self.policy == CachePolicy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None, size_hint: Optional[int] = None) -> bool:
        """Put value into cache."""
        with self._lock:
            # Estimate size if not provided
            if size_hint is None:
                size_hint = self._estimate_size(value)
            
            # Check if we need to evict entries
            while (self.current_size_bytes + size_hint > self.max_size_bytes and 
                   len(self.cache) > 0):
                if not self._evict_one():
                    break  # Could not evict anything
            
            # If still too large, reject
            if self.current_size_bytes + size_hint > self.max_size_bytes:
                logger.warning(f"Cache entry too large: {size_hint} bytes, max: {self.max_size_bytes}")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_hint,
                created_time=time.time(),
                last_accessed=time.time(),
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_size_bytes += size_hint
            
            # Update tracking structures
            if self.policy == CachePolicy.LRU:
                self.access_order.append(key)
            elif self.policy == CachePolicy.FIFO:
                self.insertion_order.append(key)
            
            return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.cache:
            return False
        
        victim_key = None
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            while self.access_order and victim_key is None:
                candidate = self.access_order.popleft()
                if candidate in self.cache:
                    victim_key = candidate
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            min_frequency = min(self.access_frequency[k] for k in self.cache.keys())
            for key in self.cache.keys():
                if self.access_frequency[key] == min_frequency:
                    victim_key = key
                    break
        
        elif self.policy == CachePolicy.FIFO:
            # Remove first inserted
            while self.insertion_order and victim_key is None:
                candidate = self.insertion_order.popleft()
                if candidate in self.cache:
                    victim_key = candidate
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired
            ]
            if expired_keys:
                victim_key = expired_keys[0]
            else:
                # Fallback to oldest
                victim_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_time)
        
        elif self.policy == CachePolicy.SIZE_AWARE:
            # Remove largest entries first
            victim_key = max(self.cache.keys(), key=lambda k: self.cache[k].size_bytes)
        
        if victim_key:
            self._remove_entry(victim_key)
            self.evictions += 1
            return True
        
        return False
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update tracking structures."""
        if key not in self.cache:
            return
        
        entry = self.cache[key]
        self.current_size_bytes -= entry.size_bytes
        del self.cache[key]
        
        # Clean up tracking structures
        if key in self.access_frequency:
            del self.access_frequency[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        if key in self.insertion_order:
            self.insertion_order.remove(key)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, (str, bytes)):
            return len(value.encode('utf-8') if isinstance(value, str) else value)
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            # Rough estimate for other objects
            return 1024  # 1KB default
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.access_order.clear()
            self.access_frequency.clear()
            self.insertion_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "entries": len(self.cache),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.current_size_bytes / self.max_size_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "policy": self.policy.value,
            }


class AdaptiveWorkerPool:
    """Self-scaling worker pool that adapts to load patterns."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_queue_size: int = 5,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scale_check_interval: float = 10.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_queue_size = target_queue_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        
        # Worker management
        self.workers: Dict[str, WorkerStats] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.current_workers = min_workers
        
        # Load balancing
        self.strategy = LoadBalancingStrategy.ADAPTIVE
        self.round_robin_counter = 0
        
        # Scaling state
        self.scaling_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance tracking
        self.request_history = deque(maxlen=1000)
        self.scaling_events = deque(maxlen=100)
        
        self._lock = threading.Lock()
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize the minimum number of workers."""
        for i in range(self.min_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerStats(worker_id=worker_id)
    
    async def start_scaling(self) -> None:
        """Start automatic scaling monitoring."""
        if self.running:
            return
        
        self.running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Adaptive worker pool scaling started")
    
    async def stop_scaling(self) -> None:
        """Stop automatic scaling monitoring."""
        self.running = False
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Adaptive worker pool scaling stopped")
    
    async def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        while self.running:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(self.scale_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_scaling_needs(self) -> None:
        """Check if scaling is needed and perform scaling actions."""
        with self._lock:
            # Calculate current metrics
            total_active_requests = sum(w.active_requests for w in self.workers.values())
            avg_queue_depth = sum(w.queue_depth for w in self.workers.values()) / len(self.workers)
            avg_cpu_usage = sum(w.cpu_usage for w in self.workers.values()) / len(self.workers)
            
            # Recent request rate
            current_time = time.time()
            recent_requests = [
                req for req in self.request_history
                if current_time - req['timestamp'] < 60  # Last minute
            ]
            request_rate = len(recent_requests) / 60.0
            
            # Scaling decision logic
            should_scale_up = (
                (avg_queue_depth > self.target_queue_size * self.scale_up_threshold) or
                (avg_cpu_usage > self.scale_up_threshold) or
                (request_rate > self.current_workers * 2)  # More than 2 requests per worker per second
            )
            
            should_scale_down = (
                (avg_queue_depth < self.target_queue_size * self.scale_down_threshold) and
                (avg_cpu_usage < self.scale_down_threshold) and
                (request_rate < self.current_workers * 0.5)  # Less than 0.5 requests per worker per second
            )
            
            # Execute scaling
            if should_scale_up and self.current_workers < self.max_workers:
                await self._scale_up()
            elif should_scale_down and self.current_workers > self.min_workers:
                await self._scale_down()
    
    async def _scale_up(self) -> None:
        """Add more workers to the pool."""
        new_worker_id = f"worker_{self.current_workers}"
        self.workers[new_worker_id] = WorkerStats(worker_id=new_worker_id)
        self.current_workers += 1
        
        # Record scaling event
        self.scaling_events.append({
            'timestamp': time.time(),
            'action': 'scale_up',
            'workers': self.current_workers,
            'reason': 'high_load'
        })
        
        logger.info(f"Scaled up to {self.current_workers} workers")
    
    async def _scale_down(self) -> None:
        """Remove workers from the pool."""
        if self.current_workers <= self.min_workers:
            return
        
        # Find worker with least active requests
        idle_workers = [
            (worker_id, stats) for worker_id, stats in self.workers.items()
            if stats.active_requests == 0
        ]
        
        if idle_workers:
            # Remove the most recently added idle worker
            worker_to_remove = max(idle_workers, key=lambda x: x[0])  # Highest ID = most recent
            del self.workers[worker_to_remove[0]]
            self.current_workers -= 1
            
            # Record scaling event
            self.scaling_events.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'workers': self.current_workers,
                'reason': 'low_load'
            })
            
            logger.info(f"Scaled down to {self.current_workers} workers")
    
    def select_worker(self) -> str:
        """Select the best worker based on load balancing strategy."""
        with self._lock:
            if not self.workers:
                raise RuntimeError("No workers available")
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker_ids = list(self.workers.keys())
                selected = worker_ids[self.round_robin_counter % len(worker_ids)]
                self.round_robin_counter += 1
                return selected
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(self.workers.keys(), 
                          key=lambda w: self.workers[w].active_requests)
            
            elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
                return min(self.workers.keys(), 
                          key=lambda w: self.workers[w].avg_response_time_ms)
            
            elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return min(self.workers.keys(),
                          key=lambda w: (self.workers[w].cpu_usage + 
                                       self.workers[w].memory_usage_mb / 1000))
            
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                # Composite scoring based on multiple factors
                def worker_score(worker_id: str) -> float:
                    stats = self.workers[worker_id]
                    return (
                        stats.active_requests * 0.4 +
                        stats.avg_response_time_ms / 1000 * 0.3 +
                        stats.cpu_usage * 0.2 +
                        stats.queue_depth * 0.1
                    )
                
                return min(self.workers.keys(), key=worker_score)
            
            else:
                # Default to round robin
                worker_ids = list(self.workers.keys())
                return worker_ids[0]
    
    def record_request_start(self, worker_id: str) -> None:
        """Record the start of a request on a worker."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_requests += 1
                self.workers[worker_id].total_requests += 1
                self.workers[worker_id].last_request_time = time.time()
                
                self.request_history.append({
                    'timestamp': time.time(),
                    'worker_id': worker_id,
                    'type': 'start'
                })
    
    def record_request_end(self, worker_id: str, duration_ms: float, success: bool = True) -> None:
        """Record the completion of a request on a worker."""
        with self._lock:
            if worker_id in self.workers:
                stats = self.workers[worker_id]
                stats.active_requests = max(0, stats.active_requests - 1)
                
                if not success:
                    stats.total_errors += 1
                
                # Update average response time
                if stats.total_requests > 0:
                    alpha = 0.1  # Exponential moving average factor
                    stats.avg_response_time_ms = (
                        (1 - alpha) * stats.avg_response_time_ms + 
                        alpha * duration_ms
                    )
                
                self.request_history.append({
                    'timestamp': time.time(),
                    'worker_id': worker_id,
                    'type': 'end',
                    'duration_ms': duration_ms,
                    'success': success
                })
    
    def update_worker_resources(self, worker_id: str, cpu_usage: float, memory_mb: float) -> None:
        """Update worker resource usage statistics."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].cpu_usage = cpu_usage
                self.workers[worker_id].memory_usage_mb = memory_mb
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._lock:
            total_requests = sum(w.total_requests for w in self.workers.values())
            total_errors = sum(w.total_errors for w in self.workers.values())
            total_active = sum(w.active_requests for w in self.workers.values())
            avg_response_time = sum(w.avg_response_time_ms for w in self.workers.values()) / len(self.workers) if self.workers else 0
            
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': total_errors / total_requests if total_requests > 0 else 0,
                'active_requests': total_active,
                'avg_response_time_ms': avg_response_time,
                'strategy': self.strategy.value,
                'recent_scaling_events': list(self.scaling_events)[-10:],  # Last 10 events
                'worker_details': {
                    worker_id: {
                        'active_requests': stats.active_requests,
                        'total_requests': stats.total_requests,
                        'error_rate': stats.total_errors / stats.total_requests if stats.total_requests > 0 else 0,
                        'avg_response_time_ms': stats.avg_response_time_ms,
                        'cpu_usage': stats.cpu_usage,
                        'memory_usage_mb': stats.memory_usage_mb,
                        'health_status': stats.health_status
                    }
                    for worker_id, stats in self.workers.items()
                }
            }


class ModelPreloader:
    """Intelligent model preloading system for reduced cold start latency."""
    
    def __init__(self, cache_size_mb: int = 1024):
        self.cache = IntelligentCache(max_size_mb=cache_size_mb, policy=CachePolicy.LFU)
        self.preload_queue = asyncio.Queue()
        self.preload_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Usage prediction
        self.model_usage_history = defaultdict(list)
        self.preload_candidates = set()
        
        self._lock = threading.Lock()
    
    async def start_preloading(self) -> None:
        """Start background preloading task."""
        if self.running:
            return
        
        self.running = True
        self.preload_task = asyncio.create_task(self._preload_loop())
        logger.info("Model preloader started")
    
    async def stop_preloading(self) -> None:
        """Stop background preloading task."""
        self.running = False
        
        if self.preload_task:
            self.preload_task.cancel()
            try:
                await self.preload_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Model preloader stopped")
    
    async def _preload_loop(self) -> None:
        """Background preloading loop."""
        while self.running:
            try:
                # Wait for preload request or timeout
                try:
                    model_info = await asyncio.wait_for(
                        self.preload_queue.get(), timeout=30.0
                    )
                    await self._preload_model(model_info)
                except asyncio.TimeoutError:
                    # Periodic maintenance
                    await self._update_preload_candidates()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Preload loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _preload_model(self, model_info: Dict[str, Any]) -> None:
        """Preload a specific model."""
        model_path = model_info['path']
        model_id = model_info.get('id', model_path)
        
        try:
            # Check if already cached
            if self.cache.get(model_id) is not None:
                logger.debug(f"Model {model_id} already cached")
                return
            
            # Load model (this would interface with the actual model loading)
            logger.info(f"Preloading model: {model_id}")
            
            # For now, create a placeholder
            model_data = {
                'id': model_id,
                'path': model_path,
                'preloaded_time': time.time(),
                'size': model_info.get('size', 1024 * 1024)  # 1MB default
            }
            
            # Cache the preloaded model
            self.cache.put(model_id, model_data, size_hint=model_data['size'])
            
            logger.info(f"Successfully preloaded model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to preload model {model_id}: {e}")
    
    async def _update_preload_candidates(self) -> None:
        """Update list of models that should be preloaded."""
        with self._lock:
            current_time = time.time()
            
            # Analyze usage patterns
            for model_id, usage_times in self.model_usage_history.items():
                # Remove old usage records (older than 24 hours)
                recent_usage = [
                    t for t in usage_times 
                    if current_time - t < 86400
                ]
                self.model_usage_history[model_id] = recent_usage
                
                # Models used frequently in the last hour are candidates
                recent_frequent_usage = [
                    t for t in recent_usage
                    if current_time - t < 3600  # Last hour
                ]
                
                if len(recent_frequent_usage) >= 3:  # Used 3+ times in last hour
                    self.preload_candidates.add(model_id)
                elif len(recent_usage) < 2:  # Infrequently used
                    self.preload_candidates.discard(model_id)
    
    def record_model_usage(self, model_id: str) -> None:
        """Record that a model was used (for prediction)."""
        with self._lock:
            self.model_usage_history[model_id].append(time.time())
    
    async def request_preload(self, model_path: str, model_id: Optional[str] = None, priority: bool = False) -> None:
        """Request that a model be preloaded."""
        model_info = {
            'path': model_path,
            'id': model_id or model_path,
            'priority': priority,
            'requested_time': time.time()
        }
        
        if priority:
            # For high priority, preload immediately
            await self._preload_model(model_info)
        else:
            # Queue for background preloading
            await self.preload_queue.put(model_info)
    
    def get_cached_model(self, model_id: str) -> Optional[Any]:
        """Get a preloaded model from cache."""
        self.record_model_usage(model_id)
        return self.cache.get(model_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preloader statistics."""
        with self._lock:
            return {
                'cache_stats': self.cache.get_stats(),
                'preload_candidates': len(self.preload_candidates),
                'models_tracked': len(self.model_usage_history),
                'queue_size': self.preload_queue.qsize(),
                'running': self.running
            }


class ScalingManager:
    """Central manager for all scaling and performance optimization features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.intelligent_cache = IntelligentCache(
            max_size_mb=self.config.get('cache_size_mb', 512),
            policy=CachePolicy(self.config.get('cache_policy', 'lru')),
            default_ttl_seconds=self.config.get('cache_ttl_seconds')
        )
        
        self.worker_pool = AdaptiveWorkerPool(
            min_workers=self.config.get('min_workers', 2),
            max_workers=self.config.get('max_workers', 20),
            target_queue_size=self.config.get('target_queue_size', 5)
        )
        
        self.model_preloader = ModelPreloader(
            cache_size_mb=self.config.get('preload_cache_mb', 1024)
        )
        
        # Performance optimization
        self.batch_processor = None
        self.auto_optimization = self.config.get('auto_optimization', True)
        
        # Metrics and monitoring
        self.performance_metrics = defaultdict(list)
        self.optimization_events = deque(maxlen=1000)
        
    async def initialize(self) -> None:
        """Initialize scaling manager and start background tasks."""
        await self.worker_pool.start_scaling()
        await self.model_preloader.start_preloading()
        
        logger.info("Scaling manager initialized with auto-optimization enabled")
    
    async def shutdown(self) -> None:
        """Shutdown scaling manager and cleanup resources."""
        await self.worker_pool.stop_scaling()
        await self.model_preloader.stop_preloading()
        
        logger.info("Scaling manager shut down")
    
    async def execute_with_scaling(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with full scaling optimizations."""
        
        # Select optimal worker
        worker_id = self.worker_pool.select_worker()
        self.worker_pool.record_request_start(worker_id)
        
        start_time = time.time()
        success = True
        result = None
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(operation_name, args, kwargs)
            result = self.intelligent_cache.get(cache_key)
            
            if result is not None:
                logger.debug(f"Cache hit for {operation_name}")
                return result
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Cache result if cacheable
            if self._is_cacheable(operation_name, result):
                self.intelligent_cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            success = False
            raise
        finally:
            # Record completion
            duration_ms = (time.time() - start_time) * 1000
            self.worker_pool.record_request_end(worker_id, duration_ms, success)
            
            # Update performance metrics
            self.performance_metrics[operation_name].append({
                'timestamp': time.time(),
                'duration_ms': duration_ms,
                'success': success,
                'worker_id': worker_id,
                'cache_hit': result is not None
            })
    
    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation and arguments."""
        # Create deterministic key from inputs
        key_data = {
            'operation': operation_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _is_cacheable(self, operation_name: str, result: Any) -> bool:
        """Determine if operation result should be cached."""
        # Don't cache very large results
        if isinstance(result, torch.Tensor) and result.numel() > 1000000:  # 1M elements
            return False
        
        # Don't cache error results
        if isinstance(result, Exception):
            return False
        
        # Cache model inference results
        if 'inference' in operation_name.lower() or 'forward' in operation_name.lower():
            return True
        
        # Cache by default for most operations
        return True
    
    async def optimize_model_loading(self, model_path: str, model_id: Optional[str] = None) -> None:
        """Optimize model loading through preloading."""
        await self.model_preloader.request_preload(model_path, model_id)
    
    def get_cached_model(self, model_id: str) -> Optional[Any]:
        """Get preloaded model if available."""
        return self.model_preloader.get_cached_model(model_id)
    
    def adaptive_batch_size(self, current_load: float, base_batch_size: int = 16) -> int:
        """Adaptively adjust batch size based on current system load."""
        if current_load < 0.3:
            return min(base_batch_size * 2, 64)  # Increase batch size when load is low
        elif current_load > 0.8:
            return max(base_batch_size // 2, 1)  # Decrease batch size when load is high
        else:
            return base_batch_size
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance metrics."""
        current_time = time.time()
        
        # Aggregate performance metrics
        operation_summary = {}
        for op_name, metrics in self.performance_metrics.items():
            recent_metrics = [m for m in metrics if current_time - m['timestamp'] < 300]  # Last 5 minutes
            
            if recent_metrics:
                durations = [m['duration_ms'] for m in recent_metrics]
                cache_hits = sum(1 for m in recent_metrics if m.get('cache_hit', False))
                successes = sum(1 for m in recent_metrics if m['success'])
                
                operation_summary[op_name] = {
                    'total_requests': len(recent_metrics),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'min_duration_ms': min(durations),
                    'max_duration_ms': max(durations),
                    'success_rate': successes / len(recent_metrics),
                    'cache_hit_rate': cache_hits / len(recent_metrics),
                    'throughput_rps': len(recent_metrics) / 300  # Requests per second
                }
        
        return {
            'cache_stats': self.intelligent_cache.get_stats(),
            'worker_pool_stats': self.worker_pool.get_pool_stats(),
            'preloader_stats': self.model_preloader.get_stats(),
            'operation_performance': operation_summary,
            'system_resources': self._get_system_resources(),
            'optimization_events': list(self.optimization_events)[-20:],  # Last 20 events
        }
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.warning(f"Failed to get system resources: {e}")
            return {
                'cpu_usage_percent': 0,
                'memory_usage_percent': 0,
                'memory_available_gb': 0,
                'load_average': [0, 0, 0]
            }