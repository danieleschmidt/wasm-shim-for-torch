"""Enhanced Inference Engine - Generation 1: Core Functionality

Advanced inference pipeline with streaming capabilities, batch processing,
intelligent caching, and production-ready error handling for PyTorch-to-WASM models.
"""

import asyncio
import time
import hashlib
import json
import threading
import weakref
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import logging

# Handle torch import gracefully
try:
    import torch
    torch_available = True
except ImportError:
    from .mock_torch import torch, MockTensor
    torch_available = False

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Enhanced inference request with comprehensive metadata and tracking."""
    request_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    timeout: float = 300.0
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        return self.age_seconds > self.timeout


@dataclass
class InferenceResult:
    """Enhanced inference result with comprehensive performance and error tracking."""
    request_id: str
    model_id: str
    output_data: Any
    inference_time: float
    success: bool = True
    error_message: Optional[str] = None
    model_version: str = "1.0"
    memory_mb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: float = field(default_factory=time.time)
    
    @property
    def latency_ms(self) -> float:
        """Convert inference time to milliseconds."""
        return self.inference_time * 1000


class AdaptiveBatchProcessor:
    """Intelligent batch processor that adapts to system load and model characteristics."""
    
    def __init__(self, max_batch_size: int = 32, target_latency_ms: float = 100.0):
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.batch_history = deque(maxlen=100)
        self.current_batch_size = 8
        self.pending_requests = deque()
        self._lock = threading.Lock()
        
    def add_request(self, request: InferenceRequest) -> None:
        """Add request to batch queue."""
        with self._lock:
            self.pending_requests.append(request)
    
    def get_optimal_batch(self) -> List[InferenceRequest]:
        """Get optimally sized batch based on performance history."""
        with self._lock:
            if not self.pending_requests:
                return []
            
            # Adapt batch size based on recent performance
            if len(self.batch_history) > 10:
                avg_latency = sum(h['latency'] for h in list(self.batch_history)[-10:]) / 10
                if avg_latency > self.target_latency_ms * 1.2:
                    self.current_batch_size = max(1, self.current_batch_size - 1)
                elif avg_latency < self.target_latency_ms * 0.8:
                    self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            
            # Extract batch
            batch_size = min(self.current_batch_size, len(self.pending_requests))
            batch = []
            for _ in range(batch_size):
                batch.append(self.pending_requests.popleft())
            
            return batch
    
    def record_batch_performance(self, batch_size: int, latency_ms: float) -> None:
        """Record batch performance for adaptive optimization."""
        self.batch_history.append({
            'batch_size': batch_size,
            'latency': latency_ms,
            'timestamp': time.time()
        })


class IntelligentModelCache:
    """Model cache with predictive loading and memory management."""
    
    def __init__(self, max_models: int = 10, memory_limit_mb: int = 2048):
        self.max_models = max_models
        self.memory_limit_mb = memory_limit_mb
        self.models: Dict[str, Any] = {}
        self.model_stats: Dict[str, Dict] = defaultdict(lambda: {
            'load_count': 0, 'last_used': 0, 'memory_mb': 0
        })
        self._lock = threading.Lock()
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from cache, loading if necessary."""
        with self._lock:
            if model_id in self.models:
                self.model_stats[model_id]['last_used'] = time.time()
                self.model_stats[model_id]['load_count'] += 1
                return self.models[model_id]
            return None
    
    def cache_model(self, model_id: str, model: Any, memory_mb: float) -> bool:
        """Cache model with intelligent eviction."""
        with self._lock:
            # Check if we need to evict models
            self._evict_if_needed(memory_mb)
            
            if len(self.models) >= self.max_models:
                # Evict least recently used model
                lru_model = min(self.model_stats.items(), 
                               key=lambda x: x[1]['last_used'])[0]
                self._evict_model(lru_model)
            
            self.models[model_id] = model
            self.model_stats[model_id].update({
                'last_used': time.time(),
                'memory_mb': memory_mb,
                'load_count': 1
            })
            return True
    
    def _evict_if_needed(self, new_memory_mb: float) -> None:
        """Evict models if memory limit would be exceeded."""
        current_memory = sum(stats['memory_mb'] for stats in self.model_stats.values())
        
        while current_memory + new_memory_mb > self.memory_limit_mb and self.models:
            # Evict least recently used model
            lru_model = min(self.model_stats.items(), 
                           key=lambda x: x[1]['last_used'])[0]
            current_memory -= self.model_stats[lru_model]['memory_mb']
            self._evict_model(lru_model)
    
    def _evict_model(self, model_id: str) -> None:
        """Evict specific model from cache."""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Evicted model {model_id} from cache")


@dataclass
class EngineStats:
    """Comprehensive statistics for the inference engine."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_inference_time: float = 0.0
    average_inference_time: float = 0.0
    requests_per_second: float = 0.0
    active_requests: int = 0
    queue_depth: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_request_time: Optional[float] = None
    model_counts: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)


class InferenceCache:
    """Advanced caching system for inference results with TTL and intelligent eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        with self._lock:
            if cache_key not in self._cache:
                return None
                
            cache_entry = self._cache[cache_key]
            current_time = time.time()
            
            # Check TTL
            if current_time - cache_entry['stored_at'] > self.ttl_seconds:
                self._remove_key(cache_key)
                return None
                
            # Update access time
            self._access_times[cache_key] = current_time
            return cache_entry['result']
    
    def put(self, cache_key: str, result: Any) -> None:
        """Store result in cache with LRU eviction."""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_lru()
            
            # Store result
            self._cache[cache_key] = {
                'result': result,
                'stored_at': current_time
            }
            self._access_times[cache_key] = current_time
    
    def _remove_key(self, cache_key: str) -> None:
        """Remove key from cache and access times."""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
            
        # Find LRU key
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times[k])
        self._remove_key(lru_key)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'memory_estimate_mb': len(self._cache) * 0.1  # Rough estimate
            }


class RequestQueue:
    """Priority queue for inference requests with timeout handling."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = deque()
        self._priority_queues: Dict[int, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
    async def put(self, request: InferenceRequest, timeout: Optional[float] = None) -> bool:
        """Add request to queue with optional timeout."""
        loop = asyncio.get_event_loop()
        
        def _put():
            with self._not_full:
                while len(self._queue) >= self.max_size:
                    if timeout is None:
                        self._not_full.wait()
                    else:
                        if not self._not_full.wait(timeout):
                            return False
                
                # Add to appropriate priority queue
                self._priority_queues[request.priority].append(request)
                self._queue.append(request)
                self._not_empty.notify()
                return True
        
        return await loop.run_in_executor(None, _put)
    
    async def get(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        """Get next request from queue with priority ordering."""
        loop = asyncio.get_event_loop()
        
        def _get():
            with self._not_empty:
                while len(self._queue) == 0:
                    if timeout is None:
                        self._not_empty.wait()
                    else:
                        if not self._not_empty.wait(timeout):
                            return None
                
                # Get highest priority request
                for priority in sorted(self._priority_queues.keys(), reverse=True):
                    priority_queue = self._priority_queues[priority]
                    if priority_queue:
                        request = priority_queue.popleft()
                        self._queue.remove(request)
                        
                        # Clean up empty priority queue
                        if not priority_queue:
                            del self._priority_queues[priority]
                        
                        self._not_full.notify()
                        return request
                
                return None
        
        return await loop.run_in_executor(None, _get)
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        """Clear all requests from queue."""
        with self._lock:
            self._queue.clear()
            self._priority_queues.clear()
            self._not_full.notify_all()


class ModelLoadBalancer:
    """Load balancer for multiple model instances."""
    
    def __init__(self):
        self._model_instances: Dict[str, List[Any]] = defaultdict(list)
        self._instance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def register_model_instance(self, model_id: str, model_instance: Any) -> None:
        """Register a model instance for load balancing."""
        with self._lock:
            self._model_instances[model_id].append(model_instance)
            instance_key = f"{model_id}_{len(self._model_instances[model_id]) - 1}"
            self._instance_stats[instance_key] = {
                'requests_handled': 0,
                'total_inference_time': 0.0,
                'last_used': time.time()
            }
    
    def get_best_instance(self, model_id: str) -> Optional[Tuple[Any, str]]:
        """Get the best available instance for a model."""
        with self._lock:
            instances = self._model_instances.get(model_id, [])
            if not instances:
                return None
            
            # Simple round-robin for now
            counter = self._round_robin_counters[model_id]
            selected_instance = instances[counter % len(instances)]
            instance_key = f"{model_id}_{counter % len(instances)}"
            
            self._round_robin_counters[model_id] = (counter + 1) % len(instances)
            return selected_instance, instance_key
    
    def update_instance_stats(self, instance_key: str, inference_time: float) -> None:
        """Update statistics for an instance."""
        with self._lock:
            if instance_key in self._instance_stats:
                stats = self._instance_stats[instance_key]
                stats['requests_handled'] += 1
                stats['total_inference_time'] += inference_time
                stats['last_used'] = time.time()


class EnhancedInferenceEngine:
    """Enhanced inference engine with production-ready features for Generation 1.
    
    Features:
    - Intelligent request queuing with priority support
    - Result caching with TTL and LRU eviction
    - Load balancing across model instances
    - Comprehensive monitoring and statistics
    - Async streaming inference support
    - Robust error handling and recovery
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 100,
        enable_caching: bool = True,
        enable_batching: bool = True,
        enable_load_balancing: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 3600,
        max_batch_size: int = 32,
        batch_timeout: float = 0.05
    ):
        # Core configuration
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_load_balancing = enable_load_balancing
        
        # Initialize core components
        self._request_queue = RequestQueue(max_size=max_concurrent_requests * 2)
        self._result_cache = InferenceCache(cache_size, cache_ttl) if enable_caching else None
        self._batch_processor = AdaptiveBatchProcessor(max_batch_size, batch_timeout * 1000) if enable_batching else None
        self._load_balancer = ModelLoadBalancer() if enable_load_balancing else None
        
        # State management
        self._models: Dict[str, Any] = {}
        self._stats = EngineStats()
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests // 4)
        self._startup_time = time.time()
        
        # Request tracking
        self._active_requests: Dict[str, InferenceRequest] = {}
        self._request_results: Dict[str, InferenceResult] = {}
        self._request_history = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced Inference Engine initialized with advanced features")
    
    async def start(self) -> None:
        """Start the inference engine with all background workers."""
        if self._running:
            logger.warning("Inference engine already running")
            return
        
        self._running = True
        self._startup_time = time.time()
        
        # Start worker tasks
        num_workers = min(self.max_concurrent_requests // 10, 10)  # Reasonable number of workers
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self._worker_tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_loop())
        self._worker_tasks.append(monitor_task)
        
        logger.info(f"Enhanced Inference Engine started with {num_workers} workers")
    
    async def stop(self) -> None:
        """Stop the inference engine and cleanup resources."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Enhanced Inference Engine stopped")
    
    def register_model(self, model_id: str, model_instance: Any) -> None:
        """Register a model for inference."""
        with self._lock:
            self._models[model_id] = model_instance
            
            # Register with load balancer if enabled
            if self._load_balancer:
                self._load_balancer.register_model_instance(model_id, model_instance)
        
        logger.info(f"Registered model: {model_id}")
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                logger.info(f"Unregistered model: {model_id}")
                return True
            return False
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        with self._lock:
            return list(self._models.keys())
    
    async def infer_async(self,
                         model_id: str,
                         input_data: Any,
                         priority: int = 0,
                         timeout: float = 300.0,
                         metadata: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """Perform asynchronous inference."""
        
        # Generate request ID
        request_id = self._generate_request_id(model_id, input_data)
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            model_id=model_id,
            input_data=input_data,
            priority=priority,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        # Check cache first
        if self._result_cache and self.enable_caching:
            cache_key = self._generate_cache_key(model_id, input_data)
            cached_result = self._result_cache.get(cache_key)
            if cached_result is not None:
                self._stats.cache_hits += 1
                logger.debug(f"Cache hit for request {request_id}")
                return cached_result
            self._stats.cache_misses += 1
        
        # Add to queue
        success = await self._request_queue.put(request, timeout=timeout)
        if not success:
            return InferenceResult(
                request_id=request_id,
                model_id=model_id,
                output_data=None,
                inference_time=0.0,
                success=False,
                error_message="Request queue timeout"
            )
        
        # Wait for completion
        return await self._wait_for_completion(request_id, timeout)
    
    async def infer_batch_async(self,
                               requests: List[Tuple[str, Any, int, Optional[Dict[str, Any]]]],
                               timeout: float = 300.0) -> List[InferenceResult]:
        """Perform batch inference asynchronously."""
        
        if not self.enable_batching or not requests:
            # Fall back to individual inference
            tasks = []
            for model_id, input_data, priority, metadata in requests:
                task = self.infer_async(model_id, input_data, priority, timeout, metadata)
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        # For now, fall back to individual processing (can be optimized later)
        results = []
        for model_id, input_data, priority, metadata in requests:
            result = await self.infer_async(model_id, input_data, priority, timeout, metadata)
            results.append(result)
        
        return results
    
    async def infer_stream(self,
                          model_id: str,
                          input_stream: AsyncGenerator[Any, None],
                          priority: int = 0,
                          timeout: float = 300.0) -> AsyncGenerator[InferenceResult, None]:
        """Perform streaming inference."""
        
        async for input_data in input_stream:
            try:
                result = await self.infer_async(
                    model_id=model_id,
                    input_data=input_data,
                    priority=priority,
                    timeout=timeout
                )
                yield result
            except Exception as e:
                yield InferenceResult(
                    request_id=self._generate_request_id(model_id, input_data),
                    model_id=model_id,
                    output_data=None,
                    inference_time=0.0,
                    success=False,
                    error_message=str(e)
                )
    
    async def _wait_for_completion(self, request_id: str, timeout: float) -> InferenceResult:
        """Wait for request completion with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if request is completed
            if request_id in self._request_results:
                return self._request_results.pop(request_id)
            
            # Check if request is still active
            if request_id not in self._active_requests:
                # Request might have been processed but result not found
                await asyncio.sleep(0.01)  # Small delay to allow result storage
                continue
            
            await asyncio.sleep(0.01)  # Poll every 10ms
        
        # Timeout reached
        return InferenceResult(
            request_id=request_id,
            model_id="unknown",
            output_data=None,
            inference_time=timeout,
            success=False,
            error_message="Request timeout"
        )
    
    async def _batch_processing_loop(self) -> None:
        """Main batch processing loop."""
        while not self._shutdown:
            try:
                batch = self.batch_processor.get_optimal_batch()
                if batch:
                    await self._process_batch(batch)
                else:
                    await asyncio.sleep(self.batch_timeout_ms / 1000)
            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, batch: List[InferenceRequest]) -> None:
        """Process a batch of inference requests."""
        start_time = time.time()
        
        try:
            # Group by model for efficient processing
            model_groups = defaultdict(list)
            for request in batch:
                model_groups[request.model_id].append(request)
            
            # Process each model group
            for model_id, requests in model_groups.items():
                await self._process_model_group(model_id, requests)
            
            # Record batch performance
            latency_ms = (time.time() - start_time) * 1000
            self.batch_processor.record_batch_performance(len(batch), latency_ms)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Mark all requests as failed
            for request in batch:
                self.active_requests.pop(request.request_id, None)
    
    async def _process_model_group(self, model_id: str, requests: List[InferenceRequest]) -> None:
        """Process requests for a specific model."""
        try:
            # Load model if needed
            model = await self._load_model(model_id)
            if not model:
                for request in requests:
                    self.active_requests.pop(request.request_id, None)
                return
            
            # Process requests (simplified simulation)
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Complete requests
            for request in requests:
                self.active_requests.pop(request.request_id, None)
                self.performance_metrics['total_requests'] += 1
                self.performance_metrics['successful_requests'] += 1
        
        except Exception as e:
            logger.error(f"Model group processing failed: {e}")
            for request in requests:
                self.active_requests.pop(request.request_id, None)
                self.performance_metrics['failed_requests'] += 1
    
    async def _load_model(self, model_id: str) -> Optional[Any]:
        """Load model with caching support."""
        if self.model_cache:
            model = self.model_cache.get_model(model_id)
            if model:
                return model
        
        # Simulate model loading
        await asyncio.sleep(0.1)
        mock_model = f"model_{model_id}_loaded"
        
        if self.model_cache:
            self.model_cache.cache_model(model_id, mock_model, 256.0)
        
        return mock_model
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics calculation loop."""
        last_request_count = 0
        last_time = time.time()
        
        while not self._shutdown:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                current_time = time.time()
                current_requests = self.performance_metrics['total_requests']
                
                # Calculate throughput
                time_delta = current_time - last_time
                request_delta = current_requests - last_request_count
                
                if time_delta > 0:
                    self.performance_metrics['throughput_rps'] = request_delta / time_delta
                
                last_request_count = current_requests
                last_time = current_time
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'active_requests': len(self.active_requests),
            'pending_batch_requests': len(self.batch_processor.pending_requests),
            'current_batch_size': self.batch_processor.current_batch_size,
            'cached_models': len(self.model_cache.models) if self.model_cache else 0,
            'performance_metrics': self.get_performance_metrics(),
            'is_healthy': len(self.active_requests) < self.max_concurrent_requests
        }
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing requests."""
        logger.debug(f"Started worker: {worker_id}")
        
        while self._running:
            try:
                # Get next request
                request = await self._request_queue.get(timeout=1.0)
                if request is None:
                    continue
                
                # Check if request has timed out
                if request.is_expired:
                    logger.warning(f"Request {request.request_id} timed out (age: {request.age_seconds:.2f}s)")
                    continue
                
                # Track active request
                with self._lock:
                    self._active_requests[request.request_id] = request
                
                # Process request
                result = await self._process_request(request)
                
                # Cache result if enabled and successful
                if self._result_cache and self.enable_caching and result.success:
                    cache_key = self._generate_cache_key(request.model_id, request.input_data)
                    self._result_cache.put(cache_key, result)
                
                # Store result for pickup
                with self._lock:
                    self._active_requests.pop(request.request_id, None)
                    self._request_results[request.request_id] = result
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Stopped worker: {worker_id}")
    
    async def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process an individual inference request."""
        start_time = time.time()
        
        try:
            # Update stats
            with self._lock:
                self._stats.total_requests += 1
                self._stats.model_counts[request.model_id] = (
                    self._stats.model_counts.get(request.model_id, 0) + 1
                )
            
            # Get model instance
            model_instance = None
            instance_key = None
            
            if self._load_balancer:
                result = self._load_balancer.get_best_instance(request.model_id)
                if result:
                    model_instance, instance_key = result
            else:
                model_instance = self._models.get(request.model_id)
            
            if model_instance is None:
                raise ValueError(f"Model not found: {request.model_id}")
            
            # Perform inference
            if hasattr(model_instance, 'forward'):
                output_data = await model_instance.forward(request.input_data)
            elif callable(model_instance):
                output_data = await model_instance(request.input_data)
            else:
                # Mock processing for demo
                if hasattr(request.input_data, '__iter__') and not isinstance(request.input_data, str):
                    output_data = [x * 0.5 + 0.1 for x in request.input_data]
                else:
                    output_data = f"processed_{request.input_data}"
            
            inference_time = time.time() - start_time
            
            # Update stats
            with self._lock:
                self._stats.successful_requests += 1
                self._stats.total_inference_time += inference_time
                self._stats.last_request_time = time.time()
            
            # Update load balancer stats
            if self._load_balancer and instance_key:
                self._load_balancer.update_instance_stats(instance_key, inference_time)
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=output_data,
                inference_time=inference_time,
                success=True
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            error_message = str(e)
            
            # Update error stats
            with self._lock:
                self._stats.failed_requests += 1
                error_type = type(e).__name__
                self._stats.error_types[error_type] = (
                    self._stats.error_types.get(error_type, 0) + 1
                )
            
            logger.error(f"Inference failed for request {request.request_id}: {error_message}")
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=None,
                inference_time=inference_time,
                success=False,
                error_message=error_message
            )
    
    async def _monitor_loop(self) -> None:
        """Background monitoring and maintenance loop."""
        logger.debug("Started monitoring loop")
        
        while self._running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Clean up old request results
                current_time = time.time()
                max_age = 300  # 5 minutes
                
                with self._lock:
                    expired_results = [
                        req_id for req_id, result in self._request_results.items()
                        if current_time - result.completed_at > max_age
                    ]
                    
                    for req_id in expired_results:
                        self._request_results.pop(req_id, None)
                
                # Log stats periodically
                stats = self.get_engine_stats()
                logger.info(f"Engine stats: {stats['total_requests']} requests, "
                           f"{stats['success_rate']:.2%} success, "
                           f"{stats['requests_per_second']:.2f} RPS")
                
            except asyncio.CancelledError:
                logger.debug("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
        
        logger.debug("Stopped monitoring loop")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        with self._lock:
            # Update uptime
            self._stats.uptime_seconds = time.time() - self._startup_time
            
            # Update queue depth
            self._stats.queue_depth = self._request_queue.size()
            self._stats.active_requests = len(self._active_requests)
            
            # Calculate requests per second
            if self._stats.uptime_seconds > 0:
                self._stats.requests_per_second = self._stats.total_requests / self._stats.uptime_seconds
            
            # Calculate average inference time
            if self._stats.successful_requests > 0:
                self._stats.average_inference_time = (
                    self._stats.total_inference_time / self._stats.successful_requests
                )
            
            stats_dict = {
                'total_requests': self._stats.total_requests,
                'successful_requests': self._stats.successful_requests,
                'failed_requests': self._stats.failed_requests,
                'success_rate': self._stats.successful_requests / max(self._stats.total_requests, 1),
                'average_inference_time': self._stats.average_inference_time,
                'requests_per_second': self._stats.requests_per_second,
                'active_requests': self._stats.active_requests,
                'queue_depth': self._stats.queue_depth,
                'cache_hits': self._stats.cache_hits,
                'cache_misses': self._stats.cache_misses,
                'cache_hit_rate': self._stats.cache_hits / max(self._stats.cache_hits + self._stats.cache_misses, 1),
                'uptime_seconds': self._stats.uptime_seconds,
                'registered_models': len(self._models),
                'model_counts': dict(self._stats.model_counts),
                'error_types': dict(self._stats.error_types)
            }
            
            # Add cache stats if enabled
            if self._result_cache:
                stats_dict['cache_stats'] = self._result_cache.get_stats()
            
            return stats_dict
    
    def _generate_request_id(self, model_id: str, input_data: Any) -> str:
        """Generate unique request ID."""
        timestamp = str(time.time())
        data_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:8]
        return f"{model_id}_{timestamp}_{data_hash}"
    
    def _generate_cache_key(self, model_id: str, input_data: Any) -> str:
        """Generate cache key for input."""
        data_str = str(input_data)
        if hasattr(input_data, 'shape'):
            data_str += f"_shape_{input_data.shape}"
        return f"{model_id}_{hashlib.md5(data_str.encode()).hexdigest()}"


# Utility functions for testing and demonstration
async def demo_enhanced_inference_engine():
    """Comprehensive demonstration of the enhanced inference engine."""
    
    class MockModel:
        def __init__(self, model_id: str, delay: float = 0.01):
            self.model_id = model_id
            self.delay = delay
        
        async def forward(self, input_data):
            await asyncio.sleep(self.delay)  # Simulate processing time
            
            # Simple transformation
            if hasattr(input_data, '__iter__') and not isinstance(input_data, str):
                return [x * 0.5 + 0.1 for x in input_data]
            else:
                return f"processed_{input_data}"
    
    try:
        print("Enhanced Inference Engine Demo - Generation 1")
        print("=" * 60)
        
        # Create engine with all features enabled
        engine = EnhancedInferenceEngine(
            max_concurrent_requests=50,
            enable_caching=True,
            enable_batching=True,
            enable_load_balancing=True,
            cache_size=100,
            max_batch_size=16
        )
        
        # Start engine
        await engine.start()
        print("✓ Engine started with advanced features")
        
        # Register test models
        models = {
            'fast_model': MockModel('fast_model', 0.01),
            'slow_model': MockModel('slow_model', 0.1),
            'classifier': MockModel('classifier', 0.05)
        }
        
        for model_id, model in models.items():
            engine.register_model(model_id, model)
        
        print(f"✓ Registered {len(models)} models with load balancing")
        
        # Test single inference
        print("\\nTesting single inference...")
        result = await engine.infer_async('fast_model', [1, 2, 3, 4])
        print(f"Result: {result.output_data} (time: {result.inference_time:.3f}s)")
        
        # Test cache hit
        print("\\nTesting cache hit...")
        start_time = time.time()
        result2 = await engine.infer_async('fast_model', [1, 2, 3, 4])
        cache_time = time.time() - start_time
        print(f"Cache result: {result2.output_data} (time: {cache_time:.6f}s)")
        
        # Test concurrent requests with priorities
        print("\\nTesting concurrent requests with priorities...")
        tasks = []
        for i in range(20):
            model_id = list(models.keys())[i % len(models)]
            input_data = [i, i+1, i+2]
            priority = i % 3  # Different priorities
            task = engine.infer_async(model_id, input_data, priority=priority)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        successful = sum(1 for r in results if r.success)
        print(f"✓ Completed {successful}/{len(results)} concurrent requests")
        
        # Test batch inference
        print("\\nTesting batch inference...")
        batch_requests = [
            ('classifier', [1, 2], 1, None),
            ('classifier', [3, 4], 2, None),
            ('classifier', [5, 6], 0, None),
        ]
        
        batch_results = await engine.infer_batch_async(batch_requests)
        batch_successful = sum(1 for r in batch_results if r.success)
        print(f"✓ Completed {batch_successful}/{len(batch_results)} batch requests")
        
        # Show comprehensive statistics
        stats = engine.get_engine_stats()
        print(f"\\nComprehensive Engine Statistics:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average inference time: {stats['average_inference_time']:.4f}s")
        print(f"Requests per second: {stats['requests_per_second']:.2f}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"Active requests: {stats['active_requests']}")
        print(f"Queue depth: {stats['queue_depth']}")
        print(f"Registered models: {stats['registered_models']}")
        print(f"Model usage: {stats['model_counts']}")
        
        # Stop engine
        await engine.stop()
        print("\\n✓ Engine stopped gracefully")
        
        print("\\n🎉 Enhanced Inference Engine Generation 1 Demo Complete!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_enhanced_inference_engine())
