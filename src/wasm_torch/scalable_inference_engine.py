"""
Scalable Inference Engine - Generation 3: Make It Scale
High-performance, horizontally scalable inference system with advanced optimization.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import weakref
import pickle
import zlib

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for the inference engine."""
    
    FIXED = auto()           # Fixed number of workers
    ADAPTIVE = auto()        # Adaptive based on load
    PREDICTIVE = auto()      # Predictive scaling based on patterns
    BURST = auto()           # Burst scaling for traffic spikes


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = auto()     # Round-robin distribution
    LEAST_CONNECTIONS = auto()  # Route to worker with fewest connections
    WEIGHTED_ROUND_ROBIN = auto()  # Weighted based on worker performance
    PERFORMANCE_BASED = auto()     # Route based on historical performance


@dataclass
class WorkerMetrics:
    """Metrics for individual workers."""
    
    worker_id: str
    active_requests: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_activity: float = field(default_factory=time.time)
    performance_score: float = 1.0
    
    def calculate_performance_score(self) -> float:
        """Calculate performance score based on metrics."""
        if self.total_requests == 0:
            return 1.0
        
        success_rate = self.successful_requests / self.total_requests
        latency_penalty = min(self.average_latency / 1.0, 1.0)  # Penalty after 1s
        load_penalty = min(self.active_requests / 10.0, 1.0)   # Penalty after 10 requests
        
        score = success_rate * (1.0 - latency_penalty) * (1.0 - load_penalty)
        self.performance_score = max(0.1, score)  # Minimum score of 0.1
        
        return self.performance_score


@dataclass
class InferenceJob:
    """Scalable inference job with advanced features."""
    
    job_id: str
    model_id: str
    input_data: Any
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.result is not None or self.error is not None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        """Get total time from creation to completion."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None


class AdvancedCache:
    """
    Advanced caching system with intelligent eviction and compression.
    """
    
    def __init__(self, max_size: int = 1000, compression_enabled: bool = True):
        self.max_size = max_size
        self.compression_enabled = compression_enabled
        self._cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, access_count)
        self._access_order: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'total_size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking."""
        with self._lock:
            if key in self._cache:
                value, timestamp, access_count = self._cache[key]
                
                # Update access count and timestamp
                self._cache[key] = (value, time.time(), access_count + 1)
                
                # Update LRU order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                # Decompress if needed
                if isinstance(value, bytes) and self.compression_enabled:
                    try:
                        decompressed = pickle.loads(zlib.decompress(value))
                        self._stats['hits'] += 1
                        return decompressed
                    except Exception:
                        # If decompression fails, treat as miss
                        del self._cache[key]
                        if key in self._access_order:
                            self._access_order.remove(key)
                        self._stats['misses'] += 1
                        return None
                
                self._stats['hits'] += 1
                return value
            
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache with optional TTL."""
        try:
            with self._lock:
                # Check if we need to evict
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_lru()
                
                # Compress large objects
                stored_value = value
                if self.compression_enabled and self._should_compress(value):
                    try:
                        compressed = zlib.compress(pickle.dumps(value))
                        if len(compressed) < len(pickle.dumps(value)):  # Only if beneficial
                            stored_value = compressed
                            self._stats['compressions'] += 1
                    except Exception:
                        pass  # Fall back to uncompressed storage
                
                # Store with timestamp and access count
                self._cache[key] = (stored_value, time.time(), 1)
                
                # Update LRU order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                return True
                
        except Exception as e:
            logger.error(f"Cache put error: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats['evictions'] += len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'compressions': self._stats['compressions'],
                'compression_enabled': self.compression_enabled
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1
    
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed."""
        try:
            serialized_size = len(pickle.dumps(value))
            return serialized_size > 1024  # Compress objects larger than 1KB
        except Exception:
            return False


class WorkerPool:
    """
    Advanced worker pool with dynamic scaling and load balancing.
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 10,
        scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_strategy = scaling_strategy
        self.load_balancing = load_balancing
        
        # Worker management
        self._workers: Dict[str, asyncio.Task] = {}
        self._worker_metrics: Dict[str, WorkerMetrics] = {}
        self._worker_queues: Dict[str, asyncio.Queue] = {}
        self._job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._result_futures: Dict[str, asyncio.Future] = {}
        
        # Scaling control
        self._last_scale_time = time.time()
        self._scale_cooldown = 30.0  # 30 seconds between scaling decisions
        self._load_history: deque = deque(maxlen=100)  # Track load history
        
        # Control state
        self._running = False
        self._lock = threading.RLock()
        
        # Round-robin counter for load balancing
        self._round_robin_counter = 0
    
    async def start(self) -> bool:
        """Start the worker pool."""
        try:
            if self._running:
                return True
            
            logger.info(f"Starting worker pool with {self.min_workers} workers")
            self._running = True
            
            # Start initial workers
            for i in range(self.min_workers):
                await self._create_worker(f"worker-{i}")
            
            # Start scaling monitor
            asyncio.create_task(self._scaling_monitor())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start worker pool: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the worker pool gracefully."""
        try:
            logger.info("Stopping worker pool")
            self._running = False
            
            # Cancel all workers
            for worker_id, worker_task in self._workers.items():
                if not worker_task.done():
                    worker_task.cancel()
            
            # Wait for workers to complete
            if self._workers:
                await asyncio.gather(*self._workers.values(), return_exceptions=True)
            
            # Clear resources
            self._workers.clear()
            self._worker_metrics.clear()
            self._worker_queues.clear()
            
            logger.info("Worker pool stopped")
            
        except Exception as e:
            logger.error(f"Error stopping worker pool: {e}")
    
    async def submit_job(self, job: InferenceJob) -> str:
        """Submit a job for processing."""
        if not self._running:
            raise RuntimeError("Worker pool is not running")
        
        # Create future for result
        future = asyncio.Future()
        self._result_futures[job.job_id] = future
        
        # Add to job queue with priority
        await self._job_queue.put((-job.priority, job.created_at, job))
        
        logger.debug(f"Job {job.job_id} submitted to worker pool")
        return job.job_id
    
    async def get_result(self, job_id: str, timeout: Optional[float] = None) -> InferenceJob:
        """Get result for a submitted job."""
        if job_id not in self._result_futures:
            raise ValueError(f"Job {job_id} not found")
        
        future = self._result_futures[job_id]
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        finally:
            # Clean up future
            if job_id in self._result_futures:
                del self._result_futures[job_id]
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._lock:
            total_active_requests = sum(
                metrics.active_requests for metrics in self._worker_metrics.values()
            )
            
            avg_performance = sum(
                metrics.performance_score for metrics in self._worker_metrics.values()
            ) / len(self._worker_metrics) if self._worker_metrics else 0.0
            
            return {
                'worker_count': len(self._workers),
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'active_requests': total_active_requests,
                'pending_jobs': self._job_queue.qsize(),
                'average_performance': avg_performance,
                'scaling_strategy': self.scaling_strategy.name,
                'load_balancing': self.load_balancing.name,
                'worker_metrics': {
                    worker_id: {
                        'active_requests': metrics.active_requests,
                        'total_requests': metrics.total_requests,
                        'success_rate': metrics.successful_requests / max(metrics.total_requests, 1),
                        'average_latency': metrics.average_latency,
                        'performance_score': metrics.performance_score
                    } for worker_id, metrics in self._worker_metrics.items()
                }
            }
    
    async def _create_worker(self, worker_id: str) -> bool:
        """Create a new worker."""
        try:
            if worker_id in self._workers:
                return False
            
            # Create worker queue
            worker_queue = asyncio.Queue(maxsize=100)
            self._worker_queues[worker_id] = worker_queue
            
            # Create worker metrics
            self._worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            
            # Start worker task
            worker_task = asyncio.create_task(self._worker_loop(worker_id))
            self._workers[worker_id] = worker_task
            
            logger.info(f"Created worker: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create worker {worker_id}: {e}")
            return False
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        logger.info(f"Worker {worker_id} started")
        metrics = self._worker_metrics[worker_id]
        
        try:
            while self._running:
                try:
                    # Get next job from global queue
                    try:
                        priority, timestamp, job = await asyncio.wait_for(
                            self._job_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    # Process job
                    await self._process_job_on_worker(worker_id, job)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    await asyncio.sleep(1.0)
        
        except asyncio.CancelledError:
            pass
        finally:
            logger.info(f"Worker {worker_id} stopped")
    
    async def _process_job_on_worker(self, worker_id: str, job: InferenceJob) -> None:
        """Process a job on a specific worker."""
        metrics = self._worker_metrics[worker_id]
        start_time = time.time()
        
        try:
            # Update metrics
            metrics.active_requests += 1
            metrics.total_requests += 1
            metrics.last_activity = start_time
            
            job.started_at = start_time
            job.worker_id = worker_id
            
            # Simulate inference processing
            result = await self._run_inference(job)
            
            # Record success
            job.result = result
            job.completed_at = time.time()
            metrics.successful_requests += 1
            
        except Exception as e:
            # Record failure
            job.error = str(e)
            job.completed_at = time.time()
            metrics.failed_requests += 1
            logger.error(f"Job {job.job_id} failed on worker {worker_id}: {e}")
        
        finally:
            # Update metrics
            metrics.active_requests = max(0, metrics.active_requests - 1)
            
            if job.execution_time:
                # Update average latency with exponential moving average
                if metrics.average_latency == 0:
                    metrics.average_latency = job.execution_time
                else:
                    alpha = 0.1  # Smoothing factor
                    metrics.average_latency = (
                        alpha * job.execution_time + 
                        (1 - alpha) * metrics.average_latency
                    )
            
            # Calculate performance score
            metrics.calculate_performance_score()
            
            # Complete the future
            if job.job_id in self._result_futures:
                future = self._result_futures[job.job_id]
                if not future.done():
                    future.set_result(job)
    
    async def _run_inference(self, job: InferenceJob) -> Any:
        """Run inference for a job (mock implementation)."""
        # Simulate processing time based on input complexity
        if isinstance(job.input_data, (list, tuple)):
            processing_time = min(0.1 + len(job.input_data) * 0.001, 2.0)
        elif isinstance(job.input_data, str):
            processing_time = min(0.05 + len(job.input_data) * 0.0001, 1.0)
        else:
            processing_time = 0.1
        
        await asyncio.sleep(processing_time)
        
        # Mock result based on model type
        if 'classifier' in job.model_id.lower():
            return {
                'prediction': 'positive' if hash(str(job.input_data)) % 2 == 0 else 'negative',
                'confidence': 0.8 + (hash(str(job.input_data)) % 20) / 100,
                'model_id': job.model_id,
                'processing_time': processing_time
            }
        else:
            return {
                'output': hash(str(job.input_data)) % 1000,
                'model_id': job.model_id,
                'processing_time': processing_time
            }
    
    async def _scaling_monitor(self) -> None:
        """Monitor load and make scaling decisions."""
        while self._running:
            try:
                current_time = time.time()
                
                # Calculate current load
                total_active = sum(
                    metrics.active_requests for metrics in self._worker_metrics.values()
                )
                pending_jobs = self._job_queue.qsize()
                current_load = total_active + pending_jobs
                
                # Track load history
                self._load_history.append((current_time, current_load))
                
                # Make scaling decision
                if current_time - self._last_scale_time > self._scale_cooldown:
                    await self._make_scaling_decision(current_load)
                
                # Sleep before next check
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _make_scaling_decision(self, current_load: int) -> None:
        """Make scaling decision based on current load."""
        current_workers = len(self._workers)
        
        if self.scaling_strategy == ScalingStrategy.FIXED:
            return  # No scaling for fixed strategy
        
        elif self.scaling_strategy == ScalingStrategy.ADAPTIVE:
            # Scale up if load is high
            if current_load > current_workers * 2 and current_workers < self.max_workers:
                new_worker_id = f"worker-{current_workers}"
                if await self._create_worker(new_worker_id):
                    self._last_scale_time = time.time()
                    logger.info(f"Scaled up: added {new_worker_id}")
            
            # Scale down if load is low
            elif current_load < current_workers * 0.5 and current_workers > self.min_workers:
                await self._remove_worker()
                self._last_scale_time = time.time()
                logger.info("Scaled down: removed worker")
        
        elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
            # Analyze load trends
            if len(self._load_history) >= 5:
                recent_loads = [load for _, load in self._load_history[-5:]]
                load_trend = sum(recent_loads[-3:]) - sum(recent_loads[:2])
                
                if load_trend > 0 and current_workers < self.max_workers:
                    # Predictive scale up
                    new_worker_id = f"worker-{current_workers}"
                    if await self._create_worker(new_worker_id):
                        self._last_scale_time = time.time()
                        logger.info(f"Predictive scale up: added {new_worker_id}")
    
    async def _remove_worker(self) -> bool:
        """Remove a worker (least loaded one)."""
        if len(self._workers) <= self.min_workers:
            return False
        
        # Find least loaded worker
        min_load = float('inf')
        worker_to_remove = None
        
        for worker_id, metrics in self._worker_metrics.items():
            if metrics.active_requests < min_load:
                min_load = metrics.active_requests
                worker_to_remove = worker_id
        
        if worker_to_remove and min_load == 0:  # Only remove if idle
            # Cancel worker task
            if worker_to_remove in self._workers:
                worker_task = self._workers[worker_to_remove]
                worker_task.cancel()
                
                # Clean up
                del self._workers[worker_to_remove]
                del self._worker_metrics[worker_to_remove]
                if worker_to_remove in self._worker_queues:
                    del self._worker_queues[worker_to_remove]
                
                return True
        
        return False


class ScalableInferenceEngine:
    """
    High-performance, scalable inference engine with advanced optimization.
    Generation 3: Make It Scale - Optimized for performance and horizontal scaling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.worker_pool = WorkerPool(
            min_workers=self.config.get('min_workers', 2),
            max_workers=self.config.get('max_workers', 20),
            scaling_strategy=ScalingStrategy(self.config.get('scaling_strategy', 2)),
            load_balancing=LoadBalancingStrategy(self.config.get('load_balancing', 4))
        )
        
        self.cache = AdvancedCache(
            max_size=self.config.get('cache_size', 10000),
            compression_enabled=self.config.get('cache_compression', True)
        )
        
        # Performance tracking
        self._request_counter = 0
        self._start_time = time.time()
        self._performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'average_latency': 0.0,
            'p95_latency': 0.0,
            'requests_per_second': 0.0
        }
        
        self._latency_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._running = False
    
    async def start(self) -> bool:
        """Start the scalable inference engine."""
        try:
            if self._running:
                return True
            
            logger.info("Starting scalable inference engine")
            
            # Start worker pool
            success = await self.worker_pool.start()
            if not success:
                return False
            
            self._running = True
            self._start_time = time.time()
            
            # Start performance monitor
            asyncio.create_task(self._performance_monitor())
            
            logger.info("Scalable inference engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scalable inference engine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the scalable inference engine."""
        try:
            logger.info("Stopping scalable inference engine")
            self._running = False
            
            # Stop worker pool
            await self.worker_pool.stop()
            
            logger.info("Scalable inference engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scalable inference engine: {e}")
    
    async def infer(
        self,
        model_id: str,
        input_data: Any,
        priority: int = 1,
        timeout: float = 30.0,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform high-performance inference with caching and scaling.
        
        Args:
            model_id: Model identifier
            input_data: Input data for inference
            priority: Job priority (higher number = higher priority)
            timeout: Operation timeout
            use_cache: Whether to use caching
            
        Returns:
            Inference result dictionary
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = None
            if use_cache:
                cache_key = self._generate_cache_key(model_id, input_data)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self._record_cache_hit()
                    return {
                        'result': cached_result,
                        'cached': True,
                        'latency': time.time() - start_time,
                        'model_id': model_id
                    }
            
            # Create inference job
            job_id = f"job_{self._request_counter}_{int(time.time() * 1000000)}"
            job = InferenceJob(
                job_id=job_id,
                model_id=model_id,
                input_data=input_data,
                priority=priority,
                timeout=timeout
            )
            
            with self._lock:
                self._request_counter += 1
                self._performance_metrics['total_requests'] += 1
            
            # Submit job to worker pool
            await self.worker_pool.submit_job(job)
            
            # Wait for result
            completed_job = await self.worker_pool.get_result(job_id, timeout=timeout)
            
            # Process result
            latency = time.time() - start_time
            self._record_latency(latency)
            
            if completed_job.error:
                with self._lock:
                    self._performance_metrics['failed_requests'] += 1
                raise RuntimeError(f"Inference failed: {completed_job.error}")
            
            # Cache result if successful
            if use_cache and cache_key and completed_job.result:
                self.cache.put(cache_key, completed_job.result)
            
            with self._lock:
                self._performance_metrics['successful_requests'] += 1
            
            return {
                'result': completed_job.result,
                'cached': False,
                'latency': latency,
                'execution_time': completed_job.execution_time,
                'worker_id': completed_job.worker_id,
                'model_id': model_id
            }
            
        except Exception as e:
            latency = time.time() - start_time
            self._record_latency(latency)
            
            with self._lock:
                self._performance_metrics['failed_requests'] += 1
            
            logger.error(f"Inference error: {e}")
            raise
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Calculate requests per second
            if uptime > 0:
                rps = self._performance_metrics['total_requests'] / uptime
            else:
                rps = 0.0
            
            # Calculate latency percentiles
            p95_latency = 0.0
            if self._latency_history:
                sorted_latencies = sorted(self._latency_history)
                p95_index = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else 0.0
            
            return {
                'uptime_seconds': uptime,
                'performance': {
                    **self._performance_metrics,
                    'requests_per_second': rps,
                    'p95_latency': p95_latency
                },
                'worker_pool': self.worker_pool.get_pool_stats(),
                'cache': self.cache.get_stats(),
                'running': self._running
            }
    
    def _generate_cache_key(self, model_id: str, input_data: Any) -> str:
        """Generate cache key for input."""
        try:
            # Create deterministic hash of input
            input_str = json.dumps(input_data, sort_keys=True)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
            return f"{model_id}:{input_hash}"
        except Exception:
            # Fall back to string representation
            input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()[:16]
            return f"{model_id}:{input_hash}"
    
    def _record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._performance_metrics['cache_hits'] += 1
    
    def _record_latency(self, latency: float) -> None:
        """Record request latency."""
        with self._lock:
            self._latency_history.append(latency)
            
            # Update average latency with exponential moving average
            if self._performance_metrics['average_latency'] == 0:
                self._performance_metrics['average_latency'] = latency
            else:
                alpha = 0.1
                self._performance_metrics['average_latency'] = (
                    alpha * latency + 
                    (1 - alpha) * self._performance_metrics['average_latency']
                )
    
    async def _performance_monitor(self) -> None:
        """Monitor performance and log statistics."""
        while self._running:
            try:
                stats = self.get_engine_stats()
                
                logger.info(
                    f"Engine stats - RPS: {stats['performance']['requests_per_second']:.2f}, "
                    f"Avg Latency: {stats['performance']['average_latency']:.3f}s, "
                    f"Workers: {stats['worker_pool']['worker_count']}, "
                    f"Cache Hit Rate: {stats['cache']['hit_rate']:.2%}"
                )
                
                await asyncio.sleep(30.0)  # Log every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(10.0)


# Example usage and testing
async def demo_scalable_inference_engine():
    """Demonstration of scalable inference engine."""
    config = {
        'min_workers': 3,
        'max_workers': 15,
        'scaling_strategy': ScalingStrategy.ADAPTIVE.value,
        'cache_size': 5000,
        'cache_compression': True
    }
    
    engine = ScalableInferenceEngine(config)
    
    try:
        # Start engine
        success = await engine.start()
        if not success:
            print("Failed to start scalable inference engine")
            return
        
        print("Scalable inference engine started")
        
        # Simulate load testing
        print("Simulating inference load...")
        
        # Create test tasks
        test_models = ['classifier_v1', 'regressor_v2', 'nlp_model_v3']
        test_inputs = [
            [1, 2, 3, 4, 5],
            "Hello world, this is a test input",
            {'features': [0.1, 0.2, 0.3], 'metadata': {'version': 1}},
            list(range(100))  # Larger input
        ]
        
        # Submit concurrent requests
        tasks = []
        for i in range(50):  # 50 concurrent requests
            model_id = test_models[i % len(test_models)]
            input_data = test_inputs[i % len(test_inputs)]
            priority = 1 + (i % 3)  # Vary priority
            
            task = asyncio.create_task(
                engine.infer(model_id, input_data, priority=priority)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes
        
        print(f"Load test completed:")
        print(f"  Successful requests: {successes}")
        print(f"  Failed requests: {failures}")
        
        # Show final statistics
        stats = engine.get_engine_stats()
        print(f"\nFinal Engine Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    finally:
        # Stop engine
        await engine.stop()
        print("\nScalable inference engine stopped")


if __name__ == "__main__":
    asyncio.run(demo_scalable_inference_engine())