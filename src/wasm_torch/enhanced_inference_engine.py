"""Enhanced inference engine with adaptive optimization and intelligent batching."""

import asyncio
import time
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import threading
import weakref


logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Inference request with metadata and tracking."""
    request_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    timeout: float = 30.0
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Inference result with performance metrics."""
    request_id: str
    output_data: Any
    latency_ms: float
    memory_mb: float
    model_version: str = "1.0"
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


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


class EnhancedInferenceEngine:
    """Enhanced inference engine with adaptive optimization and intelligent features."""
    
    def __init__(
        self,
        max_concurrent_requests: int = 100,
        batch_timeout_ms: float = 50.0,
        enable_caching: bool = True,
        enable_monitoring: bool = True
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        
        # Core components
        self.batch_processor = AdaptiveBatchProcessor()
        self.model_cache = IntelligentModelCache() if enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency_ms': 0.0,
            'throughput_rps': 0.0
        }
        
        # Background processing
        self._shutdown = False
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._metrics_update_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Inference Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the inference engine."""
        self._batch_processor_task = asyncio.create_task(self._batch_processing_loop())
        if self.enable_monitoring:
            self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())
        
        logger.info("Enhanced Inference Engine started")
    
    async def submit_request(self, request: InferenceRequest) -> InferenceResult:
        """Submit inference request and return result."""
        if len(self.active_requests) >= self.max_concurrent_requests:
            raise RuntimeError("Maximum concurrent requests exceeded")
        
        self.active_requests[request.request_id] = request
        self.batch_processor.add_request(request)
        
        # Wait for completion
        return await self._wait_for_completion(request.request_id, request.timeout)
    
    async def _wait_for_completion(self, request_id: str, timeout: float) -> InferenceResult:
        """Wait for request completion with timeout."""
        start_time = time.time()
        
        while request_id in self.active_requests:
            if time.time() - start_time > timeout:
                self.active_requests.pop(request_id, None)
                return InferenceResult(
                    request_id=request_id,
                    output_data=None,
                    latency_ms=(time.time() - start_time) * 1000,
                    memory_mb=0.0,
                    success=False,
                    error="Request timeout"
                )
            
            await asyncio.sleep(0.001)  # 1ms check interval
        
        # Request completed, get result from storage (simplified)
        latency_ms = (time.time() - start_time) * 1000
        return InferenceResult(
            request_id=request_id,
            output_data=f"result_for_{request_id}",  # Placeholder
            latency_ms=latency_ms,
            memory_mb=128.0,  # Placeholder
            success=True
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
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the inference engine."""
        logger.info("Shutting down Enhanced Inference Engine")
        self._shutdown = True
        
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        if self._metrics_update_task:
            self._metrics_update_task.cancel()
        
        self.executor.shutdown(wait=True)
        logger.info("Enhanced Inference Engine shutdown complete")
