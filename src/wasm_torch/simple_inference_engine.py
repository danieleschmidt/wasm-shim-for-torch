"""
Simple Inference Engine - Generation 1: Make It Work
Lightweight, dependency-free inference engine for WASM-Torch.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Simple inference request structure."""
    
    request_id: str
    model_id: str
    input_data: Any
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    priority: int = 1  # Lower number = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Simple inference result structure."""
    
    request_id: str
    model_id: str
    output_data: Any
    success: bool
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleModelRegistry:
    """
    Simple model registry for managing loaded models.
    No external dependencies, just basic in-memory storage.
    """
    
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def register_model(
        self, 
        model_id: str, 
        model_data: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model in the registry."""
        try:
            with self._lock:
                self._models[model_id] = {
                    'data': model_data,
                    'metadata': metadata or {},
                    'registered_at': time.time(),
                    'access_count': 0,
                    'last_accessed': None
                }
            
            logger.info(f"Registered model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Retrieve a model from the registry."""
        try:
            with self._lock:
                if model_id not in self._models:
                    return None
                
                model_info = self._models[model_id]
                model_info['access_count'] += 1
                model_info['last_accessed'] = time.time()
                
                return model_info['data']
                
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def unregister_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        try:
            with self._lock:
                if model_id in self._models:
                    del self._models[model_id]
                    logger.info(f"Unregistered model: {model_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister model {model_id}: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        with self._lock:
            return list(self._models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific model."""
        with self._lock:
            if model_id not in self._models:
                return None
            
            info = self._models[model_id].copy()
            # Don't return the actual model data, just metadata
            info.pop('data', None)
            return info


class SimpleInferenceEngine:
    """
    Simple inference engine - Generation 1 implementation.
    Focuses on basic functionality with minimal dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', 4)
        self.max_queue_size = self.config.get('max_queue_size', 100)
        
        # Core components
        self.model_registry = SimpleModelRegistry()
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.result_cache: Dict[str, InferenceResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # State management
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._statistics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_execution_time': 0.0,
            'peak_queue_size': 0
        }
        
    async def start(self) -> bool:
        """Start the inference engine."""
        try:
            if self._running:
                logger.warning("Inference engine is already running")
                return True
            
            logger.info("Starting Simple Inference Engine")
            self._running = True
            
            # Start worker tasks
            for i in range(self.max_workers):
                task = asyncio.create_task(
                    self._worker_loop(f"worker-{i}")
                )
                self._worker_tasks.append(task)
            
            logger.info(f"Started {len(self._worker_tasks)} worker tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start inference engine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the inference engine gracefully."""
        try:
            logger.info("Stopping Simple Inference Engine")
            self._running = False
            
            # Cancel worker tasks
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Simple Inference Engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping inference engine: {e}")
    
    async def submit_request(self, request: InferenceRequest) -> bool:
        """Submit an inference request for processing."""
        try:
            if not self._running:
                logger.error("Inference engine is not running")
                return False
            
            # Check queue capacity
            if self.request_queue.full():
                logger.warning("Request queue is full, rejecting request")
                return False
            
            await self.request_queue.put(request)
            self._statistics['total_requests'] += 1
            
            # Update peak queue size
            queue_size = self.request_queue.qsize()
            if queue_size > self._statistics['peak_queue_size']:
                self._statistics['peak_queue_size'] = queue_size
            
            logger.debug(f"Submitted request: {request.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            return False
    
    async def get_result(
        self, 
        request_id: str, 
        timeout: float = 30.0
    ) -> Optional[InferenceResult]:
        """Get the result of an inference request."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        logger.warning(f"Timeout waiting for result: {request_id}")
        return None
    
    def register_model(
        self, 
        model_id: str, 
        model_data: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model for inference."""
        return self.model_registry.register_model(model_id, model_data, metadata)
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        return self.model_registry.unregister_model(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return self.model_registry.list_models()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self._statistics.copy()
        stats.update({
            'running': self._running,
            'active_workers': len(self._worker_tasks),
            'queue_size': self.request_queue.qsize(),
            'registered_models': len(self.model_registry.list_models()),
            'cached_results': len(self.result_cache)
        })
        return stats
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing inference requests."""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                result = await self._process_request(request, worker_id)
                
                # Store result in cache
                self.result_cache[request.request_id] = result
                
                # Update statistics
                if result.success:
                    self._statistics['successful_requests'] += 1
                else:
                    self._statistics['failed_requests'] += 1
                
                # Update average execution time
                total_requests = (
                    self._statistics['successful_requests'] + 
                    self._statistics['failed_requests']
                )
                if total_requests > 0:
                    current_avg = self._statistics['average_execution_time']
                    new_avg = (
                        (current_avg * (total_requests - 1) + result.execution_time) / 
                        total_requests
                    )
                    self._statistics['average_execution_time'] = new_avg
                
                logger.debug(f"Worker {worker_id} processed request {request.request_id}")
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_request(
        self, 
        request: InferenceRequest, 
        worker_id: str
    ) -> InferenceResult:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            # Get the model
            model = self.model_registry.get_model(request.model_id)
            if model is None:
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    output_data=None,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Model not found: {request.model_id}"
                )
            
            # Perform inference (simplified mock implementation)
            output = await self._run_inference(model, request.input_data, request)
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=output,
                success=True,
                execution_time=execution_time,
                metadata={
                    'worker_id': worker_id,
                    'processed_at': time.time()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=None,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata={
                    'worker_id': worker_id,
                    'processed_at': time.time()
                }
            )
    
    async def _run_inference(
        self, 
        model: Any, 
        input_data: Any, 
        request: InferenceRequest
    ) -> Any:
        """
        Run actual inference on the model.
        This is a simplified implementation for Generation 1.
        """
        # Simulate processing time
        processing_time = 0.01 + (hash(str(input_data)) % 100) / 10000
        await asyncio.sleep(processing_time)
        
        # Mock inference result based on input
        if isinstance(input_data, (list, tuple)):
            # Return sum for numeric inputs
            try:
                numeric_sum = sum(float(x) for x in input_data if isinstance(x, (int, float)))
                return {'prediction': numeric_sum, 'confidence': 0.95}
            except:
                return {'prediction': len(input_data), 'confidence': 0.8}
        
        elif isinstance(input_data, str):
            # Return text analysis for string inputs
            return {
                'prediction': len(input_data.split()),
                'sentiment': 'positive' if len(input_data) % 2 == 0 else 'negative',
                'confidence': 0.85
            }
        
        elif isinstance(input_data, dict):
            # Return aggregated results for dict inputs
            return {
                'prediction': len(input_data),
                'keys': list(input_data.keys()),
                'confidence': 0.90
            }
        
        else:
            # Default response
            return {
                'prediction': str(type(input_data).__name__),
                'confidence': 0.75
            }


# Utility functions for testing and demonstration
async def demo_simple_inference_engine():
    """Demonstration of the simple inference engine."""
    engine = SimpleInferenceEngine({
        'max_workers': 2,
        'max_queue_size': 10
    })
    
    try:
        # Start engine
        success = await engine.start()
        if not success:
            print("Failed to start inference engine")
            return
        
        # Register a mock model
        mock_model = {'type': 'simple_classifier', 'version': '1.0'}
        engine.register_model('test_model', mock_model)
        
        # Submit some test requests
        test_inputs = [
            [1, 2, 3, 4, 5],
            "Hello world, this is a test",
            {'key1': 'value1', 'key2': 'value2'},
            42
        ]
        
        print("Submitting inference requests...")
        request_ids = []
        
        for i, input_data in enumerate(test_inputs):
            request = InferenceRequest(
                request_id=f"test_request_{i}",
                model_id='test_model',
                input_data=input_data
            )
            
            success = await engine.submit_request(request)
            if success:
                request_ids.append(request.request_id)
                print(f"Submitted: {request.request_id}")
        
        # Get results
        print("\nGetting results...")
        for request_id in request_ids:
            result = await engine.get_result(request_id, timeout=5.0)
            if result:
                print(f"Result for {request_id}: {result.output_data}")
            else:
                print(f"No result for {request_id}")
        
        # Show statistics
        stats = engine.get_statistics()
        print(f"\nEngine statistics:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Average execution time: {stats['average_execution_time']:.4f}s")
        
    finally:
        # Stop engine
        await engine.stop()


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demo_simple_inference_engine())