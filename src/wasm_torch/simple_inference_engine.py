"""Simple Inference Engine - Generation 1: Basic Functionality

Lightweight inference engine for PyTorch-to-WASM models with essential features
for quick deployment and testing.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json

# Handle torch import gracefully
try:
    import torch
    torch_available = True
except ImportError:
    from .mock_torch import torch, MockTensor
    torch_available = False

logger = logging.getLogger(__name__)


@dataclass
class SimpleRequest:
    """Simple request structure for basic inference."""
    request_id: str
    model_id: str
    input_data: Any
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0


@dataclass
class SimpleResult:
    """Simple result structure for basic inference."""
    request_id: str
    model_id: str
    output_data: Any
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class SimpleInferenceEngine:
    """Simple inference engine with essential features for quick deployment."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._models: Dict[str, Any] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }
        self._lock = threading.Lock()
        
        logger.info(f"Simple Inference Engine initialized with {max_workers} workers")
    
    def register_model(self, model_id: str, model: Any) -> None:
        """Register a model for inference."""
        with self._lock:
            self._models[model_id] = model
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
    
    async def infer(self, model_id: str, input_data: Any, timeout: float = 30.0) -> SimpleResult:
        """Perform simple inference."""
        request_id = f"{model_id}_{int(time.time() * 1000000)}"
        request = SimpleRequest(
            request_id=request_id,
            model_id=model_id,
            input_data=input_data,
            timeout=timeout
        )
        
        return await self._process_request(request)
    
    async def infer_batch(self, requests: List[tuple], timeout: float = 30.0) -> List[SimpleResult]:
        """Perform batch inference."""
        tasks = []
        for model_id, input_data in requests:
            task = self.infer(model_id, input_data, timeout)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _process_request(self, request: SimpleRequest) -> SimpleResult:
        """Process an inference request."""
        start_time = time.time()
        
        try:
            # Get model
            model = self._models.get(request.model_id)
            if model is None:
                raise ValueError(f"Model not found: {request.model_id}")
            
            # Process with thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            output_data = await loop.run_in_executor(
                self._thread_pool,
                self._run_model,
                model,
                request.input_data
            )
            
            processing_time = time.time() - start_time
            
            # Update stats
            with self._lock:
                self._stats['total_requests'] += 1
                self._stats['successful_requests'] += 1
                total_time = self._stats['average_processing_time'] * (self._stats['successful_requests'] - 1)
                self._stats['average_processing_time'] = (total_time + processing_time) / self._stats['successful_requests']
            
            return SimpleResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=output_data,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            # Update error stats
            with self._lock:
                self._stats['total_requests'] += 1
                self._stats['failed_requests'] += 1
            
            logger.error(f"Inference failed for {request.request_id}: {error_message}")
            
            return SimpleResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output_data=None,
                processing_time=processing_time,
                success=False,
                error=error_message
            )
    
    def _run_model(self, model, input_data):
        """Run model inference (CPU-bound operation)."""
        if hasattr(model, 'forward'):
            return model.forward(input_data)
        elif hasattr(model, '__call__'):
            return model(input_data)
        else:
            # Simple processing for demo
            if isinstance(input_data, list):
                return [x * 0.5 + 0.1 for x in input_data]
            elif isinstance(input_data, (int, float)):
                return input_data * 0.5 + 0.1
            else:
                return f"processed_{input_data}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return self._stats.copy()
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        self._thread_pool.shutdown(wait=True)
        logger.info("Simple Inference Engine shutdown")


class SimpleTensorOps:
    """Simple tensor operations for WASM compatibility."""
    
    @staticmethod
    def relu(tensor):
        """ReLU activation."""
        if torch_available and hasattr(tensor, 'clamp_'):
            return tensor.clamp_(min=0.0)
        elif isinstance(tensor, list):
            return [max(0.0, x) for x in tensor]
        else:
            return max(0.0, tensor)
    
    @staticmethod
    def linear(input_tensor, weight, bias=None):
        """Linear transformation."""
        if torch_available and hasattr(input_tensor, 'matmul'):
            output = torch.matmul(input_tensor, weight.T)
            if bias is not None:
                output = output + bias
            return output
        else:
            # Simple mock implementation
            if isinstance(input_tensor, list) and isinstance(weight, list):
                # Simplified matrix multiplication
                if len(weight) > 0 and len(input_tensor) == len(weight[0]):
                    output = []
                    for row in weight:
                        value = sum(a * b for a, b in zip(input_tensor, row))
                        output.append(value)
                    
                    if bias and len(bias) == len(output):
                        output = [o + b for o, b in zip(output, bias)]
                    
                    return output
            
            return input_tensor
    
    @staticmethod
    def softmax(tensor):
        """Softmax activation."""
        if torch_available and hasattr(tensor, 'softmax'):
            return torch.softmax(tensor, dim=-1)
        elif isinstance(tensor, list):
            # Simple softmax implementation
            import math
            exp_values = [math.exp(x - max(tensor)) for x in tensor]
            sum_exp = sum(exp_values)
            return [x / sum_exp for x in exp_values]
        else:
            return tensor


class SimpleModel:
    """Simple model implementation for testing."""
    
    def __init__(self, layers: List[Dict[str, Any]]):
        self.layers = layers
        self.ops = SimpleTensorOps()
    
    def forward(self, input_data):
        """Forward pass through the model."""
        current_input = input_data
        
        for layer in self.layers:
            layer_type = layer.get('type', 'unknown')
            
            if layer_type == 'linear':
                weight = layer.get('weight', [[1.0]])
                bias = layer.get('bias', None)
                current_input = self.ops.linear(current_input, weight, bias)
            
            elif layer_type == 'relu':
                current_input = self.ops.relu(current_input)
            
            elif layer_type == 'softmax':
                current_input = self.ops.softmax(current_input)
            
            else:
                logger.warning(f"Unknown layer type: {layer_type}")
        
        return current_input
    
    @classmethod
    def create_classifier(cls, input_size: int, hidden_size: int, num_classes: int):
        """Create a simple classifier model."""
        import random
        
        # Generate random weights for demo
        hidden_weights = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        output_weights = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(num_classes)]
        
        layers = [
            {'type': 'linear', 'weight': hidden_weights, 'bias': [0.1] * hidden_size},
            {'type': 'relu'},
            {'type': 'linear', 'weight': output_weights, 'bias': [0.0] * num_classes},
            {'type': 'softmax'}
        ]
        
        return cls(layers)


# Demo function
async def demo_simple_inference_engine():
    """Demonstration of the simple inference engine."""
    
    try:
        print("Simple Inference Engine Demo - Generation 1")
        print("=" * 50)
        
        # Create engine
        engine = SimpleInferenceEngine(max_workers=2)
        print("✓ Engine created")
        
        # Create test models
        classifier = SimpleModel.create_classifier(4, 8, 3)
        simple_processor = lambda x: [val * 2 + 1 for val in x] if isinstance(x, list) else x * 2 + 1
        
        # Register models
        engine.register_model('classifier', classifier)
        engine.register_model('processor', simple_processor)
        print(f"✓ Registered models: {engine.list_models()}")
        
        # Test single inference
        print("\\nTesting single inference...")
        result1 = await engine.infer('classifier', [0.5, -0.3, 0.8, 0.1])
        print(f"Classifier result: {result1.output_data} (time: {result1.processing_time:.4f}s)")
        
        result2 = await engine.infer('processor', [1, 2, 3, 4])
        print(f"Processor result: {result2.output_data} (time: {result2.processing_time:.4f}s)")
        
        # Test batch inference
        print("\\nTesting batch inference...")
        batch_requests = [
            ('classifier', [0.2, 0.4, -0.1, 0.6]),
            ('processor', [5, 6, 7]),
            ('classifier', [-0.3, 0.9, 0.2, -0.5])
        ]
        
        batch_results = await engine.infer_batch(batch_requests)
        print(f"✓ Processed {len(batch_results)} requests in batch")
        
        for i, result in enumerate(batch_results):
            print(f"  Result {i+1}: {result.output_data[:3] if isinstance(result.output_data, list) else result.output_data} "
                  f"(time: {result.processing_time:.4f}s)")
        
        # Show statistics
        stats = engine.get_stats()
        print(f"\\nEngine Statistics:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Average processing time: {stats['average_processing_time']:.4f}s")
        
        # Test error handling
        print("\\nTesting error handling...")
        error_result = await engine.infer('nonexistent_model', [1, 2, 3])
        print(f"Error result: success={error_result.success}, error={error_result.error}")
        
        # Shutdown
        engine.shutdown()
        print("\\n✓ Engine shutdown complete")
        
        print("\\n🎉 Simple Inference Engine Demo Complete!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_simple_inference_engine())