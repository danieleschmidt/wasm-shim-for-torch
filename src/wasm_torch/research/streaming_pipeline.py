"""Streaming multi-modal inference pipeline for real-time applications.

This module implements a novel streaming inference system that can process
multiple data modalities (vision, audio, text) concurrently with adaptive
resource allocation and quality-of-service guarantees.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from collections import deque
import threading

import numpy as np
import torch
import torch.nn as nn

from ..security import log_security_event
from ..validation import validate_tensor_safe
from ..performance import PerformanceProfiler


logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported data modalities."""
    
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    SENSOR = "sensor"
    MULTIMODAL = "multimodal"


class PriorityLevel(Enum):
    """Processing priority levels."""
    
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class StreamingConfig:
    """Configuration for streaming inference pipeline."""
    
    max_batch_size: int = 8
    max_latency_ms: int = 100
    buffer_size: int = 1000
    adaptive_batching: bool = True
    quality_aware: bool = True
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        "memory_mb": 512,
        "cpu_percent": 80,
    })


@dataclass
class InferenceRequest:
    """Single inference request with metadata."""
    
    request_id: str
    modality: ModalityType
    data: Union[torch.Tensor, np.ndarray, str, bytes]
    priority: PriorityLevel = PriorityLevel.NORMAL
    timestamp: float = field(default_factory=time.time)
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[callable] = None


@dataclass
class InferenceResponse:
    """Response from inference with timing information."""
    
    request_id: str
    result: Any
    latency_ms: float
    queue_time_ms: float
    processing_time_ms: float
    model_name: str
    confidence: Optional[float] = None
    error: Optional[str] = None


class DataProcessor(ABC):
    """Abstract base class for modality-specific data processors."""
    
    @abstractmethod
    async def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess raw data for inference."""
        pass
    
    @abstractmethod
    async def postprocess(self, output: torch.Tensor) -> Any:
        """Postprocess model output."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        pass


class VisionProcessor(DataProcessor):
    """Vision data processor for image/video streams."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    async def preprocess(self, data: Union[torch.Tensor, np.ndarray, bytes]) -> torch.Tensor:
        """Preprocess image data."""
        
        if isinstance(data, bytes):
            # Simulate image decoding
            await asyncio.sleep(0.001)  # Simulate decode time
            # In practice would decode with PIL/OpenCV
            tensor = torch.randn(3, *self.target_size)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = data
        
        validate_tensor_safe(tensor, "vision_input")
        
        # Normalize and resize
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
        # Simulate resizing and normalization
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.target_size, mode='bilinear', align_corners=False
        )
        
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    async def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """Postprocess vision model output."""
        
        validate_tensor_safe(output, "vision_output")
        
        # Apply softmax for classification
        if len(output.shape) == 2:  # Classification
            probs = torch.softmax(output, dim=1)
            top_k = torch.topk(probs, k=5, dim=1)
            
            return {
                "predictions": [
                    {"class_id": int(idx), "confidence": float(prob)}
                    for idx, prob in zip(top_k.indices[0], top_k.values[0])
                ],
                "raw_output": output.tolist(),
            }
        
        # For other output shapes (detection, segmentation)
        return {"raw_output": output.tolist()}
    
    def validate_input(self, data: Any) -> bool:
        """Validate vision input."""
        if isinstance(data, torch.Tensor):
            return len(data.shape) >= 2 and data.numel() > 0
        elif isinstance(data, np.ndarray):
            return len(data.shape) >= 2 and data.size > 0
        elif isinstance(data, bytes):
            return len(data) > 0
        return False


class AudioProcessor(DataProcessor):
    """Audio data processor for audio streams."""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
    async def preprocess(self, data: Union[torch.Tensor, np.ndarray, bytes]) -> torch.Tensor:
        """Preprocess audio data."""
        
        if isinstance(data, bytes):
            # Simulate audio decoding
            await asyncio.sleep(0.002)  # Simulate decode time
            # In practice would decode with librosa/torchaudio
            tensor = torch.randn(self.chunk_size)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = data
        
        validate_tensor_safe(tensor, "audio_input")
        
        # Ensure proper shape
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        # Pad or truncate to fixed size
        if tensor.shape[-1] > self.chunk_size:
            tensor = tensor[:, :self.chunk_size]
        elif tensor.shape[-1] < self.chunk_size:
            padding = self.chunk_size - tensor.shape[-1]
            tensor = torch.nn.functional.pad(tensor, (0, padding))
        
        # Apply mel-spectrogram transform
        # Simulate with random features
        n_mels = 80
        mel_features = torch.randn(tensor.shape[0], n_mels, 100)
        
        return mel_features
    
    async def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """Postprocess audio model output."""
        
        validate_tensor_safe(output, "audio_output")
        
        # For speech recognition or audio classification
        if len(output.shape) == 3:  # Sequence output
            # Apply log softmax for sequence prediction
            log_probs = torch.log_softmax(output, dim=-1)
            
            # Simulate CTC decoding or attention decoding
            return {
                "transcription": "sample transcription",
                "confidence": float(torch.mean(torch.max(log_probs, dim=-1)[0])),
                "raw_output": output.tolist(),
            }
        
        return {"raw_output": output.tolist()}
    
    def validate_input(self, data: Any) -> bool:
        """Validate audio input."""
        if isinstance(data, torch.Tensor):
            return len(data.shape) >= 1 and data.numel() > 0
        elif isinstance(data, np.ndarray):
            return len(data.shape) >= 1 and data.size > 0
        elif isinstance(data, bytes):
            return len(data) > 0
        return False


class TextProcessor(DataProcessor):
    """Text data processor for NLP tasks."""
    
    def __init__(self, vocab_size: int = 50000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    async def preprocess(self, data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Preprocess text data."""
        
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, list):
            texts = data
        else:
            # Already tokenized
            tensor = data
            validate_tensor_safe(tensor, "text_input")
            return tensor
        
        # Simulate tokenization
        await asyncio.sleep(0.001)  # Simulate tokenization time
        
        batch_size = len(texts)
        # Generate random token IDs (in practice would use real tokenizer)
        token_ids = torch.randint(0, self.vocab_size, (batch_size, self.max_length))
        
        # Create attention mask
        attention_mask = torch.ones_like(token_ids)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }
    
    async def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        """Postprocess text model output."""
        
        validate_tensor_safe(output, "text_output")
        
        # For classification tasks
        if len(output.shape) == 2:
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1)
            
            return {
                "prediction": int(prediction[0]),
                "confidence": float(torch.max(probs[0])),
                "raw_output": output.tolist(),
            }
        
        # For generation tasks
        elif len(output.shape) == 3:
            # Simulate text generation
            return {
                "generated_text": "sample generated text",
                "raw_output": output.tolist(),
            }
        
        return {"raw_output": output.tolist()}
    
    def validate_input(self, data: Any) -> bool:
        """Validate text input."""
        if isinstance(data, str):
            return len(data.strip()) > 0
        elif isinstance(data, list):
            return len(data) > 0 and all(isinstance(x, str) for x in data)
        elif isinstance(data, torch.Tensor):
            return data.numel() > 0
        return False


class AdaptiveBatcher:
    """Adaptive batching system for optimal throughput."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.pending_requests: Dict[ModalityType, deque] = {
            modality: deque() for modality in ModalityType
        }
        self.batch_timers: Dict[ModalityType, Optional[float]] = {
            modality: None for modality in ModalityType
        }
        self.stats = {
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
        }
    
    def add_request(self, request: InferenceRequest) -> None:
        """Add request to appropriate batch queue."""
        
        # Sort by priority
        queue = self.pending_requests[request.modality]
        
        # Insert based on priority
        inserted = False
        for i, existing_request in enumerate(queue):
            if request.priority.value < existing_request.priority.value:
                queue.insert(i, request)
                inserted = True
                break
        
        if not inserted:
            queue.append(request)
        
        # Start batch timer if first request
        if self.batch_timers[request.modality] is None:
            self.batch_timers[request.modality] = time.time()
    
    def should_process_batch(self, modality: ModalityType) -> bool:
        """Determine if batch should be processed now."""
        
        queue = self.pending_requests[modality]
        if not queue:
            return False
        
        # Check batch size threshold
        if len(queue) >= self.config.max_batch_size:
            return True
        
        # Check time threshold
        batch_timer = self.batch_timers[modality]
        if batch_timer and (time.time() - batch_timer) * 1000 >= self.config.max_latency_ms:
            return True
        
        # Check priority - process immediately if critical
        if queue[0].priority == PriorityLevel.CRITICAL:
            return True
        
        return False
    
    def get_batch(self, modality: ModalityType) -> List[InferenceRequest]:
        """Get next batch for processing."""
        
        queue = self.pending_requests[modality]
        if not queue:
            return []
        
        batch_size = min(len(queue), self.config.max_batch_size)
        batch = [queue.popleft() for _ in range(batch_size)]
        
        # Reset timer
        self.batch_timers[modality] = time.time() if queue else None
        
        # Update statistics
        self.stats["total_batches"] += 1
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + batch_size) /
            self.stats["total_batches"]
        )
        
        return batch


class QualityOfServiceManager:
    """Manages quality of service guarantees and resource allocation."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.resource_usage = {
            "memory_mb": 0.0,
            "cpu_percent": 0.0,
        }
        self.qos_metrics = {
            "p95_latency": 0.0,
            "throughput": 0.0,
            "error_rate": 0.0,
        }
    
    def can_accept_request(self, request: InferenceRequest) -> bool:
        """Check if system can accept new request."""
        
        # Check resource limits
        if (self.resource_usage["memory_mb"] >= self.config.resource_limits["memory_mb"] or
            self.resource_usage["cpu_percent"] >= self.config.resource_limits["cpu_percent"]):
            
            # Only accept critical priority requests when at capacity
            if request.priority != PriorityLevel.CRITICAL:
                return False
        
        # Check timeout constraints
        current_time = time.time()
        if (current_time - request.timestamp) * 1000 >= request.timeout_ms:
            return False
        
        return True
    
    def track_request(self, request: InferenceRequest) -> None:
        """Track active request for resource management."""
        self.active_requests[request.request_id] = request
        
        # Estimate resource usage
        base_memory = 50  # Base memory per request in MB
        if request.modality == ModalityType.VISION:
            base_memory += 100
        elif request.modality == ModalityType.AUDIO:
            base_memory += 50
        
        self.resource_usage["memory_mb"] += base_memory
        self.resource_usage["cpu_percent"] += 5  # Base CPU usage per request
    
    def complete_request(self, request_id: str, response: InferenceResponse) -> None:
        """Mark request as complete and update metrics."""
        
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            
            # Update resource usage
            base_memory = 50
            if request.modality == ModalityType.VISION:
                base_memory += 100
            elif request.modality == ModalityType.AUDIO:
                base_memory += 50
            
            self.resource_usage["memory_mb"] -= base_memory
            self.resource_usage["cpu_percent"] -= 5
        
        # Update QoS metrics
        self._update_qos_metrics(response)
    
    def _update_qos_metrics(self, response: InferenceResponse) -> None:
        """Update quality of service metrics."""
        
        # Simple moving average for metrics
        alpha = 0.1  # Smoothing factor
        
        # Update latency (P95 approximation)
        self.qos_metrics["p95_latency"] = (
            (1 - alpha) * self.qos_metrics["p95_latency"] +
            alpha * response.latency_ms
        )
        
        # Update throughput (requests per second)
        self.qos_metrics["throughput"] = (
            (1 - alpha) * self.qos_metrics["throughput"] +
            alpha * (1000.0 / response.latency_ms if response.latency_ms > 0 else 0)
        )
        
        # Update error rate
        error_indicator = 1.0 if response.error else 0.0
        self.qos_metrics["error_rate"] = (
            (1 - alpha) * self.qos_metrics["error_rate"] +
            alpha * error_indicator
        )


class StreamingInferencePipeline:
    """Main streaming inference pipeline for multi-modal processing."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.is_running = False
        
        # Core components
        self.batcher = AdaptiveBatcher(config)
        self.qos_manager = QualityOfServiceManager(config)
        self.profiler = PerformanceProfiler()
        
        # Data processors
        self.processors: Dict[ModalityType, DataProcessor] = {
            ModalityType.VISION: VisionProcessor(),
            ModalityType.AUDIO: AudioProcessor(),
            ModalityType.TEXT: TextProcessor(),
        }
        
        # Model registry
        self.models: Dict[ModalityType, nn.Module] = {}
        
        # Processing queues and workers
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_size)
        self.output_callbacks: Dict[str, callable] = {}
        
        # Statistics
        self.stats = {
            "requests_processed": 0,
            "total_latency": 0.0,
            "errors": 0,
        }
    
    def register_model(self, modality: ModalityType, model: nn.Module) -> None:
        """Register model for specific modality."""
        
        model.eval()  # Set to evaluation mode
        self.models[modality] = model
        
        logger.info(f"Registered {modality.value} model with "
                   f"{sum(p.numel() for p in model.parameters())} parameters")
    
    async def submit_request(self, request: InferenceRequest) -> str:
        """Submit inference request to pipeline."""
        
        # Validate request
        if not self._validate_request(request):
            raise ValueError(f"Invalid request: {request.request_id}")
        
        # Check QoS constraints
        if not self.qos_manager.can_accept_request(request):
            raise RuntimeError("System at capacity, request rejected")
        
        # Add to processing queue
        await self.input_queue.put(request)
        self.qos_manager.track_request(request)
        
        log_security_event("inference_request_submitted", {
            "request_id": request.request_id,
            "modality": request.modality.value,
            "priority": request.priority.value,
        })
        
        return request.request_id
    
    def _validate_request(self, request: InferenceRequest) -> bool:
        """Validate inference request."""
        
        # Check if modality is supported
        if request.modality not in self.processors:
            logger.error(f"Unsupported modality: {request.modality}")
            return False
        
        # Check if model is registered
        if request.modality not in self.models:
            logger.error(f"No model registered for modality: {request.modality}")
            return False
        
        # Validate input data
        processor = self.processors[request.modality]
        if not processor.validate_input(request.data):
            logger.error(f"Invalid input data for modality: {request.modality}")
            return False
        
        return True
    
    async def start(self) -> None:
        """Start the streaming inference pipeline."""
        
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        logger.info("Starting streaming inference pipeline")
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._input_processor()),
            asyncio.create_task(self._batch_scheduler()),
        ]
        
        # Start modality-specific processors
        for modality in ModalityType:
            if modality in self.models:
                tasks.append(asyncio.create_task(self._modality_processor(modality)))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the streaming inference pipeline."""
        
        self.is_running = False
        logger.info("Stopping streaming inference pipeline")
        
        # Wait for pending requests to complete
        while not self.input_queue.empty():
            await asyncio.sleep(0.1)
    
    async def _input_processor(self) -> None:
        """Process incoming requests and add to batches."""
        
        while self.is_running:
            try:
                # Get request with timeout
                request = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
                
                # Add to batcher
                self.batcher.add_request(request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Input processor error: {e}")
    
    async def _batch_scheduler(self) -> None:
        """Schedule batch processing based on adaptive batching policy."""
        
        while self.is_running:
            try:
                # Check each modality for ready batches
                for modality in ModalityType:
                    if modality in self.models and self.batcher.should_process_batch(modality):
                        batch = self.batcher.get_batch(modality)
                        if batch:
                            # Signal batch processor
                            await asyncio.sleep(0.001)  # Yield control
                
                await asyncio.sleep(0.01)  # Check every 10ms
                
            except Exception as e:
                logger.error(f"Batch scheduler error: {e}")
    
    async def _modality_processor(self, modality: ModalityType) -> None:
        """Process batches for specific modality."""
        
        processor = self.processors[modality]
        model = self.models[modality]
        
        while self.is_running:
            try:
                # Check for ready batch
                if self.batcher.should_process_batch(modality):
                    batch = self.batcher.get_batch(modality)
                    if batch:
                        await self._process_batch(batch, modality, processor, model)
                
                await asyncio.sleep(0.01)  # Check every 10ms
                
            except Exception as e:
                logger.error(f"Modality processor error for {modality}: {e}")
    
    async def _process_batch(
        self,
        batch: List[InferenceRequest],
        modality: ModalityType,
        processor: DataProcessor,
        model: nn.Module,
    ) -> None:
        """Process batch of requests for given modality."""
        
        batch_start_time = time.time()
        
        try:
            # Preprocess batch
            preprocessed_inputs = []
            for request in batch:
                start_time = time.time()
                processed_input = await processor.preprocess(request.data)
                preprocessed_inputs.append(processed_input)
                
                # Track preprocessing time
                preprocess_time = (time.time() - start_time) * 1000
                
            # Create batch tensor
            if modality == ModalityType.TEXT:
                # Special handling for text (dict inputs)
                batch_input = {
                    key: torch.cat([inp[key] for inp in preprocessed_inputs], dim=0)
                    for key in preprocessed_inputs[0].keys()
                }
            else:
                batch_input = torch.cat(preprocessed_inputs, dim=0)
            
            # Run inference
            inference_start = time.time()
            with torch.no_grad():
                with self.profiler.profile_operation(f"{modality.value}_inference"):
                    if isinstance(batch_input, dict):
                        batch_output = model(**batch_input)
                    else:
                        batch_output = model(batch_input)
            
            inference_time = (time.time() - inference_start) * 1000
            
            # Split batch output and postprocess
            if len(batch_output.shape) > 1:
                outputs = torch.split(batch_output, 1, dim=0)
            else:
                outputs = [batch_output]
            
            # Process results
            for i, (request, output) in enumerate(zip(batch, outputs)):
                try:
                    postprocess_start = time.time()
                    result = await processor.postprocess(output)
                    postprocess_time = (time.time() - postprocess_start) * 1000
                    
                    # Create response
                    total_latency = (time.time() - request.timestamp) * 1000
                    queue_time = (batch_start_time - request.timestamp) * 1000
                    processing_time = inference_time + postprocess_time
                    
                    response = InferenceResponse(
                        request_id=request.request_id,
                        result=result,
                        latency_ms=total_latency,
                        queue_time_ms=queue_time,
                        processing_time_ms=processing_time,
                        model_name=f"{modality.value}_model",
                        confidence=result.get("confidence") if isinstance(result, dict) else None,
                    )
                    
                    # Update statistics
                    self.stats["requests_processed"] += 1
                    self.stats["total_latency"] += total_latency
                    
                    # Complete request in QoS manager
                    self.qos_manager.complete_request(request.request_id, response)
                    
                    # Call callback if provided
                    if request.callback:
                        await request.callback(response)
                    
                    logger.debug(f"Processed request {request.request_id} in {total_latency:.1f}ms")
                    
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    self.stats["errors"] += 1
                    
                    # Create error response
                    error_response = InferenceResponse(
                        request_id=request.request_id,
                        result=None,
                        latency_ms=(time.time() - request.timestamp) * 1000,
                        queue_time_ms=0,
                        processing_time_ms=0,
                        model_name=f"{modality.value}_model",
                        error=str(e),
                    )
                    
                    if request.callback:
                        await request.callback(error_response)
        
        except Exception as e:
            logger.error(f"Batch processing error for {modality}: {e}")
            self.stats["errors"] += len(batch)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        
        avg_latency = (
            self.stats["total_latency"] / self.stats["requests_processed"]
            if self.stats["requests_processed"] > 0 else 0
        )
        
        return {
            "requests_processed": self.stats["requests_processed"],
            "average_latency_ms": avg_latency,
            "error_rate": self.stats["errors"] / max(self.stats["requests_processed"], 1),
            "qos_metrics": self.qos_manager.qos_metrics,
            "resource_usage": self.qos_manager.resource_usage,
            "batch_stats": self.batcher.stats,
            "active_requests": len(self.qos_manager.active_requests),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of pipeline components."""
        
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time(),
        }
        
        # Check if pipeline is running
        health_status["components"]["pipeline"] = {
            "status": "running" if self.is_running else "stopped"
        }
        
        # Check models
        for modality, model in self.models.items():
            try:
                # Quick inference test
                test_input = torch.randn(1, 3, 224, 224)  # Generic test input
                with torch.no_grad():
                    _ = model(test_input)
                
                health_status["components"][f"{modality.value}_model"] = {
                    "status": "healthy"
                }
            except Exception as e:
                health_status["components"][f"{modality.value}_model"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        # Check resource usage
        if self.qos_manager.resource_usage["memory_mb"] > self.config.resource_limits["memory_mb"] * 0.9:
            health_status["status"] = "warning"
            health_status["warnings"] = ["High memory usage"]
        
        return health_status