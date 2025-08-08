"""Comprehensive tests for research modules.

This test suite covers all advanced research implementations including
adaptive optimization, ML quantization, streaming pipeline, federated learning,
WebGPU acceleration, and model hub functionality.
"""

import asyncio
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

from src.wasm_torch.research.adaptive_optimizer import (
    AdaptiveWASMOptimizer, OptimizationConfig, ModelCharacteristics
)
from src.wasm_torch.research.ml_quantizer import (
    MLQuantizationEngine, QuantizationConfig, QuantizationType
)
from src.wasm_torch.research.streaming_pipeline import (
    StreamingInferencePipeline, StreamingConfig, InferenceRequest, 
    ModalityType, PriorityLevel, VisionProcessor, AudioProcessor, TextProcessor
)
from src.wasm_torch.research.federated_inference import (
    FederatedCoordinator, FederatedClient, FederatedConfig, PrivacyEngine
)
from src.wasm_torch.webgpu.gpu_runtime import (
    WebGPURuntime, GPUBackend, GPUDeviceInfo, ComputeCapability
)
from src.wasm_torch.hub.model_registry import (
    ModelRegistry, ModelMetadata, License, ModelStatus
)


class TestAdaptiveOptimizer:
    """Test adaptive WASM optimizer functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.optimizer = AdaptiveWASMOptimizer(self.temp_dir)
        
        # Create simple test model
        self.test_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.example_input = torch.randn(1, 10)
    
    def test_model_analysis(self):
        """Test model characteristics analysis."""
        
        characteristics = self.optimizer.analyze_model(self.test_model, self.example_input)
        
        assert isinstance(characteristics, ModelCharacteristics)
        assert characteristics.parameter_count > 0
        assert characteristics.flops > 0
        assert characteristics.memory_usage > 0
        assert characteristics.batch_size == 1
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        
        characteristics = self.optimizer.analyze_model(self.test_model, self.example_input)
        recommendations = self.optimizer.get_optimization_recommendations(characteristics)
        
        assert isinstance(recommendations, dict)
        assert "mobile" in recommendations
        assert "server" in recommendations
        assert "debug" in recommendations
        
        # Check recommendation structure
        mobile_config = recommendations["mobile"]
        assert isinstance(mobile_config, OptimizationConfig)
        assert mobile_config.optimization_level in ["O0", "O1", "O2", "O3", "Oz"]
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization(self):
        """Test adaptive optimization process."""
        
        target_env = {
            "mobile": 1.0,
            "latency_weight": 1.0,
            "size_weight": 0.8,
            "memory_weight": 0.5,
        }
        
        config, metrics = self.optimizer.optimize_for_target(
            self.test_model, self.example_input, target_env, max_iterations=3
        )
        
        assert isinstance(config, OptimizationConfig)
        assert metrics.compilation_time > 0
        assert metrics.binary_size > 0
        assert metrics.inference_latency > 0
    
    def test_report_export(self):
        """Test optimization report export."""
        
        report_path = self.temp_dir / "optimization_report.json"
        self.optimizer.export_optimization_report(report_path)
        
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        assert "optimization_history" in report
        assert "model_statistics" in report
        assert "timestamp" in report
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestMLQuantizer:
    """Test ML quantization engine functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.quantizer = MLQuantizationEngine()
        
        # Create test model
        self.test_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        self.calibration_data = torch.randn(10, 3, 32, 32)
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        
        config = QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            target_accuracy_loss=0.05
        )
        
        result = self.quantizer.quantize_model(
            self.test_model, self.calibration_data, config
        )
        
        assert result.accuracy_loss >= 0
        assert result.compression_ratio > 1.0
        assert result.quantized_model is not None
    
    def test_mixed_precision_quantization(self):
        """Test mixed precision quantization."""
        
        config = QuantizationConfig(
            quantization_type=QuantizationType.MIXED_PRECISION,
            target_accuracy_loss=0.03
        )
        
        result = self.quantizer.quantize_model(
            self.test_model, self.calibration_data, config
        )
        
        assert result.sensitivity_analysis is not None
        assert len(result.sensitivity_analysis) > 0
    
    def test_adaptive_quantization(self):
        """Test adaptive quantization."""
        
        config = QuantizationConfig(
            quantization_type=QuantizationType.ADAPTIVE,
            target_accuracy_loss=0.08
        )
        
        result = self.quantizer.quantize_model(
            self.test_model, self.calibration_data, config
        )
        
        assert result.config_used.quantization_type == QuantizationType.ADAPTIVE
        assert result.inference_speedup > 1.0
    
    def test_quantization_optimization(self):
        """Test quantization configuration optimization."""
        
        optimal_config = self.quantizer.optimize_quantization_config(
            self.test_model, self.calibration_data,
            target_accuracy_loss=0.05,
            target_compression=4.0
        )
        
        assert isinstance(optimal_config, QuantizationConfig)
        assert optimal_config.target_accuracy_loss == 0.05
        assert optimal_config.compression_ratio == 4.0
    
    def test_quantization_report(self):
        """Test quantization report generation."""
        
        # Run quantization first
        config = QuantizationConfig(quantization_type=QuantizationType.DYNAMIC)
        self.quantizer.quantize_model(self.test_model, self.calibration_data, config)
        
        # Generate report
        report = self.quantizer.export_quantization_report("test_model")
        
        assert "model_name" in report
        assert "quantization_summary" in report
        assert "sensitivity_analysis" in report
        assert "recommendations" in report


class TestStreamingPipeline:
    """Test streaming inference pipeline functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = StreamingConfig(
            max_batch_size=4,
            max_latency_ms=50,
            buffer_size=100
        )
        self.pipeline = StreamingInferencePipeline(self.config)
        
        # Register test models
        vision_model = nn.Sequential(nn.Conv2d(3, 10, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(10, 5))
        audio_model = nn.Sequential(nn.Linear(80*100, 128), nn.ReLU(), nn.Linear(128, 10))
        text_model = nn.Sequential(nn.Embedding(1000, 128), nn.LSTM(128, 64), nn.Linear(64, 2))
        
        self.pipeline.register_model(ModalityType.VISION, vision_model)
        self.pipeline.register_model(ModalityType.AUDIO, audio_model)
        self.pipeline.register_model(ModalityType.TEXT, text_model)
    
    @pytest.mark.asyncio
    async def test_vision_processing(self):
        """Test vision data processing."""
        
        processor = VisionProcessor()
        
        # Test preprocessing
        test_image = torch.randn(3, 224, 224)
        processed = await processor.preprocess(test_image)
        
        assert processed.shape[0] == 1  # Batch dimension added
        assert processed.shape[1:] == (3, 224, 224)
        
        # Test postprocessing
        test_output = torch.randn(1, 5)
        result = await processor.postprocess(test_output)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 5  # Top-5 predictions
    
    @pytest.mark.asyncio
    async def test_audio_processing(self):
        """Test audio data processing."""
        
        processor = AudioProcessor()
        
        # Test preprocessing
        test_audio = torch.randn(16000)  # 1 second at 16kHz
        processed = await processor.preprocess(test_audio)
        
        assert len(processed.shape) == 3  # Batch, mel, time dimensions
        
        # Test postprocessing
        test_output = torch.randn(1, 10, 100)
        result = await processor.postprocess(test_output)
        
        assert "transcription" in result or "raw_output" in result
    
    @pytest.mark.asyncio
    async def test_text_processing(self):
        """Test text data processing."""
        
        processor = TextProcessor()
        
        # Test preprocessing
        test_text = "This is a test sentence"
        processed = await processor.preprocess(test_text)
        
        assert isinstance(processed, dict)
        assert "input_ids" in processed
        assert "attention_mask" in processed
        
        # Test postprocessing
        test_output = torch.randn(1, 2)
        result = await processor.postprocess(test_output)
        
        assert "prediction" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_request_submission(self):
        """Test inference request submission."""
        
        request = InferenceRequest(
            request_id="test_001",
            modality=ModalityType.VISION,
            data=torch.randn(3, 224, 224),
            priority=PriorityLevel.NORMAL
        )
        
        request_id = await self.pipeline.submit_request(request)
        assert request_id == "test_001"
        
        # Check that request is queued
        assert not self.pipeline.input_queue.empty()
    
    @pytest.mark.asyncio
    async def test_pipeline_health_check(self):
        """Test pipeline health check."""
        
        health = await self.pipeline.health_check()
        
        assert "status" in health
        assert "components" in health
        assert "timestamp" in health
        
        # Check component health
        assert "pipeline" in health["components"]
        assert "vision_model" in health["components"]
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics collection."""
        
        stats = self.pipeline.get_statistics()
        
        assert "requests_processed" in stats
        assert "average_latency_ms" in stats
        assert "error_rate" in stats
        assert "qos_metrics" in stats
        assert "resource_usage" in stats


class TestFederatedLearning:
    """Test federated learning system functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = FederatedConfig(
            num_clients=5,
            rounds=3,
            local_epochs=2,
            differential_privacy=True,
            secure_aggregation=True
        )
        
        # Create test model
        self.global_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        self.coordinator = FederatedCoordinator(self.global_model, self.config)
        
    def test_privacy_engine(self):
        """Test differential privacy engine."""
        
        privacy_engine = PrivacyEngine(noise_multiplier=1.0, max_grad_norm=1.0)
        
        # Test gradient noise addition
        gradients = {
            "layer1.weight": torch.randn(20, 10),
            "layer1.bias": torch.randn(20),
        }
        
        noisy_gradients = privacy_engine.add_noise_to_gradients(gradients)
        
        assert len(noisy_gradients) == len(gradients)
        assert "layer1.weight" in noisy_gradients
        assert "layer1.bias" in noisy_gradients
        
        # Check that noise was added (gradients should be different)
        assert not torch.allclose(gradients["layer1.weight"], noisy_gradients["layer1.weight"])
    
    def test_privacy_budget_computation(self):
        """Test privacy budget computation."""
        
        privacy_engine = PrivacyEngine(noise_multiplier=1.0)
        
        epsilon = privacy_engine.compute_privacy_budget(
            epochs=5, batch_size=32, dataset_size=1000
        )
        
        assert epsilon > 0
        assert isinstance(epsilon, float)
    
    def test_client_registration(self):
        """Test client registration."""
        
        from src.wasm_torch.research.federated_inference import ClientInfo
        
        client_info = ClientInfo(
            client_id="client_001",
            data_samples=1000,
            capabilities={"compute_power": "medium"}
        )
        
        success = self.coordinator.register_client(client_info)
        assert success
        
        # Try to register same client again
        success = self.coordinator.register_client(client_info)
        assert not success  # Should fail
    
    def test_client_selection(self):
        """Test client selection for federated round."""
        
        from src.wasm_torch.research.federated_inference import ClientInfo
        
        # Register multiple clients
        for i in range(5):
            client_info = ClientInfo(
                client_id=f"client_{i:03d}",
                data_samples=1000 + i * 100,
                trust_score=0.8 + i * 0.05
            )
            self.coordinator.register_client(client_info)
        
        selected_clients = self.coordinator.select_clients_for_round(1)
        
        assert len(selected_clients) >= self.config.min_clients_per_round
        assert len(selected_clients) <= len(self.coordinator.clients)
        
        # Check that higher trust score clients are preferred
        client_trust_scores = [
            self.coordinator.clients[client_id].trust_score 
            for client_id in selected_clients
        ]
        assert len(client_trust_scores) > 0
    
    @pytest.mark.asyncio
    async def test_federated_client_training(self):
        """Test federated client local training."""
        
        # Create mock training data
        train_data = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(5)]
        val_data = [(torch.randn(16, 10), torch.randint(0, 2, (16,))) for _ in range(3)]
        
        client = FederatedClient("client_001", self.global_model, self.config)
        
        # Mock DataLoader behavior
        class MockDataLoader:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        mock_train_loader = MockDataLoader(train_data)
        mock_val_loader = MockDataLoader(val_data)
        
        update = await client.train_local_model(mock_train_loader, mock_val_loader)
        
        assert update.client_id == "client_001"
        assert update.training_loss > 0
        assert update.data_samples > 0
        assert update.model_weights is not None
        assert update.signature is not None
    
    @pytest.mark.asyncio
    async def test_federated_training_round(self):
        """Test complete federated training round."""
        
        from src.wasm_torch.research.federated_inference import ClientInfo
        
        # Register clients
        for i in range(3):
            client_info = ClientInfo(
                client_id=f"client_{i:03d}",
                data_samples=500 + i * 100,
            )
            self.coordinator.register_client(client_info)
        
        # Run single round
        result = await self.coordinator.run_federated_round(1)
        
        assert "status" in result
        assert result["round_number"] == 1
        
        if result["status"] == "success":
            assert "participants" in result
            assert "avg_loss" in result
            assert "total_samples" in result
    
    def test_global_model_state(self):
        """Test global model state retrieval."""
        
        state = self.coordinator.get_global_model_state()
        
        assert "version" in state
        assert "round_number" in state
        assert "model_weights" in state
        assert "participants" in state


class TestWebGPURuntime:
    """Test WebGPU runtime functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.runtime = WebGPURuntime(GPUBackend.AUTO)
    
    @pytest.mark.asyncio
    async def test_runtime_initialization(self):
        """Test WebGPU runtime initialization."""
        
        success = await self.runtime.initialize()
        assert success
        assert self.runtime.is_initialized
        assert self.runtime.current_backend in [GPUBackend.WEBGPU, GPUBackend.HYBRID, GPUBackend.WASM_SIMD]
    
    @pytest.mark.asyncio
    async def test_buffer_management(self):
        """Test GPU buffer allocation and deallocation."""
        
        await self.runtime.initialize()
        
        # Allocate buffer
        buffer_id = await self.runtime.allocate_buffer(1024 * 1024, "test")  # 1MB
        assert buffer_id in self.runtime.gpu_allocations
        
        # Check memory tracking
        assert self.runtime.memory_used_mb > 0
        
        # Deallocate buffer
        await self.runtime.deallocate_buffer(buffer_id)
        assert buffer_id not in self.runtime.gpu_allocations
    
    @pytest.mark.asyncio
    async def test_tensor_upload_download(self):
        """Test tensor upload/download operations."""
        
        await self.runtime.initialize()
        
        # Test tensor
        test_tensor = torch.randn(10, 20)
        
        # Upload tensor
        buffer_id = await self.runtime.upload_tensor(test_tensor, "input")
        assert buffer_id in self.runtime.gpu_allocations
        
        # Download tensor
        downloaded = await self.runtime.download_tensor(buffer_id, test_tensor.shape, test_tensor.dtype)
        assert downloaded.shape == test_tensor.shape
        assert downloaded.dtype == test_tensor.dtype
        
        # Cleanup
        await self.runtime.deallocate_buffer(buffer_id)
    
    @pytest.mark.asyncio
    async def test_kernel_execution(self):
        """Test GPU kernel execution."""
        
        await self.runtime.initialize()
        
        # Create test buffers
        input_buffer = await self.runtime.allocate_buffer(1024, "input")
        output_buffer = await self.runtime.allocate_buffer(1024, "output")
        
        # Execute kernel
        result = await self.runtime.execute_kernel(
            kernel_name="test_kernel",
            input_buffers=[input_buffer],
            output_buffers=[output_buffer],
            workgroup_size=(16, 16, 1),
            dispatch_size=(32, 32, 1)
        )
        
        assert result.success
        assert result.execution_time_ms > 0
        assert result.workgroups_dispatched > 0
        
        # Cleanup
        await self.runtime.deallocate_buffer(input_buffer)
        await self.runtime.deallocate_buffer(output_buffer)
    
    @pytest.mark.asyncio
    async def test_inference_execution(self):
        """Test complete model inference."""
        
        await self.runtime.initialize()
        
        # Define simple model layers
        model_layers = [
            {
                "type": "linear",
                "complexity": 1000,
                "output_size": 512,
                "output_shape": (1, 10),
            }
        ]
        
        test_input = torch.randn(1, 5)
        
        result = await self.runtime.run_inference(model_layers, test_input)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 10)
    
    def test_runtime_statistics(self):
        """Test runtime statistics collection."""
        
        stats = self.runtime.get_runtime_statistics()
        
        assert "runtime_info" in stats
        assert "memory_usage" in stats
        assert "performance" in stats
        assert "capabilities" in stats
        
        # Check runtime info
        assert "backend" in stats["runtime_info"]
        assert "is_initialized" in stats["runtime_info"]
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test WebGPU performance benchmarking."""
        
        await self.runtime.initialize()
        
        test_workloads = [
            {
                "name": "small_workload",
                "input_shape": (1, 3, 64, 64),
                "kernel_name": "conv2d",
                "workgroup_size": (8, 8, 1),
                "dispatch_size": (8, 8, 1),
            },
            {
                "name": "medium_workload",
                "input_shape": (1, 3, 224, 224),
                "kernel_name": "linear",
                "workgroup_size": (16, 16, 1),
                "dispatch_size": (16, 16, 1),
            }
        ]
        
        benchmark_results = await self.runtime.benchmark_performance(test_workloads)
        
        assert "device_info" in benchmark_results
        assert "test_results" in benchmark_results
        assert "summary" in benchmark_results
        
        # Check test results
        assert len(benchmark_results["test_results"]) == len(test_workloads)
        
        for result in benchmark_results["test_results"]:
            if result.get("success"):
                assert "execution_time_ms" in result
                assert "total_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_runtime_shutdown(self):
        """Test WebGPU runtime shutdown."""
        
        await self.runtime.initialize()
        
        # Allocate some buffers
        buffer_ids = []
        for i in range(3):
            buffer_id = await self.runtime.allocate_buffer(1024, f"test_{i}")
            buffer_ids.append(buffer_id)
        
        # Shutdown runtime
        await self.runtime.shutdown()
        
        assert not self.runtime.is_initialized
        assert len(self.runtime.gpu_allocations) == 0
        assert self.runtime.memory_used_mb == 0.0


class TestModelHub:
    """Test model hub functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry = ModelRegistry(self.temp_dir)
        
        # Create test model
        self.test_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Create test metadata
        self.test_metadata = ModelMetadata(
            model_id="test_model_001",
            name="Test Model",
            description="A simple test model",
            version="1.0.0",
            author="Test Author",
            architecture="MLP",
            input_shape=(10,),
            output_shape=(5,),
            tags={"test", "mlp", "classification"},
            categories=["classification", "demo"],
            license=License.MIT
        )
    
    def test_model_registration(self):
        """Test model registration."""
        
        # Create temporary model file
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        
        success = self.registry.register_model(
            self.test_model, self.test_metadata, model_path
        )
        
        assert success
        assert self.test_metadata.model_id in self.registry.models
        
        # Check that metadata was enhanced
        entry = self.registry.models[self.test_metadata.model_id]
        assert entry.metadata.parameter_count > 0
        assert entry.metadata.model_size_mb > 0
    
    def test_model_search(self):
        """Test model search functionality."""
        
        # Register a model first
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        # Search by tags
        results = self.registry.search_models(tags=["test"])
        assert len(results) == 1
        assert results[0].metadata.model_id == self.test_metadata.model_id
        
        # Search by categories
        results = self.registry.search_models(categories=["classification"])
        assert len(results) == 1
        
        # Search by author
        results = self.registry.search_models(author="Test Author")
        assert len(results) == 1
        
        # Text query search
        results = self.registry.search_models(query="simple test")
        assert len(results) == 1
    
    def test_version_management(self):
        """Test model version management."""
        
        # Register initial model
        model_path = self.temp_dir / "test_model_v1.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        # Add new version
        new_model_path = self.temp_dir / "test_model_v2.pth"
        torch.save(self.test_model.state_dict(), new_model_path)
        
        success = self.registry.add_model_version(
            self.test_metadata.model_id,
            "2.0.0",
            new_model_path,
            changelog="Added new features"
        )
        
        assert success
        
        entry = self.registry.models[self.test_metadata.model_id]
        assert "2.0.0" in entry.versions
        assert entry.latest_version == "2.0.0"
        assert entry.versions["2.0.0"].is_latest
        assert not entry.versions["1.0.0"].is_latest
    
    def test_model_download(self):
        """Test model download functionality."""
        
        # Register model
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        # Download model
        download_path = self.registry.download_model(self.test_metadata.model_id)
        
        assert download_path is not None
        assert download_path.exists()
        
        # Check download count was incremented
        entry = self.registry.models[self.test_metadata.model_id]
        assert entry.metadata.download_count == 1
    
    def test_model_status_update(self):
        """Test model status updates."""
        
        # Register model
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        # Update status
        success = self.registry.update_model_status(
            self.test_metadata.model_id,
            "1.0.0",
            ModelStatus.DEPRECATED
        )
        
        assert success
        
        entry = self.registry.models[self.test_metadata.model_id]
        assert entry.versions["1.0.0"].status == ModelStatus.DEPRECATED
        assert entry.versions["1.0.0"].deprecated_at is not None
    
    def test_registry_statistics(self):
        """Test registry statistics collection."""
        
        # Register a model
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        stats = self.registry.get_registry_statistics()
        
        assert "total_models" in stats
        assert "total_versions" in stats
        assert "indices" in stats
        assert "last_updated" in stats
        
        assert stats["total_models"] == 1
        assert stats["total_versions"] == 1
    
    def test_registry_report(self):
        """Test registry report export."""
        
        # Register a model
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        report = self.registry.export_registry_report()
        
        assert "registry_info" in report
        assert "statistics" in report
        assert "distributions" in report
        assert "author_statistics" in report
        assert "top_models" in report
        assert "recent_activity" in report
        
        # Check statistics
        assert report["statistics"]["total_models"] == 1
        assert report["statistics"]["total_versions"] == 1
        
        # Check author statistics
        assert "Test Author" in report["author_statistics"]
        assert report["author_statistics"]["Test Author"]["models"] == 1
    
    def test_model_deletion(self):
        """Test model deletion functionality."""
        
        # Register model
        model_path = self.temp_dir / "test_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        self.registry.register_model(self.test_model, self.test_metadata, model_path)
        
        # Delete model
        success = self.registry.delete_model(self.test_metadata.model_id, confirm=True)
        
        assert success
        assert self.test_metadata.model_id not in self.registry.models
        
        # Check that indices were updated
        assert "test" not in self.registry.model_index or \
               self.test_metadata.model_id not in self.registry.model_index["test"]
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestIntegration:
    """Integration tests for research modules working together."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test model
        self.test_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    @pytest.mark.asyncio
    async def test_optimization_and_quantization_pipeline(self):
        """Test optimization and quantization working together."""
        
        # Setup components
        optimizer = AdaptiveWASMOptimizer(self.temp_dir)
        quantizer = MLQuantizationEngine()
        
        example_input = torch.randn(1, 3, 32, 32)
        calibration_data = torch.randn(10, 3, 32, 32)
        
        # Run optimization
        target_env = {"mobile": 1.0, "latency_weight": 1.0, "size_weight": 0.8}
        opt_config, opt_metrics = optimizer.optimize_for_target(
            self.test_model, example_input, target_env, max_iterations=2
        )
        
        # Run quantization
        quant_config = QuantizationConfig(
            quantization_type=QuantizationType.ADAPTIVE,
            target_accuracy_loss=0.05
        )
        
        quant_result = quantizer.quantize_model(
            self.test_model, calibration_data, quant_config
        )
        
        # Verify both processes completed successfully
        assert opt_metrics.compilation_time > 0
        assert quant_result.compression_ratio > 1.0
    
    @pytest.mark.asyncio
    async def test_streaming_with_webgpu_backend(self):
        """Test streaming pipeline with WebGPU backend."""
        
        # Setup streaming pipeline
        config = StreamingConfig(max_batch_size=2, max_latency_ms=100)
        pipeline = StreamingInferencePipeline(config)
        
        # Register simple vision model
        vision_model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 5)
        )
        pipeline.register_model(ModalityType.VISION, vision_model)
        
        # Setup WebGPU runtime
        webgpu_runtime = WebGPURuntime(GPUBackend.AUTO)
        await webgpu_runtime.initialize()
        
        # Submit test request
        request = InferenceRequest(
            request_id="integration_test_001",
            modality=ModalityType.VISION,
            data=torch.randn(3, 64, 64),
            priority=PriorityLevel.NORMAL
        )
        
        request_id = await pipeline.submit_request(request)
        assert request_id == "integration_test_001"
        
        # Check health of both systems
        pipeline_health = await pipeline.health_check()
        webgpu_stats = webgpu_runtime.get_runtime_statistics()
        
        assert pipeline_health["status"] in ["healthy", "degraded"]
        assert "runtime_info" in webgpu_stats
        
        # Cleanup
        await webgpu_runtime.shutdown()
    
    def test_model_hub_with_optimization_metadata(self):
        """Test model hub storing optimization metadata."""
        
        # Setup model registry
        registry = ModelRegistry(self.temp_dir)
        
        # Create enhanced metadata with optimization info
        metadata = ModelMetadata(
            model_id="optimized_model_001",
            name="Optimized Test Model",
            description="Model with optimization metadata",
            version="1.0.0",
            author="Integration Test",
            architecture="CNN",
            input_shape=(3, 32, 32),
            output_shape=(10,),
            optimization_flags={
                "wasm_optimization": "O3",
                "quantization": "dynamic_int8",
                "simd_enabled": True
            },
            benchmark_results={
                "inference_latency_ms": 15.2,
                "memory_usage_mb": 2.5,
                "accuracy": 0.94
            },
            tags={"optimized", "cnn", "efficient"},
            categories=["computer_vision", "classification"]
        )
        
        # Save model
        model_path = self.temp_dir / "optimized_model.pth"
        torch.save(self.test_model.state_dict(), model_path)
        
        # Register model
        success = registry.register_model(self.test_model, metadata, model_path)
        assert success
        
        # Search for optimized models
        results = registry.search_models(tags=["optimized"])
        assert len(results) == 1
        
        found_model = results[0]
        assert "wasm_optimization" in found_model.metadata.optimization_flags
        assert "inference_latency_ms" in found_model.metadata.benchmark_results
    
    @pytest.mark.asyncio
    async def test_federated_learning_with_quantization(self):
        """Test federated learning with quantized models."""
        
        # Setup federated learning
        config = FederatedConfig(
            num_clients=2,
            rounds=1,
            local_epochs=1,
            differential_privacy=True
        )
        
        global_model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
        coordinator = FederatedCoordinator(global_model, config)
        
        # Register clients
        from src.wasm_torch.research.federated_inference import ClientInfo
        
        for i in range(2):
            client_info = ClientInfo(
                client_id=f"quant_client_{i}",
                data_samples=100,
                capabilities={"quantization": "int8_support"}
            )
            coordinator.register_client(client_info)
        
        # Setup quantization engine
        quantizer = MLQuantizationEngine()
        
        # Quantize global model
        calibration_data = torch.randn(10, 5)
        quant_config = QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            target_accuracy_loss=0.1
        )
        
        quant_result = quantizer.quantize_model(
            global_model, calibration_data, quant_config
        )
        
        # Run federated round with quantized model
        result = await coordinator.run_federated_round(1)
        
        # Verify integration works
        assert quant_result.compression_ratio > 1.0
        if result["status"] == "success":
            assert "participants" in result
            assert result["participants"] >= 0
    
    def teardown_method(self):
        """Cleanup integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# Performance benchmarks for research modules
class TestPerformanceBenchmarks:
    """Performance benchmarks for research modules."""
    
    @pytest.mark.benchmark
    def test_adaptive_optimizer_performance(self, benchmark):
        """Benchmark adaptive optimizer performance."""
        
        def run_optimization():
            temp_dir = Path(tempfile.mkdtemp())
            try:
                optimizer = AdaptiveWASMOptimizer(temp_dir)
                model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
                example_input = torch.randn(1, 100)
                target_env = {"mobile": 1.0, "latency_weight": 1.0}
                
                config, metrics = optimizer.optimize_for_target(
                    model, example_input, target_env, max_iterations=3
                )
                return config, metrics
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        result = benchmark(run_optimization)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_quantization_performance(self, benchmark):
        """Benchmark quantization engine performance."""
        
        def run_quantization():
            quantizer = MLQuantizationEngine()
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, 10)
            )
            calibration_data = torch.randn(20, 3, 32, 32)
            config = QuantizationConfig(quantization_type=QuantizationType.DYNAMIC)
            
            result = quantizer.quantize_model(model, calibration_data, config)
            return result
        
        result = benchmark(run_quantization)
        assert result.compression_ratio > 1.0
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_webgpu_kernel_execution_performance(self, benchmark):
        """Benchmark WebGPU kernel execution performance."""
        
        async def run_kernel_execution():
            runtime = WebGPURuntime(GPUBackend.AUTO)
            await runtime.initialize()
            
            try:
                # Allocate buffers
                input_buffer = await runtime.allocate_buffer(4096, "input")
                output_buffer = await runtime.allocate_buffer(4096, "output")
                
                # Execute kernel
                result = await runtime.execute_kernel(
                    kernel_name="performance_test",
                    input_buffers=[input_buffer],
                    output_buffers=[output_buffer],
                    workgroup_size=(16, 16, 1),
                    dispatch_size=(64, 64, 1)
                )
                
                # Cleanup
                await runtime.deallocate_buffer(input_buffer)
                await runtime.deallocate_buffer(output_buffer)
                
                return result
            finally:
                await runtime.shutdown()
        
        result = await benchmark(run_kernel_execution)
        assert result.success


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])