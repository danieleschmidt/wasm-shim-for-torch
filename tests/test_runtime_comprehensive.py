"""Comprehensive tests for WASM runtime functionality."""

import pytest
import asyncio
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from wasm_torch.runtime import (
    WASMRuntime, WASMModel, MemoryManager, OperationRegistry,
    WASMOperation, LinearOperation, ReLUOperation, Conv2dOperation,
    BatchNormOperation, AddOperation, MulOperation, RuntimeStats
)


class TestWASMRuntime:
    """Test WASM runtime functionality."""
    
    @pytest.mark.asyncio
    async def test_runtime_initialization(self):
        """Test runtime initialization."""
        runtime = WASMRuntime(
            simd=True,
            threads=4,
            memory_limit_mb=512,
            timeout_seconds=60.0,
            enable_monitoring=True
        )
        
        assert runtime.simd is True
        assert runtime.threads == 4
        assert runtime.memory_limit_mb == 512
        assert runtime.timeout_seconds == 60.0
        assert runtime.enable_monitoring is True
        assert runtime._initialized is False
        
        # Initialize runtime
        initialized_runtime = await runtime.init()
        assert initialized_runtime is runtime
        assert runtime._initialized is True
        assert runtime._startup_time is not None
        
        # Clean up
        await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_runtime_model_loading(self):
        """Test model loading."""
        runtime = WASMRuntime()
        await runtime.init()
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as f:
            model_path = Path(f.name)
        
        # Create metadata file
        metadata_path = model_path.with_suffix('.json')
        metadata = {
            "graph": {
                "operations": [
                    {"kind": "aten::linear", "attributes": {}},
                    {"kind": "aten::relu", "attributes": {}}
                ],
                "parameters": {
                    "weight": {"shape": [10, 5], "dtype": "torch.float32"},
                    "bias": {"shape": [10], "dtype": "torch.float32"}
                }
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        try:
            # Load model
            model = await runtime.load_model(model_path)
            assert isinstance(model, WASMModel)
            assert model._is_loaded is True
            assert len(model.operations) == 2
            assert len(model.parameters) == 2
            
        finally:
            # Clean up
            model_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_runtime_model_loading_no_metadata(self):
        """Test model loading without metadata."""
        runtime = WASMRuntime()
        await runtime.init()
        
        # Create temporary model file without metadata
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as f:
            model_path = Path(f.name)
        
        try:
            # Load model (should create minimal structure)
            model = await runtime.load_model(model_path)
            assert isinstance(model, WASMModel)
            assert model._is_loaded is True
            assert len(model.operations) >= 2  # Minimal structure
            
        finally:
            model_path.unlink(missing_ok=True)
            await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_runtime_model_loading_nonexistent_file(self):
        """Test loading non-existent model file."""
        runtime = WASMRuntime()
        await runtime.init()
        
        nonexistent_path = Path("/nonexistent/model.wasm")
        
        with pytest.raises(FileNotFoundError):
            await runtime.load_model(nonexistent_path)
        
        await runtime.cleanup()
    
    def test_runtime_stats(self):
        """Test runtime statistics."""
        runtime = WASMRuntime()
        
        stats = runtime.get_runtime_stats()
        assert isinstance(stats, dict)
        assert 'uptime_seconds' in stats
        assert 'inference_count' in stats
        assert 'error_count' in stats
        assert 'health_status' in stats
    
    @pytest.mark.asyncio
    async def test_runtime_health_monitoring(self):
        """Test health monitoring."""
        runtime = WASMRuntime(enable_monitoring=True)
        await runtime.init()
        
        # Wait a bit for health monitoring to run
        await asyncio.sleep(0.1)
        
        stats = runtime.get_runtime_stats()
        assert stats['last_health_check'] is not None
        
        await runtime.cleanup()


class TestWASMModel:
    """Test WASM model functionality."""
    
    @pytest.mark.asyncio
    async def test_model_forward_pass(self):
        """Test model forward pass."""
        runtime = WASMRuntime()
        await runtime.init()
        
        model = WASMModel(Path("dummy.wasm"), runtime)
        
        # Manually set up model state
        model.operations = [
            {"kind": "aten::linear", "attributes": {}},
            {"kind": "aten::relu", "attributes": {}}
        ]
        model.parameters = {
            "weight": torch.randn(5, 10),
            "bias": torch.randn(5)
        }
        model._is_loaded = True
        
        # Test forward pass
        input_tensor = torch.randn(1, 10)
        output = await model.forward(input_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1  # Batch size preserved
        
        await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_forward_pass_not_loaded(self):
        """Test forward pass on unloaded model."""
        runtime = WASMRuntime()
        await runtime.init()
        
        model = WASMModel(Path("dummy.wasm"), runtime)
        model._is_loaded = False
        
        input_tensor = torch.randn(1, 10)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.forward(input_tensor)
        
        await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_forward_pass_invalid_input(self):
        """Test forward pass with invalid input."""
        runtime = WASMRuntime()
        await runtime.init()
        
        model = WASMModel(Path("dummy.wasm"), runtime)
        model._is_loaded = True
        model.operations = []
        model.parameters = {}
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            await model.forward("not a tensor")
        
        await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_memory_stats(self):
        """Test model memory statistics."""
        runtime = WASMRuntime()
        await runtime.init()
        
        model = WASMModel(Path("dummy.wasm"), runtime)
        model._is_loaded = True
        model.parameters = {
            "weight": torch.randn(100, 100),  # ~40KB
            "bias": torch.randn(100)          # ~400B
        }
        model.operations = [{"kind": "aten::linear"}]
        
        stats = await model.get_memory_stats()
        
        assert stats['model_loaded'] is True
        assert stats['parameter_count'] == 10100  # 100*100 + 100
        assert stats['parameter_bytes'] > 0
        assert stats['operations_count'] == 1
        
        await runtime.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_memory_stats_not_loaded(self):
        """Test memory stats for unloaded model."""
        runtime = WASMRuntime()
        await runtime.init()
        
        model = WASMModel(Path("dummy.wasm"), runtime)
        model._is_loaded = False
        
        stats = await model.get_memory_stats()
        assert stats['model_loaded'] is False
        
        await runtime.cleanup()


class TestMemoryManager:
    """Test memory manager functionality."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        manager = MemoryManager(limit_mb=100)
        
        assert manager.limit_bytes == 100 * 1024 * 1024
        assert manager.allocated_bytes == 0
        assert manager.peak_bytes == 0
    
    def test_memory_allocation(self):
        """Test memory allocation."""
        manager = MemoryManager(limit_mb=1)  # 1MB limit
        
        # Allocate within limit
        success = manager.allocate(512 * 1024)  # 512KB
        assert success is True
        assert manager.allocated_bytes == 512 * 1024
        
        # Allocate beyond limit
        success = manager.allocate(600 * 1024)  # 600KB (total would be 1.1MB)
        assert success is False
        assert manager.allocated_bytes == 512 * 1024  # Unchanged
    
    def test_memory_deallocation(self):
        """Test memory deallocation."""
        manager = MemoryManager(limit_mb=1)
        
        manager.allocate(512 * 1024)
        manager.deallocate(256 * 1024)
        
        assert manager.allocated_bytes == 256 * 1024
        
        # Can't deallocate below zero
        manager.deallocate(500 * 1024)
        assert manager.allocated_bytes == 0
    
    def test_memory_peak_tracking(self):
        """Test peak memory tracking."""
        manager = MemoryManager(limit_mb=1)
        
        manager.allocate(300 * 1024)
        manager.allocate(200 * 1024)
        assert manager.peak_bytes == 500 * 1024
        
        manager.deallocate(100 * 1024)
        assert manager.peak_bytes == 500 * 1024  # Peak unchanged
    
    def test_memory_stats(self):
        """Test memory statistics."""
        manager = MemoryManager(limit_mb=1)
        manager.allocate(256 * 1024)
        
        stats = manager.get_stats()
        
        assert stats['allocated_bytes'] == 256 * 1024
        assert stats['limit_bytes'] == 1 * 1024 * 1024
        assert stats['available_bytes'] == (1 * 1024 * 1024) - (256 * 1024)
        assert stats['peak_bytes'] == 256 * 1024


class TestOperationRegistry:
    """Test operation registry functionality."""
    
    def test_operation_registration(self):
        """Test operation registration."""
        registry = OperationRegistry()
        mock_operation = Mock(spec=WASMOperation)
        
        registry.register("test_op", mock_operation)
        
        retrieved = registry.get("test_op")
        assert retrieved is mock_operation
        
        # Test non-existent operation
        assert registry.get("nonexistent") is None
    
    def test_operation_listing(self):
        """Test operation listing."""
        registry = OperationRegistry()
        
        mock_op1 = Mock(spec=WASMOperation)
        mock_op2 = Mock(spec=WASMOperation)
        
        registry.register("op1", mock_op1)
        registry.register("op2", mock_op2)
        
        operations = registry.list_operations()
        assert set(operations) == {"op1", "op2"}


class TestWASMOperations:
    """Test WASM operation implementations."""
    
    @pytest.mark.asyncio
    async def test_linear_operation(self):
        """Test linear operation."""
        operation = LinearOperation()
        
        input_tensor = torch.randn(2, 10)
        op_info = {"attributes": {}}
        parameters = {
            "weight": torch.randn(5, 10),
            "bias": torch.randn(5)
        }
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        assert output.shape == (2, 5)  # Batch size 2, output features 5
        assert isinstance(output, torch.Tensor)
    
    @pytest.mark.asyncio
    async def test_relu_operation(self):
        """Test ReLU operation."""
        operation = ReLUOperation()
        
        input_tensor = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        op_info = {"attributes": {}}
        parameters = {}
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        expected = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
        assert torch.allclose(output, expected)
    
    @pytest.mark.asyncio
    async def test_conv2d_operation(self):
        """Test Conv2D operation."""
        operation = Conv2dOperation()
        
        # NCHW format input
        input_tensor = torch.randn(1, 3, 32, 32)
        op_info = {
            "attributes": {
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [0, 0]
            }
        }
        parameters = {}
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        # Should be some kind of processed tensor
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4  # Still NCHW format
    
    @pytest.mark.asyncio
    async def test_batch_norm_operation(self):
        """Test batch normalization operation."""
        operation = BatchNormOperation()
        
        input_tensor = torch.randn(4, 10)  # Batch of 4, features 10
        op_info = {"attributes": {}}
        parameters = {}
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        assert output.shape == input_tensor.shape
        # Output should be normalized (mean close to 0, std close to 1)
        assert abs(output.mean().item()) < 0.1
        assert abs(output.std().item() - 1.0) < 0.2
    
    @pytest.mark.asyncio
    async def test_add_operation(self):
        """Test addition operation."""
        operation = AddOperation()
        
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op_info = {"attributes": {}}
        parameters = {}
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        # Should add 0.1 to each element
        expected = input_tensor + 0.1
        assert torch.allclose(output, expected)
    
    @pytest.mark.asyncio
    async def test_mul_operation(self):
        """Test multiplication operation."""
        operation = MulOperation()
        
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op_info = {"attributes": {}}
        parameters = {}
        runtime = Mock()
        
        output = await operation.execute(input_tensor, op_info, parameters, runtime)
        
        # Should multiply by 0.9
        expected = input_tensor * 0.9
        assert torch.allclose(output, expected)


class TestRuntimeStats:
    """Test runtime statistics."""
    
    def test_runtime_stats_initialization(self):
        """Test runtime stats initialization."""
        stats = RuntimeStats()
        
        assert stats.inference_count == 0
        assert stats.total_inference_time == 0.0
        assert stats.last_inference_time is None
        assert stats.error_count == 0
        assert stats.memory_peak_mb == 0.0
        assert stats.models_loaded == 0
        assert stats.models_failed == 0
        assert stats.startup_time is None
        assert stats.last_health_check is None
        assert len(stats.warnings) == 0


class TestRuntimeIntegration:
    """Integration tests for runtime components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_inference(self):
        """Test complete end-to-end inference pipeline."""
        runtime = WASMRuntime(
            simd=True,
            threads=2,
            memory_limit_mb=256,
            enable_monitoring=True
        )
        await runtime.init()
        
        # Create a simple model
        model = WASMModel(Path("test_model.wasm"), runtime)
        model.operations = [
            {"kind": "aten::linear", "attributes": {}},
            {"kind": "aten::relu", "attributes": {}}
        ]
        model.parameters = {
            "weight": torch.randn(5, 10),
            "bias": torch.randn(5)
        }
        model._is_loaded = True
        
        # Run inference multiple times
        for i in range(3):
            input_tensor = torch.randn(1, 10)
            output = await model.forward(input_tensor)
            
            assert isinstance(output, torch.Tensor)
            assert output.shape[1] == 5  # Output features
        
        # Check runtime stats
        stats = runtime.get_runtime_stats()
        assert stats['inference_count'] >= 3
        assert stats['total_inference_time'] > 0
        
        # Get model memory stats
        memory_stats = await model.get_memory_stats()
        assert memory_stats['model_loaded'] is True
        assert memory_stats['parameter_count'] > 0
        
        await runtime.cleanup()
    
    @pytest.mark.asyncio 
    async def test_concurrent_inference(self):
        """Test concurrent inference operations."""
        runtime = WASMRuntime(threads=4, enable_monitoring=True)
        await runtime.init()
        
        model = WASMModel(Path("concurrent_test.wasm"), runtime)
        model.operations = [{"kind": "aten::relu", "attributes": {}}]
        model.parameters = {}
        model._is_loaded = True
        
        # Run multiple concurrent inferences
        async def run_inference(input_val):
            input_tensor = torch.full((1, 5), input_val)
            return await model.forward(input_tensor)
        
        tasks = [run_inference(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 5)
        
        await runtime.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])