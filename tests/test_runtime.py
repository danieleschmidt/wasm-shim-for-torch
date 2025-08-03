"""Tests for WASM runtime functionality."""

import pytest
import torch
import torch.nn as nn
import asyncio
from pathlib import Path
import tempfile
import json

from wasm_torch.runtime import WASMRuntime, WASMModel, MemoryManager, OperationRegistry
from wasm_torch.runtime import LinearOperation, ReLUOperation


class TestMemoryManager:
    """Test memory management functionality."""
    
    def test_memory_allocation(self):
        """Test memory allocation and tracking."""
        manager = MemoryManager(limit_mb=10)  # 10MB limit
        
        # Test successful allocation
        assert manager.allocate(1024) == True  # 1KB
        assert manager.allocated_bytes == 1024
        
        # Test allocation exceeding limit
        large_size = 11 * 1024 * 1024  # 11MB
        assert manager.allocate(large_size) == False
        assert manager.allocated_bytes == 1024  # Should remain unchanged
        
    def test_memory_deallocation(self):
        """Test memory deallocation."""
        manager = MemoryManager(limit_mb=10)
        
        manager.allocate(2048)
        assert manager.allocated_bytes == 2048
        
        manager.deallocate(1024)
        assert manager.allocated_bytes == 1024
        
        # Test underflow protection
        manager.deallocate(5000)
        assert manager.allocated_bytes == 0
        
    def test_memory_stats(self):
        """Test memory statistics."""
        manager = MemoryManager(limit_mb=1)
        
        manager.allocate(512)
        manager.allocate(256)
        
        stats = manager.get_stats()
        assert stats["allocated_bytes"] == 768
        assert stats["peak_bytes"] == 768
        assert stats["limit_bytes"] == 1024 * 1024
        assert stats["available_bytes"] == 1024 * 1024 - 768


class TestOperationRegistry:
    """Test operation registry functionality."""
    
    def test_operation_registration(self):
        """Test registering operations."""
        registry = OperationRegistry()
        linear_op = LinearOperation()
        
        registry.register("aten::linear", linear_op)
        
        retrieved_op = registry.get("aten::linear")
        assert retrieved_op is linear_op
        
    def test_unknown_operation(self):
        """Test retrieving unknown operation."""
        registry = OperationRegistry()
        
        unknown_op = registry.get("unknown::operation")
        assert unknown_op is None
        
    def test_list_operations(self):
        """Test listing registered operations."""
        registry = OperationRegistry()
        
        registry.register("aten::linear", LinearOperation())
        registry.register("aten::relu", ReLUOperation())
        
        ops = registry.list_operations()
        assert "aten::linear" in ops
        assert "aten::relu" in ops
        assert len(ops) == 2


class TestLinearOperation:
    """Test linear operation implementation."""
    
    @pytest.mark.asyncio
    async def test_linear_forward(self):
        """Test linear operation forward pass."""
        op = LinearOperation()
        
        # Create test tensors
        input_tensor = torch.randn(2, 5)
        parameters = {
            "weight": torch.randn(3, 5),  # Output features: 3, Input features: 5
            "bias": torch.randn(3)
        }
        
        # Mock runtime and operation info
        runtime = None
        op_info = {"attributes": {}}
        
        output = await op.execute(input_tensor, op_info, parameters, runtime)
        
        # Check output shape
        assert output.shape == (2, 3)
        
        # Verify computation manually
        expected = torch.matmul(input_tensor, parameters["weight"].T) + parameters["bias"]
        torch.testing.assert_close(output, expected)


class TestReLUOperation:
    """Test ReLU operation implementation."""
    
    @pytest.mark.asyncio
    async def test_relu_forward(self):
        """Test ReLU operation forward pass."""
        op = ReLUOperation()
        
        # Create test tensor with positive and negative values
        input_tensor = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        
        # Mock parameters and runtime
        parameters = {}
        runtime = None
        op_info = {"attributes": {}}
        
        output = await op.execute(input_tensor, op_info, parameters, runtime)
        
        # Check that negative values are clamped to 0
        expected = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
        torch.testing.assert_close(output, expected)


class TestWASMRuntime:
    """Test WASM runtime functionality."""
    
    @pytest.mark.asyncio
    async def test_runtime_initialization(self):
        """Test runtime initialization."""
        runtime = WASMRuntime(simd=True, threads=2, memory_limit_mb=512)
        
        # Test initialization
        initialized_runtime = await runtime.init()
        assert initialized_runtime is runtime
        assert runtime._initialized is True
        
        # Check that components are initialized
        assert hasattr(runtime, '_thread_pool')
        assert hasattr(runtime, '_memory_manager')
        assert hasattr(runtime, '_op_registry')
        
        # Cleanup
        await runtime.cleanup()
        
    @pytest.mark.asyncio
    async def test_model_loading_nonexistent(self):
        """Test loading non-existent model file."""
        runtime = WASMRuntime()
        
        with pytest.raises(FileNotFoundError):
            await runtime.load_model("nonexistent_model.wasm")
            
    @pytest.mark.asyncio
    async def test_model_loading_with_metadata(self):
        """Test loading model with metadata file."""
        runtime = WASMRuntime()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model file
            model_file = temp_path / "test_model.wasm"
            model_file.write_text("dummy wasm content")
            
            # Create metadata file
            metadata = {
                "model_info": {
                    "input_shape": [1, 10],
                    "input_dtype": "torch.float32"
                },
                "graph": {
                    "operations": [
                        {"kind": "aten::linear", "attributes": {}},
                        {"kind": "aten::relu", "attributes": {}}
                    ],
                    "parameters": {
                        "weight": {"shape": [5, 10], "dtype": "torch.float32"},
                        "bias": {"shape": [5], "dtype": "torch.float32"}
                    }
                }
            }
            
            metadata_file = temp_path / "test_model.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Load model
            model = await runtime.load_model(model_file)
            
            assert model._is_loaded
            assert len(model.operations) == 2
            assert len(model.parameters) == 2
            
            await runtime.cleanup()


class TestWASMModel:
    """Test WASM model functionality."""
    
    @pytest.mark.asyncio
    async def test_model_forward_pass(self):
        """Test model forward pass."""
        runtime = WASMRuntime()
        await runtime.init()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model file
            model_file = temp_path / "test_model.wasm"
            model_file.write_text("dummy wasm content")
            
            # Load model (will create minimal structure)
            model = await runtime.load_model(model_file)
            
            # Test forward pass
            input_tensor = torch.randn(1, 10)
            output = await model.forward(input_tensor)
            
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == input_tensor.shape[0]  # Batch dimension preserved
            
            await runtime.cleanup()
            
    @pytest.mark.asyncio
    async def test_model_forward_unloaded(self):
        """Test forward pass on unloaded model."""
        runtime = WASMRuntime()
        model = WASMModel(Path("dummy.wasm"), runtime)
        
        input_tensor = torch.randn(1, 10)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.forward(input_tensor)
            
    @pytest.mark.asyncio
    async def test_model_memory_stats(self):
        """Test model memory statistics."""
        runtime = WASMRuntime()
        await runtime.init()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model file
            model_file = temp_path / "test_model.wasm"
            model_file.write_text("dummy wasm content")
            
            # Load model
            model = await runtime.load_model(model_file)
            
            # Get memory stats
            stats = await model.get_memory_stats()
            
            assert stats["model_loaded"] is True
            assert "parameter_bytes" in stats
            assert "parameter_count" in stats
            assert "operations_count" in stats
            assert stats["operations_count"] > 0
            
            await runtime.cleanup()
            
    @pytest.mark.asyncio
    async def test_model_memory_stats_unloaded(self):
        """Test memory stats for unloaded model."""
        runtime = WASMRuntime()
        model = WASMModel(Path("dummy.wasm"), runtime)
        
        stats = await model.get_memory_stats()
        assert stats["model_loaded"] is False


@pytest.mark.integration
class TestRuntimeIntegration:
    """Integration tests for runtime functionality."""
    
    @pytest.mark.asyncio
    async def test_full_inference_pipeline(self):
        """Test complete inference pipeline."""
        # Create a simple PyTorch model
        pytorch_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create WASM runtime
        runtime = WASMRuntime(simd=True, threads=2)
        await runtime.init()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create model file and metadata
            model_file = temp_path / "model.wasm"
            model_file.write_text("dummy wasm content")
            
            # Create realistic metadata
            metadata = {
                "model_info": {
                    "input_shape": [1, 10],
                    "input_dtype": "torch.float32"
                },
                "graph": {
                    "operations": [
                        {"kind": "aten::linear", "attributes": {}},
                        {"kind": "aten::relu", "attributes": {}},
                        {"kind": "aten::linear", "attributes": {}}
                    ],
                    "parameters": {
                        "fc1.weight": {"shape": [5, 10], "dtype": "torch.float32"},
                        "fc1.bias": {"shape": [5], "dtype": "torch.float32"},
                        "fc2.weight": {"shape": [1, 5], "dtype": "torch.float32"},
                        "fc2.bias": {"shape": [1], "dtype": "torch.float32"}
                    }
                }
            }
            
            metadata_file = temp_path / "model.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Load and test model
            model = await runtime.load_model(model_file)
            
            # Test multiple inferences
            for _ in range(3):
                input_tensor = torch.randn(1, 10)
                output = await model.forward(input_tensor)
                
                assert isinstance(output, torch.Tensor)
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()
            
            # Check memory stats
            stats = await model.get_memory_stats()
            assert stats["model_loaded"] is True
            assert stats["operations_count"] == 3
            
            await runtime.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])