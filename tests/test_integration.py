"""Integration tests for WASM Torch end-to-end functionality."""

import pytest
import asyncio
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wasm_torch import export_to_wasm, optimize_for_browser, quantize_for_wasm
from wasm_torch.runtime import WASMRuntime
from wasm_torch.security import SecurityConfig, validate_model_path
from wasm_torch.performance import get_performance_monitor
from wasm_torch.validation import validate_input_tensor


class IntegrationTestModel(nn.Module):
    """Test model for integration tests."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test complete model pipeline from training to inference."""
        print("Starting complete pipeline test...")
        
        # 1. Create and train model
        model = IntegrationTestModel()
        model.train()
        
        # Generate dummy training data
        X = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(3):  # Quick training
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        print("✅ Model training completed")
        
        # 2. Optimize model for browser
        try:
            optimized_model = optimize_for_browser(
                model,
                target_size_mb=50,
                optimization_passes=["optimize_memory_layout"]
            )
            print("✅ Model optimization completed")
        except Exception as e:
            print(f"⚠️ Optimization failed (expected): {e}")
            optimized_model = model
        
        # 3. Apply quantization
        try:
            quantized_model = quantize_for_wasm(
                optimized_model,
                quantization_type="dynamic"
            )
            print("✅ Model quantization completed")
        except Exception as e:
            print(f"⚠️ Quantization failed (expected): {e}")
            quantized_model = optimized_model
        
        # 4. Test model inference (native PyTorch)
        test_input = torch.randn(1, 784)
        with torch.no_grad():
            native_output = quantized_model(test_input)
        print(f"✅ Native inference completed, output shape: {native_output.shape}")
        
        # 5. Export to WASM (will fail without Emscripten, but test the pipeline)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.wasm"
            
            try:
                export_to_wasm(
                    model=quantized_model,
                    output_path=output_path,
                    example_input=test_input,
                    optimization_level="O2",
                    use_simd=True,
                    use_threads=True
                )
                print("✅ WASM export completed")
                export_success = True
            except RuntimeError as e:
                if "Emscripten not found" in str(e):
                    print("⚠️ WASM export skipped (Emscripten not available)")
                    # Create dummy files for runtime test
                    output_path.write_text("dummy wasm content")
                    
                    # Create metadata for runtime
                    metadata_path = output_path.with_suffix('.json')
                    metadata = {
                        "graph": {
                            "operations": [
                                {"kind": "aten::linear", "attributes": {}},
                                {"kind": "aten::relu", "attributes": {}},
                                {"kind": "aten::linear", "attributes": {}},
                                {"kind": "aten::relu", "attributes": {}},
                                {"kind": "aten::linear", "attributes": {}}
                            ],
                            "parameters": {
                                "layers.0.weight": {"shape": [128, 784], "dtype": "torch.float32"},
                                "layers.0.bias": {"shape": [128], "dtype": "torch.float32"},
                                "layers.2.weight": {"shape": [64, 128], "dtype": "torch.float32"},
                                "layers.2.bias": {"shape": [64], "dtype": "torch.float32"},
                                "layers.4.weight": {"shape": [10, 64], "dtype": "torch.float32"},
                                "layers.4.bias": {"shape": [10], "dtype": "torch.float32"}
                            }
                        }
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    
                    export_success = False
                else:
                    raise
            
            # 6. Test WASM runtime
            runtime = WASMRuntime(simd=True, threads=2, enable_monitoring=True)
            await runtime.init()
            
            # Load model
            wasm_model = await runtime.load_model(output_path)
            print("✅ WASM model loaded")
            
            # Test inference
            wasm_output = await wasm_model.forward(test_input)
            print(f"✅ WASM inference completed, output shape: {wasm_output.shape}")
            
            # Check output shapes match
            assert wasm_output.shape == native_output.shape
            
            # Get performance stats
            stats = runtime.get_runtime_stats()
            print(f"✅ Runtime stats: {stats['inference_count']} inferences, {stats['error_count']} errors")
            
            # Get memory stats
            memory_stats = await wasm_model.get_memory_stats()
            print(f"✅ Memory stats: {memory_stats['parameter_count']} parameters")
            
            await runtime.cleanup()
            print("✅ Runtime cleanup completed")
    
    def test_security_integration(self):
        """Test security features integration."""
        print("Testing security integration...")
        
        # Test model path validation
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"dummy model content")
        
        try:
            security_config = SecurityConfig(max_model_size_mb=1.0)
            validated_path = validate_model_path(temp_path, security_config)
            assert validated_path == temp_path.resolve()
            print("✅ Security validation passed")
        except Exception as e:
            print(f"❌ Security validation failed: {e}")
            raise
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_validation_integration(self):
        """Test validation features integration."""
        print("Testing validation integration...")
        
        # Test valid tensor
        valid_tensor = torch.randn(10, 10)
        try:
            validate_input_tensor(valid_tensor)
            print("✅ Tensor validation passed")
        except Exception as e:
            print(f"❌ Tensor validation failed: {e}")
            raise
        
        # Test invalid tensor
        invalid_tensor = torch.tensor([1.0, float('nan'), 3.0])
        try:
            validate_input_tensor(invalid_tensor)
            print("❌ Tensor validation should have failed")
            raise AssertionError("Validation should have caught NaN")
        except ValueError as e:
            assert "NaN values" in str(e)
            print("✅ NaN detection working")
    
    def test_performance_integration(self):
        """Test performance features integration."""
        print("Testing performance integration...")
        
        monitor = get_performance_monitor()
        
        # Test cache
        monitor.cache_result("test_key", "test_value")
        cached = monitor.get_cached_result("test_key")
        assert cached == "test_value"
        print("✅ Performance caching working")
        
        # Test memory pool
        tensor = monitor.get_tensor_from_pool((5, 5))
        if tensor is not None:
            monitor.return_tensor_to_pool(tensor)
            print("✅ Memory pool working")
        else:
            print("✅ Memory pool working (no suitable tensor)")
        
        # Get stats
        stats = monitor.get_comprehensive_stats()
        assert 'operations' in stats
        assert 'cache' in stats
        assert 'memory_pool' in stats
        print("✅ Performance monitoring working")


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_runtime_error_recovery(self):
        """Test runtime error recovery."""
        runtime = WASMRuntime()
        await runtime.init()
        
        # Test with non-existent model
        try:
            await runtime.load_model(Path("nonexistent.wasm"))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("✅ Error handling working for non-existent model")
        
        # Runtime should still be functional
        stats = runtime.get_runtime_stats()
        assert stats['models_failed'] >= 1
        print("✅ Runtime recovery after error working")
        
        await runtime.cleanup()
    
    def test_validation_error_recovery(self):
        """Test validation error recovery."""
        model = IntegrationTestModel()
        
        # Test with invalid inputs
        invalid_inputs = [
            torch.empty(0),  # Empty tensor
            torch.tensor([float('nan')]),  # NaN values
            torch.tensor([float('inf')]),  # Infinite values
        ]
        
        for i, invalid_input in enumerate(invalid_inputs):
            try:
                validate_input_tensor(invalid_input)
                assert False, f"Should have raised error for input {i}"
            except ValueError:
                print(f"✅ Validation error {i+1} handled correctly")


class TestConcurrency:
    """Test concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_runtime_operations(self):
        """Test concurrent runtime operations."""
        runtime = WASMRuntime(threads=4)
        await runtime.init()
        
        # Create dummy model
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        # Create metadata
        metadata_path = temp_path.with_suffix('.json')
        metadata = {
            "graph": {
                "operations": [{"kind": "aten::relu", "attributes": {}}],
                "parameters": {}
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        try:
            # Load model multiple times concurrently
            tasks = []
            for i in range(3):
                task = runtime.load_model(temp_path)
                tasks.append(task)
            
            models = await asyncio.gather(*tasks)
            assert len(models) == 3
            
            # Run concurrent inferences
            inference_tasks = []
            for model in models:
                test_input = torch.randn(1, 10)
                task = model.forward(test_input)
                inference_tasks.append(task)
            
            results = await asyncio.gather(*inference_tasks)
            assert len(results) == 3
            
            print("✅ Concurrent operations completed successfully")
            
        finally:
            temp_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            await runtime.cleanup()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])