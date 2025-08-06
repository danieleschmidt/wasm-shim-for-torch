"""Comprehensive tests for export functionality."""

import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from wasm_torch.export import (
    export_to_wasm, _validate_export_inputs, _trace_model, _convert_to_ir,
    _generate_compilation_units, _compile_to_wasm, _check_emscripten,
    register_custom_op, get_custom_operators, _check_unsupported_modules,
    _estimate_compilation_time, _check_system_requirements
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class ComplexTestModel(nn.Module):
    """More complex model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestExportValidation:
    """Test export input validation."""
    
    def test_validate_export_inputs_valid(self):
        """Test validation with valid inputs."""
        model = SimpleTestModel()
        model.eval()
        example_input = torch.randn(1, 10)
        
        # Should not raise any exception
        _validate_export_inputs(model, example_input, "O2")
    
    def test_validate_export_inputs_invalid_model(self):
        """Test validation with invalid model."""
        with pytest.raises(ValueError, match="model must be a torch.nn.Module"):
            _validate_export_inputs("not a model", torch.randn(1, 10), "O2")
    
    def test_validate_export_inputs_invalid_input(self):
        """Test validation with invalid input."""
        model = SimpleTestModel()
        
        with pytest.raises(ValueError, match="example_input must be a torch.Tensor"):
            _validate_export_inputs(model, "not a tensor", "O2")
    
    def test_validate_export_inputs_empty_tensor(self):
        """Test validation with empty tensor."""
        model = SimpleTestModel()
        empty_tensor = torch.empty(0)
        
        with pytest.raises(ValueError, match="example_input cannot be empty"):
            _validate_export_inputs(model, empty_tensor, "O2")
    
    def test_validate_export_inputs_nan_tensor(self):
        """Test validation with NaN values."""
        model = SimpleTestModel()
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        with pytest.raises(ValueError, match="example_input contains NaN values"):
            _validate_export_inputs(model, nan_tensor, "O2")
    
    def test_validate_export_inputs_inf_tensor(self):
        """Test validation with infinite values."""
        model = SimpleTestModel()
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        with pytest.raises(ValueError, match="example_input contains infinite values"):
            _validate_export_inputs(model, inf_tensor, "O2")
    
    def test_validate_export_inputs_large_tensor(self):
        """Test validation with oversized tensor."""
        model = SimpleTestModel()
        
        # Mock tensor size to appear large
        with patch.object(torch.Tensor, 'numel', return_value=26214400):  # 100MB of float32
            with patch.object(torch.Tensor, 'element_size', return_value=4):
                large_tensor = torch.randn(10, 10)  # Small for memory, large for test
                with pytest.raises(ValueError, match="Input tensor too large"):
                    _validate_export_inputs(model, large_tensor, "O2")
    
    def test_validate_export_inputs_invalid_optimization_level(self):
        """Test validation with invalid optimization level."""
        model = SimpleTestModel()
        example_input = torch.randn(1, 10)
        
        with pytest.raises(ValueError, match="optimization_level must be one of"):
            _validate_export_inputs(model, example_input, "Invalid")
    
    def test_validate_export_inputs_training_mode_warning(self):
        """Test validation with model in training mode."""
        model = SimpleTestModel()
        model.train()  # Set to training mode
        example_input = torch.randn(1, 10)
        
        # Should not raise exception but should set to eval mode
        _validate_export_inputs(model, example_input, "O2")
        assert not model.training  # Should be in eval mode now


class TestModelTracing:
    """Test model tracing functionality."""
    
    def test_trace_model_success(self):
        """Test successful model tracing."""
        model = SimpleTestModel()
        model.eval()
        example_input = torch.randn(1, 10)
        
        traced_model = _trace_model(model, example_input)
        
        assert isinstance(traced_model, torch.jit.ScriptModule)
        
        # Test that traced model produces same output
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            assert torch.allclose(original_output, traced_output, rtol=1e-4)
    
    def test_trace_model_failure(self):
        """Test model tracing failure."""
        class ProblematicModel(nn.Module):
            def forward(self, x):
                # Dynamic behavior that can't be traced
                if x.sum() > 0:
                    return x * 2
                else:
                    return x * 3
        
        model = ProblematicModel()
        example_input = torch.randn(1, 10)
        
        # May raise tracing error (implementation dependent)
        try:
            traced_model = _trace_model(model, example_input)
            # If tracing succeeds, verify it works
            assert isinstance(traced_model, torch.jit.ScriptModule)
        except ValueError as e:
            assert "Failed to trace model" in str(e)
    
    def test_trace_model_runtime_error(self):
        """Test model tracing with runtime error."""
        class FailingModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Intentional failure")
        
        model = FailingModel()
        example_input = torch.randn(1, 10)
        
        with pytest.raises(ValueError, match="Model forward pass failed"):
            _trace_model(model, example_input)


class TestIRConversion:
    """Test intermediate representation conversion."""
    
    def test_convert_to_ir(self):
        """Test IR conversion."""
        model = SimpleTestModel()
        model.eval()
        example_input = torch.randn(1, 10)
        
        traced_model = torch.jit.trace(model, example_input)
        
        ir_data = _convert_to_ir(traced_model, example_input, use_simd=True, use_threads=True)
        
        assert isinstance(ir_data, dict)
        assert "model_info" in ir_data
        assert "graph" in ir_data
        assert "optimization" in ir_data
        assert "metadata" in ir_data
        
        # Check model info
        model_info = ir_data["model_info"]
        assert model_info["input_shape"] == list(example_input.shape)
        assert model_info["input_dtype"] == str(example_input.dtype)
        
        # Check optimization settings
        optimization = ir_data["optimization"]
        assert optimization["use_simd"] is True
        assert optimization["use_threads"] is True
        
        # Check metadata
        metadata = ir_data["metadata"]
        assert "torch_version" in metadata
        assert "export_version" in metadata


class TestCompilationUnits:
    """Test compilation unit generation."""
    
    def test_generate_compilation_units(self):
        """Test compilation unit generation."""
        model = SimpleTestModel()
        example_input = torch.randn(1, 10)
        traced_model = torch.jit.trace(model, example_input)
        ir_data = _convert_to_ir(traced_model, example_input, True, True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            files = _generate_compilation_units(ir_data, temp_path)
            
            # Check that all expected files are generated
            expected_files = ["runtime", "operations", "header", "interface", "cmake"]
            for file_type in expected_files:
                assert file_type in files
                assert files[file_type].exists()
                assert files[file_type].stat().st_size > 0  # File has content
    
    def test_generate_runtime_cpp(self):
        """Test C++ runtime generation."""
        ir_data = {
            "optimization": {"use_simd": True, "use_threads": True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            from wasm_torch.export import _generate_runtime_cpp
            
            output_path = temp_path / "runtime.cpp"
            _generate_runtime_cpp(ir_data, output_path)
            
            assert output_path.exists()
            content = output_path.read_text()
            
            # Check for expected content
            assert "#include <emscripten.h>" in content
            assert "WASMModel" in content
            assert "forward" in content
            assert "use_simd = true" in content
            assert "use_threads = true" in content


class TestCustomOperators:
    """Test custom operator registration."""
    
    def test_register_custom_op(self):
        """Test custom operator registration."""
        
        @register_custom_op("test_custom_op")
        def custom_function(x, y):
            return x + y
        
        # Check that operator was registered
        custom_ops = get_custom_operators()
        assert "test_custom_op" in custom_ops
        
        op_info = custom_ops["test_custom_op"]
        assert op_info["function"] is custom_function
        assert op_info["name"] == "test_custom_op"
    
    def test_get_custom_operators_empty(self):
        """Test getting custom operators when none registered."""
        # Clear any existing custom operators
        if hasattr(register_custom_op, '_custom_ops'):
            delattr(register_custom_op, '_custom_ops')
        
        custom_ops = get_custom_operators()
        assert isinstance(custom_ops, dict)
        assert len(custom_ops) == 0


class TestUnsupportedModules:
    """Test unsupported module detection."""
    
    def test_check_unsupported_modules_none(self):
        """Test with model containing no unsupported modules."""
        model = SimpleTestModel()
        
        unsupported = _check_unsupported_modules(model)
        assert isinstance(unsupported, list)
        assert len(unsupported) == 0
    
    def test_check_unsupported_modules_with_unsupported(self):
        """Test with model containing unsupported modules."""
        class ModelWithUnsupported(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 20)  # LSTM is potentially unsupported
                self.linear = nn.Linear(20, 1)
            
            def forward(self, x):
                x, _ = self.lstm(x)
                return self.linear(x[-1])
        
        model = ModelWithUnsupported()
        
        unsupported = _check_unsupported_modules(model)
        assert len(unsupported) > 0
        assert any("LSTM" in item for item in unsupported)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_compilation_time(self):
        """Test compilation time estimation."""
        small_model = SimpleTestModel(input_size=5, hidden_size=3, output_size=1)
        large_model = SimpleTestModel(input_size=1000, hidden_size=500, output_size=100)
        
        small_time = _estimate_compilation_time(small_model)
        large_time = _estimate_compilation_time(large_model)
        
        assert isinstance(small_time, float)
        assert isinstance(large_time, float)
        assert small_time > 0
        assert large_time > small_time  # Larger model takes longer
    
    def test_check_system_requirements(self):
        """Test system requirements checking."""
        requirements = _check_system_requirements()
        
        assert isinstance(requirements, dict)
        expected_keys = ['sufficient_memory', 'sufficient_disk', 'emscripten_available']
        for key in expected_keys:
            assert key in requirements
            assert isinstance(requirements[key], bool)
    
    def test_check_emscripten(self):
        """Test Emscripten availability check."""
        available = _check_emscripten()
        assert isinstance(available, bool)
        # Don't assert the specific value since it depends on environment


class TestExportIntegration:
    """Integration tests for export functionality."""
    
    @patch('wasm_torch.export._check_emscripten')
    @patch('wasm_torch.export._compile_to_wasm')
    def test_export_to_wasm_success(self, mock_compile, mock_check_emscripten):
        """Test successful WASM export."""
        mock_check_emscripten.return_value = True
        mock_compile.return_value = None  # Success
        
        model = SimpleTestModel()
        model.eval()
        example_input = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.wasm') as temp_file:
            output_path = Path(temp_file.name)
            
            # Should not raise exception
            export_to_wasm(
                model=model,
                output_path=output_path,
                example_input=example_input,
                optimization_level="O2",
                use_simd=True,
                use_threads=True
            )
            
            # Check that compile function was called
            mock_compile.assert_called_once()
    
    def test_export_to_wasm_validation_failure(self):
        """Test export with validation failure."""
        model = SimpleTestModel()
        invalid_input = "not a tensor"
        
        with tempfile.NamedTemporaryFile(suffix='.wasm') as temp_file:
            output_path = Path(temp_file.name)
            
            with pytest.raises(ValueError, match="example_input must be a torch.Tensor"):
                export_to_wasm(
                    model=model,
                    output_path=output_path,
                    example_input=invalid_input,
                    optimization_level="O2"
                )
    
    @patch('wasm_torch.export._check_emscripten')
    def test_export_to_wasm_no_emscripten(self, mock_check_emscripten):
        """Test export without Emscripten."""
        mock_check_emscripten.return_value = False
        
        model = SimpleTestModel()
        model.eval()
        example_input = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.wasm') as temp_file:
            output_path = Path(temp_file.name)
            
            with pytest.raises(RuntimeError, match="Emscripten not found"):
                export_to_wasm(
                    model=model,
                    output_path=output_path,
                    example_input=example_input,
                    optimization_level="O2"
                )


class TestErrorHandling:
    """Test error handling in export functions."""
    
    def test_export_model_tracing_error(self):
        """Test export with model that can't be traced."""
        class UntracableModel(nn.Module):
            def forward(self, x):
                # Very dynamic behavior
                for i in range(x.shape[0]):
                    if torch.rand(1).item() > 0.5:
                        x[i] *= 2
                return x
        
        model = UntracableModel()
        example_input = torch.randn(3, 5)
        
        with tempfile.NamedTemporaryFile(suffix='.wasm') as temp_file:
            output_path = Path(temp_file.name)
            
            # May succeed or fail depending on PyTorch version and model complexity
            try:
                export_to_wasm(
                    model=model,
                    output_path=output_path,
                    example_input=example_input,
                    optimization_level="O1"
                )
            except RuntimeError as e:
                assert "Failed to export model to WASM" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])