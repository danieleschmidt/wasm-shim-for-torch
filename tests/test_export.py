"""Tests for model export functionality."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from wasm_torch.export import export_to_wasm, register_custom_op, get_custom_operators


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleModel()


@pytest.fixture
def example_input():
    """Create example input tensor."""
    return torch.randn(1, 10)


def test_export_input_validation(simple_model, example_input, tmp_path):
    """Test input validation for export_to_wasm."""
    output_path = tmp_path / "model.wasm"
    
    # Test invalid model type
    with pytest.raises(ValueError, match="model must be a torch.nn.Module"):
        export_to_wasm("not_a_model", output_path, example_input)
    
    # Test invalid input tensor
    with pytest.raises(ValueError, match="example_input must be a torch.Tensor"):
        export_to_wasm(simple_model, output_path, "not_a_tensor")
    
    # Test invalid optimization level
    with pytest.raises(ValueError, match="optimization_level must be one of"):
        export_to_wasm(simple_model, output_path, example_input, optimization_level="O9")


def test_export_requires_emscripten(simple_model, example_input, tmp_path):
    """Test that export fails gracefully without Emscripten."""
    output_path = tmp_path / "model.wasm"
    
    with patch('wasm_torch.export._check_emscripten', return_value=False):
        with pytest.raises(RuntimeError, match="Emscripten not found"):
            export_to_wasm(simple_model, output_path, example_input)


def test_model_tracing(simple_model, example_input, tmp_path):
    """Test PyTorch model tracing."""
    from wasm_torch.export import _trace_model
    
    traced_model = _trace_model(simple_model, example_input)
    assert isinstance(traced_model, torch.jit.ScriptModule)
    
    # Test that traced model produces same output
    simple_model.eval()
    with torch.no_grad():
        original_output = simple_model(example_input)
        traced_output = traced_model(example_input)
        
    # Allow for small numerical differences
    assert torch.allclose(original_output, traced_output, atol=1e-6)


def test_ir_conversion(simple_model, example_input):
    """Test conversion to intermediate representation."""
    from wasm_torch.export import _trace_model, _convert_to_ir
    
    traced_model = _trace_model(simple_model, example_input)
    ir_data = _convert_to_ir(traced_model, example_input, use_simd=True, use_threads=True)
    
    # Validate IR structure
    assert "model_info" in ir_data
    assert "graph" in ir_data
    assert "optimization" in ir_data
    assert "metadata" in ir_data
    
    # Check model info
    assert ir_data["model_info"]["input_shape"] == list(example_input.shape)
    assert ir_data["model_info"]["input_dtype"] == str(example_input.dtype)
    
    # Check optimization settings
    assert ir_data["optimization"]["use_simd"] is True
    assert ir_data["optimization"]["use_threads"] is True


def test_register_custom_op():
    """Test custom operator registration."""
    # Clear existing custom operators
    if hasattr(register_custom_op, '_custom_ops'):
        register_custom_op._custom_ops.clear()
    
    @register_custom_op("test_op")
    def test_op(x: torch.Tensor) -> torch.Tensor:
        return x * 2
    
    # Check registration
    custom_ops = get_custom_operators()
    assert "test_op" in custom_ops
    assert custom_ops["test_op"]["name"] == "test_op"
    assert custom_ops["test_op"]["function"] == test_op


def test_custom_operator_usage():
    """Test that registered custom operators work."""
    # Clear existing custom operators
    if hasattr(register_custom_op, '_custom_ops'):
        register_custom_op._custom_ops.clear()
    
    @register_custom_op("double")
    def double_op(x):
        return x * 2
    
    # Test the operator function itself
    result = double_op(5)
    assert result == 10
    
    # Test it's registered
    custom_ops = get_custom_operators()
    assert len(custom_ops) == 1
    assert "double" in custom_ops


@pytest.mark.parametrize("optimization_level", ["O0", "O1", "O2", "O3"])
def test_export_optimization_levels_validation(simple_model, example_input, tmp_path, optimization_level):
    """Test different optimization levels pass validation."""
    from wasm_torch.export import _validate_export_inputs
    
    # These should not raise exceptions
    _validate_export_inputs(simple_model, example_input, optimization_level)


def test_compilation_unit_generation(simple_model, example_input, tmp_path):
    """Test generation of compilation units."""
    from wasm_torch.export import _trace_model, _convert_to_ir, _generate_compilation_units
    
    traced_model = _trace_model(simple_model, example_input)
    ir_data = _convert_to_ir(traced_model, example_input, use_simd=True, use_threads=True)
    
    files = _generate_compilation_units(ir_data, tmp_path)
    
    # Check that all expected files are generated
    expected_files = ["runtime", "operations", "header", "interface", "cmake"]
    for file_type in expected_files:
        assert file_type in files
        assert files[file_type].exists()
        assert files[file_type].stat().st_size > 0  # File is not empty
    
    # Check C++ files have reasonable content
    cpp_content = files["runtime"].read_text()
    assert "#include <emscripten.h>" in cpp_content
    assert "WASMModel" in cpp_content
    
    # Check header file
    header_content = files["header"].read_text()
    assert "ModelConfig" in header_content
    assert "ModelOperation" in header_content


@pytest.mark.slow
def test_full_export_with_mock_emscripten(simple_model, example_input, tmp_path):
    """Test full export pipeline with mocked Emscripten."""
    output_path = tmp_path / "model.wasm"
    
    with patch('wasm_torch.export._check_emscripten', return_value=True), \
         patch('subprocess.run') as mock_run, \
         patch('shutil.copy2') as mock_copy:
        
        # Mock successful build
        mock_run.return_value.returncode = 0
        
        # Mock output files exist
        with patch('pathlib.Path.exists', return_value=True):
            # This should complete without error
            export_to_wasm(simple_model, output_path, example_input)
        
        # Verify subprocess calls were made
        assert mock_run.call_count >= 2  # cmake + make
        assert mock_copy.call_count >= 1  # copy output files