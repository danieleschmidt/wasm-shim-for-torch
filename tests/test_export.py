"""Tests for model export functionality."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from wasm_torch.export import export_to_wasm, register_custom_op


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


def test_export_to_wasm_not_implemented(simple_model, example_input, tmp_path):
    """Test that export_to_wasm raises NotImplementedError."""
    output_path = tmp_path / "model.wasm"
    
    with pytest.raises(NotImplementedError):
        export_to_wasm(simple_model, output_path, example_input)


def test_register_custom_op_not_implemented():
    """Test that register_custom_op raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        @register_custom_op("test_op")
        def test_op(x):
            return x


@pytest.mark.parametrize("optimization_level", ["O0", "O1", "O2", "O3"])
def test_export_optimization_levels(simple_model, example_input, tmp_path, optimization_level):
    """Test different optimization levels."""
    output_path = tmp_path / f"model_{optimization_level}.wasm"
    
    with pytest.raises(NotImplementedError):
        export_to_wasm(
            simple_model,
            output_path,
            example_input,
            optimization_level=optimization_level
        )