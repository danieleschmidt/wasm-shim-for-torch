"""Tests for validation utilities."""

import pytest
import torch
from pathlib import Path
from wasm_torch.validation import (
    validate_input_tensor,
    validate_intermediate_tensor,
    validate_output_tensor,
    sanitize_file_path
)


class TestTensorValidation:
    """Test tensor validation functions."""
    
    def test_validate_input_tensor_valid(self):
        """Test validation with valid tensor."""
        tensor = torch.randn(2, 3)
        # Should not raise any exception
        validate_input_tensor(tensor)
    
    def test_validate_input_tensor_invalid_type(self):
        """Test validation with invalid input type."""
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            validate_input_tensor([1, 2, 3])
    
    def test_validate_input_tensor_empty(self):
        """Test validation with empty tensor."""
        empty_tensor = torch.empty(0)
        with pytest.raises(ValueError, match="Input tensor cannot be empty"):
            validate_input_tensor(empty_tensor)
    
    def test_validate_input_tensor_nan(self):
        """Test validation with NaN values."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="Input tensor contains NaN values"):
            validate_input_tensor(tensor)
    
    def test_validate_input_tensor_inf(self):
        """Test validation with infinite values."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError, match="Input tensor contains infinite values"):
            validate_input_tensor(tensor)
    
    def test_validate_input_tensor_too_large(self):
        """Test validation with very large tensor."""
        # Create tensor larger than 1GB limit
        # This would require too much memory, so we'll mock the size check
        import unittest.mock
        with unittest.mock.patch.object(torch.Tensor, 'numel', return_value=268435456):  # 1GB of float32
            with unittest.mock.patch.object(torch.Tensor, 'element_size', return_value=4):
                tensor = torch.randn(10, 10)  # Small tensor for actual memory usage
                with pytest.raises(ValueError, match="Input tensor too large"):
                    validate_input_tensor(tensor)
    
    def test_validate_intermediate_tensor_valid(self):
        """Test intermediate tensor validation with valid tensor."""
        tensor = torch.randn(2, 3)
        # Should not raise any exception
        validate_intermediate_tensor(tensor, 0, "test_op")
    
    def test_validate_intermediate_tensor_nan(self):
        """Test intermediate tensor validation with NaN."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(RuntimeError, match="NaN values detected after operation 0 \\(test_op\\)"):
            validate_intermediate_tensor(tensor, 0, "test_op")
    
    def test_validate_intermediate_tensor_empty(self):
        """Test intermediate tensor validation with empty tensor."""
        tensor = torch.empty(0)
        with pytest.raises(RuntimeError, match="Empty tensor produced by operation 0 \\(test_op\\)"):
            validate_intermediate_tensor(tensor, 0, "test_op")
    
    def test_validate_output_tensor_valid(self):
        """Test output tensor validation with valid tensor."""
        tensor = torch.randn(2, 3)
        # Should not raise any exception
        validate_output_tensor(tensor)
    
    def test_validate_output_tensor_nan(self):
        """Test output tensor validation with NaN."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(RuntimeError, match="Final output contains NaN values"):
            validate_output_tensor(tensor)
    
    def test_validate_output_tensor_inf(self):
        """Test output tensor validation with infinite values."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(RuntimeError, match="Final output contains infinite values"):
            validate_output_tensor(tensor)
    
    def test_validate_output_tensor_empty(self):
        """Test output tensor validation with empty tensor."""
        tensor = torch.empty(0)
        with pytest.raises(RuntimeError, match="Final output is empty"):
            validate_output_tensor(tensor)


class TestFileSanitization:
    """Test file path sanitization."""
    
    def test_sanitize_file_path_safe(self, tmp_path):
        """Test sanitization with safe path."""
        safe_path = tmp_path / "model.wasm"
        safe_path.touch()
        
        result = sanitize_file_path(str(safe_path), {'.wasm'})
        assert Path(result) == safe_path.resolve()
    
    def test_sanitize_file_path_traversal(self):
        """Test sanitization blocks directory traversal."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            sanitize_file_path("../../../etc/passwd")
    
    def test_sanitize_file_path_invalid_extension(self, tmp_path):
        """Test sanitization blocks invalid extensions."""
        bad_path = tmp_path / "model.exe"
        bad_path.touch()
        
        with pytest.raises(ValueError, match="File extension .exe not allowed"):
            sanitize_file_path(str(bad_path), {'.wasm', '.pth'})
    
    def test_sanitize_file_path_system_directory(self):
        """Test sanitization blocks system directories."""
        with pytest.raises(ValueError, match="Access to system directory .* not allowed"):
            sanitize_file_path("/etc/hosts")