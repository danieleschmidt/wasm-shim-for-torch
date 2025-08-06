"""Input validation and safety utilities for WASM Torch."""

import torch
from typing import Optional


def validate_input_tensor(tensor: torch.Tensor) -> None:
    """Validate input tensor for inference."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Input must be a torch.Tensor, got {type(tensor)}")
    
    if tensor.numel() == 0:
        raise ValueError("Input tensor cannot be empty")
    
    if torch.isnan(tensor).any():
        raise ValueError("Input tensor contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValueError("Input tensor contains infinite values")
    
    # Check tensor size (prevent memory issues)
    tensor_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
    if tensor_size_mb > 1000:  # 1GB limit
        raise ValueError(f"Input tensor too large ({tensor_size_mb:.1f}MB). Maximum 1GB allowed.")


def validate_intermediate_tensor(tensor: torch.Tensor, op_index: int, op_type: str) -> None:
    """Validate intermediate tensor during inference."""
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN values detected after operation {op_index} ({op_type})")
    
    if torch.isinf(tensor).any():
        raise RuntimeError(f"Infinite values detected after operation {op_index} ({op_type})")
    
    if tensor.numel() == 0:
        raise RuntimeError(f"Empty tensor produced by operation {op_index} ({op_type})")


def validate_output_tensor(tensor: torch.Tensor) -> None:
    """Validate final output tensor."""
    if torch.isnan(tensor).any():
        raise RuntimeError("Final output contains NaN values")
    
    if torch.isinf(tensor).any():
        raise RuntimeError("Final output contains infinite values")
    
    if tensor.numel() == 0:
        raise RuntimeError("Final output is empty")


def sanitize_file_path(path: str, allowed_extensions: Optional[set] = None) -> str:
    """Sanitize file path to prevent directory traversal attacks.
    
    Args:
        path: File path to sanitize
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Sanitized file path
        
    Raises:
        ValueError: If path is unsafe or has disallowed extension
    """
    import os
    from pathlib import Path
    
    # Convert to Path object for safer handling
    path_obj = Path(path).resolve()
    
    # Check for directory traversal attempts
    if ".." in str(path_obj):
        raise ValueError(f"Path traversal detected in: {path}")
    
    # Check allowed extensions if specified
    if allowed_extensions and path_obj.suffix.lower() not in allowed_extensions:
        raise ValueError(f"File extension {path_obj.suffix} not allowed. Allowed: {allowed_extensions}")
    
    # Ensure path doesn't access system directories
    restricted_paths = {"/etc", "/usr", "/bin", "/sbin", "/root", "/home"}
    for restricted in restricted_paths:
        if str(path_obj).startswith(restricted):
            raise ValueError(f"Access to system directory {restricted} not allowed")
    
    return str(path_obj)