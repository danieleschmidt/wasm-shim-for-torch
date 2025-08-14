"""Input validation and safety utilities for WASM Torch."""

import torch
import logging
import sys
from typing import Optional, Union, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


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


def validate_tensor_safe(tensor: torch.Tensor, name: str) -> None:
    """Enhanced tensor validation with detailed logging and safety checks.
    
    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error reporting
        
    Raises:
        ValueError: If tensor fails validation
        RuntimeError: If tensor contains unsafe values
    """
    try:
        # Basic type checking
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        # Shape validation
        if len(tensor.shape) == 0:
            logger.warning(f"{name} is a scalar tensor")
        elif len(tensor.shape) > 8:
            raise ValueError(f"{name} has too many dimensions ({len(tensor.shape)}), max 8 allowed")
        
        # Size validation
        if tensor.numel() == 0:
            raise ValueError(f"{name} cannot be empty")
        
        # Memory usage check
        tensor_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
        if tensor_size_mb > 2048:  # 2GB limit
            raise ValueError(f"{name} too large ({tensor_size_mb:.1f}MB), max 2GB allowed")
        
        # Data type validation
        if tensor.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            logger.warning(f"{name} has unsupported dtype {tensor.dtype}, may cause issues")
        
        # Value validation
        if torch.isnan(tensor).any():
            raise RuntimeError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise RuntimeError(f"{name} contains infinite values")
        
        # Range validation for floating point tensors
        if tensor.dtype.is_floating_point:
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()
            
            if tensor_min < -1e6 or tensor_max > 1e6:
                logger.warning(f"{name} has extreme values: min={tensor_min:.2e}, max={tensor_max:.2e}")
        
        logger.debug(f"✓ {name} validation passed: shape={tensor.shape}, dtype={tensor.dtype}")
        
    except Exception as e:
        logger.error(f"Tensor validation failed for {name}: {e}")
        raise


def validate_model_compatibility(model: torch.nn.Module, example_input: torch.Tensor) -> Dict[str, Any]:
    """Validate model compatibility for WASM export with comprehensive checks.
    
    Args:
        model: PyTorch model to validate
        example_input: Example input for forward pass testing
        
    Returns:
        Dictionary with validation results and warnings
        
    Raises:
        ValueError: If model is incompatible
        RuntimeError: If validation fails
    """
    results = {
        "compatible": True,
        "warnings": [],
        "errors": [],
        "model_info": {},
        "recommendations": []
    }
    
    try:
        # Model type validation
        if not isinstance(model, torch.nn.Module):
            results["errors"].append("Model must be a torch.nn.Module instance")
            results["compatible"] = False
            return results
        
        # Model mode check
        if model.training:
            results["warnings"].append("Model is in training mode, should be in eval mode for export")
            
        # Parameter analysis
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        results["model_info"].update({
            "parameter_count": param_count,
            "parameter_size_mb": param_size_mb,
            "has_parameters": param_count > 0
        })
        
        if param_count == 0:
            results["warnings"].append("Model has no parameters")
        elif param_count > 100_000_000:  # 100M parameters
            results["warnings"].append(f"Large model ({param_count:,} parameters), may be slow to export")
            
        if param_size_mb > 500:  # 500MB
            results["warnings"].append(f"Large model size ({param_size_mb:.1f}MB), consider quantization")
        
        # Module compatibility check
        unsupported_modules = []
        supported_modules = {
            "Linear", "Conv1d", "Conv2d", "Conv3d", "ReLU", "ReLU6", "LeakyReLU",
            "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
            "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
            "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d",
            "Dropout", "Dropout2d", "Dropout3d", "Identity", "Flatten"
        }
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in supported_modules and module_type != "Sequential" and "Container" not in module_type:
                if module_type not in ["Module"]:  # Skip base Module class
                    unsupported_modules.append(f"{name} ({module_type})")
        
        if unsupported_modules:
            results["warnings"].append(f"Potentially unsupported modules: {', '.join(unsupported_modules[:5])}")
            if len(unsupported_modules) > 5:
                results["warnings"].append(f"... and {len(unsupported_modules) - 5} more")
        
        # Forward pass validation
        try:
            validate_tensor_safe(example_input, "example_input")
            
            model.eval()
            with torch.no_grad():
                output = model(example_input)
                validate_tensor_safe(output, "model_output")
                
                results["model_info"].update({
                    "input_shape": list(example_input.shape),
                    "output_shape": list(output.shape),
                    "input_dtype": str(example_input.dtype),
                    "output_dtype": str(output.dtype)
                })
                
        except Exception as e:
            results["errors"].append(f"Forward pass failed: {str(e)}")
            results["compatible"] = False
        
        # Performance recommendations
        if param_count > 50_000_000:
            results["recommendations"].append("Consider model quantization for better performance")
        if len(unsupported_modules) > 0:
            results["recommendations"].append("Review unsupported modules for potential issues")
        if param_size_mb > 100:
            results["recommendations"].append("Consider using model compression techniques")
        
        # Final compatibility assessment
        if len(results["errors"]) == 0:
            logger.info(f"Model compatibility check passed with {len(results['warnings'])} warnings")
        else:
            logger.error(f"Model compatibility check failed: {results['errors']}")
            results["compatible"] = False
            
    except Exception as e:
        logger.error(f"Model validation error: {e}")
        results["errors"].append(f"Validation exception: {str(e)}")
        results["compatible"] = False
    
    return results


def validate_system_resources() -> Dict[str, Any]:
    """Validate system resources for WASM compilation and execution.
    
    Returns:
        Dictionary with system resource validation results
    """
    resources = {
        "sufficient": True,
        "warnings": [],
        "details": {},
        "recovery_suggestions": []
    }
    
    try:
        # Memory check with adaptive thresholds
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            total_gb = memory_info.total / (1024**3)
            
            resources["details"]["memory"] = {
                "total_gb": round(total_gb, 1),
                "available_gb": round(available_gb, 1),
                "usage_percent": memory_info.percent,
                "threshold_warning": 2.0,
                "threshold_critical": 1.0
            }
            
            if available_gb < 2.0:
                resources["warnings"].append(f"Low memory ({available_gb:.1f}GB available)")
                resources["recovery_suggestions"].append("Close other applications to free memory")
                if available_gb < 1.0:
                    resources["sufficient"] = False
                    resources["recovery_suggestions"].append("Consider using a machine with more RAM")
            elif available_gb < 4.0:
                resources["recovery_suggestions"].append("Monitor memory usage during compilation")
                    
        except ImportError:
            resources["warnings"].append("psutil not available, cannot check memory")
        
        # Disk space check with cleanup suggestions
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            resources["details"]["disk"] = {
                "free_gb": round(free_gb, 1),
                "total_gb": round(disk_usage.total / (1024**3), 1),
                "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1)
            }
            
            if free_gb < 2.0:
                resources["warnings"].append(f"Low disk space ({free_gb:.1f}GB free)")
                resources["recovery_suggestions"].append("Clean temporary files and caches")
                if free_gb < 0.5:
                    resources["sufficient"] = False
                    resources["recovery_suggestions"].append("Free up disk space or use different directory")
            elif free_gb < 5.0:
                resources["recovery_suggestions"].append("Monitor disk usage during build process")
                    
        except Exception as e:
            resources["warnings"].append(f"Cannot check disk space: {e}")
        
        # CPU info
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            resources["details"]["cpu"] = {"cores": cpu_count}
            
            if cpu_count < 2:
                resources["warnings"].append("Limited CPU cores available")
                
        except Exception:
            resources["warnings"].append("Cannot determine CPU count")
        
        # Python version check
        python_version = sys.version_info
        resources["details"]["python"] = {
            "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "supported": python_version >= (3, 10)
        }
        
        if python_version < (3, 10):
            resources["warnings"].append(f"Python {python_version.major}.{python_version.minor} may not be fully supported")
        
        logger.debug(f"System resource validation: {resources}")
        
    except Exception as e:
        logger.error(f"System resource validation failed: {e}")
        resources["warnings"].append(f"Resource validation error: {e}")
    
    return resources


def validate_compilation_environment() -> Dict[str, bool]:
    """Validate that the compilation environment is properly set up.
    
    Returns:
        Dictionary with compilation environment status
    """
    env_status = {
        "emscripten_available": False,
        "cmake_available": False,
        "ninja_available": False,
        "python_dev_available": False
    }
    
    try:
        # Check Emscripten
        import subprocess
        try:
            result = subprocess.run(["emcc", "--version"], capture_output=True, timeout=10)
            env_status["emscripten_available"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check CMake
        try:
            result = subprocess.run(["cmake", "--version"], capture_output=True, timeout=5)
            env_status["cmake_available"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check Ninja
        try:
            result = subprocess.run(["ninja", "--version"], capture_output=True, timeout=5)
            env_status["ninja_available"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check Python development headers
        try:
            import sysconfig
            include_dir = sysconfig.get_path('include')
            python_h = Path(include_dir) / "Python.h"
            env_status["python_dev_available"] = python_h.exists()
        except Exception:
            pass
        
        logger.info(f"Compilation environment status: {env_status}")
        
    except Exception as e:
        logger.error(f"Environment validation error: {e}")
    
    return env_status