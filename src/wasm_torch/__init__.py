"""WASM Shim for Torch: Browser-native PyTorch Inference.

This package provides WebAssembly System Interface (WASI-NN) shim 
to run PyTorch 2.4+ models inside the browser with SIMD & threads.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Check for torch availability and handle gracefully
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    # Use mock torch implementation
    try:
        from .mock_torch import torch
        import warnings
        warnings.warn("PyTorch not available, using mock implementation for testing")
        torch_available = False
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import mock torch: {e}")

# Import main functionality
try:
    if torch_available:
        # Use real implementations
        from .export import export_to_wasm, register_custom_op, get_custom_operators
        from .runtime import WASMRuntime  
        from .optimize import optimize_for_browser, quantize_for_wasm
    else:
        # Use mock-compatible implementations
        from .basic_model_loader import MockExporter, MockWASMRuntime, MockOptimizer
        
        # Create compatibility layer
        def export_to_wasm(*args, **kwargs):
            exporter = MockExporter()
            return exporter.export_to_wasm(*args, **kwargs)
        
        def register_custom_op(name):
            def decorator(func):
                return func
            return decorator
        
        def get_custom_operators():
            return {}
        
        WASMRuntime = MockWASMRuntime
        
        def optimize_for_browser(*args, **kwargs):
            optimizer = MockOptimizer()
            return optimizer.optimize_for_browser(*args, **kwargs)
        
        def quantize_for_wasm(*args, **kwargs):
            optimizer = MockOptimizer()
            return optimizer.quantize_for_wasm(*args, **kwargs)
    
    __all__ = [
        "export_to_wasm",
        "register_custom_op",
        "get_custom_operators", 
        "WASMRuntime", 
        "optimize_for_browser",
        "quantize_for_wasm",
        "torch_available"
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Critical modules could not be imported: {e}")
    
    # Fallback implementations
    def export_to_wasm(*args, **kwargs):
        raise NotImplementedError("WASM export not available")
    
    def register_custom_op(*args, **kwargs):
        raise NotImplementedError("Custom operators not available")
    
    def get_custom_operators():
        return {}
    
    class WASMRuntime:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("WASM runtime not available")
    
    def optimize_for_browser(*args, **kwargs):
        raise NotImplementedError("Browser optimization not available")
    
    def quantize_for_wasm(*args, **kwargs):
        raise NotImplementedError("WASM quantization not available")
    
    __all__ = [
        "export_to_wasm",
        "register_custom_op", 
        "get_custom_operators",
        "WASMRuntime",
        "optimize_for_browser", 
        "quantize_for_wasm",
        "torch_available"
    ]