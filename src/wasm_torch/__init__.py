"""WASM Shim for Torch: Browser-native PyTorch Inference.

This package provides WebAssembly System Interface (WASI-NN) shim 
to run PyTorch 2.4+ models inside the browser with SIMD & threads.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Gracefully handle missing torch dependency
try:
    from .export import export_to_wasm, register_custom_op, get_custom_operators
    from .runtime import WASMRuntime
    from .optimize import optimize_for_browser, quantize_for_wasm
    
    __all__ = [
        "export_to_wasm",
        "register_custom_op",
        "get_custom_operators", 
        "WASMRuntime", 
        "optimize_for_browser",
        "quantize_for_wasm",
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported due to missing dependencies: {e}")
    
    # Provide mock implementations for testing
    def export_to_wasm(*args, **kwargs):
        raise ImportError("PyTorch not available - mock implementation")
    
    def register_custom_op(*args, **kwargs):
        raise ImportError("PyTorch not available - mock implementation")
    
    def get_custom_operators():
        return {}
    
    class WASMRuntime:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available - mock implementation")
    
    def optimize_for_browser(*args, **kwargs):
        raise ImportError("PyTorch not available - mock implementation")
    
    def quantize_for_wasm(*args, **kwargs):
        raise ImportError("PyTorch not available - mock implementation")
    
    __all__ = [
        "export_to_wasm",
        "register_custom_op", 
        "get_custom_operators",
        "WASMRuntime",
        "optimize_for_browser",
        "quantize_for_wasm",
    ]