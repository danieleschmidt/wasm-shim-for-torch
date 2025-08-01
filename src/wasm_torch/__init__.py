"""WASM Shim for Torch: Browser-native PyTorch Inference.

This package provides WebAssembly System Interface (WASI-NN) shim 
to run PyTorch 2.4+ models inside the browser with SIMD & threads.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

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