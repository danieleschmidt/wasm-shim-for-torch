# wasm-shim-for-torch

A WASI-NN shim layer for running PyTorch/ONNX models in WebAssembly environments.

## Overview

This library implements the **WASI-NN interface** in Python, backed by ONNX Runtime. It provides:

- **WasiNNModel**: Load ONNX models and run inference (compatible with WASI-NN API surface)
- **WasmMemoryBuffer**: Simulated WebAssembly linear memory (flat byte array with read/write/grow)
- **PyTorch → ONNX converter**: Export PyTorch models for use with the shim
- **WasiNNRuntime**: Full runtime context management (graphs + execution contexts)

> Note: Actual WASM compilation is out of scope. This implements the host-side shim interface.

## Quick Start

```python
from wasm_shim.model import WasiNNModel
from wasm_shim.converter import torch_to_onnx, build_demo_model
import numpy as np

# Export PyTorch model to ONNX
model = build_demo_model()
onnx_bytes = torch_to_onnx(model, input_shape=(1, 4))

# Load via WASI-NN shim
shim = WasiNNModel()
shim.load_from_bytes(onnx_bytes)

# Run inference
output = shim.infer({shim.input_names[0]: np.random.randn(1, 4).astype(np.float32)})
print(output)
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
