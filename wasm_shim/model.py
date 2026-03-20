"""
WASI-NN shim for running ONNX/PyTorch models in WebAssembly environments.
Implements the WasiNNModel interface with ONNX Runtime backend.
"""
from __future__ import annotations
import io
import os
import struct
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union
import onnxruntime as ort


class WasmMemoryBuffer:
    """
    Mock WebAssembly linear memory buffer.
    Simulates WASM memory as a flat byte array with read/write ops.
    """
    def __init__(self, size_bytes: int = 64 * 1024 * 1024):  # 64MB default
        self._memory = bytearray(size_bytes)
        self._size = size_bytes
        self._cursor = 0

    @property
    def size(self) -> int:
        return self._size

    def write(self, offset: int, data: bytes) -> int:
        """Write bytes at offset, return bytes written."""
        end = offset + len(data)
        if end > self._size:
            raise MemoryError(f"Write exceeds WASM memory: {end} > {self._size}")
        self._memory[offset:end] = data
        return len(data)

    def read(self, offset: int, length: int) -> bytes:
        """Read bytes from offset."""
        if offset + length > self._size:
            raise MemoryError(f"Read exceeds WASM memory: {offset + length} > {self._size}")
        return bytes(self._memory[offset:offset + length])

    def write_f32_array(self, offset: int, arr: np.ndarray) -> int:
        """Write float32 array to memory."""
        flat = arr.astype(np.float32).flatten()
        data = struct.pack(f"{len(flat)}f", *flat)
        return self.write(offset, data)

    def read_f32_array(self, offset: int, count: int) -> np.ndarray:
        """Read float32 array from memory."""
        data = self.read(offset, count * 4)
        return np.array(struct.unpack(f"{count}f", data), dtype=np.float32)

    def grow(self, pages: int = 1, page_size: int = 65536) -> int:
        """Grow memory by N pages (WASM page = 64KB)."""
        new_bytes = pages * page_size
        self._memory.extend(bytearray(new_bytes))
        self._size += new_bytes
        return pages


class WasiNNModel:
    """
    WASI-NN compatible model interface backed by ONNX Runtime.
    Supports loading .onnx models and running inference via the WASI-NN API surface.
    """
    GRAPH_ENCODING_ONNX = 1
    EXECUTION_TARGET_CPU = 0
    EXECUTION_TARGET_GPU = 1

    def __init__(self, memory: Optional[WasmMemoryBuffer] = None):
        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[str] = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_shapes: dict[str, list] = {}
        self._memory = memory or WasmMemoryBuffer()
        self._loaded = False

    def load(self, model_path: Union[str, Path]) -> "WasiNNModel":
        """Load an ONNX model from disk."""
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(model_path, sess_options=opts,
                                              providers=["CPUExecutionProvider"])
        self._model_path = model_path
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_shapes = {i.name: i.shape for i in self._session.get_inputs()}
        self._loaded = True
        return self

    def load_from_bytes(self, model_bytes: bytes) -> "WasiNNModel":
        """Load an ONNX model from bytes (e.g., from WASM memory)."""
        opts = ort.SessionOptions()
        self._session = ort.InferenceSession(model_bytes, sess_options=opts,
                                              providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_shapes = {i.name: i.shape for i in self._session.get_inputs()}
        self._loaded = True
        return self

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. Returns dict of output_name -> numpy array."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        outputs = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, outputs))

    def infer_from_memory(self, input_offsets: dict[str, tuple[int, tuple]]) -> dict[str, np.ndarray]:
        """
        Run inference reading inputs from WASM memory.
        input_offsets: {name: (memory_offset, shape)}
        """
        inputs = {}
        for name, (offset, shape) in input_offsets.items():
            count = int(np.prod(shape))
            flat = self._memory.read_f32_array(offset, count)
            inputs[name] = flat.reshape(shape)
        return self.infer(inputs)

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_input_shape(self, name: str) -> list:
        return self._input_shapes.get(name, [])
