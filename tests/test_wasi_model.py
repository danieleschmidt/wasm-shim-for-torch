import numpy as np
import pytest
from wasm_shim.model import WasiNNModel, WasmMemoryBuffer

def test_model_not_loaded_raises(tiny_onnx_bytes):
    m = WasiNNModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m.infer({})

def test_load_from_bytes(tiny_onnx_bytes):
    m = WasiNNModel()
    m.load_from_bytes(tiny_onnx_bytes)
    assert m.is_loaded

def test_load_from_file(tiny_onnx_file):
    m = WasiNNModel()
    m.load(tiny_onnx_file)
    assert m.is_loaded

def test_load_missing_file_raises():
    m = WasiNNModel()
    with pytest.raises(FileNotFoundError):
        m.load("/nonexistent/path/model.onnx")

def test_input_names(tiny_onnx_bytes):
    m = WasiNNModel()
    m.load_from_bytes(tiny_onnx_bytes)
    assert len(m.input_names) > 0

def test_output_names(tiny_onnx_bytes):
    m = WasiNNModel()
    m.load_from_bytes(tiny_onnx_bytes)
    assert len(m.output_names) > 0

def test_infer_returns_dict(tiny_onnx_bytes):
    m = WasiNNModel()
    m.load_from_bytes(tiny_onnx_bytes)
    inp = {m.input_names[0]: np.random.randn(1, 4).astype(np.float32)}
    result = m.infer(inp)
    assert isinstance(result, dict)
    assert len(result) > 0

def test_infer_output_shape(tiny_onnx_bytes):
    m = WasiNNModel()
    m.load_from_bytes(tiny_onnx_bytes)
    inp = {m.input_names[0]: np.random.randn(1, 4).astype(np.float32)}
    result = m.infer(inp)
    for v in result.values():
        assert isinstance(v, np.ndarray)

def test_infer_from_memory(tiny_onnx_bytes):
    mem = WasmMemoryBuffer(1024 * 1024)
    m = WasiNNModel(memory=mem)
    m.load_from_bytes(tiny_onnx_bytes)
    arr = np.random.randn(1, 4).astype(np.float32)
    mem.write_f32_array(0, arr)
    result = m.infer_from_memory({m.input_names[0]: (0, (1, 4))})
    assert isinstance(result, dict)
