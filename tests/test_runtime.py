import numpy as np
import pytest
from wasm_shim.runtime import WasiNNRuntime

def test_load_model(tiny_onnx_bytes):
    rt = WasiNNRuntime()
    gid = rt.load(tiny_onnx_bytes)
    assert gid == 0

def test_init_context(tiny_onnx_bytes):
    rt = WasiNNRuntime()
    gid = rt.load(tiny_onnx_bytes)
    ctx_id = rt.init_execution_context(gid)
    assert ctx_id == 0

def test_full_inference_pipeline(tiny_onnx_bytes):
    rt = WasiNNRuntime()
    gid = rt.load(tiny_onnx_bytes)
    ctx_id = rt.init_execution_context(gid)
    # Get input name
    model = rt._graphs[gid].model
    inp_name = model.input_names[0]
    tensor = np.random.randn(1, 4).astype(np.float32)
    rt.set_input(ctx_id, inp_name, tensor)
    outputs = rt.compute(ctx_id)
    assert isinstance(outputs, dict)

def test_multiple_contexts(tiny_onnx_bytes):
    rt = WasiNNRuntime()
    gid = rt.load(tiny_onnx_bytes)
    c1 = rt.init_execution_context(gid)
    c2 = rt.init_execution_context(gid)
    assert c1 != c2

def test_invalid_graph_raises(tiny_onnx_bytes):
    rt = WasiNNRuntime()
    with pytest.raises(KeyError):
        rt.init_execution_context(999)
