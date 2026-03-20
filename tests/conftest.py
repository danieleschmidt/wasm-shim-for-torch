import pytest
import numpy as np
import io
import os
import sys
import tempfile

# Build and export a tiny ONNX model for tests
@pytest.fixture(scope="session")
def tiny_onnx_bytes():
    try:
        import torch
        import torch.nn as nn
        from wasm_shim.converter import torch_to_onnx, build_demo_model
        model = build_demo_model()
        return torch_to_onnx(model, (1, 4))
    except ImportError:
        # Build ONNX directly without torch if not available
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        # Create a simple identity-like model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])
        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph([node], "tiny", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        buf = io.BytesIO()
        onnx.save_model(model, buf)
        return buf.getvalue()

@pytest.fixture(scope="session")
def tiny_onnx_file(tiny_onnx_bytes, tmp_path_factory):
    p = tmp_path_factory.mktemp("models") / "tiny.onnx"
    p.write_bytes(tiny_onnx_bytes)
    return str(p)
