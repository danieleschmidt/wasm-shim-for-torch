"""Convert PyTorch models to ONNX format for use with WasiNNModel."""
from __future__ import annotations
import io
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Union, Optional


def torch_to_onnx(model, input_shape: tuple, output_path: Optional[str] = None,
                  opset: int = 17, input_names=None, output_names=None) -> bytes:
    """
    Export a PyTorch model to ONNX format.
    Returns ONNX bytes. Optionally saves to output_path.
    """
    import torch
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    model.eval()
    dummy_input = torch.randn(*input_shape)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buf,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch_size"} for name in input_names + output_names}
    )
    onnx_bytes = buf.getvalue()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(onnx_bytes)
    return onnx_bytes


def build_demo_model():
    """Build a small PyTorch model for demonstration purposes."""
    import torch
    import torch.nn as nn

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 3),
            )

        def forward(self, x):
            return self.net(x)

    return TinyMLP()
