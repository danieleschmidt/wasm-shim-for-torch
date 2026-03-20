"""WASI-NN runtime context — manages graphs and execution contexts."""
from __future__ import annotations
from typing import Optional
import numpy as np
from wasm_shim.model import WasiNNModel, WasmMemoryBuffer


class WasiNNGraph:
    """Represents a loaded neural network graph in the WASI-NN API."""
    def __init__(self, graph_id: int, model: WasiNNModel):
        self.graph_id = graph_id
        self.model = model


class WasiNNExecutionContext:
    """Execution context for running inference on a graph."""
    def __init__(self, ctx_id: int, graph: WasiNNGraph):
        self.ctx_id = ctx_id
        self.graph = graph
        self._inputs: dict[str, np.ndarray] = {}
        self._outputs: dict[str, np.ndarray] = {}

    def set_input(self, name: str, tensor: np.ndarray):
        self._inputs[name] = tensor

    def compute(self) -> dict[str, np.ndarray]:
        self._outputs = self.graph.model.infer(self._inputs)
        return self._outputs

    def get_output(self, name: str) -> np.ndarray:
        if name not in self._outputs:
            raise KeyError(f"Output '{name}' not found. Call compute() first.")
        return self._outputs[name]


class WasiNNRuntime:
    """
    WASI-NN runtime host. Manages graphs and contexts.
    Mimics the wasi-nn spec: load -> init_execution_context -> set_input -> compute -> get_output
    """
    def __init__(self):
        self._graphs: dict[int, WasiNNGraph] = {}
        self._contexts: dict[int, WasiNNExecutionContext] = {}
        self._next_graph_id = 0
        self._next_ctx_id = 0
        self.memory = WasmMemoryBuffer()

    def load(self, model_bytes: bytes, encoding: int = WasiNNModel.GRAPH_ENCODING_ONNX) -> int:
        """Load a model and return graph_id."""
        m = WasiNNModel(self.memory)
        m.load_from_bytes(model_bytes)
        gid = self._next_graph_id
        self._graphs[gid] = WasiNNGraph(gid, m)
        self._next_graph_id += 1
        return gid

    def init_execution_context(self, graph_id: int) -> int:
        if graph_id not in self._graphs:
            raise KeyError(f"Graph {graph_id} not found")
        ctx_id = self._next_ctx_id
        self._contexts[ctx_id] = WasiNNExecutionContext(ctx_id, self._graphs[graph_id])
        self._next_ctx_id += 1
        return ctx_id

    def set_input(self, ctx_id: int, name: str, tensor: np.ndarray):
        self._contexts[ctx_id].set_input(name, tensor)

    def compute(self, ctx_id: int) -> dict[str, np.ndarray]:
        return self._contexts[ctx_id].compute()

    def get_output(self, ctx_id: int, name: str) -> np.ndarray:
        return self._contexts[ctx_id].get_output(name)
