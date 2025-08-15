"""Mock PyTorch module for development environment."""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


class MockTensor:
    """Mock PyTorch tensor for development."""
    
    def __init__(self, data: Any, dtype: str = "float32"):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def dtype(self) -> str:
        return str(self.data.dtype)
    
    def __str__(self) -> str:
        return f"MockTensor({self.data})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def numpy(self) -> np.ndarray:
        return self.data
    
    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self.shape
        return self.shape[dim]


class MockModule:
    """Mock PyTorch nn.Module."""
    
    def __init__(self):
        self._parameters: Dict[str, MockTensor] = {}
        self._modules: Dict[str, "MockModule"] = {}
        
    def __call__(self, x: MockTensor) -> MockTensor:
        return self.forward(x)
        
    def forward(self, x: MockTensor) -> MockTensor:
        # Simple passthrough for mock
        return x
    
    def parameters(self) -> List[MockTensor]:
        return list(self._parameters.values())
    
    def eval(self) -> "MockModule":
        return self
    
    def train(self, mode: bool = True) -> "MockModule":
        return self


class MockLinear(MockModule):
    """Mock PyTorch Linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MockTensor(np.random.randn(out_features, in_features))
        self.bias = MockTensor(np.random.randn(out_features))
        
    def forward(self, x: MockTensor) -> MockTensor:
        # Simple matrix multiplication
        output = np.dot(x.data, self.weight.data.T) + self.bias.data
        return MockTensor(output)


class MockReLU(MockModule):
    """Mock PyTorch ReLU activation."""
    
    def forward(self, x: MockTensor) -> MockTensor:
        return MockTensor(np.maximum(0, x.data))


class MockNN:
    """Mock torch.nn module."""
    Module = MockModule
    Linear = MockLinear
    ReLU = MockReLU


def randn(*shape: int) -> MockTensor:
    """Mock torch.randn function."""
    return MockTensor(np.random.randn(*shape))


def zeros(*shape: int) -> MockTensor:
    """Mock torch.zeros function."""
    return MockTensor(np.zeros(shape))


def ones(*shape: int) -> MockTensor:
    """Mock torch.ones function."""
    return MockTensor(np.ones(shape))


def tensor(data: Any, dtype: str = "float32") -> MockTensor:
    """Mock torch.tensor function.""" 
    return MockTensor(data, dtype)


# Mock torch module interface
nn = MockNN()