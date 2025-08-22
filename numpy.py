"""
Mock numpy module for system-wide compatibility.
This allows tests to import numpy even when not installed.
"""

from src.wasm_torch.mock_dependencies import np, MockNumPy, MockNdarray

# Export all numpy API
zeros = np.zeros
array = np.array
random = np.random()
eye = np.eye
dot = np.dot
linalg = np.linalg()
ndarray = MockNdarray
float32 = 'float32'
int32 = 'int32'

# Version info
__version__ = '1.24.0'

# Re-export for compatibility
__all__ = ['zeros', 'array', 'random', 'eye', 'dot', 'linalg', 'ndarray', 'float32', 'int32']