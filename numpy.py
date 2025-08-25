"""
Mock numpy module for system-wide compatibility.
This allows tests to import numpy even when not installed.
"""

import math
from src.wasm_torch.mock_dependencies import np, MockNumPy, MockNdarray

# Mathematical constants
pi = math.pi
e = math.e

def radians(degrees):
    """Convert degrees to radians."""
    return math.radians(degrees)

def degrees(radians):
    """Convert radians to degrees."""
    return math.degrees(radians)

def sin(x):
    """Sine function."""
    return math.sin(x)

def cos(x):
    """Cosine function."""
    return math.cos(x)

def exp(x):
    """Exponential function."""
    return math.exp(x)

def sqrt(x):
    """Square root function."""
    return math.sqrt(x)

def log(x):
    """Natural logarithm."""
    return math.log(x)

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
__all__ = ['zeros', 'array', 'random', 'eye', 'dot', 'linalg', 'ndarray', 'float32', 'int32', 
           'pi', 'e', 'radians', 'degrees', 'sin', 'cos', 'exp', 'sqrt', 'log']