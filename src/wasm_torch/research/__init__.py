"""Research modules for advanced WASM optimization and ML techniques."""

from .adaptive_optimizer import AdaptiveWASMOptimizer

try:
    from .ml_quantizer import MLQuantizationEngine
    _HAS_QUANTIZER = True
except ImportError:
    _HAS_QUANTIZER = False

try:
    from .federated_inference import FederatedInferenceSystem
    _HAS_FEDERATED = True
except ImportError:
    _HAS_FEDERATED = False

try:
    from .streaming_pipeline import StreamingInferencePipeline
    _HAS_STREAMING = True
except ImportError:
    _HAS_STREAMING = False

__all__ = ["AdaptiveWASMOptimizer"]

if _HAS_QUANTIZER:
    __all__.append("MLQuantizationEngine")
if _HAS_FEDERATED:
    __all__.append("FederatedInferenceSystem")
if _HAS_STREAMING:
    __all__.append("StreamingInferencePipeline")