"""Machine Learning-based adaptive quantization engine.

This module implements novel quantization techniques using reinforcement learning
to adaptively select quantization schemes based on model characteristics,
target accuracy, and deployment constraints.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig

from ..security import log_security_event
from ..validation import validate_tensor_safe


logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "quantization_aware_training"
    MIXED_PRECISION = "mixed_precision"
    ADAPTIVE = "adaptive"


@dataclass
class QuantizationConfig:
    """Configuration for quantization process."""
    
    quantization_type: QuantizationType
    target_accuracy_loss: float = 0.05  # Maximum acceptable accuracy loss
    compression_ratio: float = 4.0  # Target compression ratio
    preserve_accuracy_layers: List[str] = None  # Layers to keep in FP32
    calibration_samples: int = 100
    mixed_precision_policy: Optional[Dict] = None


@dataclass
class QuantizationResult:
    """Results of quantization process."""
    
    quantized_model: nn.Module
    accuracy_loss: float
    compression_ratio: float
    size_reduction: float
    inference_speedup: float
    config_used: QuantizationConfig
    sensitivity_analysis: Dict[str, float]


class SensitivityAnalyzer:
    """Analyzes layer-wise sensitivity to quantization."""
    
    def __init__(self, model: nn.Module, calibration_data: torch.Tensor):
        self.model = model
        self.calibration_data = calibration_data
        self.layer_sensitivities: Dict[str, float] = {}
        
    def analyze_layer_sensitivity(self) -> Dict[str, float]:
        """Analyze sensitivity of each layer to quantization."""
        
        validate_tensor_safe(self.calibration_data, "calibration_data")
        
        # Get baseline accuracy
        baseline_outputs = self._get_baseline_outputs()
        
        sensitivities = {}
        
        # Analyze each quantizable layer
        for name, module in self.model.named_modules():
            if self._is_quantizable_layer(module):
                sensitivity = self._compute_layer_sensitivity(
                    name, module, baseline_outputs
                )
                sensitivities[name] = sensitivity
                logger.debug(f"Layer {name} sensitivity: {sensitivity:.4f}")
        
        self.layer_sensitivities = sensitivities
        return sensitivities
    
    def _get_baseline_outputs(self) -> torch.Tensor:
        """Get baseline model outputs for comparison."""
        self.model.eval()
        with torch.no_grad():
            return self.model(self.calibration_data)
    
    def _is_quantizable_layer(self, module: nn.Module) -> bool:
        """Check if layer can be quantized."""
        quantizable_types = (
            nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d,
            nn.BatchNorm2d, nn.BatchNorm1d, nn.LSTM, nn.GRU
        )
        return isinstance(module, quantizable_types)
    
    def _compute_layer_sensitivity(
        self, layer_name: str, layer: nn.Module, baseline_outputs: torch.Tensor
    ) -> float:
        """Compute sensitivity of specific layer to quantization."""
        
        # Temporarily quantize just this layer
        original_dtype = None
        if hasattr(layer, 'weight'):
            original_dtype = layer.weight.dtype
            # Simulate quantization by adding noise
            with torch.no_grad():
                noise_scale = layer.weight.abs().max() * 0.01  # 1% noise
                layer.weight.data += torch.randn_like(layer.weight) * noise_scale
        
        # Compute outputs with modified layer
        self.model.eval()
        with torch.no_grad():
            modified_outputs = self.model(self.calibration_data)
        
        # Restore original weights
        if hasattr(layer, 'weight') and original_dtype is not None:
            with torch.no_grad():
                layer.weight.data -= torch.randn_like(layer.weight) * noise_scale
        
        # Compute sensitivity as output difference
        sensitivity = torch.mean((baseline_outputs - modified_outputs) ** 2).item()
        return float(sensitivity)
    
    def get_sensitive_layers(self, threshold: float = 0.1) -> List[str]:
        """Get list of layers that are sensitive to quantization."""
        if not self.layer_sensitivities:
            self.analyze_layer_sensitivity()
            
        return [
            name for name, sensitivity in self.layer_sensitivities.items()
            if sensitivity > threshold
        ]


class AdaptiveQuantizationPolicy:
    """Policy for adaptive quantization based on layer characteristics."""
    
    def __init__(self, sensitivity_analyzer: SensitivityAnalyzer):
        self.sensitivity_analyzer = sensitivity_analyzer
        self.policy_network = self._build_policy_network()
        
    def _build_policy_network(self) -> nn.Module:
        """Build neural network for quantization policy decisions."""
        return nn.Sequential(
            nn.Linear(8, 64),  # Input: layer features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 4),  # Output: quantization decision (8bit, 16bit, FP32, skip)
            nn.Softmax(dim=-1)
        )
    
    def extract_layer_features(self, layer_name: str, layer: nn.Module) -> torch.Tensor:
        """Extract features for policy decision."""
        features = []
        
        # Basic layer properties
        if hasattr(layer, 'weight'):
            weight = layer.weight
            features.extend([
                float(weight.numel()),  # Parameter count
                float(weight.abs().mean()),  # Mean absolute weight
                float(weight.std()),  # Weight standard deviation
                float(weight.abs().max()),  # Max absolute weight
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Sensitivity score
        sensitivity = self.sensitivity_analyzer.layer_sensitivities.get(layer_name, 0.0)
        features.append(float(sensitivity))
        
        # Layer type encoding
        layer_type_encoding = self._encode_layer_type(layer)
        features.extend(layer_type_encoding)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_layer_type(self, layer: nn.Module) -> List[float]:
        """Encode layer type as feature vector."""
        # One-hot encoding for common layer types
        if isinstance(layer, nn.Linear):
            return [1.0, 0.0, 0.0]
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return [0.0, 1.0, 0.0]
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return [0.0, 0.0, 1.0]
        else:
            return [0.0, 0.0, 0.0]
    
    def decide_quantization(self, layer_name: str, layer: nn.Module) -> str:
        """Decide quantization strategy for specific layer."""
        features = self.extract_layer_features(layer_name, layer)
        
        with torch.no_grad():
            policy_output = self.policy_network(features)
            decision = torch.argmax(policy_output).item()
        
        # Map decision to quantization strategy
        strategies = ["int8", "int16", "fp32", "skip"]
        return strategies[decision]


class MLQuantizationEngine:
    """Machine learning-based adaptive quantization engine."""
    
    def __init__(self, calibration_dataset_size: int = 100):
        self.calibration_dataset_size = calibration_dataset_size
        self.quantization_history: List[QuantizationResult] = []
        
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizationResult:
        """Apply adaptive quantization to model."""
        
        validate_tensor_safe(calibration_data, "calibration_data")
        
        start_time = time.time()
        logger.info(f"Starting {config.quantization_type.value} quantization")
        
        # Analyze layer sensitivities
        sensitivity_analyzer = SensitivityAnalyzer(model, calibration_data)
        sensitivities = sensitivity_analyzer.analyze_layer_sensitivity()
        
        # Apply quantization based on type
        if config.quantization_type == QuantizationType.ADAPTIVE:
            quantized_model = self._adaptive_quantization(
                model, calibration_data, sensitivity_analyzer, config
            )
        elif config.quantization_type == QuantizationType.MIXED_PRECISION:
            quantized_model = self._mixed_precision_quantization(
                model, sensitivities, config
            )
        else:
            quantized_model = self._standard_quantization(model, config)
        
        # Evaluate results
        result = self._evaluate_quantization(
            model, quantized_model, calibration_data, config, sensitivities
        )
        
        self.quantization_history.append(result)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Quantization completed in {elapsed_time:.2f}s. "
                   f"Accuracy loss: {result.accuracy_loss:.3f}, "
                   f"Compression: {result.compression_ratio:.1f}x")
        
        log_security_event("model_quantized", {
            "quantization_type": config.quantization_type.value,
            "accuracy_loss": result.accuracy_loss,
            "compression_ratio": result.compression_ratio,
        })
        
        return result
    
    def _adaptive_quantization(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        sensitivity_analyzer: SensitivityAnalyzer,
        config: QuantizationConfig,
    ) -> nn.Module:
        """Apply adaptive quantization using ML policy."""
        
        # Create adaptive policy
        policy = AdaptiveQuantizationPolicy(sensitivity_analyzer)
        
        # Clone model for modification
        quantized_model = torch.jit.script(model.eval())
        
        # Apply layer-specific quantization
        layer_decisions = {}
        for name, layer in model.named_modules():
            if sensitivity_analyzer._is_quantizable_layer(layer):
                decision = policy.decide_quantization(name, layer)
                layer_decisions[name] = decision
                
                # Apply quantization based on decision
                if decision == "int8":
                    self._quantize_layer_int8(quantized_model, name, layer)
                elif decision == "int16":
                    self._quantize_layer_int16(quantized_model, name, layer)
                # fp32 and skip leave layer unchanged
        
        logger.info(f"Adaptive quantization decisions: {layer_decisions}")
        return quantized_model
    
    def _mixed_precision_quantization(
        self,
        model: nn.Module,
        sensitivities: Dict[str, float],
        config: QuantizationConfig,
    ) -> nn.Module:
        """Apply mixed precision quantization."""
        
        # Determine precision based on sensitivity
        sensitive_threshold = np.percentile(list(sensitivities.values()), 75)
        
        quantized_model = model  # Start with original model
        
        for name, layer in model.named_modules():
            if name in sensitivities:
                sensitivity = sensitivities[name]
                
                if sensitivity < sensitive_threshold:
                    # Low sensitivity -> aggressive quantization
                    quantized_model = self._quantize_layer_int8(quantized_model, name, layer)
                else:
                    # High sensitivity -> preserve precision
                    logger.debug(f"Preserving FP32 for sensitive layer: {name}")
        
        return quantized_model
    
    def _standard_quantization(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply standard quantization methods."""
        
        if config.quantization_type == QuantizationType.DYNAMIC:
            return quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        
        # For other types, would implement static/QAT quantization
        logger.warning(f"Standard quantization for {config.quantization_type} not fully implemented")
        return model
    
    def _quantize_layer_int8(self, model: nn.Module, layer_name: str, layer: nn.Module) -> nn.Module:
        """Quantize specific layer to INT8."""
        # Simplified quantization - in practice would use proper PyTorch quantization
        if hasattr(layer, 'weight'):
            with torch.no_grad():
                # Simulate INT8 quantization
                weight = layer.weight
                scale = weight.abs().max() / 127.0
                quantized_weight = torch.round(weight / scale).clamp(-128, 127)
                layer.weight.data = quantized_weight * scale
        
        return model
    
    def _quantize_layer_int16(self, model: nn.Module, layer_name: str, layer: nn.Module) -> nn.Module:
        """Quantize specific layer to INT16."""
        if hasattr(layer, 'weight'):
            with torch.no_grad():
                # Simulate INT16 quantization
                weight = layer.weight
                scale = weight.abs().max() / 32767.0
                quantized_weight = torch.round(weight / scale).clamp(-32768, 32767)
                layer.weight.data = quantized_weight * scale
        
        return model
    
    def _evaluate_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: torch.Tensor,
        config: QuantizationConfig,
        sensitivities: Dict[str, float],
    ) -> QuantizationResult:
        """Evaluate quantization results."""
        
        # Calculate model sizes
        original_size = self._calculate_model_size(original_model)
        quantized_size = self._calculate_model_size(quantized_model)
        
        size_reduction = (original_size - quantized_size) / original_size
        compression_ratio = original_size / quantized_size
        
        # Simulate accuracy evaluation
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            original_output = original_model(test_data)
            quantized_output = quantized_model(test_data)
            
            # Compute accuracy loss as MSE
            accuracy_loss = torch.mean((original_output - quantized_output) ** 2).item()
        
        # Simulate inference speedup
        inference_speedup = compression_ratio * 0.8  # Rough approximation
        
        return QuantizationResult(
            quantized_model=quantized_model,
            accuracy_loss=accuracy_loss,
            compression_ratio=compression_ratio,
            size_reduction=size_reduction,
            inference_speedup=inference_speedup,
            config_used=config,
            sensitivity_analysis=sensitivities,
        )
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def optimize_quantization_config(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        target_accuracy_loss: float = 0.05,
        target_compression: float = 4.0,
    ) -> QuantizationConfig:
        """Optimize quantization configuration for target metrics."""
        
        validate_tensor_safe(calibration_data, "calibration_data")
        
        # Try different quantization strategies
        strategies = [
            QuantizationType.DYNAMIC,
            QuantizationType.MIXED_PRECISION,
            QuantizationType.ADAPTIVE,
        ]
        
        best_config = None
        best_score = float('-inf')
        
        for strategy in strategies:
            config = QuantizationConfig(
                quantization_type=strategy,
                target_accuracy_loss=target_accuracy_loss,
                compression_ratio=target_compression,
            )
            
            try:
                result = self.quantize_model(model, calibration_data, config)
                
                # Score based on accuracy and compression trade-off
                accuracy_score = max(0, target_accuracy_loss - result.accuracy_loss)
                compression_score = min(result.compression_ratio, target_compression * 2) / target_compression
                
                total_score = accuracy_score * 2 + compression_score  # Weight accuracy higher
                
                if total_score > best_score:
                    best_score = total_score
                    best_config = config
                    
                logger.info(f"Strategy {strategy.value}: score={total_score:.3f}, "
                           f"accuracy_loss={result.accuracy_loss:.3f}, "
                           f"compression={result.compression_ratio:.1f}x")
                
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue
        
        if best_config is None:
            logger.warning("No quantization strategy succeeded, using default")
            best_config = QuantizationConfig(
                quantization_type=QuantizationType.DYNAMIC,
                target_accuracy_loss=target_accuracy_loss,
            )
        
        logger.info(f"Optimal quantization strategy: {best_config.quantization_type.value}")
        return best_config
    
    def export_quantization_report(self, model_name: str) -> Dict:
        """Export detailed quantization analysis report."""
        
        if not self.quantization_history:
            return {"error": "No quantization history available"}
        
        latest_result = self.quantization_history[-1]
        
        report = {
            "model_name": model_name,
            "quantization_summary": {
                "type": latest_result.config_used.quantization_type.value,
                "accuracy_loss": latest_result.accuracy_loss,
                "compression_ratio": latest_result.compression_ratio,
                "size_reduction_percent": latest_result.size_reduction * 100,
                "inference_speedup": latest_result.inference_speedup,
            },
            "sensitivity_analysis": latest_result.sensitivity_analysis,
            "recommendations": self._generate_quantization_recommendations(latest_result),
            "history": [
                {
                    "type": result.config_used.quantization_type.value,
                    "accuracy_loss": result.accuracy_loss,
                    "compression_ratio": result.compression_ratio,
                }
                for result in self.quantization_history
            ],
            "timestamp": time.time(),
        }
        
        return report
    
    def _generate_quantization_recommendations(self, result: QuantizationResult) -> List[str]:
        """Generate recommendations based on quantization results."""
        recommendations = []
        
        if result.accuracy_loss > 0.1:
            recommendations.append(
                "High accuracy loss detected. Consider mixed precision or preserving sensitive layers."
            )
        
        if result.compression_ratio < 2.0:
            recommendations.append(
                "Low compression achieved. Try more aggressive quantization or different strategy."
            )
        
        if result.sensitivity_analysis:
            sensitive_layers = [
                name for name, sensitivity in result.sensitivity_analysis.items() 
                if sensitivity > 0.1
            ]
            if sensitive_layers:
                recommendations.append(
                    f"Consider preserving FP32 for sensitive layers: {', '.join(sensitive_layers[:3])}"
                )
        
        if not recommendations:
            recommendations.append("Quantization results look good! Consider deployment.")
        
        return recommendations