"""Smart model optimizer with ML-powered optimization strategies."""

import asyncio
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import numpy as np
from collections import defaultdict
import threading


logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Model optimization profile with performance characteristics."""
    model_id: str
    input_shapes: List[Tuple[int, ...]]
    param_count: int
    model_size_mb: float
    target_platform: str = "browser"
    target_latency_ms: float = 100.0
    memory_constraint_mb: float = 512.0
    accuracy_threshold: float = 0.95
    

@dataclass
class OptimizationStrategy:
    """Optimization strategy with specific techniques and parameters."""
    name: str
    techniques: List[str]
    parameters: Dict[str, Any]
    expected_speedup: float
    expected_size_reduction: float
    accuracy_impact: float
    compatibility_score: float
    

@dataclass
class OptimizationResult:
    """Result of model optimization with metrics."""
    strategy_name: str
    optimized_size_mb: float
    optimization_time_s: float
    predicted_latency_ms: float
    accuracy_retention: float
    success: bool
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ModelAnalyzer:
    """Analyzes models to determine optimal optimization strategies."""
    
    def __init__(self):
        self.architecture_patterns = {
            'cnn': ['conv', 'pool', 'bn'],
            'transformer': ['attention', 'layernorm', 'linear'],
            'rnn': ['lstm', 'gru', 'rnn'],
            'hybrid': ['conv', 'attention', 'linear']
        }
        
        self.optimization_rules = {
            'cnn': ['quantization', 'pruning', 'knowledge_distillation'],
            'transformer': ['attention_pruning', 'quantization', 'layer_fusion'],
            'rnn': ['quantization', 'cell_fusion'],
            'hybrid': ['selective_optimization', 'quantization']
        }
    
    def analyze_model_architecture(self, model_info: Dict[str, Any]) -> str:
        """Analyze model architecture to determine type."""
        layer_types = model_info.get('layer_types', [])
        layer_counts = defaultdict(int)
        
        for layer in layer_types:
            layer_name = layer.lower()
            for arch_type, patterns in self.architecture_patterns.items():
                for pattern in patterns:
                    if pattern in layer_name:
                        layer_counts[arch_type] += 1
        
        if not layer_counts:
            return 'unknown'
        
        return max(layer_counts, key=layer_counts.get)
    
    def estimate_optimization_potential(self, profile: OptimizationProfile) -> Dict[str, float]:
        """Estimate optimization potential for different techniques."""
        potential = {
            'quantization': 0.5 if profile.param_count > 1000000 else 0.3,
            'pruning': 0.3 if profile.model_size_mb > 100 else 0.2,
            'knowledge_distillation': 0.4 if profile.param_count > 5000000 else 0.2,
            'layer_fusion': 0.15,
            'weight_sharing': 0.25 if profile.param_count > 1000000 else 0.1
        }
        
        # Adjust based on target constraints
        if profile.memory_constraint_mb < profile.model_size_mb * 0.5:
            potential['quantization'] *= 1.5
            potential['pruning'] *= 1.3
        
        return potential


class AdaptiveOptimizer:
    """Adaptive optimizer that learns from optimization results."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.model_analyzer = ModelAnalyzer()
        
        # Pre-defined optimization strategies
        self.strategies = {
            'aggressive_quantization': OptimizationStrategy(
                name='aggressive_quantization',
                techniques=['int8_quantization', 'weight_pruning'],
                parameters={'quantization_bits': 8, 'pruning_ratio': 0.3},
                expected_speedup=2.5,
                expected_size_reduction=0.75,
                accuracy_impact=0.02,
                compatibility_score=0.9
            ),
            'balanced_optimization': OptimizationStrategy(
                name='balanced_optimization',
                techniques=['int16_quantization', 'selective_pruning'],
                parameters={'quantization_bits': 16, 'pruning_ratio': 0.1},
                expected_speedup=1.8,
                expected_size_reduction=0.4,
                accuracy_impact=0.005,
                compatibility_score=0.95
            ),
            'conservative_optimization': OptimizationStrategy(
                name='conservative_optimization',
                techniques=['layer_fusion', 'constant_folding'],
                parameters={'fusion_threshold': 0.8},
                expected_speedup=1.3,
                expected_size_reduction=0.15,
                accuracy_impact=0.001,
                compatibility_score=0.99
            )
        }
    
    def select_optimal_strategy(self, profile: OptimizationProfile) -> OptimizationStrategy:
        """Select optimal optimization strategy based on profile and history."""
        architecture_type = self.model_analyzer.analyze_model_architecture({
            'layer_types': ['linear', 'relu', 'conv2d'],  # Mock data
            'param_count': profile.param_count
        })
        
        optimization_potential = self.model_analyzer.estimate_optimization_potential(profile)
        
        # Score strategies based on profile requirements
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            score = 0.0
            
            # Size constraint scoring
            if profile.model_size_mb > profile.memory_constraint_mb:
                size_reduction_needed = 1 - (profile.memory_constraint_mb / profile.model_size_mb)
                if strategy.expected_size_reduction >= size_reduction_needed:
                    score += 3.0
                else:
                    score += strategy.expected_size_reduction / size_reduction_needed
            else:
                score += 2.0  # Baseline for meeting size requirements
            
            # Latency constraint scoring
            predicted_latency = profile.target_latency_ms / strategy.expected_speedup
            if predicted_latency <= profile.target_latency_ms:
                score += 2.0
            else:
                score += 2.0 * (profile.target_latency_ms / predicted_latency)
            
            # Accuracy constraint scoring
            if strategy.accuracy_impact <= (1.0 - profile.accuracy_threshold):
                score += 2.0
            else:
                score += 1.0
            
            # Historical performance bonus
            if name in self.strategy_performance:
                avg_performance = np.mean(self.strategy_performance[name])
                score += avg_performance * 0.5
            
            strategy_scores[name] = score
        
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        return self.strategies[best_strategy_name]
    
    def record_optimization_result(self, profile: OptimizationProfile, 
                                   result: OptimizationResult) -> None:
        """Record optimization result for future learning."""
        performance_score = 0.0
        
        if result.success:
            # Calculate performance score based on multiple factors
            size_score = min(1.0, profile.memory_constraint_mb / result.optimized_size_mb)
            latency_score = min(1.0, profile.target_latency_ms / result.predicted_latency_ms)
            accuracy_score = result.accuracy_retention
            
            performance_score = (size_score + latency_score + accuracy_score) / 3.0
        
        self.strategy_performance[result.strategy_name].append(performance_score)
        
        self.optimization_history.append({
            'profile': profile,
            'result': result,
            'performance_score': performance_score,
            'timestamp': time.time()
        })
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        insights = {
            'total_optimizations': len(self.optimization_history),
            'success_rate': sum(1 for h in self.optimization_history 
                               if h['result'].success) / len(self.optimization_history),
            'strategy_performance': {
                name: {
                    'average_score': np.mean(scores),
                    'success_count': len(scores),
                    'best_score': max(scores) if scores else 0.0
                }
                for name, scores in self.strategy_performance.items()
            }
        }
        
        return insights


class SmartModelOptimizer:
    """Smart model optimizer with ML-powered optimization strategies."""
    
    def __init__(self, enable_learning: bool = True):
        self.enable_learning = enable_learning
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.optimization_queue: List[Tuple[OptimizationProfile, asyncio.Future]] = []
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        logger.info("Smart Model Optimizer initialized")
    
    async def optimize_model(self, profile: OptimizationProfile) -> OptimizationResult:
        """Optimize model using intelligent strategy selection."""
        start_time = time.time()
        
        try:
            # Select optimal strategy
            strategy = self.adaptive_optimizer.select_optimal_strategy(profile)
            logger.info(f"Selected optimization strategy: {strategy.name}")
            
            # Track active optimization
            with self._lock:
                self.active_optimizations[profile.model_id] = {
                    'strategy': strategy.name,
                    'start_time': start_time,
                    'status': 'running'
                }
            
            # Perform optimization (mock implementation)
            result = await self._execute_optimization(profile, strategy)
            
            # Record result for learning
            if self.enable_learning:
                self.adaptive_optimizer.record_optimization_result(profile, result)
            
            # Update tracking
            with self._lock:
                if profile.model_id in self.active_optimizations:
                    self.active_optimizations[profile.model_id]['status'] = 'completed'
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {profile.model_id}: {e}")
            
            result = OptimizationResult(
                strategy_name="failed",
                optimized_size_mb=profile.model_size_mb,
                optimization_time_s=time.time() - start_time,
                predicted_latency_ms=0.0,
                accuracy_retention=0.0,
                success=False,
                error_message=str(e)
            )
            
            # Still record failed attempts for learning
            if self.enable_learning:
                self.adaptive_optimizer.record_optimization_result(profile, result)
            
            return result
        
        finally:
            with self._lock:
                self.active_optimizations.pop(profile.model_id, None)
    
    async def _execute_optimization(self, profile: OptimizationProfile, 
                                    strategy: OptimizationStrategy) -> OptimizationResult:
        """Execute specific optimization strategy."""
        # Mock optimization execution with realistic simulation
        optimization_time = 2.0 + (profile.model_size_mb / 100.0)  # Simulate processing time
        await asyncio.sleep(min(optimization_time, 10.0))  # Cap at 10 seconds for demo
        
        # Calculate optimized metrics based on strategy
        optimized_size = profile.model_size_mb * (1.0 - strategy.expected_size_reduction)
        predicted_latency = profile.target_latency_ms / strategy.expected_speedup
        accuracy_retention = 1.0 - strategy.accuracy_impact
        
        # Add some realistic variance
        variance_factor = np.random.normal(1.0, 0.05)  # 5% variance
        optimized_size *= variance_factor
        predicted_latency *= variance_factor
        accuracy_retention *= np.random.normal(1.0, 0.01)  # 1% accuracy variance
        
        return OptimizationResult(
            strategy_name=strategy.name,
            optimized_size_mb=max(0.1, optimized_size),
            optimization_time_s=optimization_time,
            predicted_latency_ms=max(1.0, predicted_latency),
            accuracy_retention=min(1.0, max(0.0, accuracy_retention)),
            success=True,
            artifacts={
                'optimization_techniques': strategy.techniques,
                'parameters_used': strategy.parameters,
                'compatibility_score': strategy.compatibility_score
            }
        )
    
    def get_active_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active optimizations."""
        with self._lock:
            return self.active_optimizations.copy()
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get optimization insights and recommendations."""
        return self.adaptive_optimizer.get_optimization_insights()
    
    async def batch_optimize(self, profiles: List[OptimizationProfile]) -> List[OptimizationResult]:
        """Optimize multiple models concurrently."""
        tasks = [self.optimize_model(profile) for profile in profiles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = OptimizationResult(
                    strategy_name="error",
                    optimized_size_mb=profiles[i].model_size_mb,
                    optimization_time_s=0.0,
                    predicted_latency_ms=0.0,
                    accuracy_retention=0.0,
                    success=False,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
