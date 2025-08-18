"""Neural Architecture Search (NAS) for WASM-Torch

Autonomous neural architecture search and optimization system that continuously
evolves model architectures for optimal WASM performance.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

class ArchitectureType(Enum):
    """Neural architecture types optimized for WASM"""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional" 
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    OPTIMIZED_SPARSE = "optimized_sparse"

class OptimizationObjective(Enum):
    """Optimization objectives for NAS"""
    INFERENCE_SPEED = "inference_speed"
    MODEL_SIZE = "model_size"
    MEMORY_USAGE = "memory_usage"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ACCURACY = "accuracy"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class LayerSpec:
    """Specification for a neural network layer"""
    layer_type: str
    input_size: int
    output_size: int
    activation: str = "relu"
    parameters: Dict[str, Any] = field(default_factory=dict)
    wasm_optimized: bool = True

@dataclass
class ArchitectureSpec:
    """Complete neural architecture specification"""
    layers: List[LayerSpec]
    architecture_type: ArchitectureType
    total_parameters: int = 0
    estimated_inference_time: float = 0.0
    estimated_memory_usage: float = 0.0
    wasm_compatibility_score: float = 1.0
    
    def __post_init__(self):
        self.total_parameters = sum(
            layer.input_size * layer.output_size + layer.output_size 
            for layer in self.layers
        )

@dataclass
class PerformanceMetrics:
    """Performance metrics for architecture evaluation"""
    inference_time: float
    memory_usage: float
    model_size: float
    accuracy: float
    energy_consumption: float
    wasm_optimization_score: float
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate composite performance score"""
        if weights is None:
            weights = {
                "inference_time": 0.3,
                "memory_usage": 0.2,
                "model_size": 0.2,
                "accuracy": 0.2,
                "wasm_optimization_score": 0.1
            }
        
        # Normalize metrics (lower is better for most, higher for accuracy)
        normalized_inference = 1.0 / (self.inference_time + 1e-6)
        normalized_memory = 1.0 / (self.memory_usage + 1e-6)
        normalized_size = 1.0 / (self.model_size + 1e-6)
        normalized_accuracy = self.accuracy
        normalized_wasm = self.wasm_optimization_score
        
        score = (
            weights["inference_time"] * normalized_inference +
            weights["memory_usage"] * normalized_memory +
            weights["model_size"] * normalized_size +
            weights["accuracy"] * normalized_accuracy +
            weights["wasm_optimization_score"] * normalized_wasm
        )
        
        return score

class ArchitectureGenerator(ABC):
    """Base class for architecture generators"""
    
    @abstractmethod
    def generate_architecture(self, 
                            input_shape: Tuple[int, ...], 
                            output_shape: Tuple[int, ...],
                            constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Generate a neural architecture"""
        pass

class EvolutionaryGenerator(ArchitectureGenerator):
    """Evolutionary algorithm for architecture generation"""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[ArchitectureSpec] = []
        self.generation = 0
        
    def generate_architecture(self, 
                            input_shape: Tuple[int, ...], 
                            output_shape: Tuple[int, ...],
                            constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Generate architecture using evolutionary algorithm"""
        if not self.population:
            self._initialize_population(input_shape, output_shape, constraints)
        
        # Select parent architectures
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        # Crossover
        if random.random() < self.crossover_rate:
            offspring = self._crossover(parent1, parent2)
        else:
            offspring = random.choice([parent1, parent2])
        
        # Mutation
        if random.random() < self.mutation_rate:
            offspring = self._mutate(offspring, constraints)
        
        return offspring
    
    def _initialize_population(self, 
                             input_shape: Tuple[int, ...], 
                             output_shape: Tuple[int, ...],
                             constraints: Dict[str, Any]):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.population_size):
            arch = self._create_random_architecture(input_shape, output_shape, constraints)
            self.population.append(arch)
    
    def _create_random_architecture(self, 
                                  input_shape: Tuple[int, ...], 
                                  output_shape: Tuple[int, ...],
                                  constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Create random architecture within constraints"""
        layers = []
        current_size = input_shape[0] if input_shape else 784
        
        # Random number of hidden layers
        num_layers = random.randint(2, constraints.get("max_layers", 8))
        
        # Hidden layers
        for i in range(num_layers - 1):
            layer_size = random.choice([64, 128, 256, 512, 1024])
            layer_size = min(layer_size, constraints.get("max_layer_size", 1024))
            
            activation = random.choice(["relu", "tanh", "sigmoid", "gelu"])
            
            layers.append(LayerSpec(
                layer_type="linear",
                input_size=current_size,
                output_size=layer_size,
                activation=activation,
                wasm_optimized=True
            ))
            current_size = layer_size
        
        # Output layer
        output_size = output_shape[0] if output_shape else 10
        layers.append(LayerSpec(
            layer_type="linear",
            input_size=current_size,
            output_size=output_size,
            activation="linear",
            wasm_optimized=True
        ))
        
        arch_type = random.choice(list(ArchitectureType))
        
        return ArchitectureSpec(
            layers=layers,
            architecture_type=arch_type
        )
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureSpec:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        # For now, return random selection (would use fitness in real implementation)
        return random.choice(tournament)
    
    def _crossover(self, parent1: ArchitectureSpec, parent2: ArchitectureSpec) -> ArchitectureSpec:
        """Crossover two parent architectures"""
        # Simple layer-wise crossover
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_layers - 1)
        
        new_layers = (
            parent1.layers[:crossover_point] + 
            parent2.layers[crossover_point:min_layers]
        )
        
        return ArchitectureSpec(
            layers=new_layers,
            architecture_type=random.choice([parent1.architecture_type, parent2.architecture_type])
        )
    
    def _mutate(self, architecture: ArchitectureSpec, constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Mutate architecture"""
        new_layers = []
        
        for layer in architecture.layers:
            if random.random() < 0.3:  # 30% chance to mutate each layer
                # Mutate layer size
                size_multiplier = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
                new_output_size = int(layer.output_size * size_multiplier)
                new_output_size = max(1, min(new_output_size, constraints.get("max_layer_size", 1024)))
                
                # Mutate activation
                new_activation = random.choice(["relu", "tanh", "sigmoid", "gelu"])
                
                new_layer = LayerSpec(
                    layer_type=layer.layer_type,
                    input_size=layer.input_size,
                    output_size=new_output_size,
                    activation=new_activation,
                    wasm_optimized=layer.wasm_optimized
                )
            else:
                new_layer = layer
            
            new_layers.append(new_layer)
        
        return ArchitectureSpec(
            layers=new_layers,
            architecture_type=architecture.architecture_type
        )

class WASMOptimizedGenerator(ArchitectureGenerator):
    """Generator specialized for WASM-optimized architectures"""
    
    def __init__(self):
        self.wasm_preferred_sizes = [64, 128, 256, 512]  # SIMD-friendly sizes
        self.wasm_preferred_activations = ["relu", "gelu"]  # Fast on WASM
        
    def generate_architecture(self, 
                            input_shape: Tuple[int, ...], 
                            output_shape: Tuple[int, ...],
                            constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Generate WASM-optimized architecture"""
        layers = []
        current_size = input_shape[0] if input_shape else 784
        
        # Determine optimal layer count for WASM
        max_layers = constraints.get("max_layers", 6)
        optimal_layers = min(4, max_layers)  # WASM prefers fewer, wider layers
        
        # Create layers with WASM-friendly sizes
        for i in range(optimal_layers - 1):
            # Choose SIMD-friendly layer size
            layer_size = self._choose_wasm_friendly_size(
                current_size, 
                constraints.get("max_layer_size", 512)
            )
            
            # WASM-optimized activation
            activation = random.choice(self.wasm_preferred_activations)
            
            layers.append(LayerSpec(
                layer_type="linear",
                input_size=current_size,
                output_size=layer_size,
                activation=activation,
                parameters={"wasm_simd_optimized": True},
                wasm_optimized=True
            ))
            current_size = layer_size
        
        # Output layer
        output_size = output_shape[0] if output_shape else 10
        layers.append(LayerSpec(
            layer_type="linear",
            input_size=current_size,
            output_size=output_size,
            activation="linear",
            parameters={"wasm_simd_optimized": True},
            wasm_optimized=True
        ))
        
        architecture = ArchitectureSpec(
            layers=layers,
            architecture_type=ArchitectureType.OPTIMIZED_SPARSE
        )
        
        # Calculate WASM compatibility score
        architecture.wasm_compatibility_score = self._calculate_wasm_score(architecture)
        
        return architecture
    
    def _choose_wasm_friendly_size(self, current_size: int, max_size: int) -> int:
        """Choose SIMD-friendly layer size"""
        # Prefer sizes that are multiples of 64 (SIMD vector size)
        candidates = [size for size in self.wasm_preferred_sizes if size <= max_size]
        
        if not candidates:
            candidates = [64]  # Fallback
        
        return random.choice(candidates)
    
    def _calculate_wasm_score(self, architecture: ArchitectureSpec) -> float:
        """Calculate WASM compatibility score"""
        score = 1.0
        
        # Prefer fewer layers
        if len(architecture.layers) <= 4:
            score += 0.2
        elif len(architecture.layers) <= 6:
            score += 0.1
        
        # Prefer SIMD-friendly sizes
        simd_friendly_layers = sum(
            1 for layer in architecture.layers 
            if layer.output_size in self.wasm_preferred_sizes
        )
        score += 0.3 * (simd_friendly_layers / len(architecture.layers))
        
        # Prefer WASM-optimized activations
        wasm_activations = sum(
            1 for layer in architecture.layers
            if layer.activation in self.wasm_preferred_activations
        )
        score += 0.2 * (wasm_activations / len(architecture.layers))
        
        return min(score, 2.0)  # Cap at 2.0

class ArchitectureEvaluator:
    """Evaluates architecture performance"""
    
    def __init__(self, evaluation_timeout: float = 30.0):
        self.evaluation_timeout = evaluation_timeout
        self.evaluation_cache: Dict[str, PerformanceMetrics] = {}
        
    async def evaluate_architecture(self, 
                                   architecture: ArchitectureSpec,
                                   test_data: Optional[np.ndarray] = None) -> PerformanceMetrics:
        """Evaluate architecture performance"""
        # Check cache
        arch_hash = self._hash_architecture(architecture)
        if arch_hash in self.evaluation_cache:
            return self.evaluation_cache[arch_hash]
        
        # Simulate evaluation (in real implementation, would train and test)
        start_time = time.time()
        
        # Estimate performance based on architecture characteristics
        metrics = await self._estimate_performance(architecture)
        
        # Cache result
        self.evaluation_cache[arch_hash] = metrics
        
        return metrics
    
    async def _estimate_performance(self, architecture: ArchitectureSpec) -> PerformanceMetrics:
        """Estimate performance metrics"""
        # Simulate evaluation time
        await asyncio.sleep(0.1)
        
        # Calculate estimated metrics
        total_params = architecture.total_parameters
        num_layers = len(architecture.layers)
        
        # Inference time (ms) - based on parameters and layers
        base_inference_time = 5.0 + (total_params / 100000) * 10.0 + num_layers * 2.0
        wasm_speedup = architecture.wasm_compatibility_score
        inference_time = base_inference_time / wasm_speedup
        
        # Memory usage (MB) - based on parameters
        memory_usage = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        memory_usage *= (2.0 - architecture.wasm_compatibility_score)  # WASM optimization reduces memory
        
        # Model size (MB) - similar to memory but compressed
        model_size = memory_usage * 0.7
        
        # Accuracy - simulated based on architecture complexity
        complexity_score = min(1.0, total_params / 1000000)  # Normalize by 1M params
        layer_diversity = len(set(layer.activation for layer in architecture.layers)) / 4.0
        accuracy = 0.6 + 0.3 * complexity_score + 0.1 * layer_diversity
        accuracy = min(0.95, accuracy)  # Cap at 95%
        
        # Energy consumption - inverse of WASM optimization
        energy_consumption = inference_time * (2.0 - architecture.wasm_compatibility_score)
        
        return PerformanceMetrics(
            inference_time=inference_time,
            memory_usage=memory_usage,
            model_size=model_size,
            accuracy=accuracy,
            energy_consumption=energy_consumption,
            wasm_optimization_score=architecture.wasm_compatibility_score
        )
    
    def _hash_architecture(self, architecture: ArchitectureSpec) -> str:
        """Generate hash for architecture caching"""
        arch_str = f"{architecture.architecture_type.value}_"
        arch_str += "_".join([
            f"{layer.layer_type}_{layer.input_size}_{layer.output_size}_{layer.activation}"
            for layer in architecture.layers
        ])
        return str(hash(arch_str))

class NeuralArchitectureSearchEngine:
    """Main NAS engine for WASM-Torch"""
    
    def __init__(self, 
                 generators: Optional[List[ArchitectureGenerator]] = None,
                 evaluator: Optional[ArchitectureEvaluator] = None,
                 search_budget: int = 100):
        self.generators = generators or [
            EvolutionaryGenerator(),
            WASMOptimizedGenerator()
        ]
        self.evaluator = evaluator or ArchitectureEvaluator()
        self.search_budget = search_budget
        
        # Search state
        self.search_history: List[Dict[str, Any]] = []
        self.best_architectures: List[Tuple[ArchitectureSpec, PerformanceMetrics]] = []
        self.search_iteration = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def search_optimal_architecture(self, 
                                        input_shape: Tuple[int, ...],
                                        output_shape: Tuple[int, ...],
                                        optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE,
                                        constraints: Optional[Dict[str, Any]] = None) -> Tuple[ArchitectureSpec, PerformanceMetrics]:
        """Search for optimal neural architecture"""
        constraints = constraints or {}
        search_start_time = time.time()
        
        self.logger.info(f"🔍 Starting NAS with budget {self.search_budget}")
        
        # Initialize search
        best_architecture = None
        best_metrics = None
        best_score = -float('inf')
        
        for iteration in range(self.search_budget):
            self.search_iteration = iteration
            
            # Generate candidate architecture
            generator = random.choice(self.generators)
            candidate = generator.generate_architecture(input_shape, output_shape, constraints)
            
            # Evaluate candidate
            try:
                metrics = await self.evaluator.evaluate_architecture(candidate)
                
                # Calculate score based on objective
                score = self._calculate_objective_score(metrics, optimization_objective)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_architecture = candidate
                    best_metrics = metrics
                    
                    self.logger.info(f"🚀 New best architecture found: score={score:.3f}")
                
                # Record search step
                self._record_search_step(candidate, metrics, score)
                
                # Update generator fitness (for evolutionary)
                if isinstance(generator, EvolutionaryGenerator):
                    await self._update_evolutionary_population(generator, candidate, metrics)
                
            except Exception as e:
                self.logger.error(f"Architecture evaluation failed: {e}")
                continue
            
            # Progress logging
            if (iteration + 1) % 10 == 0:
                elapsed = time.time() - search_start_time
                self.logger.info(f"NAS progress: {iteration + 1}/{self.search_budget}, "
                               f"best_score={best_score:.3f}, elapsed={elapsed:.1f}s")
        
        # Finalize search
        search_time = time.time() - search_start_time
        self.logger.info(f"✅ NAS completed in {search_time:.1f}s, best score: {best_score:.3f}")
        
        if best_architecture is None:
            raise RuntimeError("No valid architecture found during search")
        
        return best_architecture, best_metrics
    
    def _calculate_objective_score(self, 
                                  metrics: PerformanceMetrics, 
                                  objective: OptimizationObjective) -> float:
        """Calculate score based on optimization objective"""
        if objective == OptimizationObjective.INFERENCE_SPEED:
            return 1000.0 / (metrics.inference_time + 1e-6)
        elif objective == OptimizationObjective.MODEL_SIZE:
            return 100.0 / (metrics.model_size + 1e-6)
        elif objective == OptimizationObjective.MEMORY_USAGE:
            return 100.0 / (metrics.memory_usage + 1e-6)
        elif objective == OptimizationObjective.ENERGY_EFFICIENCY:
            return 100.0 / (metrics.energy_consumption + 1e-6)
        elif objective == OptimizationObjective.ACCURACY:
            return metrics.accuracy * 100.0
        else:  # MULTI_OBJECTIVE
            return metrics.get_composite_score()
    
    def _record_search_step(self, 
                           architecture: ArchitectureSpec, 
                           metrics: PerformanceMetrics,
                           score: float):
        """Record search step for analysis"""
        with self._lock:
            step_record = {
                "iteration": self.search_iteration,
                "timestamp": time.time(),
                "architecture_type": architecture.architecture_type.value,
                "total_parameters": architecture.total_parameters,
                "num_layers": len(architecture.layers),
                "wasm_compatibility": architecture.wasm_compatibility_score,
                "metrics": {
                    "inference_time": metrics.inference_time,
                    "memory_usage": metrics.memory_usage,
                    "model_size": metrics.model_size,
                    "accuracy": metrics.accuracy,
                    "wasm_optimization_score": metrics.wasm_optimization_score
                },
                "objective_score": score
            }
            
            self.search_history.append(step_record)
            
            # Keep best architectures
            self.best_architectures.append((architecture, metrics))
            self.best_architectures.sort(key=lambda x: self._calculate_objective_score(x[1], OptimizationObjective.MULTI_OBJECTIVE), reverse=True)
            self.best_architectures = self.best_architectures[:10]  # Keep top 10
    
    async def _update_evolutionary_population(self, 
                                            generator: EvolutionaryGenerator,
                                            architecture: ArchitectureSpec,
                                            metrics: PerformanceMetrics):
        """Update evolutionary generator population"""
        # Replace worst individual with new architecture if better
        if len(generator.population) >= generator.population_size:
            # Simple replacement strategy
            generator.population[-1] = architecture
    
    def get_search_report(self) -> Dict[str, Any]:
        """Get comprehensive search report"""
        with self._lock:
            if not self.search_history:
                return {"status": "no_search_performed"}
            
            scores = [step["objective_score"] for step in self.search_history]
            best_architectures_summary = []
            
            for arch, metrics in self.best_architectures[:5]:  # Top 5
                best_architectures_summary.append({
                    "architecture_type": arch.architecture_type.value,
                    "total_parameters": arch.total_parameters,
                    "layers": len(arch.layers),
                    "wasm_score": arch.wasm_compatibility_score,
                    "inference_time": metrics.inference_time,
                    "accuracy": metrics.accuracy,
                    "composite_score": metrics.get_composite_score()
                })
            
            return {
                "search_iterations": len(self.search_history),
                "best_score": max(scores) if scores else 0.0,
                "score_improvement": max(scores) - scores[0] if len(scores) > 0 else 0.0,
                "convergence_rate": self._calculate_convergence_rate(),
                "architecture_type_distribution": self._get_architecture_type_distribution(),
                "best_architectures": best_architectures_summary,
                "search_efficiency": len(self.best_architectures) / len(self.search_history) if self.search_history else 0.0
            }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate search convergence rate"""
        if len(self.search_history) < 10:
            return 0.0
        
        recent_scores = [step["objective_score"] for step in self.search_history[-10:]]
        early_scores = [step["objective_score"] for step in self.search_history[:10]]
        
        return (np.mean(recent_scores) - np.mean(early_scores)) / max(np.mean(early_scores), 1e-6)
    
    def _get_architecture_type_distribution(self) -> Dict[str, int]:
        """Get distribution of architecture types tried"""
        distribution = {}
        for step in self.search_history:
            arch_type = step["architecture_type"]
            distribution[arch_type] = distribution.get(arch_type, 0) + 1
        return distribution

# Global NAS engine
_global_nas_engine: Optional[NeuralArchitectureSearchEngine] = None

def get_nas_engine() -> NeuralArchitectureSearchEngine:
    """Get global NAS engine"""
    global _global_nas_engine
    if _global_nas_engine is None:
        _global_nas_engine = NeuralArchitectureSearchEngine()
    return _global_nas_engine

async def search_optimal_wasm_architecture(input_shape: Tuple[int, ...],
                                         output_shape: Tuple[int, ...],
                                         optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE,
                                         search_budget: int = 50) -> Tuple[ArchitectureSpec, PerformanceMetrics]:
    """Search for optimal WASM architecture"""
    engine = get_nas_engine()
    engine.search_budget = search_budget
    
    return await engine.search_optimal_architecture(
        input_shape=input_shape,
        output_shape=output_shape,
        optimization_objective=optimization_objective,
        constraints={
            "max_layers": 6,
            "max_layer_size": 512,
            "prefer_wasm_optimized": True
        }
    )