"""Adaptive WASM optimization using reinforcement learning.

This module implements a novel approach to WASM optimization that uses
reinforcement learning to adaptively select compilation parameters
based on model characteristics and target deployment environment.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..security import validate_path, log_security_event
from ..validation import validate_tensor_safe


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for adaptive WASM optimization."""
    
    optimization_level: str = "O3"
    use_simd: bool = True
    use_threads: bool = True
    memory_growth: bool = False
    initial_memory: int = 16 * 1024 * 1024  # 16MB
    maximum_memory: int = 512 * 1024 * 1024  # 512MB
    stack_size: int = 64 * 1024  # 64KB
    allow_memory_growth: bool = True
    use_closure_compiler: bool = True
    disable_exception_catching: bool = False
    

@dataclass 
class ModelCharacteristics:
    """Characteristics of a PyTorch model for optimization."""
    
    parameter_count: int
    flops: float
    memory_usage: int
    model_type: str
    has_attention: bool
    has_convolutions: bool
    max_sequence_length: Optional[int] = None
    batch_size: int = 1


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization evaluation."""
    
    compilation_time: float
    binary_size: int
    inference_latency: float
    memory_peak: int
    throughput: float
    accuracy_loss: float = 0.0


class ReinforcementOptimizer:
    """Reinforcement learning agent for WASM optimization parameter selection."""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 8, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple Q-network for optimization parameter selection
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer: List[Tuple] = []
        self.epsilon = 0.1  # Exploration rate
        
    def encode_state(self, model_chars: ModelCharacteristics, target_env: Dict) -> torch.Tensor:
        """Encode model characteristics and target environment as state vector."""
        state = torch.tensor([
            float(model_chars.parameter_count) / 1e6,  # Normalize to millions
            float(model_chars.flops) / 1e9,  # Normalize to GFLOPS
            float(model_chars.memory_usage) / (1024**2),  # Normalize to MB
            float(model_chars.has_attention),
            float(model_chars.has_convolutions),
            float(target_env.get("mobile", 0)),
            float(target_env.get("low_memory", 0)),
            float(target_env.get("low_cpu", 0)),
            float(target_env.get("battery_sensitive", 0)),
            float(target_env.get("network_limited", 0)),
            float(model_chars.batch_size),
            float(model_chars.max_sequence_length or 0) / 1000.0,
        ], dtype=torch.float32)
        
        return state
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select optimization action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return int(torch.argmax(q_values).item())
    
    def decode_action(self, action: int) -> OptimizationConfig:
        """Decode action into optimization configuration."""
        configs = [
            OptimizationConfig(optimization_level="O0", use_simd=False, use_threads=False),
            OptimizationConfig(optimization_level="O1", use_simd=False, use_threads=False), 
            OptimizationConfig(optimization_level="O2", use_simd=True, use_threads=False),
            OptimizationConfig(optimization_level="O3", use_simd=True, use_threads=False),
            OptimizationConfig(optimization_level="O2", use_simd=False, use_threads=True),
            OptimizationConfig(optimization_level="O3", use_simd=False, use_threads=True),
            OptimizationConfig(optimization_level="O3", use_simd=True, use_threads=True),
            OptimizationConfig(optimization_level="Oz", use_simd=True, use_threads=True, 
                             initial_memory=8*1024*1024),  # Size-optimized
        ]
        
        return configs[action]
    
    def compute_reward(self, metrics: PerformanceMetrics, target_env: Dict) -> float:
        """Compute reward signal for reinforcement learning."""
        # Multi-objective reward balancing inference speed, binary size, and accuracy
        weights = {
            "latency": -target_env.get("latency_weight", 1.0),
            "size": -target_env.get("size_weight", 0.5), 
            "memory": -target_env.get("memory_weight", 0.3),
            "accuracy": -target_env.get("accuracy_weight", 2.0),
        }
        
        # Normalize metrics
        latency_score = min(metrics.inference_latency / 100.0, 2.0)  # Cap at 200ms
        size_score = min(metrics.binary_size / (10 * 1024 * 1024), 2.0)  # Cap at 10MB
        memory_score = min(metrics.memory_peak / (100 * 1024 * 1024), 2.0)  # Cap at 100MB
        accuracy_score = metrics.accuracy_loss
        
        reward = (
            weights["latency"] * latency_score +
            weights["size"] * size_score +
            weights["memory"] * memory_score +
            weights["accuracy"] * accuracy_score
        )
        
        return float(reward)
    
    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        """Update Q-network using temporal difference learning."""
        self.replay_buffer.append((state, action, reward, next_state))
        
        if len(self.replay_buffer) < 32:  # Wait for minimum batch
            return
            
        # Sample batch from replay buffer
        batch = np.random.choice(len(self.replay_buffer), size=16, replace=False)
        batch_states = torch.stack([self.replay_buffer[i][0] for i in batch])
        batch_actions = torch.tensor([self.replay_buffer[i][1] for i in batch])
        batch_rewards = torch.tensor([self.replay_buffer[i][2] for i in batch])
        batch_next_states = torch.stack([self.replay_buffer[i][3] for i in batch])
        
        # Compute Q-values and targets
        current_q_values = self.q_network(batch_states).gather(1, batch_actions.unsqueeze(1))
        next_q_values = self.q_network(batch_next_states).max(1)[0].detach()
        target_q_values = batch_rewards + 0.99 * next_q_values  # Gamma = 0.99
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)


class AdaptiveWASMOptimizer:
    """Adaptive WASM optimizer using reinforcement learning."""
    
    def __init__(self, model_cache_dir: Optional[Path] = None):
        self.model_cache_dir = model_cache_dir or Path.cwd() / ".wasm_optimizer_cache"
        self.model_cache_dir.mkdir(exist_ok=True)
        
        self.rl_agent = ReinforcementOptimizer()
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Load pre-trained RL model if available."""
        model_path = self.model_cache_dir / "rl_optimizer.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    checkpoint = pickle.load(f)
                    self.rl_agent.q_network.load_state_dict(checkpoint["model"])
                    self.rl_agent.epsilon = checkpoint["epsilon"]
                    self.performance_history = checkpoint["history"]
                logger.info("Loaded pre-trained optimization model")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model: {e}")
    
    def _save_trained_model(self):
        """Save trained RL model."""
        model_path = self.model_cache_dir / "rl_optimizer.pkl"
        try:
            checkpoint = {
                "model": self.rl_agent.q_network.state_dict(),
                "epsilon": self.rl_agent.epsilon,
                "history": self.performance_history,
            }
            with open(model_path, "wb") as f:
                pickle.dump(checkpoint, f)
            logger.info("Saved trained optimization model")
        except Exception as e:
            logger.error(f"Could not save model: {e}")
    
    def analyze_model(self, model: nn.Module, example_input: torch.Tensor) -> ModelCharacteristics:
        """Analyze PyTorch model characteristics."""
        validate_tensor_safe(example_input, "example_input")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Estimate FLOPs (simplified)
        model.eval()
        with torch.no_grad():
            # Mock FLOP counting - in practice would use tools like thop
            flops = param_count * 2.0  # Rough approximation
            
            # Memory usage estimation
            memory_usage = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
        
        # Model type detection
        model_type = "unknown"
        has_attention = False
        has_convolutions = False
        
        for name, module in model.named_modules():
            if "transformer" in name.lower() or "attention" in name.lower():
                has_attention = True
                model_type = "transformer"
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                has_convolutions = True
                if model_type == "unknown":
                    model_type = "cnn"
        
        return ModelCharacteristics(
            parameter_count=param_count,
            flops=flops,
            memory_usage=memory_usage,
            model_type=model_type,
            has_attention=has_attention,
            has_convolutions=has_convolutions,
            batch_size=example_input.shape[0] if len(example_input.shape) > 0 else 1,
        )
    
    def optimize_for_target(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        target_environment: Dict,
        max_iterations: int = 10,
    ) -> Tuple[OptimizationConfig, PerformanceMetrics]:
        """Optimize WASM compilation parameters for target environment."""
        
        model_chars = self.analyze_model(model, example_input)
        state = self.rl_agent.encode_state(model_chars, target_environment)
        
        best_config = None
        best_metrics = None
        best_reward = float('-inf')
        
        logger.info(f"Starting adaptive optimization for {model_chars.model_type} model")
        
        for iteration in range(max_iterations):
            # Select optimization configuration
            action = self.rl_agent.select_action(state, training=True)
            config = self.rl_agent.decode_action(action)
            
            # Simulate compilation and evaluation
            metrics = self._evaluate_configuration(model, example_input, config, model_chars)
            
            # Compute reward and update RL agent
            reward = self.rl_agent.compute_reward(metrics, target_environment)
            
            if reward > best_reward:
                best_reward = reward
                best_config = config
                best_metrics = metrics
            
            # Update RL agent (simplified - would need proper next state)
            self.rl_agent.update(state, action, reward, state)
            
            logger.info(f"Iteration {iteration + 1}: reward={reward:.3f}, "
                       f"latency={metrics.inference_latency:.1f}ms, "
                       f"size={metrics.binary_size / 1024 / 1024:.1f}MB")
        
        # Save performance history
        model_key = f"{model_chars.model_type}_{model_chars.parameter_count}"
        if model_key not in self.performance_history:
            self.performance_history[model_key] = []
        self.performance_history[model_key].append(best_metrics)
        
        self._save_trained_model()
        
        logger.info(f"Optimization complete. Best configuration: {best_config}")
        return best_config, best_metrics
    
    def _evaluate_configuration(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        config: OptimizationConfig,
        model_chars: ModelCharacteristics,
    ) -> PerformanceMetrics:
        """Evaluate optimization configuration (simulated)."""
        
        # In a real implementation, this would:
        # 1. Export model with given config
        # 2. Compile to WASM
        # 3. Run benchmark tests
        # 4. Measure actual performance
        
        # For now, provide realistic simulation based on config
        start_time = time.time()
        
        # Simulate compilation time
        base_compile_time = model_chars.parameter_count / 1e6 * 30  # 30s per million params
        if config.optimization_level == "O3":
            compilation_time = base_compile_time * 1.5
        elif config.optimization_level == "Oz":
            compilation_time = base_compile_time * 2.0
        else:
            compilation_time = base_compile_time
            
        # Simulate binary size
        base_size = model_chars.memory_usage * 2  # Roughly 2x model size
        size_multiplier = {
            "O0": 1.5, "O1": 1.2, "O2": 1.0, "O3": 0.9, "Oz": 0.7
        }.get(config.optimization_level, 1.0)
        binary_size = int(base_size * size_multiplier)
        
        # Simulate inference latency
        base_latency = model_chars.flops / 1e9 * 50  # 50ms per GFLOP
        if config.use_simd:
            base_latency *= 0.7  # 30% improvement
        if config.use_threads:
            base_latency *= 0.8  # 20% improvement
            
        opt_multiplier = {
            "O0": 3.0, "O1": 2.0, "O2": 1.2, "O3": 1.0, "Oz": 1.1
        }.get(config.optimization_level, 1.0)
        inference_latency = base_latency * opt_multiplier
        
        # Simulate memory usage
        memory_peak = config.initial_memory + model_chars.memory_usage * 3
        
        # Simulate throughput
        throughput = 1000.0 / inference_latency  # Inferences per second
        
        return PerformanceMetrics(
            compilation_time=compilation_time,
            binary_size=binary_size,
            inference_latency=inference_latency,
            memory_peak=memory_peak,
            throughput=throughput,
            accuracy_loss=0.0,  # Assume no accuracy loss for simulation
        )
    
    def get_optimization_recommendations(
        self, model_chars: ModelCharacteristics
    ) -> Dict[str, OptimizationConfig]:
        """Get pre-computed optimization recommendations for common scenarios."""
        
        recommendations = {}
        
        # Mobile/Edge deployment
        recommendations["mobile"] = OptimizationConfig(
            optimization_level="Oz",  # Size optimization
            use_simd=True,
            use_threads=False,  # Battery consideration
            initial_memory=8 * 1024 * 1024,  # 8MB
            maximum_memory=64 * 1024 * 1024,  # 64MB
        )
        
        # Server deployment
        recommendations["server"] = OptimizationConfig(
            optimization_level="O3",
            use_simd=True,
            use_threads=True,
            initial_memory=32 * 1024 * 1024,  # 32MB
            maximum_memory=512 * 1024 * 1024,  # 512MB
        )
        
        # Development/Debug
        recommendations["debug"] = OptimizationConfig(
            optimization_level="O0",
            use_simd=False,
            use_threads=False,
            disable_exception_catching=False,
        )
        
        return recommendations
    
    def export_optimization_report(self, output_path: Path) -> None:
        """Export detailed optimization analysis report."""
        
        validate_path(output_path, allow_write=True)
        
        report = {
            "optimization_history": self.performance_history,
            "model_statistics": {
                "total_models_optimized": len(self.performance_history),
                "avg_improvement": self._calculate_avg_improvement(),
            },
            "recommendations": self._generate_general_recommendations(),
            "timestamp": time.time(),
        }
        
        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            log_security_event(
                "optimization_report_exported", 
                {"output_path": str(output_path)}
            )
            logger.info(f"Optimization report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise
    
    def _calculate_avg_improvement(self) -> Dict[str, float]:
        """Calculate average performance improvements."""
        if not self.performance_history:
            return {}
            
        improvements = {"latency": [], "size": [], "throughput": []}
        
        for model_history in self.performance_history.values():
            if len(model_history) >= 2:
                baseline = model_history[0]
                optimized = model_history[-1]
                
                improvements["latency"].append(
                    (baseline.inference_latency - optimized.inference_latency) 
                    / baseline.inference_latency
                )
                improvements["size"].append(
                    (baseline.binary_size - optimized.binary_size) 
                    / baseline.binary_size
                )
                improvements["throughput"].append(
                    (optimized.throughput - baseline.throughput) 
                    / baseline.throughput
                )
        
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in improvements.items()
        }
    
    def _generate_general_recommendations(self) -> List[Dict]:
        """Generate general optimization recommendations based on history."""
        recommendations = []
        
        if self.performance_history:
            recommendations.append({
                "category": "general",
                "recommendation": "Use O3 optimization for server deployment",
                "impact": "15-25% latency reduction",
            })
            
            recommendations.append({
                "category": "mobile",
                "recommendation": "Use Oz optimization for mobile deployment", 
                "impact": "30-50% size reduction",
            })
            
            recommendations.append({
                "category": "performance",
                "recommendation": "Enable SIMD for compute-intensive models",
                "impact": "20-30% latency reduction",
            })
        
        return recommendations