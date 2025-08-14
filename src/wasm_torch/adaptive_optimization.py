"""Adaptive optimization system for WASM Torch runtime performance."""

import asyncio
import time
import threading
import statistics
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics used for adaptive optimization decisions."""
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    batch_efficiency: float = 0.0
    optimization_score: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for adaptive optimization."""
    batch_size: int = 1
    thread_count: int = 1
    cache_size: int = 128
    memory_pool_size: int = 10
    optimization_level: str = "O2"
    use_simd: bool = True
    prefetch_enabled: bool = True
    adaptive_batching: bool = True
    target_latency_ms: float = 100.0
    target_throughput: float = 10.0


class AdaptiveOptimizer:
    """ML-powered adaptive optimization system."""
    
    def __init__(self, learning_rate: float = 0.1, history_size: int = 100):
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.config_history: deque = deque(maxlen=history_size)
        self.current_config = OptimizationConfig()
        self.optimization_weights = {
            'latency': 0.4,
            'throughput': 0.3,
            'memory': 0.2,
            'error_rate': 0.1
        }
        self.learning_enabled = True
        self.lock = threading.Lock()
        
    def record_metrics(self, metrics: OptimizationMetrics, config: OptimizationConfig) -> None:
        """Record performance metrics with current configuration."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.config_history.append(config)
            
            if self.learning_enabled and len(self.metrics_history) >= 10:
                self._update_optimization_strategy()
    
    def _calculate_optimization_score(self, metrics: OptimizationMetrics, 
                                    target_config: OptimizationConfig) -> float:
        """Calculate optimization score based on current metrics."""
        score = 0.0
        
        # Latency score (lower is better)
        latency_target = target_config.target_latency_ms
        latency_score = max(0, 1 - (metrics.latency_p95 / latency_target))
        score += self.optimization_weights['latency'] * latency_score
        
        # Throughput score (higher is better)
        throughput_target = target_config.target_throughput
        throughput_score = min(1, metrics.throughput / throughput_target)
        score += self.optimization_weights['throughput'] * throughput_score
        
        # Memory efficiency score (lower usage is better)
        memory_score = max(0, 1 - (metrics.memory_usage / 100))
        score += self.optimization_weights['memory'] * memory_score
        
        # Error rate score (lower is better)
        error_score = max(0, 1 - metrics.error_rate)
        score += self.optimization_weights['error_rate'] * error_score
        
        return score
    
    def _update_optimization_strategy(self) -> None:
        """Update optimization strategy based on historical performance."""
        if len(self.metrics_history) < 5:
            return
        
        recent_metrics = list(self.metrics_history)[-5:]
        recent_configs = list(self.config_history)[-5:]
        
        # Calculate average performance for recent configurations
        best_score = -1
        best_config = None
        
        for metrics, config in zip(recent_metrics, recent_configs):
            score = self._calculate_optimization_score(metrics, config)
            if score > best_score:
                best_score = score
                best_config = config
        
        if best_config and best_score > 0.7:  # Only apply if significantly better
            logger.info(f"Adopting better configuration with score {best_score:.3f}")
            self._apply_config_gradual(best_config)
    
    def _apply_config_gradual(self, target_config: OptimizationConfig) -> None:
        """Gradually apply new configuration to avoid sudden performance drops."""
        # Gradually adjust batch size
        if target_config.batch_size != self.current_config.batch_size:
            diff = target_config.batch_size - self.current_config.batch_size
            step = max(1, abs(diff) // 4)  # Adjust in steps
            if diff > 0:
                self.current_config.batch_size = min(
                    target_config.batch_size,
                    self.current_config.batch_size + step
                )
            else:
                self.current_config.batch_size = max(
                    target_config.batch_size,
                    self.current_config.batch_size - step
                )
        
        # Apply other configuration changes
        self.current_config.thread_count = target_config.thread_count
        self.current_config.cache_size = target_config.cache_size
        self.current_config.prefetch_enabled = target_config.prefetch_enabled
    
    def suggest_optimization(self, current_metrics: OptimizationMetrics) -> OptimizationConfig:
        """Suggest optimization based on current performance."""
        suggested_config = OptimizationConfig()
        
        # Adaptive batch sizing based on latency
        if current_metrics.latency_p95 > suggested_config.target_latency_ms:
            # High latency - reduce batch size
            suggested_config.batch_size = max(1, self.current_config.batch_size - 1)
        elif current_metrics.latency_p95 < suggested_config.target_latency_ms * 0.5:
            # Low latency - increase batch size
            suggested_config.batch_size = min(32, self.current_config.batch_size + 1)
        else:
            suggested_config.batch_size = self.current_config.batch_size
        
        # Thread count optimization
        if current_metrics.cpu_usage < 50 and current_metrics.throughput < suggested_config.target_throughput:
            suggested_config.thread_count = min(8, self.current_config.thread_count + 1)
        elif current_metrics.cpu_usage > 90:
            suggested_config.thread_count = max(1, self.current_config.thread_count - 1)
        else:
            suggested_config.thread_count = self.current_config.thread_count
        
        # Cache optimization
        if current_metrics.cache_hit_rate < 0.7:
            suggested_config.cache_size = min(512, self.current_config.cache_size * 2)
        elif current_metrics.cache_hit_rate > 0.95 and current_metrics.memory_usage > 80:
            suggested_config.cache_size = max(64, self.current_config.cache_size // 2)
        else:
            suggested_config.cache_size = self.current_config.cache_size
        
        # Memory pool optimization
        if current_metrics.memory_usage > 80:
            suggested_config.memory_pool_size = max(5, self.current_config.memory_pool_size - 2)
        elif current_metrics.memory_usage < 40:
            suggested_config.memory_pool_size = min(50, self.current_config.memory_pool_size + 2)
        else:
            suggested_config.memory_pool_size = self.current_config.memory_pool_size
        
        return suggested_config


class AutoScaler:
    """Auto-scaling system for WASM Torch workloads."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.target_cpu_utilization = 70.0
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 30.0
        self.cooldown_period = 60.0  # seconds
        self.last_scale_time = 0
        self.metrics_window = deque(maxlen=20)  # 20 data points for decisions
        
    def should_scale(self, metrics: OptimizationMetrics) -> Tuple[bool, int]:
        """Determine if scaling is needed and return target instance count."""
        current_time = time.time()
        
        # Add current metrics to window
        self.metrics_window.append(metrics)
        
        # Don't scale during cooldown period
        if current_time - self.last_scale_time < self.cooldown_period:
            return False, self.current_instances
        
        # Need sufficient data points for decision
        if len(self.metrics_window) < 5:
            return False, self.current_instances
        
        # Calculate average metrics over window
        avg_cpu = statistics.mean(m.cpu_usage for m in self.metrics_window)
        avg_latency = statistics.mean(m.latency_p95 for m in self.metrics_window)
        avg_error_rate = statistics.mean(m.error_rate for m in self.metrics_window)
        
        # Scale up conditions
        scale_up = (
            avg_cpu > self.scale_up_threshold or
            avg_latency > 200.0 or  # 200ms threshold
            avg_error_rate > 0.05   # 5% error rate
        )
        
        # Scale down conditions
        scale_down = (
            avg_cpu < self.scale_down_threshold and
            avg_latency < 50.0 and  # Low latency
            avg_error_rate < 0.01   # Low error rate
        )
        
        if scale_up and self.current_instances < self.max_instances:
            target_instances = min(self.max_instances, self.current_instances + 1)
            logger.info(f"Scaling up: {self.current_instances} -> {target_instances}")
            return True, target_instances
        
        elif scale_down and self.current_instances > self.min_instances:
            target_instances = max(self.min_instances, self.current_instances - 1)
            logger.info(f"Scaling down: {self.current_instances} -> {target_instances}")
            return True, target_instances
        
        return False, self.current_instances
    
    def apply_scaling(self, target_instances: int) -> None:
        """Apply scaling decision."""
        self.current_instances = target_instances
        self.last_scale_time = time.time()


class LoadBalancer:
    """Intelligent load balancing for distributed WASM Torch instances."""
    
    def __init__(self):
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.round_robin_index = 0
        self.load_balancing_strategy = "least_connections"  # "round_robin", "least_connections", "weighted"
        self.health_check_interval = 30.0
        self.last_health_check = 0
        
    def register_instance(self, instance_id: str, endpoint: str, weight: float = 1.0) -> None:
        """Register a new instance for load balancing."""
        self.instances[instance_id] = {
            "endpoint": endpoint,
            "weight": weight,
            "active_connections": 0,
            "total_requests": 0,
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "last_health_check": time.time(),
            "healthy": True
        }
        logger.info(f"Registered instance {instance_id} at {endpoint}")
    
    def unregister_instance(self, instance_id: str) -> None:
        """Unregister an instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"Unregistered instance {instance_id}")
    
    def select_instance(self) -> Optional[str]:
        """Select the best instance for the next request."""
        healthy_instances = {
            k: v for k, v in self.instances.items() if v["healthy"]
        }
        
        if not healthy_instances:
            logger.warning("No healthy instances available")
            return None
        
        if self.load_balancing_strategy == "round_robin":
            return self._round_robin_selection(healthy_instances)
        elif self.load_balancing_strategy == "least_connections":
            return self._least_connections_selection(healthy_instances)
        elif self.load_balancing_strategy == "weighted":
            return self._weighted_selection(healthy_instances)
        
        return next(iter(healthy_instances.keys()))
    
    def _round_robin_selection(self, instances: Dict[str, Any]) -> str:
        """Round-robin load balancing."""
        instance_list = list(instances.keys())
        selected = instance_list[self.round_robin_index % len(instance_list)]
        self.round_robin_index = (self.round_robin_index + 1) % len(instance_list)
        return selected
    
    def _least_connections_selection(self, instances: Dict[str, Any]) -> str:
        """Select instance with least active connections."""
        return min(instances.keys(), 
                  key=lambda x: instances[x]["active_connections"])
    
    def _weighted_selection(self, instances: Dict[str, Any]) -> str:
        """Weighted selection based on performance metrics."""
        scores = {}
        for instance_id, info in instances.items():
            # Calculate score based on success rate, response time, and connections
            success_score = info["success_rate"]
            speed_score = max(0, 1 - (info["avg_response_time"] / 1000))  # Normalize to 1s
            load_score = max(0, 1 - (info["active_connections"] / 100))  # Normalize to 100 connections
            
            scores[instance_id] = (success_score * 0.4 + speed_score * 0.4 + load_score * 0.2) * info["weight"]
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def update_instance_metrics(self, instance_id: str, response_time: float, 
                              success: bool) -> None:
        """Update instance performance metrics."""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        instance["total_requests"] += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        instance["success_rate"] = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * instance["success_rate"]
        )
        
        # Update average response time (exponential moving average)
        instance["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * instance["avg_response_time"]
        )
    
    async def health_check(self) -> None:
        """Perform health checks on all instances."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        for instance_id, info in self.instances.items():
            try:
                # Simulate health check - in production, this would be an HTTP request
                # For now, mark as healthy if success rate is above threshold
                info["healthy"] = info["success_rate"] > 0.5
                info["last_health_check"] = current_time
                
            except Exception as e:
                logger.warning(f"Health check failed for {instance_id}: {e}")
                info["healthy"] = False
        
        self.last_health_check = current_time


class ConcurrencyManager:
    """Advanced concurrency management for optimal resource utilization."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_tasks = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue = asyncio.Queue()
        self.worker_count = 4
        self.workers_started = False
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "queue_size": 0,
            "avg_wait_time": 0.0
        }
        
    async def start_workers(self) -> None:
        """Start background worker tasks."""
        if self.workers_started:
            return
        
        for i in range(self.worker_count):
            asyncio.create_task(self._worker(f"worker-{i}"))
        
        self.workers_started = True
        logger.info(f"Started {self.worker_count} concurrency workers")
    
    async def _worker(self, worker_id: str) -> None:
        """Background worker for processing tasks."""
        while True:
            try:
                task_data = await self.task_queue.get()
                if task_data is None:  # Shutdown signal
                    break
                
                func, args, kwargs, future = task_data
                
                try:
                    async with self.semaphore:
                        self.active_tasks += 1
                        result = await func(*args, **kwargs)
                        future.set_result(result)
                        self.metrics["completed_tasks"] += 1
                
                except Exception as e:
                    future.set_exception(e)
                    self.metrics["failed_tasks"] += 1
                    logger.error(f"Task failed in {worker_id}: {e}")
                
                finally:
                    self.active_tasks -= 1
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task for concurrent execution."""
        await self.start_workers()
        
        future = asyncio.Future()
        task_data = (func, args, kwargs, future)
        
        self.metrics["total_tasks"] += 1
        self.metrics["queue_size"] = self.task_queue.qsize()
        
        await self.task_queue.put(task_data)
        return await future
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics."""
        total = self.metrics["total_tasks"]
        success_rate = (
            self.metrics["completed_tasks"] / total if total > 0 else 0
        )
        
        return {
            **self.metrics,
            "active_tasks": self.active_tasks,
            "success_rate": success_rate,
            "max_concurrent": self.max_concurrent
        }


# Global instances for the optimization system
adaptive_optimizer = AdaptiveOptimizer()
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()
concurrency_manager = ConcurrencyManager()


async def optimize_runtime_performance() -> Dict[str, Any]:
    """Main function to optimize runtime performance."""
    # Collect current metrics (placeholder - would be real metrics in production)
    current_metrics = OptimizationMetrics(
        latency_p95=45.0,
        throughput=15.0,
        memory_usage=60.0,
        cpu_usage=65.0,
        error_rate=0.02,
        cache_hit_rate=0.85
    )
    
    # Get optimization suggestions
    suggested_config = adaptive_optimizer.suggest_optimization(current_metrics)
    
    # Check if scaling is needed
    should_scale, target_instances = auto_scaler.should_scale(current_metrics)
    
    # Perform health checks
    await load_balancer.health_check()
    
    # Record metrics for learning
    adaptive_optimizer.record_metrics(current_metrics, suggested_config)
    
    optimization_results = {
        "current_metrics": current_metrics,
        "suggested_config": suggested_config,
        "scaling_needed": should_scale,
        "target_instances": target_instances,
        "healthy_instances": len([
            k for k, v in load_balancer.instances.items() if v["healthy"]
        ]),
        "concurrency_metrics": concurrency_manager.get_metrics()
    }
    
    logger.info(f"Performance optimization completed: {optimization_results}")
    return optimization_results