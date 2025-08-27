"""Performance Optimization - Generation 3: Scale & Optimize

Advanced performance optimization, auto-scaling, and efficiency systems
for high-throughput PyTorch-to-WASM inference at scale.
"""

import asyncio
import time
import logging
import threading
import math
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    ADAPTIVE_BATCHING = "adaptive_batching"
    MODEL_POOLING = "model_pooling"
    CACHE_OPTIMIZATION = "cache_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    PIPELINE_OPTIMIZATION = "pipeline_optimization"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for optimization."""
    timestamp: float = field(default_factory=time.time)
    throughput_rps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    batch_efficiency: float = 0.0
    model_load_time_ms: float = 0.0
    gc_time_ms: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Recommendation for performance optimization."""
    strategy: OptimizationStrategy
    priority: int  # 1 (highest) to 10 (lowest)
    expected_improvement: float  # Expected percentage improvement
    complexity: str  # "low", "medium", "high"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: str = "low"  # Resource cost: "low", "medium", "high"


class PerformanceProfiler:
    """Advanced performance profiler with intelligent insights."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._metrics_history: deque = deque(maxlen=history_size)
        self._request_times: deque = deque(maxlen=history_size)
        self._batch_sizes: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        
        # Performance thresholds for optimization triggers
        self.thresholds = {
            'high_latency_ms': 1000,  # 1 second
            'low_throughput_rps': 10,
            'high_error_rate': 0.05,  # 5%
            'low_cache_hit_rate': 0.8,  # 80%
            'high_queue_depth': 100,
            'low_batch_efficiency': 0.6  # 60%
        }
    
    def record_request(self, latency_ms: float, batch_size: int = 1, success: bool = True) -> None:
        """Record individual request metrics."""
        with self._lock:
            self._request_times.append({
                'latency_ms': latency_ms,
                'timestamp': time.time(),
                'batch_size': batch_size,
                'success': success
            })
            
            if batch_size > 0:
                self._batch_sizes.append(batch_size)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record comprehensive performance metrics."""
        with self._lock:
            self._metrics_history.append(metrics)
    
    def analyze_performance(self, window_seconds: float = 300) -> Dict[str, Any]:
        """Analyze recent performance and identify bottlenecks."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self._lock:
            # Filter recent data
            recent_requests = [
                req for req in self._request_times 
                if req['timestamp'] >= cutoff_time
            ]
            
            recent_metrics = [
                metric for metric in self._metrics_history
                if metric.timestamp >= cutoff_time
            ]
        
        if not recent_requests:
            return {'status': 'insufficient_data', 'recommendations': []}
        
        # Calculate statistics
        latencies = [req['latency_ms'] for req in recent_requests]
        success_count = sum(1 for req in recent_requests if req['success'])
        
        analysis = {
            'window_seconds': window_seconds,
            'total_requests': len(recent_requests),
            'success_rate': success_count / len(recent_requests),
            'error_rate': 1 - (success_count / len(recent_requests)),
            'throughput_rps': len(recent_requests) / window_seconds,
            'latency_stats': {
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'avg_ms': sum(latencies) / len(latencies),
                'p50_ms': self._percentile(latencies, 50),
                'p95_ms': self._percentile(latencies, 95),
                'p99_ms': self._percentile(latencies, 99)
            }
        }
        
        # Add batch analysis
        if self._batch_sizes:
            recent_batch_sizes = list(self._batch_sizes)[-100:]  # Last 100 batches
            analysis['batch_stats'] = {
                'avg_size': sum(recent_batch_sizes) / len(recent_batch_sizes),
                'max_size': max(recent_batch_sizes),
                'efficiency': self._calculate_batch_efficiency(recent_batch_sizes)
            }
        
        # Add latest system metrics if available
        if recent_metrics:
            latest_metric = recent_metrics[-1]
            analysis['system_metrics'] = {
                'cpu_utilization': latest_metric.cpu_utilization,
                'memory_usage_mb': latest_metric.memory_usage_mb,
                'cache_hit_rate': latest_metric.cache_hit_rate,
                'queue_depth': latest_metric.queue_depth
            }
        
        # Generate optimization recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_batch_efficiency(self, batch_sizes: List[int]) -> float:
        """Calculate batch processing efficiency."""
        if not batch_sizes:
            return 1.0
        
        total_items = sum(batch_sizes)
        total_batches = len(batch_sizes)
        avg_batch_size = total_items / total_batches
        
        # Efficiency is based on how close we are to optimal batch size
        optimal_batch_size = 32  # Assume 32 is optimal
        efficiency = min(avg_batch_size / optimal_batch_size, optimal_batch_size / avg_batch_size)
        return max(0.0, min(1.0, efficiency))
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # High latency optimization
        if analysis['latency_stats']['p95_ms'] > self.thresholds['high_latency_ms']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.MODEL_POOLING,
                priority=1,
                expected_improvement=30.0,
                complexity="medium",
                description=f"High P95 latency ({analysis['latency_stats']['p95_ms']:.1f}ms). "
                           f"Consider model pooling to reduce contention.",
                parameters={'pool_size': 3, 'warm_up_models': True}
            ))
        
        # Low throughput optimization
        if analysis['throughput_rps'] < self.thresholds['low_throughput_rps']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.ADAPTIVE_BATCHING,
                priority=2,
                expected_improvement=50.0,
                complexity="low",
                description=f"Low throughput ({analysis['throughput_rps']:.1f} RPS). "
                           f"Enable adaptive batching to improve efficiency.",
                parameters={'target_batch_size': 16, 'max_wait_time_ms': 50}
            ))
        
        # High error rate optimization
        if analysis['error_rate'] > self.thresholds['high_error_rate']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                priority=1,
                expected_improvement=25.0,
                complexity="medium",
                description=f"High error rate ({analysis['error_rate']:.2%}). "
                           f"May indicate resource exhaustion.",
                parameters={'enable_gc_optimization': True, 'memory_limit_mb': 4096}
            ))
        
        # Low cache hit rate optimization
        system_metrics = analysis.get('system_metrics', {})
        cache_hit_rate = system_metrics.get('cache_hit_rate', 1.0)
        if cache_hit_rate < self.thresholds['low_cache_hit_rate']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                priority=3,
                expected_improvement=20.0,
                complexity="low",
                description=f"Low cache hit rate ({cache_hit_rate:.2%}). "
                           f"Optimize caching strategy.",
                parameters={'cache_size_multiplier': 2, 'ttl_optimization': True}
            ))
        
        # High queue depth optimization
        queue_depth = system_metrics.get('queue_depth', 0)
        if queue_depth > self.thresholds['high_queue_depth']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CPU_OPTIMIZATION,
                priority=2,
                expected_improvement=35.0,
                complexity="medium",
                description=f"High queue depth ({queue_depth}). "
                           f"Consider increasing worker threads or CPU optimization.",
                parameters={'worker_multiplier': 1.5, 'enable_cpu_pinning': True}
            ))
        
        # Low batch efficiency optimization
        batch_stats = analysis.get('batch_stats', {})
        batch_efficiency = batch_stats.get('efficiency', 1.0)
        if batch_efficiency < self.thresholds['low_batch_efficiency']:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.PIPELINE_OPTIMIZATION,
                priority=4,
                expected_improvement=15.0,
                complexity="high",
                description=f"Low batch efficiency ({batch_efficiency:.2%}). "
                           f"Optimize request batching pipeline.",
                parameters={'dynamic_batch_sizing': True, 'request_coalescing': True}
            ))
        
        # Sort by priority (lower number = higher priority)
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations


class AdaptiveOptimizer:
    """Adaptive performance optimizer that automatically applies optimizations."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self._active_optimizations: Dict[OptimizationStrategy, Dict[str, Any]] = {}
        self._optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Optimization impact tracking
        self._baseline_metrics: Optional[PerformanceMetrics] = None
        self._optimization_start_time: Optional[float] = None
    
    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation."""
        with self._lock:
            if recommendation.strategy in self._active_optimizations:
                logger.warning(f"Optimization {recommendation.strategy.value} already active")
                return False
            
            optimization_record = {
                'strategy': recommendation.strategy,
                'applied_at': time.time(),
                'parameters': recommendation.parameters,
                'expected_improvement': recommendation.expected_improvement,
                'status': 'active'
            }
            
            try:
                success = self._apply_strategy(recommendation)
                if success:
                    self._active_optimizations[recommendation.strategy] = optimization_record
                    self._optimization_history.append(optimization_record)
                    logger.info(f"Applied optimization: {recommendation.strategy.value}")
                    return True
                else:
                    optimization_record['status'] = 'failed'
                    self._optimization_history.append(optimization_record)
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to apply optimization {recommendation.strategy.value}: {e}")
                optimization_record['status'] = 'error'
                optimization_record['error'] = str(e)
                self._optimization_history.append(optimization_record)
                return False
    
    def _apply_strategy(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization strategy."""
        strategy = recommendation.strategy
        params = recommendation.parameters
        
        if strategy == OptimizationStrategy.ADAPTIVE_BATCHING:
            return self._optimize_batching(params)
        elif strategy == OptimizationStrategy.MODEL_POOLING:
            return self._optimize_model_pooling(params)
        elif strategy == OptimizationStrategy.CACHE_OPTIMIZATION:
            return self._optimize_caching(params)
        elif strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
            return self._optimize_memory(params)
        elif strategy == OptimizationStrategy.CPU_OPTIMIZATION:
            return self._optimize_cpu(params)
        elif strategy == OptimizationStrategy.PIPELINE_OPTIMIZATION:
            return self._optimize_pipeline(params)
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            return False
    
    def _optimize_batching(self, params: Dict[str, Any]) -> bool:
        """Optimize adaptive batching parameters."""
        target_batch_size = params.get('target_batch_size', 16)
        max_wait_time_ms = params.get('max_wait_time_ms', 50)
        
        logger.info(f"Optimizing batching: target_size={target_batch_size}, "
                   f"max_wait={max_wait_time_ms}ms")
        
        # In a real implementation, this would adjust batch processor settings
        return True
    
    def _optimize_model_pooling(self, params: Dict[str, Any]) -> bool:
        """Optimize model pooling configuration."""
        pool_size = params.get('pool_size', 3)
        warm_up_models = params.get('warm_up_models', True)
        
        logger.info(f"Optimizing model pooling: pool_size={pool_size}, "
                   f"warm_up={warm_up_models}")
        
        # In a real implementation, this would configure model instances
        return True
    
    def _optimize_caching(self, params: Dict[str, Any]) -> bool:
        """Optimize caching strategy."""
        cache_multiplier = params.get('cache_size_multiplier', 2)
        ttl_optimization = params.get('ttl_optimization', True)
        
        logger.info(f"Optimizing cache: size_multiplier={cache_multiplier}, "
                   f"ttl_optimization={ttl_optimization}")
        
        # In a real implementation, this would adjust cache settings
        return True
    
    def _optimize_memory(self, params: Dict[str, Any]) -> bool:
        """Optimize memory management."""
        enable_gc_optimization = params.get('enable_gc_optimization', True)
        memory_limit_mb = params.get('memory_limit_mb', 4096)
        
        logger.info(f"Optimizing memory: gc_optimization={enable_gc_optimization}, "
                   f"limit={memory_limit_mb}MB")
        
        # In a real implementation, this would configure garbage collection
        return True
    
    def _optimize_cpu(self, params: Dict[str, Any]) -> bool:
        """Optimize CPU utilization."""
        worker_multiplier = params.get('worker_multiplier', 1.5)
        enable_cpu_pinning = params.get('enable_cpu_pinning', True)
        
        logger.info(f"Optimizing CPU: worker_multiplier={worker_multiplier}, "
                   f"cpu_pinning={enable_cpu_pinning}")
        
        # In a real implementation, this would adjust worker threads
        return True
    
    def _optimize_pipeline(self, params: Dict[str, Any]) -> bool:
        """Optimize processing pipeline."""
        dynamic_batch_sizing = params.get('dynamic_batch_sizing', True)
        request_coalescing = params.get('request_coalescing', True)
        
        logger.info(f"Optimizing pipeline: dynamic_batching={dynamic_batch_sizing}, "
                   f"coalescing={request_coalescing}")
        
        # In a real implementation, this would reconfigure the pipeline
        return True
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        with self._lock:
            return {
                'active_optimizations': {
                    strategy.value: details 
                    for strategy, details in self._active_optimizations.items()
                },
                'optimization_history': self._optimization_history[-10:],  # Last 10
                'total_optimizations_applied': len(self._optimization_history)
            }
    
    def auto_optimize(self, min_confidence: float = 0.8) -> List[OptimizationRecommendation]:
        """Automatically apply high-confidence optimizations."""
        analysis = self.profiler.analyze_performance()
        recommendations = analysis.get('recommendations', [])
        
        applied_optimizations = []
        
        for recommendation in recommendations:
            # Apply high-priority, low-complexity optimizations automatically
            should_apply = (
                recommendation.priority <= 2 and 
                recommendation.complexity in ['low', 'medium'] and
                recommendation.expected_improvement >= 15.0
            )
            
            if should_apply:
                success = self.apply_optimization(recommendation)
                if success:
                    applied_optimizations.append(recommendation)
        
        return applied_optimizations


class LoadBalancer:
    """Intelligent load balancer with predictive scaling."""
    
    def __init__(self, initial_capacity: int = 10):
        self.initial_capacity = initial_capacity
        self._current_capacity = initial_capacity
        self._load_history: deque = deque(maxlen=1000)
        self._instance_health: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Scaling parameters
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.min_capacity = 1
        self.max_capacity = 100
        self.scale_cooldown_seconds = 60  # Prevent rapid scaling
        self._last_scale_time = 0
    
    def record_load_metric(self, active_requests: int, queue_depth: int, 
                          response_time_ms: float) -> None:
        """Record load metrics for scaling decisions."""
        current_time = time.time()
        utilization = min(1.0, (active_requests + queue_depth) / self._current_capacity)
        
        load_metric = {
            'timestamp': current_time,
            'active_requests': active_requests,
            'queue_depth': queue_depth,
            'response_time_ms': response_time_ms,
            'utilization': utilization,
            'capacity': self._current_capacity
        }
        
        with self._lock:
            self._load_history.append(load_metric)
    
    def should_scale(self) -> Tuple[bool, str, int]:
        """Determine if scaling is needed and in which direction."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_time < self.scale_cooldown_seconds:
            return False, "cooldown", 0
        
        with self._lock:
            if len(self._load_history) < 10:  # Need enough data
                return False, "insufficient_data", 0
            
            # Analyze recent load (last 5 minutes or 10 data points, whichever is less)
            recent_metrics = list(self._load_history)[-60:]
            
            avg_utilization = sum(m['utilization'] for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m['response_time_ms'] for m in recent_metrics) / len(recent_metrics)
            
            # Scale up conditions
            if (avg_utilization > self.scale_up_threshold or 
                avg_response_time > 1000) and self._current_capacity < self.max_capacity:
                
                # Calculate suggested scale amount
                if avg_utilization > 0.9:
                    scale_amount = max(2, int(self._current_capacity * 0.5))  # Scale by 50%
                else:
                    scale_amount = max(1, int(self._current_capacity * 0.25))  # Scale by 25%
                
                new_capacity = min(self.max_capacity, self._current_capacity + scale_amount)
                return True, "scale_up", new_capacity - self._current_capacity
            
            # Scale down conditions
            elif (avg_utilization < self.scale_down_threshold and 
                  avg_response_time < 500 and 
                  self._current_capacity > self.min_capacity):
                
                scale_amount = max(1, int(self._current_capacity * 0.2))  # Scale down by 20%
                new_capacity = max(self.min_capacity, self._current_capacity - scale_amount)
                return True, "scale_down", self._current_capacity - new_capacity
            
            return False, "no_action", 0
    
    def apply_scaling(self, scale_amount: int) -> bool:
        """Apply scaling decision."""
        try:
            with self._lock:
                old_capacity = self._current_capacity
                self._current_capacity += scale_amount
                self._current_capacity = max(self.min_capacity, 
                                           min(self.max_capacity, self._current_capacity))
                self._last_scale_time = time.time()
                
                logger.info(f"Scaled capacity: {old_capacity} -> {self._current_capacity} "
                           f"(change: {scale_amount:+d})")
                
                # In a real implementation, this would actually start/stop instances
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply scaling: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self._lock:
            recent_metrics = list(self._load_history)[-10:] if self._load_history else []
            
            current_utilization = 0.0
            current_response_time = 0.0
            
            if recent_metrics:
                current_utilization = recent_metrics[-1]['utilization']
                current_response_time = recent_metrics[-1]['response_time_ms']
            
            return {
                'current_capacity': self._current_capacity,
                'current_utilization': current_utilization,
                'current_response_time_ms': current_response_time,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold,
                'min_capacity': self.min_capacity,
                'max_capacity': self.max_capacity,
                'cooldown_remaining': max(0, self.scale_cooldown_seconds - 
                                        (time.time() - self._last_scale_time))
            }


# Demo function
async def demo_performance_optimization():
    """Demonstrate performance optimization capabilities."""
    
    print("Performance Optimization Demo - Generation 3")
    print("=" * 55)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    print("✓ Performance profiler created")
    
    # Simulate some performance data
    print("\\nSimulating performance data...")
    
    # Simulate slow requests to trigger optimization
    for i in range(50):
        latency = 1500 + (i * 10)  # Increasingly slow
        batch_size = 1 + (i % 8)
        success = i % 10 != 0  # 10% error rate
        profiler.record_request(latency, batch_size, success)
    
    # Record system metrics
    metrics = PerformanceMetrics(
        throughput_rps=8.5,  # Low throughput
        latency_p95_ms=1800,  # High latency
        cpu_utilization=85.0,
        memory_usage_mb=2048,
        queue_depth=150,  # High queue depth
        cache_hit_rate=0.75,  # Low cache hit rate
        error_rate=0.12  # High error rate
    )
    profiler.record_metrics(metrics)
    
    print("✓ Simulated performance issues")
    
    # Analyze performance
    print("\\nAnalyzing performance...")
    analysis = profiler.analyze_performance(window_seconds=300)
    
    print(f"Performance Analysis:")
    print(f"  Throughput: {analysis['throughput_rps']:.1f} RPS")
    print(f"  P95 Latency: {analysis['latency_stats']['p95_ms']:.1f}ms")
    print(f"  Error Rate: {analysis['error_rate']:.2%}")
    
    # Show recommendations
    recommendations = analysis['recommendations']
    print(f"\\nOptimization Recommendations ({len(recommendations)} found):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec.strategy.value} (Priority: {rec.priority})")
        print(f"     Expected improvement: {rec.expected_improvement}%")
        print(f"     {rec.description}")
    
    # Test adaptive optimizer
    print("\\nTesting adaptive optimizer...")
    optimizer = AdaptiveOptimizer(profiler)
    
    # Apply some optimizations
    applied = optimizer.auto_optimize()
    print(f"✓ Auto-applied {len(applied)} optimizations")
    
    for opt in applied:
        print(f"  - Applied: {opt.strategy.value}")
    
    # Show optimization status
    status = optimizer.get_optimization_status()
    print(f"\\nOptimization Status:")
    print(f"Active optimizations: {len(status['active_optimizations'])}")
    print(f"Total optimizations applied: {status['total_optimizations_applied']}")
    
    # Test load balancer
    print("\\nTesting intelligent load balancer...")
    load_balancer = LoadBalancer(initial_capacity=5)
    
    # Simulate high load
    for i in range(20):
        active_requests = 8 + (i // 4)  # Increasing load
        queue_depth = i * 2
        response_time = 500 + (i * 50)  # Increasing response time
        load_balancer.record_load_metric(active_requests, queue_depth, response_time)
    
    # Check scaling decision
    should_scale, reason, amount = load_balancer.should_scale()
    print(f"Scaling decision: {reason}")
    if should_scale:
        print(f"  Recommended scaling: {amount:+d} instances")
        load_balancer.apply_scaling(amount)
    
    scaling_status = load_balancer.get_scaling_status()
    print(f"Load Balancer Status:")
    print(f"  Current capacity: {scaling_status['current_capacity']}")
    print(f"  Current utilization: {scaling_status['current_utilization']:.2%}")
    print(f"  Response time: {scaling_status['current_response_time_ms']:.1f}ms")
    
    print("\\n🚀 Performance Optimization Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_performance_optimization())