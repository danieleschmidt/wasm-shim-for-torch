"""Autonomous Inference Pipeline with Self-Optimization and Adaptive Learning."""

import asyncio
import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import statistics
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Comprehensive metrics for inference pipeline performance."""
    latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    memory_usage_mb: float = 0.0
    accuracy_score: float = 0.0
    cache_hit_ratio: float = 0.0
    optimization_score: float = 0.0
    adaptation_events: int = 0
    self_healing_actions: int = 0
    

@dataclass
class AdaptationConfig:
    """Configuration for autonomous adaptation behavior."""
    enable_real_time_optimization: bool = True
    enable_predictive_caching: bool = True
    enable_load_balancing: bool = True
    enable_auto_scaling: bool = True
    adaptation_sensitivity: float = 0.1  # 0.1 = sensitive, 1.0 = conservative
    learning_rate: float = 0.01
    performance_target_latency_ms: float = 50.0
    performance_target_throughput: float = 1000.0


class AdaptiveCacheManager:
    """ML-powered adaptive caching system with predictive capabilities."""
    
    def __init__(self, max_cache_size_mb: int = 512):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: Dict[str, Any] = {}
        self.access_patterns = defaultdict(deque)
        self.prediction_model = self._initialize_prediction_model()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "predictions": 0
        }
        
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize ML model for cache prediction."""
        return {
            "access_frequency_weights": defaultdict(float),
            "temporal_patterns": defaultdict(list),
            "prediction_accuracy": 0.0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached item with adaptive learning."""
        current_time = time.time()
        
        if key in self.cache:
            self.cache_stats["hits"] += 1
            self._record_access(key, current_time, hit=True)
            return self.cache[key]
        else:
            self.cache_stats["misses"] += 1
            self._record_access(key, current_time, hit=False)
            return None
    
    async def put(self, key: str, value: Any, predicted_access_time: Optional[float] = None):
        """Store item with predictive optimization."""
        # Check if cache needs eviction
        current_size = self._estimate_cache_size()
        if current_size > self.max_cache_size_mb * 1024 * 1024:  # Convert to bytes
            await self._intelligent_eviction()
        
        # Store with prediction metadata
        cache_entry = {
            "value": value,
            "stored_time": time.time(),
            "predicted_access_time": predicted_access_time,
            "access_count": 0,
            "priority_score": self._calculate_priority_score(key)
        }
        
        self.cache[key] = cache_entry
        
    def _record_access(self, key: str, access_time: float, hit: bool):
        """Record access pattern for learning."""
        pattern_entry = {
            "time": access_time,
            "hit": hit,
            "hour_of_day": int((access_time % 86400) / 3600),
            "day_of_week": int((access_time / 86400) % 7)
        }
        
        self.access_patterns[key].append(pattern_entry)
        
        # Keep only recent patterns
        if len(self.access_patterns[key]) > 1000:
            self.access_patterns[key].popleft()
        
        # Update prediction model
        self._update_prediction_model(key, pattern_entry)
    
    def _update_prediction_model(self, key: str, pattern: Dict[str, Any]):
        """Update ML prediction model with new access pattern."""
        # Update frequency weights
        self.prediction_model["access_frequency_weights"][key] += 1.0
        
        # Update temporal patterns
        if key not in self.prediction_model["temporal_patterns"]:
            self.prediction_model["temporal_patterns"][key] = []
        
        self.prediction_model["temporal_patterns"][key].append({
            "hour": pattern["hour_of_day"],
            "day": pattern["day_of_week"],
            "time": pattern["time"]
        })
        
        # Keep only recent temporal patterns
        if len(self.prediction_model["temporal_patterns"][key]) > 100:
            self.prediction_model["temporal_patterns"][key] = \
                self.prediction_model["temporal_patterns"][key][-100:]
    
    def _calculate_priority_score(self, key: str) -> float:
        """Calculate priority score for cache eviction."""
        frequency_score = self.prediction_model["access_frequency_weights"][key]
        recency_score = 1.0  # Recent items get higher score
        
        # Temporal pattern score
        temporal_score = 0.0
        if key in self.prediction_model["temporal_patterns"]:
            current_hour = int((time.time() % 86400) / 3600)
            current_day = int((time.time() / 86400) % 7)
            
            # Check if current time matches historical patterns
            for pattern in self.prediction_model["temporal_patterns"][key][-10:]:
                if pattern["hour"] == current_hour or pattern["day"] == current_day:
                    temporal_score += 0.1
        
        return frequency_score * 0.5 + recency_score * 0.3 + temporal_score * 0.2
    
    async def _intelligent_eviction(self):
        """Evict cache entries using ML-guided strategy."""
        # Calculate priority scores for all entries
        scored_entries = []
        for key, entry in self.cache.items():
            if isinstance(entry, dict) and "priority_score" in entry:
                scored_entries.append((key, entry["priority_score"]))
        
        # Sort by priority (ascending - lower scores evicted first)
        scored_entries.sort(key=lambda x: x[1])
        
        # Evict bottom 25% of entries
        eviction_count = max(1, len(scored_entries) // 4)
        for i in range(eviction_count):
            key = scored_entries[i][0]
            del self.cache[key]
            self.cache_stats["evictions"] += 1
            
        logger.info(f"🧹 Evicted {eviction_count} cache entries using ML guidance")
    
    def _estimate_cache_size(self) -> int:
        """Estimate current cache size in bytes."""
        # Simplified size estimation
        return len(self.cache) * 1024 * 10  # Assume ~10KB per entry
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_ratio = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_ratio": hit_ratio,
            "total_entries": len(self.cache),
            "total_requests": total_requests,
            "evictions": self.cache_stats["evictions"],
            "estimated_size_mb": self._estimate_cache_size() / (1024 * 1024)
        }


class AutonomousLoadBalancer:
    """Intelligent load balancer with adaptive routing."""
    
    def __init__(self, worker_nodes: List[str]):
        self.worker_nodes = worker_nodes
        self.node_health = {node: {"healthy": True, "load": 0.0, "latency": 0.0} 
                           for node in worker_nodes}
        self.request_history = deque(maxlen=10000)
        self.load_balancing_strategy = "adaptive_ml"
        
    async def route_request(self, request: Dict[str, Any]) -> str:
        """Route request to optimal worker node."""
        if self.load_balancing_strategy == "adaptive_ml":
            return await self._ml_guided_routing(request)
        elif self.load_balancing_strategy == "round_robin":
            return self._round_robin_routing()
        else:
            return self._weighted_routing()
    
    async def _ml_guided_routing(self, request: Dict[str, Any]) -> str:
        """Use ML to predict optimal routing."""
        # Analyze request characteristics
        request_features = self._extract_request_features(request)
        
        # Find best node based on prediction
        best_node = None
        best_score = float('inf')
        
        for node in self.worker_nodes:
            if not self.node_health[node]["healthy"]:
                continue
                
            # Predict performance for this node
            predicted_latency = self._predict_node_latency(node, request_features)
            predicted_load = self.node_health[node]["load"]
            
            # Combined score (lower is better)
            score = predicted_latency * 0.6 + predicted_load * 0.4
            
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node or self.worker_nodes[0]  # Fallback
    
    def _extract_request_features(self, request: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from request."""
        return {
            "model_size": request.get("model_size_mb", 0.0),
            "batch_size": request.get("batch_size", 1.0),
            "complexity": request.get("model_complexity", 1.0),
            "priority": request.get("priority", 0.5)
        }
    
    def _predict_node_latency(self, node: str, features: Dict[str, float]) -> float:
        """Predict latency for node based on features."""
        # Simplified ML prediction (would use trained model in production)
        base_latency = self.node_health[node]["latency"]
        complexity_factor = features["model_size"] * 0.1 + features["complexity"] * 0.05
        load_factor = self.node_health[node]["load"] * 0.2
        
        return base_latency + complexity_factor + load_factor
    
    async def update_node_health(self, node: str, latency: float, success: bool):
        """Update node health based on performance."""
        if node in self.node_health:
            self.node_health[node]["latency"] = (
                self.node_health[node]["latency"] * 0.9 + latency * 0.1
            )
            self.node_health[node]["healthy"] = success
            
            # Update load estimate
            current_time = time.time()
            recent_requests = [r for r in self.request_history 
                             if r["node"] == node and r["time"] > current_time - 60]
            self.node_health[node]["load"] = len(recent_requests) / 60.0  # requests per second


class AutonomousInferencePipeline:
    """Autonomous inference pipeline with self-optimization and adaptive learning."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.cache_manager = AdaptiveCacheManager()
        self.performance_metrics = InferenceMetrics()
        self.adaptation_history = deque(maxlen=1000)
        self.optimization_state = {
            "batch_size": 8,
            "thread_count": 4,
            "memory_allocation": 256,
            "optimization_level": "O2"
        }
        self.learning_model = self._initialize_learning_model()
        self.is_running = False
        
    def _initialize_learning_model(self) -> Dict[str, Any]:
        """Initialize adaptive learning model."""
        return {
            "performance_history": deque(maxlen=1000),
            "optimization_patterns": defaultdict(list),
            "adaptation_rules": {},
            "learning_weights": {
                "latency": 0.4,
                "throughput": 0.3,
                "memory": 0.2,
                "accuracy": 0.1
            }
        }
    
    async def start_autonomous_pipeline(self):
        """Start the autonomous inference pipeline."""
        self.is_running = True
        logger.info("🚀 Starting autonomous inference pipeline")
        
        # Start background optimization task
        asyncio.create_task(self._autonomous_optimization_loop())
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Autonomous pipeline started successfully")
    
    async def process_inference_request(self, 
                                      request: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request with autonomous optimization."""
        start_time = time.time()
        request_id = request.get("id", f"req_{int(start_time * 1000)}")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.debug(f"✅ Cache hit for request {request_id}")
                return {
                    "result": cached_result["value"],
                    "cache_hit": True,
                    "latency_ms": time.time() - start_time,
                    "request_id": request_id
                }
            
            # Process inference
            result = await self._execute_inference(request)
            
            # Cache result with prediction
            predicted_access_time = self._predict_reuse_probability(request)
            await self.cache_manager.put(cache_key, result, predicted_access_time)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000  # Convert to ms
            await self._record_performance(request, result, latency)
            
            # Trigger adaptation if needed
            if self._should_adapt(latency):
                asyncio.create_task(self._adapt_configuration(request, latency))
            
            return {
                "result": result,
                "cache_hit": False,
                "latency_ms": latency,
                "request_id": request_id,
                "optimization_state": self.optimization_state.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Inference failed for request {request_id}: {e}")
            await self._handle_inference_error(request, e)
            raise
    
    async def _execute_inference(self, request: Dict[str, Any]) -> Any:
        """Execute actual inference (placeholder for WASM runtime)."""
        # Simulate inference execution
        model_complexity = request.get("model_complexity", 1.0)
        batch_size = request.get("batch_size", 1)
        
        # Simulate processing time based on complexity
        processing_time = model_complexity * batch_size * 0.01
        await asyncio.sleep(processing_time)
        
        # Return mock result
        return {
            "predictions": np.random.random((batch_size, 10)).tolist(),
            "confidence": np.random.random(),
            "model_version": "v1.0",
            "processing_time": processing_time
        }
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key from request."""
        # Include relevant request parameters
        key_components = [
            request.get("model_id", "default"),
            str(request.get("input_hash", "")),
            str(request.get("parameters", {}))
        ]
        return "_".join(key_components)
    
    def _predict_reuse_probability(self, request: Dict[str, Any]) -> float:
        """Predict probability of cache reuse."""
        # Simplified prediction based on request patterns
        model_id = request.get("model_id", "default")
        current_hour = int((time.time() % 86400) / 3600)
        
        # Look for similar patterns in history
        similar_requests = [
            entry for entry in self.learning_model["performance_history"]
            if entry.get("model_id") == model_id and 
               abs(entry.get("hour", 0) - current_hour) <= 2
        ]
        
        if len(similar_requests) > 5:
            return 0.8  # High reuse probability
        elif len(similar_requests) > 2:
            return 0.5  # Medium reuse probability
        else:
            return 0.2  # Low reuse probability
    
    async def _record_performance(self, request: Dict[str, Any], 
                                result: Any, latency_ms: float):
        """Record performance metrics for learning."""
        performance_entry = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "model_id": request.get("model_id", "default"),
            "batch_size": request.get("batch_size", 1),
            "optimization_state": self.optimization_state.copy(),
            "success": True,
            "hour": int((time.time() % 86400) / 3600)
        }
        
        self.learning_model["performance_history"].append(performance_entry)
        
        # Update current metrics
        self.performance_metrics.latency_ms = (
            self.performance_metrics.latency_ms * 0.9 + latency_ms * 0.1
        )
        
        # Calculate throughput
        if hasattr(self, '_last_throughput_calculation'):
            time_diff = time.time() - self._last_throughput_calculation
            if time_diff >= 1.0:  # Calculate every second
                recent_requests = len([
                    entry for entry in self.learning_model["performance_history"]
                    if entry["timestamp"] > time.time() - 1.0
                ])
                self.performance_metrics.throughput_ops_sec = recent_requests
                self._last_throughput_calculation = time.time()
        else:
            self._last_throughput_calculation = time.time()
    
    def _should_adapt(self, current_latency: float) -> bool:
        """Determine if adaptation is needed."""
        target_latency = self.config.performance_target_latency_ms
        latency_threshold = target_latency * (1 + self.config.adaptation_sensitivity)
        
        return current_latency > latency_threshold
    
    async def _adapt_configuration(self, request: Dict[str, Any], latency_ms: float):
        """Adapt pipeline configuration based on performance."""
        logger.info(f"🔄 Adapting configuration due to latency: {latency_ms:.2f}ms")
        
        # Analyze recent performance
        recent_performance = list(self.learning_model["performance_history"])[-100:]
        avg_latency = statistics.mean([p["latency_ms"] for p in recent_performance])
        
        # Determine adaptation strategy
        if avg_latency > self.config.performance_target_latency_ms * 1.5:
            # Performance is poor - aggressive optimization
            await self._aggressive_optimization()
        elif avg_latency > self.config.performance_target_latency_ms * 1.2:
            # Performance is suboptimal - moderate optimization
            await self._moderate_optimization()
        else:
            # Fine-tuning only
            await self._fine_tune_optimization()
        
        # Record adaptation event
        adaptation_event = {
            "timestamp": time.time(),
            "trigger_latency": latency_ms,
            "old_state": self.optimization_state.copy(),
            "strategy": "adaptive"
        }
        
        self.adaptation_history.append(adaptation_event)
        self.performance_metrics.adaptation_events += 1
    
    async def _aggressive_optimization(self):
        """Apply aggressive optimization changes."""
        # Increase parallel processing
        self.optimization_state["thread_count"] = min(
            self.optimization_state["thread_count"] + 2, 16
        )
        
        # Increase batch size for better throughput
        self.optimization_state["batch_size"] = min(
            self.optimization_state["batch_size"] * 2, 64
        )
        
        # Increase memory allocation
        self.optimization_state["memory_allocation"] = min(
            self.optimization_state["memory_allocation"] + 128, 1024
        )
        
        # Use maximum optimization
        self.optimization_state["optimization_level"] = "O3"
        
        logger.info("⚡ Applied aggressive optimization")
    
    async def _moderate_optimization(self):
        """Apply moderate optimization changes."""
        # Slight increase in parallelism
        self.optimization_state["thread_count"] = min(
            self.optimization_state["thread_count"] + 1, 12
        )
        
        # Moderate batch size increase
        if self.optimization_state["batch_size"] < 32:
            self.optimization_state["batch_size"] = min(
                int(self.optimization_state["batch_size"] * 1.5), 32
            )
        
        logger.info("🔧 Applied moderate optimization")
    
    async def _fine_tune_optimization(self):
        """Apply fine-tuning adjustments."""
        # Small adjustments only
        if self.optimization_state["optimization_level"] == "O1":
            self.optimization_state["optimization_level"] = "O2"
        
        logger.info("🎯 Applied fine-tuning optimization")
    
    async def _autonomous_optimization_loop(self):
        """Background loop for autonomous optimization."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if len(self.learning_model["performance_history"]) > 10:
                    await self._analyze_and_optimize()
                    
            except Exception as e:
                logger.error(f"❌ Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _analyze_and_optimize(self):
        """Analyze performance patterns and optimize."""
        recent_performance = list(self.learning_model["performance_history"])[-100:]
        
        if len(recent_performance) < 10:
            return
        
        # Calculate performance trends
        latencies = [p["latency_ms"] for p in recent_performance]
        avg_latency = statistics.mean(latencies)
        latency_trend = self._calculate_trend(latencies)
        
        # Check if optimization is needed
        target_latency = self.config.performance_target_latency_ms
        
        if avg_latency > target_latency and latency_trend > 0:
            logger.info("📈 Performance degradation detected - initiating optimization")
            await self._adaptive_optimization_strategy(avg_latency, latency_trend)
        elif avg_latency < target_latency * 0.8 and latency_trend < 0:
            logger.info("📉 Performance exceeding targets - optimizing for efficiency")
            await self._efficiency_optimization(avg_latency)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values."""
        if len(values) < 5:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    async def _adaptive_optimization_strategy(self, avg_latency: float, trend: float):
        """Apply adaptive optimization based on performance analysis."""
        # Determine optimization intensity based on deviation from target
        target = self.config.performance_target_latency_ms
        deviation_ratio = avg_latency / target
        
        if deviation_ratio > 2.0:
            await self._aggressive_optimization()
        elif deviation_ratio > 1.5:
            await self._moderate_optimization()
        else:
            await self._fine_tune_optimization()
        
        self.performance_metrics.optimization_score += 0.1
    
    async def _efficiency_optimization(self, avg_latency: float):
        """Optimize for efficiency when performance exceeds targets."""
        # Reduce resource usage while maintaining performance
        if self.optimization_state["thread_count"] > 2:
            self.optimization_state["thread_count"] -= 1
        
        if self.optimization_state["memory_allocation"] > 128:
            self.optimization_state["memory_allocation"] -= 64
        
        logger.info("💡 Applied efficiency optimization")
    
    async def _monitoring_loop(self):
        """Background monitoring and health checks."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update cache statistics
                cache_stats = self.cache_manager.get_cache_statistics()
                self.performance_metrics.cache_hit_ratio = cache_stats["hit_ratio"]
                
                # Log status
                logger.info(f"📊 Pipeline Status - "
                          f"Latency: {self.performance_metrics.latency_ms:.2f}ms, "
                          f"Throughput: {self.performance_metrics.throughput_ops_sec:.1f} ops/sec, "
                          f"Cache Hit Ratio: {self.performance_metrics.cache_hit_ratio:.2%}")
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
    
    async def _handle_inference_error(self, request: Dict[str, Any], error: Exception):
        """Handle inference errors with self-healing."""
        logger.error(f"🚨 Inference error: {error}")
        
        # Implement self-healing strategies
        if "memory" in str(error).lower():
            # Memory error - reduce batch size
            self.optimization_state["batch_size"] = max(
                self.optimization_state["batch_size"] // 2, 1
            )
            logger.info("🔧 Reduced batch size due to memory error")
            
        elif "timeout" in str(error).lower():
            # Timeout error - increase resources
            self.optimization_state["thread_count"] = min(
                self.optimization_state["thread_count"] + 1, 8
            )
            logger.info("⚡ Increased thread count due to timeout")
        
        self.performance_metrics.self_healing_actions += 1
    
    async def stop_pipeline(self):
        """Stop the autonomous pipeline."""
        self.is_running = False
        logger.info("🛑 Autonomous inference pipeline stopped")
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        cache_stats = self.cache_manager.get_cache_statistics()
        
        return {
            "performance": {
                "latency_ms": self.performance_metrics.latency_ms,
                "throughput_ops_sec": self.performance_metrics.throughput_ops_sec,
                "cache_hit_ratio": self.performance_metrics.cache_hit_ratio,
                "optimization_score": self.performance_metrics.optimization_score
            },
            "adaptation": {
                "adaptation_events": self.performance_metrics.adaptation_events,
                "self_healing_actions": self.performance_metrics.self_healing_actions,
                "optimization_state": self.optimization_state.copy()
            },
            "cache": cache_stats,
            "health_status": "healthy" if self.is_running else "stopped"
        }


# Factory function for easy pipeline creation
def create_autonomous_pipeline(
    performance_target_latency_ms: float = 50.0,
    performance_target_throughput: float = 1000.0,
    enable_all_features: bool = True
) -> AutonomousInferencePipeline:
    """Create configured autonomous inference pipeline."""
    
    config = AdaptationConfig(
        enable_real_time_optimization=enable_all_features,
        enable_predictive_caching=enable_all_features,
        enable_load_balancing=enable_all_features,
        enable_auto_scaling=enable_all_features,
        performance_target_latency_ms=performance_target_latency_ms,
        performance_target_throughput=performance_target_throughput
    )
    
    return AutonomousInferencePipeline(config)


# Example usage
async def example_autonomous_pipeline():
    """Example of autonomous pipeline usage."""
    # Create and start pipeline
    pipeline = create_autonomous_pipeline(
        performance_target_latency_ms=30.0,
        performance_target_throughput=2000.0
    )
    
    await pipeline.start_autonomous_pipeline()
    
    # Process requests
    for i in range(100):
        request = {
            "id": f"req_{i}",
            "model_id": "model_v1",
            "batch_size": 4,
            "model_complexity": 1.5,
            "input_hash": f"hash_{i % 10}"  # Some repeated patterns
        }
        
        result = await pipeline.process_inference_request(request)
        logger.info(f"Request {i}: {result['latency_ms']:.2f}ms")
        
        await asyncio.sleep(0.1)  # Small delay between requests
    
    # Get final metrics
    metrics = pipeline.get_pipeline_metrics()
    logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
    
    await pipeline.stop_pipeline()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_autonomous_pipeline())