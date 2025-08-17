"""Autonomous Load Balancer for WASM-Torch

Intelligent load balancing system with predictive scaling, quantum-inspired
optimization, and autonomous health management for maximum throughput.
"""

import asyncio
import time
import logging
import random
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import heapq
from abc import ABC, abstractmethod
import weakref

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PREDICTIVE = "predictive"
    QUANTUM_OPTIMAL = "quantum_optimal"
    AUTONOMOUS_ADAPTIVE = "autonomous_adaptive"

class NodeState(Enum):
    """Node health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILING = "failing"
    OFFLINE = "offline"

@dataclass
class NodeMetrics:
    """Real-time node performance metrics"""
    node_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    queue_depth: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def health_score(self) -> float:
        """Calculate composite health score (0-1, higher is better)"""
        # Invert bad metrics and normalize
        cpu_score = max(0, 1.0 - self.cpu_usage)
        memory_score = max(0, 1.0 - self.memory_usage)
        response_time_score = max(0, 1.0 - min(1.0, self.avg_response_time / 5.0))  # 5s max
        error_rate_score = max(0, 1.0 - self.error_rate)
        queue_score = max(0, 1.0 - min(1.0, self.queue_depth / 100.0))  # 100 max queue
        
        # Weighted combination
        health = (
            0.25 * cpu_score +
            0.25 * memory_score +
            0.20 * response_time_score +
            0.20 * error_rate_score +
            0.10 * queue_score
        )
        
        return max(0.0, min(1.0, health))

@dataclass
class LoadBalancingNode:
    """Load balancing node with comprehensive tracking"""
    node_id: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 1000
    current_connections: int = 0
    state: NodeState = NodeState.HEALTHY
    metrics: NodeMetrics = field(default_factory=lambda: NodeMetrics(""))
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    circuit_breaker_open: bool = False
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        if not self.metrics.node_id:
            self.metrics.node_id = self.node_id
    
    def can_accept_request(self) -> bool:
        """Check if node can accept new request"""
        return (
            self.state in [NodeState.HEALTHY, NodeState.DEGRADED] and
            not self.circuit_breaker_open and
            self.current_connections < self.max_connections
        )
    
    def record_request_start(self):
        """Record request start"""
        self.current_connections += 1
        self.metrics.active_connections = self.current_connections
    
    def record_request_end(self, response_time: float, success: bool):
        """Record request completion"""
        self.current_connections = max(0, self.current_connections - 1)
        self.metrics.active_connections = self.current_connections
        
        # Update response time history
        self.response_times.append(response_time)
        if self.response_times:
            self.metrics.avg_response_time = statistics.mean(self.response_times)
        
        # Update failure tracking
        if not success:
            self.failure_count += 1
            self.last_failure_time = time.time()
        
        # Update error rate (based on last 100 requests)
        if len(self.response_times) >= 10:
            recent_failures = sum(1 for _ in range(min(10, len(self.response_times))))
            self.metrics.error_rate = recent_failures / min(10, len(self.response_times))

@dataclass
class PredictiveMetrics:
    """Predictive load balancing metrics"""
    predicted_load: float = 0.0
    load_trend: float = 0.0  # Rate of change
    capacity_forecast: float = 1.0
    optimal_weight: float = 1.0
    scaling_recommendation: str = "none"  # "scale_up", "scale_down", "none"

class LoadPredictor:
    """Predictive load analysis system"""
    
    def __init__(self, history_window: int = 300):  # 5 minutes
        self.history_window = history_window
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.pattern_weights = {
            "trend": 0.4,
            "seasonal": 0.3,
            "cyclical": 0.2,
            "random": 0.1
        }
    
    def update_load_data(self, node_id: str, load_metrics: Dict[str, float]):
        """Update load data for prediction"""
        timestamp = time.time()
        load_point = {
            "timestamp": timestamp,
            **load_metrics
        }
        self.load_history[node_id].append(load_point)
    
    def predict_load(self, node_id: str, forecast_horizon: int = 60) -> PredictiveMetrics:
        """Predict future load for node"""
        history = self.load_history[node_id]
        
        if len(history) < 10:
            return PredictiveMetrics()  # Insufficient data
        
        # Extract load values
        load_values = [point.get("requests_per_second", 0.0) for point in history]
        timestamps = [point["timestamp"] for point in history]
        
        # Calculate trend
        trend = self._calculate_trend(load_values, timestamps)
        
        # Predict future load
        current_load = load_values[-1] if load_values else 0.0
        predicted_load = max(0.0, current_load + trend * forecast_horizon)
        
        # Calculate capacity forecast
        capacity_history = [point.get("cpu_usage", 0.0) for point in history]
        capacity_forecast = self._forecast_capacity(capacity_history)
        
        # Generate scaling recommendation
        scaling_recommendation = self._generate_scaling_recommendation(
            predicted_load, current_load, capacity_forecast
        )
        
        return PredictiveMetrics(
            predicted_load=predicted_load,
            load_trend=trend,
            capacity_forecast=capacity_forecast,
            scaling_recommendation=scaling_recommendation
        )
    
    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate load trend using linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope if not np.isnan(slope) else 0.0
    
    def _forecast_capacity(self, capacity_values: List[float]) -> float:
        """Forecast remaining capacity"""
        if not capacity_values:
            return 1.0
        
        # Use exponential moving average for capacity prediction
        alpha = 0.3  # Smoothing factor
        forecast = capacity_values[0]
        
        for value in capacity_values[1:]:
            forecast = alpha * value + (1 - alpha) * forecast
        
        return max(0.0, 1.0 - forecast)  # Remaining capacity
    
    def _generate_scaling_recommendation(self, 
                                       predicted_load: float, 
                                       current_load: float,
                                       capacity_forecast: float) -> str:
        """Generate scaling recommendation"""
        load_increase = (predicted_load - current_load) / max(current_load, 1.0)
        
        # Scale up if significant load increase expected or low capacity
        if load_increase > 0.5 or capacity_forecast < 0.2:
            return "scale_up"
        
        # Scale down if load decreasing and high capacity
        if load_increase < -0.3 and capacity_forecast > 0.8:
            return "scale_down"
        
        return "none"

class QuantumLoadOptimizer:
    """Quantum-inspired load balancing optimization"""
    
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.quantum_state = np.random.normal(0, 1, dimension) + 1j * np.random.normal(0, 1, dimension)
        self.entanglement_matrix = self._create_entanglement_matrix()
    
    def _create_entanglement_matrix(self) -> np.ndarray:
        """Create quantum entanglement matrix"""
        matrix = np.random.normal(0, 0.1, (self.dimension, self.dimension))
        matrix = matrix + matrix.T  # Make symmetric
        return matrix + np.eye(self.dimension)  # Add identity for stability
    
    def optimize_weights(self, nodes: List[LoadBalancingNode]) -> Dict[str, float]:
        """Optimize node weights using quantum algorithms"""
        if not nodes:
            return {}
        
        # Create quantum states for each node
        node_states = {}
        for node in nodes:
            state = self._create_node_quantum_state(node)
            node_states[node.node_id] = state
        
        # Apply quantum optimization
        optimized_weights = {}
        for node_id, state in node_states.items():
            # Apply quantum evolution
            evolved_state = self._evolve_quantum_state(state)
            
            # Calculate optimal weight from quantum amplitude
            weight = self._extract_weight_from_state(evolved_state)
            optimized_weights[node_id] = weight
        
        # Normalize weights
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            for node_id in optimized_weights:
                optimized_weights[node_id] /= total_weight
        
        return optimized_weights
    
    def _create_node_quantum_state(self, node: LoadBalancingNode) -> np.ndarray:
        """Create quantum state representation of node"""
        state = np.zeros(self.dimension, dtype=complex)
        
        # Encode node metrics into quantum state
        health_score = node.metrics.health_score()
        
        # Health component
        health_amplitude = np.sqrt(health_score)
        state[:16] = health_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, 16))
        
        # Capacity component
        capacity = 1.0 - node.current_connections / max(node.max_connections, 1)
        capacity_amplitude = np.sqrt(capacity)
        state[16:32] = capacity_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, 16))
        
        # Performance component
        performance = 1.0 / max(node.metrics.avg_response_time, 0.1)
        performance_amplitude = min(1.0, performance / 10.0)  # Normalize
        state[32:48] = performance_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, 16))
        
        # Stability component (inverse of error rate)
        stability = 1.0 - node.metrics.error_rate
        stability_amplitude = np.sqrt(stability)
        state[48:] = stability_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, 16))
        
        # Normalize state
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        
        return state
    
    def _evolve_quantum_state(self, state: np.ndarray) -> np.ndarray:
        """Evolve quantum state through optimization"""
        # Apply quantum evolution operator
        evolution_time = 0.1
        hamiltonian = self.entanglement_matrix
        
        # Simplified quantum evolution: exp(-iHt)|psi>
        evolution_operator = np.exp(-1j * evolution_time * hamiltonian)
        evolved_state = evolution_operator @ state
        
        return evolved_state
    
    def _extract_weight_from_state(self, state: np.ndarray) -> float:
        """Extract weight from quantum state"""
        # Use quantum probability (amplitude squared)
        probability_density = np.abs(state) ** 2
        
        # Weight is average probability density
        weight = np.mean(probability_density)
        
        return max(0.1, min(2.0, weight))  # Clamp to reasonable range

class AutonomousLoadBalancer:
    """Main autonomous load balancing system"""
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.AUTONOMOUS_ADAPTIVE,
                 enable_prediction: bool = True,
                 enable_quantum_optimization: bool = True,
                 health_check_interval: float = 10.0):
        
        self.strategy = strategy
        self.enable_prediction = enable_prediction
        self.enable_quantum_optimization = enable_quantum_optimization
        self.health_check_interval = health_check_interval
        
        # Node management
        self.nodes: Dict[str, LoadBalancingNode] = {}
        self.round_robin_counter = 0
        
        # Advanced components
        self.predictor = LoadPredictor() if enable_prediction else None
        self.quantum_optimizer = QuantumLoadOptimizer() if enable_quantum_optimization else None
        
        # Statistics
        self.total_requests = 0
        self.total_response_time = 0.0
        self.request_distribution: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Threading
        self._lock = threading.RLock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AutonomousLoadBalancer")
    
    def add_node(self, 
                 node_id: str, 
                 endpoint: str, 
                 weight: float = 1.0,
                 max_connections: int = 1000) -> bool:
        """Add node to load balancer"""
        with self._lock:
            if node_id in self.nodes:
                self.logger.warning(f"Node {node_id} already exists")
                return False
            
            node = LoadBalancingNode(
                node_id=node_id,
                endpoint=endpoint,
                weight=weight,
                max_connections=max_connections
            )
            
            self.nodes[node_id] = node
            self.logger.info(f"Added node {node_id} at {endpoint}")
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from load balancer"""
        with self._lock:
            if node_id not in self.nodes:
                self.logger.warning(f"Node {node_id} not found")
                return False
            
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id}")
            return True
    
    async def select_node(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[LoadBalancingNode]:
        """Select optimal node for request"""
        with self._lock:
            available_nodes = [
                node for node in self.nodes.values() 
                if node.can_accept_request()
            ]
            
            if not available_nodes:
                self.logger.warning("No available nodes for request")
                return None
            
            # Select node based on strategy
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(available_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted_round_robin(available_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(available_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._select_least_response_time(available_nodes)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._select_resource_based(available_nodes)
            elif self.strategy == LoadBalancingStrategy.PREDICTIVE and self.predictor:
                return await self._select_predictive(available_nodes)
            elif self.strategy == LoadBalancingStrategy.QUANTUM_OPTIMAL and self.quantum_optimizer:
                return await self._select_quantum_optimal(available_nodes)
            elif self.strategy == LoadBalancingStrategy.AUTONOMOUS_ADAPTIVE:
                return await self._select_autonomous_adaptive(available_nodes, request_context)
            else:
                # Default to least connections
                return self._select_least_connections(available_nodes)
    
    def _select_round_robin(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Round robin selection"""
        selected_node = nodes[self.round_robin_counter % len(nodes)]
        self.round_robin_counter += 1
        return selected_node
    
    def _select_weighted_round_robin(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Weighted round robin selection"""
        # Create weighted list
        weighted_nodes = []
        for node in nodes:
            weight_count = max(1, int(node.weight * 10))
            weighted_nodes.extend([node] * weight_count)
        
        selected_node = weighted_nodes[self.round_robin_counter % len(weighted_nodes)]
        self.round_robin_counter += 1
        return selected_node
    
    def _select_least_connections(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Least connections selection"""
        return min(nodes, key=lambda n: n.current_connections)
    
    def _select_least_response_time(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Least response time selection"""
        return min(nodes, key=lambda n: n.metrics.avg_response_time)
    
    def _select_resource_based(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Resource-based selection (best health score)"""
        return max(nodes, key=lambda n: n.metrics.health_score())
    
    async def _select_predictive(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Predictive selection based on forecasted load"""
        if not self.predictor:
            return self._select_least_connections(nodes)
        
        best_node = None
        best_score = -1.0
        
        for node in nodes:
            prediction = self.predictor.predict_load(node.node_id)
            
            # Score based on predicted load and capacity
            load_factor = 1.0 / max(prediction.predicted_load, 1.0)
            capacity_factor = prediction.capacity_forecast
            health_factor = node.metrics.health_score()
            
            score = 0.4 * load_factor + 0.3 * capacity_factor + 0.3 * health_factor
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node or nodes[0]
    
    async def _select_quantum_optimal(self, nodes: List[LoadBalancingNode]) -> LoadBalancingNode:
        """Quantum-optimal selection"""
        if not self.quantum_optimizer:
            return self._select_resource_based(nodes)
        
        # Get quantum-optimized weights
        quantum_weights = self.quantum_optimizer.optimize_weights(nodes)
        
        # Weighted random selection based on quantum weights
        if quantum_weights:
            total_weight = sum(quantum_weights.values())
            if total_weight > 0:
                rand_val = random.uniform(0, total_weight)
                cumulative_weight = 0.0
                
                for node in nodes:
                    weight = quantum_weights.get(node.node_id, 0.0)
                    cumulative_weight += weight
                    if rand_val <= cumulative_weight:
                        return node
        
        # Fallback
        return self._select_resource_based(nodes)
    
    async def _select_autonomous_adaptive(self, 
                                        nodes: List[LoadBalancingNode],
                                        request_context: Optional[Dict[str, Any]] = None) -> LoadBalancingNode:
        """Autonomous adaptive selection combining multiple strategies"""
        # Combine scores from different strategies
        node_scores = {}
        
        for node in nodes:
            score = 0.0
            
            # Health score component (30%)
            health_score = node.metrics.health_score()
            score += 0.3 * health_score
            
            # Connection load component (25%)
            connection_load = 1.0 - (node.current_connections / max(node.max_connections, 1))
            score += 0.25 * connection_load
            
            # Response time component (20%)
            response_time_score = 1.0 / max(node.metrics.avg_response_time, 0.1)
            response_time_score = min(1.0, response_time_score / 10.0)  # Normalize
            score += 0.2 * response_time_score
            
            # Predictive component (15%)
            if self.predictor:
                prediction = self.predictor.predict_load(node.node_id)
                predictive_score = prediction.capacity_forecast
                score += 0.15 * predictive_score
            
            # Quantum component (10%)
            if self.quantum_optimizer:
                quantum_weights = self.quantum_optimizer.optimize_weights([node])
                quantum_score = quantum_weights.get(node.node_id, 0.5)
                score += 0.1 * quantum_score
            
            node_scores[node.node_id] = score
        
        # Select node with highest composite score
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k])
        return next(node for node in nodes if node.node_id == best_node_id)
    
    async def process_request(self, 
                             request_func: Callable,
                             *args,
                             request_context: Optional[Dict[str, Any]] = None,
                             **kwargs) -> Any:
        """Process request through load balancer"""
        start_time = time.time()
        
        # Select node
        node = await self.select_node(request_context)
        if not node:
            raise RuntimeError("No available nodes")
        
        # Track request start
        node.record_request_start()
        self.total_requests += 1
        self.request_distribution[node.node_id] += 1
        
        try:
            # Execute request
            if asyncio.iscoroutinefunction(request_func):
                result = await request_func(node.endpoint, *args, **kwargs)
            else:
                result = request_func(node.endpoint, *args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            node.record_request_end(response_time, True)
            self.total_response_time += response_time
            
            # Update predictor
            if self.predictor:
                self.predictor.update_load_data(node.node_id, {
                    "requests_per_second": node.metrics.requests_per_second,
                    "cpu_usage": node.metrics.cpu_usage,
                    "memory_usage": node.metrics.memory_usage,
                    "response_time": response_time
                })
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            node.record_request_end(response_time, False)
            
            self.logger.error(f"Request failed on node {node.node_id}: {e}")
            raise
    
    async def start(self):
        """Start load balancer background tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start optimization
        if self.enable_prediction or self.enable_quantum_optimization:
            self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Autonomous load balancer started")
    
    async def stop(self):
        """Stop load balancer background tasks"""
        self.is_running = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Autonomous load balancer stopped")
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for node in self.nodes.values():
                    await self._check_node_health(node)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def _check_node_health(self, node: LoadBalancingNode):
        """Check individual node health"""
        try:
            # Simulate health check (in real implementation, would ping endpoint)
            health_score = node.metrics.health_score()
            
            # Update node state based on health
            if health_score > 0.8:
                node.state = NodeState.HEALTHY
            elif health_score > 0.6:
                node.state = NodeState.DEGRADED
            elif health_score > 0.3:
                node.state = NodeState.OVERLOADED
            else:
                node.state = NodeState.FAILING
            
            # Circuit breaker logic
            current_time = time.time()
            if node.failure_count > 5 and node.last_failure_time:
                if current_time - node.last_failure_time < 300:  # 5 minutes
                    node.circuit_breaker_open = True
                else:
                    # Reset circuit breaker
                    node.circuit_breaker_open = False
                    node.failure_count = 0
            
        except Exception as e:
            self.logger.error(f"Health check failed for node {node.node_id}: {e}")
            node.state = NodeState.OFFLINE
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                # Optimize weights based on performance
                await self._optimize_node_weights()
                
                # Update predictions
                if self.predictor:
                    await self._update_predictions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    async def _optimize_node_weights(self):
        """Optimize node weights based on performance"""
        if not self.quantum_optimizer:
            return
        
        # Get quantum-optimized weights
        active_nodes = [node for node in self.nodes.values() if node.state != NodeState.OFFLINE]
        
        if active_nodes:
            optimized_weights = self.quantum_optimizer.optimize_weights(active_nodes)
            
            # Apply optimized weights
            for node in active_nodes:
                if node.node_id in optimized_weights:
                    new_weight = optimized_weights[node.node_id]
                    if abs(new_weight - node.weight) > 0.1:  # Significant change
                        self.logger.info(f"Updated weight for node {node.node_id}: {node.weight:.2f} -> {new_weight:.2f}")
                        node.weight = new_weight
    
    async def _update_predictions(self):
        """Update load predictions for all nodes"""
        if not self.predictor:
            return
        
        for node in self.nodes.values():
            prediction = self.predictor.predict_load(node.node_id)
            
            # Act on scaling recommendations
            if prediction.scaling_recommendation == "scale_up":
                self.logger.info(f"Scaling recommendation for {node.node_id}: scale up")
                # In real implementation, would trigger auto-scaling
            elif prediction.scaling_recommendation == "scale_down":
                self.logger.info(f"Scaling recommendation for {node.node_id}: scale down")
                # In real implementation, would trigger scale-down
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        with self._lock:
            node_status = {}
            total_capacity = 0
            used_capacity = 0
            
            for node_id, node in self.nodes.items():
                node_status[node_id] = {
                    "state": node.state.value,
                    "health_score": node.metrics.health_score(),
                    "current_connections": node.current_connections,
                    "max_connections": node.max_connections,
                    "weight": node.weight,
                    "avg_response_time": node.metrics.avg_response_time,
                    "error_rate": node.metrics.error_rate,
                    "circuit_breaker_open": node.circuit_breaker_open
                }
                
                total_capacity += node.max_connections
                used_capacity += node.current_connections
            
            avg_response_time = (self.total_response_time / self.total_requests 
                               if self.total_requests > 0 else 0.0)
            
            return {
                "strategy": self.strategy.value,
                "total_requests": self.total_requests,
                "avg_response_time": avg_response_time,
                "capacity_utilization": used_capacity / max(total_capacity, 1),
                "nodes": node_status,
                "request_distribution": dict(self.request_distribution),
                "active_nodes": len([n for n in self.nodes.values() if n.can_accept_request()]),
                "total_nodes": len(self.nodes),
                "features": {
                    "prediction_enabled": self.enable_prediction,
                    "quantum_optimization_enabled": self.enable_quantum_optimization
                }
            }

# Global load balancer instance
_global_load_balancer: Optional[AutonomousLoadBalancer] = None

def get_load_balancer() -> AutonomousLoadBalancer:
    """Get global load balancer instance"""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = AutonomousLoadBalancer()
    return _global_load_balancer

async def balanced_request(request_func: Callable, 
                          *args, 
                          request_context: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Any:
    """Execute request through load balancer"""
    lb = get_load_balancer()
    return await lb.process_request(request_func, *args, request_context=request_context, **kwargs)