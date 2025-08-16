"""Planetary Scale Deployment Engine for Global WASM-Torch Infrastructure."""

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
import hashlib
import socket

logger = logging.getLogger(__name__)


@dataclass
class GeographicRegion:
    """Represents a geographic deployment region."""
    name: str
    code: str  # ISO region code
    latitude: float
    longitude: float
    data_sovereignty_rules: List[str]
    compliance_requirements: List[str]
    network_latency_ms: float = 0.0
    available_resources: Dict[str, float] = field(default_factory=dict)
    

@dataclass
class DeploymentNode:
    """Represents a deployment node in the global infrastructure."""
    node_id: str
    region: GeographicRegion
    node_type: str  # edge, regional, core
    capacity: Dict[str, float]
    current_load: Dict[str, float] = field(default_factory=dict)
    health_status: str = "healthy"
    last_heartbeat: float = 0.0
    deployed_models: List[str] = field(default_factory=list)


@dataclass
class GlobalDeploymentConfig:
    """Configuration for planetary scale deployment."""
    regions: List[GeographicRegion]
    replication_factor: int = 3
    latency_target_ms: float = 50.0
    availability_target: float = 0.9999  # 99.99%
    data_residency_compliance: bool = True
    auto_scaling_enabled: bool = True
    disaster_recovery_enabled: bool = True
    performance_monitoring_enabled: bool = True


class ComplianceEngine:
    """Engine for managing global compliance requirements."""
    
    def __init__(self):
        self.compliance_rules = {
            "GDPR": {
                "data_residency": ["EU"],
                "encryption_required": True,
                "audit_logging": True,
                "right_to_deletion": True
            },
            "CCPA": {
                "data_residency": ["US-CA"],
                "opt_out_rights": True,
                "data_transparency": True
            },
            "PDPA": {
                "data_residency": ["SG", "TH", "MY"],
                "consent_management": True,
                "data_minimization": True
            },
            "LGPD": {
                "data_residency": ["BR"],
                "data_subject_rights": True,
                "privacy_by_design": True
            },
            "PIPEDA": {
                "data_residency": ["CA"],
                "purpose_limitation": True,
                "accountability": True
            }
        }
        
    async def validate_deployment_compliance(self, 
                                           deployment_region: GeographicRegion,
                                           data_types: List[str]) -> Dict[str, Any]:
        """Validate compliance requirements for deployment."""
        
        applicable_regulations = []
        compliance_status = {"compliant": True, "violations": [], "requirements": []}
        
        # Determine applicable regulations based on region
        for regulation, rules in self.compliance_rules.items():
            if any(region in deployment_region.compliance_requirements 
                   for region in rules.get("data_residency", [])):
                applicable_regulations.append(regulation)
        
        # Check compliance for each regulation
        for regulation in applicable_regulations:
            rules = self.compliance_rules[regulation]
            
            # Check data residency
            if "data_residency" in rules:
                if deployment_region.code not in rules["data_residency"]:
                    compliance_status["violations"].append(
                        f"{regulation}: Data residency violation in {deployment_region.code}"
                    )
                    compliance_status["compliant"] = False
            
            # Check encryption requirements
            if rules.get("encryption_required", False):
                compliance_status["requirements"].append(
                    f"{regulation}: End-to-end encryption required"
                )
            
            # Check audit logging
            if rules.get("audit_logging", False):
                compliance_status["requirements"].append(
                    f"{regulation}: Comprehensive audit logging required"
                )
        
        return {
            "applicable_regulations": applicable_regulations,
            "compliance_status": compliance_status,
            "deployment_allowed": compliance_status["compliant"]
        }
    
    async def generate_compliance_config(self, 
                                       regions: List[GeographicRegion]) -> Dict[str, Any]:
        """Generate compliance configuration for multi-region deployment."""
        
        compliance_config = {
            "encryption": {
                "in_transit": True,
                "at_rest": True,
                "key_management": "regional"
            },
            "logging": {
                "audit_enabled": True,
                "retention_days": 2555,  # 7 years
                "regional_storage": True
            },
            "data_governance": {
                "residency_enforcement": True,
                "cross_border_restrictions": [],
                "deletion_policies": {}
            }
        }
        
        # Add region-specific requirements
        for region in regions:
            for requirement in region.compliance_requirements:
                if requirement == "GDPR":
                    compliance_config["data_governance"]["deletion_policies"]["EU"] = {
                        "right_to_deletion": True,
                        "retention_limit_days": 365
                    }
                elif requirement == "CCPA":
                    compliance_config["data_governance"]["deletion_policies"]["US-CA"] = {
                        "opt_out_rights": True,
                        "data_portability": True
                    }
        
        return compliance_config


class GlobalLoadBalancer:
    """Intelligent global load balancer with geographic optimization."""
    
    def __init__(self, nodes: List[DeploymentNode]):
        self.nodes = {node.node_id: node for node in nodes}
        self.routing_table = {}
        self.performance_metrics = defaultdict(list)
        self.geo_optimizer = GeographicOptimizer()
        
    async def route_request(self, 
                          client_location: Tuple[float, float],
                          request: Dict[str, Any]) -> str:
        """Route request to optimal node based on geography and performance."""
        
        client_lat, client_lon = client_location
        
        # Get candidate nodes based on compliance
        compliance_requirements = request.get("compliance_requirements", [])
        candidate_nodes = await self._filter_compliant_nodes(compliance_requirements)
        
        if not candidate_nodes:
            raise ValueError("No compliant nodes available for request")
        
        # Calculate optimal routing
        best_node = None
        best_score = float('inf')
        
        for node_id in candidate_nodes:
            node = self.nodes[node_id]
            
            # Calculate geographic distance
            distance = self.geo_optimizer.calculate_distance(
                client_lat, client_lon,
                node.region.latitude, node.region.longitude
            )
            
            # Calculate performance score
            performance_score = await self._calculate_performance_score(node)
            
            # Combined routing score (lower is better)
            routing_score = (
                distance * 0.4 +  # Geographic proximity
                performance_score * 0.4 +  # Performance
                node.current_load.get("cpu", 0) * 0.2  # Current load
            )
            
            if routing_score < best_score:
                best_score = routing_score
                best_node = node_id
        
        # Update routing metrics
        await self._update_routing_metrics(best_node, best_score)
        
        return best_node
    
    async def _filter_compliant_nodes(self, 
                                    compliance_requirements: List[str]) -> List[str]:
        """Filter nodes based on compliance requirements."""
        compliant_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.health_status != "healthy":
                continue
            
            # Check if node region meets compliance requirements
            region_compliance = set(node.region.compliance_requirements)
            required_compliance = set(compliance_requirements)
            
            if required_compliance.issubset(region_compliance):
                compliant_nodes.append(node_id)
        
        return compliant_nodes
    
    async def _calculate_performance_score(self, node: DeploymentNode) -> float:
        """Calculate performance score for a node."""
        # Base latency from region
        base_latency = node.region.network_latency_ms
        
        # Load-based latency increase
        cpu_load = node.current_load.get("cpu", 0)
        memory_load = node.current_load.get("memory", 0)
        load_penalty = (cpu_load + memory_load) / 2 * 20  # Up to 20ms penalty
        
        # Historical performance
        node_metrics = self.performance_metrics.get(node.node_id, [])
        if node_metrics:
            historical_latency = statistics.mean(node_metrics[-10:])  # Last 10 measurements
        else:
            historical_latency = 0
        
        return base_latency + load_penalty + historical_latency
    
    async def _update_routing_metrics(self, node_id: str, score: float):
        """Update routing performance metrics."""
        self.performance_metrics[node_id].append(score)
        
        # Keep only recent metrics
        if len(self.performance_metrics[node_id]) > 100:
            self.performance_metrics[node_id] = self.performance_metrics[node_id][-100:]


class GeographicOptimizer:
    """Optimizer for geographic distribution and latency minimization."""
    
    def calculate_distance(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points."""
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r
    
    async def optimize_node_placement(self, 
                                    target_regions: List[GeographicRegion],
                                    user_distribution: Dict[str, float]) -> List[Tuple[float, float]]:
        """Optimize node placement for minimal global latency."""
        
        # Weight regions by user distribution
        weighted_points = []
        for region in target_regions:
            weight = user_distribution.get(region.code, 1.0)
            weighted_points.append((region.latitude, region.longitude, weight))
        
        # Use weighted k-means clustering for optimal placement
        optimal_locations = await self._weighted_kmeans_clustering(
            weighted_points, k=len(target_regions)
        )
        
        return optimal_locations
    
    async def _weighted_kmeans_clustering(self, 
                                        weighted_points: List[Tuple[float, float, float]],
                                        k: int) -> List[Tuple[float, float]]:
        """Perform weighted k-means clustering for node placement."""
        
        # Initialize centroids randomly
        centroids = []
        for i in range(k):
            idx = i % len(weighted_points)
            centroids.append((weighted_points[idx][0], weighted_points[idx][1]))
        
        # Iterate until convergence
        max_iterations = 100
        for iteration in range(max_iterations):
            # Assign points to clusters
            clusters = [[] for _ in range(k)]
            
            for lat, lon, weight in weighted_points:
                distances = [
                    self.calculate_distance(lat, lon, c_lat, c_lon)
                    for c_lat, c_lon in centroids
                ]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append((lat, lon, weight))
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if not cluster:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(centroids[len(new_centroids)])
                    continue
                
                # Calculate weighted centroid
                total_weight = sum(weight for _, _, weight in cluster)
                weighted_lat = sum(lat * weight for lat, _, weight in cluster) / total_weight
                weighted_lon = sum(lon * weight for _, lon, weight in cluster) / total_weight
                
                new_centroids.append((weighted_lat, weighted_lon))
            
            # Check convergence
            if self._centroids_converged(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def _centroids_converged(self, old_centroids: List[Tuple[float, float]],
                           new_centroids: List[Tuple[float, float]],
                           threshold: float = 0.01) -> bool:
        """Check if centroids have converged."""
        
        for (old_lat, old_lon), (new_lat, new_lon) in zip(old_centroids, new_centroids):
            distance = self.calculate_distance(old_lat, old_lon, new_lat, new_lon)
            if distance > threshold:
                return False
        
        return True


class AutoScaler:
    """Intelligent auto-scaling system for global deployment."""
    
    def __init__(self, nodes: List[DeploymentNode]):
        self.nodes = {node.node_id: node for node in nodes}
        self.scaling_history = deque(maxlen=1000)
        self.prediction_model = self._initialize_prediction_model()
        
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize load prediction model."""
        return {
            "hourly_patterns": defaultdict(list),
            "daily_patterns": defaultdict(list),
            "seasonal_patterns": defaultdict(list),
            "trend_coefficients": defaultdict(float)
        }
    
    async def predict_load(self, node_id: str, forecast_horizon_hours: int = 1) -> List[float]:
        """Predict future load for a node."""
        
        if node_id not in self.nodes:
            return [0.0] * forecast_horizon_hours
        
        current_time = time.time()
        current_hour = int((current_time % 86400) / 3600)
        current_day = int((current_time / 86400) % 7)
        
        predictions = []
        
        for hour_offset in range(forecast_horizon_hours):
            target_hour = (current_hour + hour_offset) % 24
            target_day = (current_day + (current_hour + hour_offset) // 24) % 7
            
            # Get historical patterns
            hourly_pattern = self.prediction_model["hourly_patterns"].get(
                f"{node_id}_{target_hour}", [0.5]
            )
            daily_pattern = self.prediction_model["daily_patterns"].get(
                f"{node_id}_{target_day}", [0.5]
            )
            
            # Calculate base prediction
            hourly_avg = statistics.mean(hourly_pattern[-20:])  # Last 20 measurements
            daily_avg = statistics.mean(daily_pattern[-10:])    # Last 10 measurements
            
            # Combine patterns with weights
            base_prediction = hourly_avg * 0.7 + daily_avg * 0.3
            
            # Apply trend
            trend = self.prediction_model["trend_coefficients"].get(node_id, 0.0)
            trended_prediction = base_prediction + trend * hour_offset
            
            # Clamp to valid range
            predictions.append(max(0.0, min(1.0, trended_prediction)))
        
        return predictions
    
    async def make_scaling_decision(self, node_id: str) -> Dict[str, Any]:
        """Make intelligent scaling decision for a node."""
        
        if node_id not in self.nodes:
            return {"action": "none", "reason": "node not found"}
        
        node = self.nodes[node_id]
        current_load = node.current_load
        
        # Get load predictions
        predicted_loads = await self.predict_load(node_id, forecast_horizon_hours=2)
        
        # Calculate scaling thresholds
        scale_up_threshold = 0.8
        scale_down_threshold = 0.3
        
        current_cpu_load = current_load.get("cpu", 0)
        predicted_max_load = max(predicted_loads)
        predicted_min_load = min(predicted_loads)
        
        # Make scaling decision
        if current_cpu_load > scale_up_threshold or predicted_max_load > scale_up_threshold:
            return await self._plan_scale_up(node_id, predicted_max_load)
        elif current_cpu_load < scale_down_threshold and predicted_min_load < scale_down_threshold:
            return await self._plan_scale_down(node_id, predicted_min_load)
        else:
            return {
                "action": "maintain",
                "current_load": current_cpu_load,
                "predicted_load_range": [predicted_min_load, predicted_max_load],
                "reason": "load within optimal range"
            }
    
    async def _plan_scale_up(self, node_id: str, predicted_load: float) -> Dict[str, Any]:
        """Plan scale-up action."""
        
        node = self.nodes[node_id]
        
        # Calculate required capacity increase
        current_capacity = node.capacity.get("cpu_cores", 4)
        load_ratio = predicted_load / 0.7  # Target 70% utilization
        required_capacity = max(current_capacity + 1, int(current_capacity * load_ratio))
        
        return {
            "action": "scale_up",
            "current_capacity": current_capacity,
            "target_capacity": required_capacity,
            "predicted_load": predicted_load,
            "reason": f"Predicted load {predicted_load:.2%} exceeds threshold",
            "priority": "high" if predicted_load > 0.9 else "medium"
        }
    
    async def _plan_scale_down(self, node_id: str, predicted_load: float) -> Dict[str, Any]:
        """Plan scale-down action."""
        
        node = self.nodes[node_id]
        
        # Calculate potential capacity reduction
        current_capacity = node.capacity.get("cpu_cores", 4)
        min_capacity = 2  # Minimum capacity for availability
        
        # Only scale down if load is consistently low
        target_capacity = max(min_capacity, current_capacity - 1)
        
        return {
            "action": "scale_down",
            "current_capacity": current_capacity,
            "target_capacity": target_capacity,
            "predicted_load": predicted_load,
            "reason": f"Predicted load {predicted_load:.2%} below threshold",
            "priority": "low"
        }
    
    async def update_load_patterns(self, node_id: str, load_measurement: float):
        """Update load patterns for improved prediction."""
        
        current_time = time.time()
        current_hour = int((current_time % 86400) / 3600)
        current_day = int((current_time / 86400) % 7)
        
        # Update hourly patterns
        hourly_key = f"{node_id}_{current_hour}"
        self.prediction_model["hourly_patterns"][hourly_key].append(load_measurement)
        
        # Update daily patterns
        daily_key = f"{node_id}_{current_day}"
        self.prediction_model["daily_patterns"][daily_key].append(load_measurement)
        
        # Update trend
        recent_measurements = self.prediction_model["hourly_patterns"][hourly_key][-10:]
        if len(recent_measurements) >= 5:
            # Calculate trend using linear regression
            x = list(range(len(recent_measurements)))
            y = recent_measurements
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 != 0:
                trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                self.prediction_model["trend_coefficients"][node_id] = trend


class PlanetaryDeploymentEngine:
    """Main engine for planetary scale deployment orchestration."""
    
    def __init__(self, config: GlobalDeploymentConfig):
        self.config = config
        self.nodes: Dict[str, DeploymentNode] = {}
        self.compliance_engine = ComplianceEngine()
        self.load_balancer = None
        self.auto_scaler = None
        self.geo_optimizer = GeographicOptimizer()
        self.deployment_history = deque(maxlen=10000)
        self.is_running = False
        
    async def initialize_global_infrastructure(self) -> Dict[str, Any]:
        """Initialize global deployment infrastructure."""
        
        logger.info("🌍 Initializing planetary scale infrastructure")
        
        # Generate compliance configuration
        compliance_config = await self.compliance_engine.generate_compliance_config(
            self.config.regions
        )
        
        # Create deployment nodes for each region
        nodes = []
        for region in self.config.regions:
            # Create multiple nodes per region based on type
            node_types = ["edge", "regional", "core"]
            
            for node_type in node_types:
                node_id = f"{region.code}-{node_type}-{int(time.time())}"
                
                # Define capacity based on node type
                if node_type == "edge":
                    capacity = {"cpu_cores": 4, "memory_gb": 16, "storage_gb": 100}
                elif node_type == "regional":
                    capacity = {"cpu_cores": 8, "memory_gb": 32, "storage_gb": 500}
                else:  # core
                    capacity = {"cpu_cores": 16, "memory_gb": 64, "storage_gb": 1000}
                
                node = DeploymentNode(
                    node_id=node_id,
                    region=region,
                    node_type=node_type,
                    capacity=capacity,
                    current_load={"cpu": 0.1, "memory": 0.1, "storage": 0.05},
                    health_status="healthy",
                    last_heartbeat=time.time()
                )
                
                nodes.append(node)
                self.nodes[node_id] = node
        
        # Initialize load balancer and auto-scaler
        self.load_balancer = GlobalLoadBalancer(nodes)
        self.auto_scaler = AutoScaler(nodes)
        
        # Optimize node placement
        user_distribution = await self._estimate_user_distribution()
        optimal_locations = await self.geo_optimizer.optimize_node_placement(
            self.config.regions, user_distribution
        )
        
        logger.info(f"✅ Initialized {len(nodes)} nodes across {len(self.config.regions)} regions")
        
        return {
            "total_nodes": len(nodes),
            "regions": len(self.config.regions),
            "compliance_config": compliance_config,
            "optimal_locations": optimal_locations,
            "global_capacity": self._calculate_global_capacity()
        }
    
    async def deploy_model_globally(self, 
                                  model_info: Dict[str, Any],
                                  deployment_strategy: str = "adaptive") -> Dict[str, Any]:
        """Deploy a model across the global infrastructure."""
        
        model_id = model_info.get("id", f"model_{int(time.time())}")
        logger.info(f"🚀 Deploying model {model_id} globally using {deployment_strategy} strategy")
        
        # Validate compliance for each region
        deployment_plan = {"successful_deployments": [], "failed_deployments": []}
        
        for region in self.config.regions:
            compliance_result = await self.compliance_engine.validate_deployment_compliance(
                region, model_info.get("data_types", [])
            )
            
            if compliance_result["deployment_allowed"]:
                # Find optimal nodes in this region
                region_nodes = [
                    node for node in self.nodes.values() 
                    if node.region.code == region.code and node.health_status == "healthy"
                ]
                
                if region_nodes:
                    # Select best node based on strategy
                    selected_node = await self._select_deployment_node(
                        region_nodes, model_info, deployment_strategy
                    )
                    
                    # Deploy to selected node
                    deployment_result = await self._deploy_to_node(
                        selected_node, model_info
                    )
                    
                    if deployment_result["success"]:
                        deployment_plan["successful_deployments"].append({
                            "node_id": selected_node.node_id,
                            "region": region.code,
                            "deployment_time": deployment_result["deployment_time"]
                        })
                    else:
                        deployment_plan["failed_deployments"].append({
                            "region": region.code,
                            "reason": deployment_result["error"]
                        })
                else:
                    deployment_plan["failed_deployments"].append({
                        "region": region.code,
                        "reason": "No healthy nodes available"
                    })
            else:
                deployment_plan["failed_deployments"].append({
                    "region": region.code,
                    "reason": f"Compliance violation: {compliance_result['compliance_status']['violations']}"
                })
        
        # Record deployment
        deployment_record = {
            "model_id": model_id,
            "timestamp": time.time(),
            "strategy": deployment_strategy,
            "results": deployment_plan
        }
        self.deployment_history.append(deployment_record)
        
        success_rate = len(deployment_plan["successful_deployments"]) / len(self.config.regions)
        
        logger.info(f"📊 Global deployment completed: {success_rate:.1%} success rate")
        
        return {
            "model_id": model_id,
            "deployment_plan": deployment_plan,
            "success_rate": success_rate,
            "global_availability": success_rate >= self.config.availability_target
        }
    
    async def _estimate_user_distribution(self) -> Dict[str, float]:
        """Estimate user distribution across regions."""
        # Simplified user distribution based on population and internet penetration
        return {
            "US-EAST": 0.15,
            "US-WEST": 0.12,
            "EU-WEST": 0.18,
            "EU-CENTRAL": 0.08,
            "ASIA-PACIFIC": 0.25,
            "CHINA": 0.12,
            "INDIA": 0.10
        }
    
    def _calculate_global_capacity(self) -> Dict[str, float]:
        """Calculate total global capacity."""
        total_capacity = {"cpu_cores": 0, "memory_gb": 0, "storage_gb": 0}
        
        for node in self.nodes.values():
            for resource, amount in node.capacity.items():
                if resource in total_capacity:
                    total_capacity[resource] += amount
        
        return total_capacity
    
    async def _select_deployment_node(self, 
                                    candidate_nodes: List[DeploymentNode],
                                    model_info: Dict[str, Any],
                                    strategy: str) -> DeploymentNode:
        """Select optimal node for deployment based on strategy."""
        
        if strategy == "adaptive":
            # Select based on current load and capacity
            best_node = None
            best_score = float('inf')
            
            for node in candidate_nodes:
                # Calculate load score (lower is better)
                cpu_load = node.current_load.get("cpu", 0)
                memory_load = node.current_load.get("memory", 0)
                load_score = (cpu_load + memory_load) / 2
                
                # Calculate capacity score
                model_size = model_info.get("size_mb", 100)
                available_memory = node.capacity["memory_gb"] * 1024 - \
                                 node.current_load.get("memory", 0) * node.capacity["memory_gb"] * 1024
                capacity_score = 1.0 if available_memory > model_size else 0.0
                
                # Combined score
                total_score = load_score * 0.7 + (1 - capacity_score) * 0.3
                
                if total_score < best_score:
                    best_score = total_score
                    best_node = node
            
            return best_node or candidate_nodes[0]
        
        elif strategy == "round_robin":
            # Simple round-robin selection
            return candidate_nodes[len(self.deployment_history) % len(candidate_nodes)]
        
        else:  # random
            return np.random.choice(candidate_nodes)
    
    async def _deploy_to_node(self, 
                            node: DeploymentNode, 
                            model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to a specific node."""
        
        start_time = time.time()
        
        try:
            # Simulate deployment process
            model_size = model_info.get("size_mb", 100)
            deployment_time = model_size / 1000  # Simulate network transfer
            
            await asyncio.sleep(deployment_time)
            
            # Update node state
            node.deployed_models.append(model_info["id"])
            
            # Update resource usage
            memory_usage = model_size / (node.capacity["memory_gb"] * 1024)
            node.current_load["memory"] = min(1.0, 
                node.current_load.get("memory", 0) + memory_usage
            )
            
            deployment_duration = time.time() - start_time
            
            return {
                "success": True,
                "deployment_time": deployment_duration,
                "node_id": node.node_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "deployment_time": time.time() - start_time
            }
    
    async def start_autonomous_management(self):
        """Start autonomous management systems."""
        
        self.is_running = True
        logger.info("🤖 Starting autonomous planetary management")
        
        # Start monitoring and scaling loops
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._performance_optimization_loop())
        
    async def _health_monitoring_loop(self):
        """Monitor health of all nodes globally."""
        
        while self.is_running:
            try:
                unhealthy_nodes = []
                
                for node_id, node in self.nodes.items():
                    # Check heartbeat
                    time_since_heartbeat = time.time() - node.last_heartbeat
                    
                    if time_since_heartbeat > 300:  # 5 minutes
                        node.health_status = "unhealthy"
                        unhealthy_nodes.append(node_id)
                    
                    # Update heartbeat (simulate)
                    node.last_heartbeat = time.time()
                
                if unhealthy_nodes:
                    logger.warning(f"⚠️ {len(unhealthy_nodes)} nodes unhealthy")
                    await self._handle_unhealthy_nodes(unhealthy_nodes)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _auto_scaling_loop(self):
        """Auto-scaling management loop."""
        
        while self.is_running:
            try:
                scaling_decisions = []
                
                for node_id in self.nodes:
                    decision = await self.auto_scaler.make_scaling_decision(node_id)
                    
                    if decision["action"] != "maintain":
                        scaling_decisions.append((node_id, decision))
                
                if scaling_decisions:
                    logger.info(f"🔄 Processing {len(scaling_decisions)} scaling decisions")
                    
                    for node_id, decision in scaling_decisions:
                        await self._execute_scaling_decision(node_id, decision)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Auto-scaling error: {e}")
                await asyncio.sleep(600)
    
    async def _performance_optimization_loop(self):
        """Performance optimization loop."""
        
        while self.is_running:
            try:
                # Analyze global performance metrics
                performance_metrics = await self._collect_global_metrics()
                
                # Identify optimization opportunities
                optimizations = await self._identify_optimizations(performance_metrics)
                
                if optimizations:
                    logger.info(f"🎯 Applying {len(optimizations)} performance optimizations")
                    
                    for optimization in optimizations:
                        await self._apply_optimization(optimization)
                
                await asyncio.sleep(900)  # Check every 15 minutes
                
            except Exception as e:
                logger.error(f"❌ Performance optimization error: {e}")
                await asyncio.sleep(1800)
    
    async def _handle_unhealthy_nodes(self, unhealthy_node_ids: List[str]):
        """Handle unhealthy nodes with failover."""
        
        for node_id in unhealthy_node_ids:
            node = self.nodes[node_id]
            
            # Find healthy nodes in the same region for failover
            region_nodes = [
                n for n in self.nodes.values()
                if n.region.code == node.region.code and 
                   n.health_status == "healthy" and 
                   n.node_id != node_id
            ]
            
            if region_nodes and node.deployed_models:
                # Migrate models to healthy nodes
                target_node = min(region_nodes, 
                                key=lambda n: n.current_load.get("cpu", 0))
                
                await self._migrate_models(node, target_node)
                
                logger.info(f"🔄 Migrated models from {node_id} to {target_node.node_id}")
    
    async def _migrate_models(self, source_node: DeploymentNode, target_node: DeploymentNode):
        """Migrate models between nodes."""
        
        for model_id in source_node.deployed_models:
            # Simulate model migration
            model_info = {"id": model_id, "size_mb": 100}  # Simplified
            
            deployment_result = await self._deploy_to_node(target_node, model_info)
            
            if deployment_result["success"]:
                target_node.deployed_models.append(model_id)
        
        # Clear source node
        source_node.deployed_models.clear()
    
    async def _execute_scaling_decision(self, node_id: str, decision: Dict[str, Any]):
        """Execute scaling decision for a node."""
        
        action = decision["action"]
        
        if action == "scale_up":
            # Increase node capacity
            node = self.nodes[node_id]
            current_cores = node.capacity["cpu_cores"]
            target_cores = decision["target_capacity"]
            
            node.capacity["cpu_cores"] = target_cores
            node.capacity["memory_gb"] = target_cores * 4  # 4GB per core
            
            logger.info(f"⬆️ Scaled up {node_id}: {current_cores} → {target_cores} cores")
            
        elif action == "scale_down":
            # Decrease node capacity
            node = self.nodes[node_id]
            current_cores = node.capacity["cpu_cores"]
            target_cores = decision["target_capacity"]
            
            node.capacity["cpu_cores"] = target_cores
            node.capacity["memory_gb"] = target_cores * 4
            
            logger.info(f"⬇️ Scaled down {node_id}: {current_cores} → {target_cores} cores")
    
    async def _collect_global_metrics(self) -> Dict[str, Any]:
        """Collect global performance metrics."""
        
        metrics = {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.health_status == "healthy"]),
            "average_cpu_load": 0.0,
            "average_memory_load": 0.0,
            "total_deployed_models": 0,
            "regional_distribution": defaultdict(int)
        }
        
        total_cpu_load = 0
        total_memory_load = 0
        healthy_count = 0
        
        for node in self.nodes.values():
            if node.health_status == "healthy":
                total_cpu_load += node.current_load.get("cpu", 0)
                total_memory_load += node.current_load.get("memory", 0)
                healthy_count += 1
                
            metrics["total_deployed_models"] += len(node.deployed_models)
            metrics["regional_distribution"][node.region.code] += 1
        
        if healthy_count > 0:
            metrics["average_cpu_load"] = total_cpu_load / healthy_count
            metrics["average_memory_load"] = total_memory_load / healthy_count
        
        return metrics
    
    async def _identify_optimizations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        
        optimizations = []
        
        # Check if average load is too high
        if metrics["average_cpu_load"] > 0.8:
            optimizations.append({
                "type": "load_balancing",
                "priority": "high",
                "action": "redistribute_load"
            })
        
        # Check if too many nodes in one region
        max_nodes_per_region = max(metrics["regional_distribution"].values())
        total_nodes = metrics["total_nodes"]
        
        if max_nodes_per_region / total_nodes > 0.5:
            optimizations.append({
                "type": "geographic_balancing",
                "priority": "medium",
                "action": "redistribute_nodes"
            })
        
        return optimizations
    
    async def _apply_optimization(self, optimization: Dict[str, Any]):
        """Apply performance optimization."""
        
        opt_type = optimization["type"]
        
        if opt_type == "load_balancing":
            # Implement load redistribution
            logger.info("⚖️ Applying load balancing optimization")
            
        elif opt_type == "geographic_balancing":
            # Implement geographic redistribution
            logger.info("🌍 Applying geographic balancing optimization")
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        
        status = {
            "infrastructure": {
                "total_nodes": len(self.nodes),
                "healthy_nodes": len([n for n in self.nodes.values() if n.health_status == "healthy"]),
                "regions": len(self.config.regions)
            },
            "deployments": {
                "total_deployments": len(self.deployment_history),
                "recent_success_rate": 0.0
            },
            "performance": {
                "global_availability": 0.0,
                "average_latency_ms": 0.0
            },
            "compliance": {
                "regions_compliant": len(self.config.regions),
                "data_sovereignty_enforced": True
            }
        }
        
        # Calculate recent success rate
        if self.deployment_history:
            recent_deployments = list(self.deployment_history)[-10:]
            successful = sum(1 for d in recent_deployments 
                           if len(d["results"]["successful_deployments"]) > 0)
            status["deployments"]["recent_success_rate"] = successful / len(recent_deployments)
        
        return status


# Factory function for easy deployment engine creation
def create_planetary_deployment_engine(
    target_regions: List[str] = None,
    availability_target: float = 0.9999,
    enable_compliance: bool = True
) -> PlanetaryDeploymentEngine:
    """Create configured planetary deployment engine."""
    
    if target_regions is None:
        target_regions = ["US-EAST", "US-WEST", "EU-WEST", "ASIA-PACIFIC"]
    
    # Define standard regions
    region_definitions = {
        "US-EAST": GeographicRegion("US East", "US-EAST", 39.0, -77.0, 
                                   ["CCPA"], ["CCPA"]),
        "US-WEST": GeographicRegion("US West", "US-WEST", 37.0, -122.0, 
                                   ["CCPA"], ["CCPA"]),
        "EU-WEST": GeographicRegion("EU West", "EU-WEST", 51.5, 0.0, 
                                   ["GDPR"], ["GDPR"]),
        "ASIA-PACIFIC": GeographicRegion("Asia Pacific", "ASIA-PACIFIC", 1.3, 103.8, 
                                       ["PDPA"], ["PDPA"])
    }
    
    regions = [region_definitions[code] for code in target_regions 
               if code in region_definitions]
    
    config = GlobalDeploymentConfig(
        regions=regions,
        availability_target=availability_target,
        data_residency_compliance=enable_compliance
    )
    
    return PlanetaryDeploymentEngine(config)


# Example usage
async def example_planetary_deployment():
    """Example of planetary scale deployment."""
    
    # Create deployment engine
    engine = create_planetary_deployment_engine(
        target_regions=["US-EAST", "US-WEST", "EU-WEST", "ASIA-PACIFIC"],
        availability_target=0.9999
    )
    
    # Initialize global infrastructure
    init_result = await engine.initialize_global_infrastructure()
    logger.info(f"🌍 Infrastructure initialized: {init_result}")
    
    # Start autonomous management
    await engine.start_autonomous_management()
    
    # Deploy a model globally
    model_info = {
        "id": "llama2-7b",
        "size_mb": 7000,
        "data_types": ["text", "embeddings"],
        "performance_requirements": {
            "max_latency_ms": 100,
            "min_throughput": 500
        }
    }
    
    deployment_result = await engine.deploy_model_globally(
        model_info, deployment_strategy="adaptive"
    )
    
    logger.info(f"🚀 Deployment result: {deployment_result}")
    
    # Monitor for a while
    await asyncio.sleep(30)
    
    # Get final status
    status = engine.get_global_status()
    logger.info(f"📊 Final status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_planetary_deployment())