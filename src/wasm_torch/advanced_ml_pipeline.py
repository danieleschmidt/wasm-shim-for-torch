"""Advanced ML Pipeline - Generation 3: Scalable ML Operations

Advanced ML pipeline with model versioning, A/B testing, feature stores,
and intelligent model management for production-scale PyTorch-to-WASM systems.
"""

import asyncio
import time
import logging
import threading
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import uuid
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating" 
    STAGING = "staging"
    PRODUCTION = "production"
    SHADOW = "shadow"  # Shadow deployment for A/B testing
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


class ExperimentStatus(Enum):
    """A/B test experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version with metadata and lineage."""
    model_id: str
    version: str
    created_at: float = field(default_factory=time.time)
    status: ModelStatus = ModelStatus.STAGING
    model_path: Optional[str] = None
    model_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_info: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'created_at': self.created_at,
            'status': self.status.value,
            'model_path': self.model_path,
            'model_hash': self.model_hash,
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics,
            'training_info': self.training_info,
            'parent_version': self.parent_version,
            'tags': self.tags
        }


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    treatment_models: Dict[str, ModelVersion]  # treatment_name -> model_version
    traffic_allocation: Dict[str, float]  # treatment_name -> percentage (0-1)
    success_metrics: List[str]
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    max_duration_hours: float = 168  # 1 week default
    created_at: float = field(default_factory=time.time)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    def __post_init__(self):
        # Validate traffic allocation sums to 1.0
        total_allocation = sum(self.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")


@dataclass
class ExperimentResult:
    """Results from A/B test experiment."""
    experiment_id: str
    treatment_name: str
    metric_name: str
    sample_size: int
    mean_value: float
    std_deviation: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    is_significant: bool = False


class FeatureStore:
    """Feature store for ML pipeline with caching and versioning."""
    
    def __init__(self, cache_size: int = 10000, ttl_seconds: float = 3600):
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self._features: Dict[str, Dict[str, Any]] = {}
        self._feature_cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def register_feature_group(self, group_name: str, features: Dict[str, Any],
                              version: str = "1.0", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a group of features."""
        with self._lock:
            feature_key = f"{group_name}:{version}"
            self._features[feature_key] = {
                'group_name': group_name,
                'version': version,
                'features': features,
                'metadata': metadata or {},
                'registered_at': time.time()
            }
            logger.info(f"Registered feature group: {feature_key}")
    
    def get_features(self, group_name: str, feature_names: List[str],
                    version: str = "1.0", entity_id: Optional[str] = None) -> Dict[str, Any]:
        """Get features for inference."""
        cache_key = f"{group_name}:{version}:{entity_id}"
        current_time = time.time()
        
        with self._lock:
            # Check cache first
            if cache_key in self._feature_cache:
                cached_data = self._feature_cache[cache_key]
                if current_time - cached_data['cached_at'] < self.ttl_seconds:
                    self._access_times[cache_key] = current_time
                    return {name: cached_data['features'].get(name) for name in feature_names}
            
            # Get from feature store
            feature_key = f"{group_name}:{version}"
            if feature_key not in self._features:
                raise ValueError(f"Feature group not found: {feature_key}")
            
            feature_group = self._features[feature_key]
            
            # Simulate feature computation/retrieval
            result = {}
            for feature_name in feature_names:
                if feature_name in feature_group['features']:
                    # In real implementation, this might involve database queries,
                    # computations, or external API calls
                    result[feature_name] = feature_group['features'][feature_name]
                else:
                    result[feature_name] = None
            
            # Cache the result
            self._cache_features(cache_key, result, current_time)
            
            return result
    
    def _cache_features(self, cache_key: str, features: Dict[str, Any], cached_at: float) -> None:
        """Cache features with LRU eviction."""
        # Evict if cache is full
        if len(self._feature_cache) >= self.cache_size:
            # Remove least recently used
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            self._feature_cache.pop(lru_key, None)
            self._access_times.pop(lru_key, None)
        
        self._feature_cache[cache_key] = {
            'features': features,
            'cached_at': cached_at
        }
        self._access_times[cache_key] = cached_at
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        with self._lock:
            return {
                'total_feature_groups': len(self._features),
                'cache_size': len(self._feature_cache),
                'cache_utilization': len(self._feature_cache) / self.cache_size,
                'feature_groups': list(self._features.keys())
            }


class ModelRegistry:
    """Model registry with versioning and lineage tracking."""
    
    def __init__(self):
        self._models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)  # model_id -> version -> ModelVersion
        self._production_models: Dict[str, str] = {}  # model_id -> production_version
        self._model_lineage: Dict[str, List[str]] = defaultdict(list)  # parent -> children versions
        self._lock = threading.RLock()
    
    def register_model(self, model_version: ModelVersion) -> None:
        """Register a new model version."""
        with self._lock:
            # Check if version already exists
            if model_version.version in self._models[model_version.model_id]:
                raise ValueError(f"Model version already exists: {model_version.model_id}:{model_version.version}")
            
            # Calculate model hash if not provided
            if not model_version.model_hash and model_version.model_path:
                model_version.model_hash = self._calculate_model_hash(model_version.model_path)
            
            # Store model version
            self._models[model_version.model_id][model_version.version] = model_version
            
            # Update lineage
            if model_version.parent_version:
                self._model_lineage[f"{model_version.model_id}:{model_version.parent_version}"].append(
                    model_version.version
                )
            
            logger.info(f"Registered model: {model_version.model_id}:{model_version.version}")
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Get a specific model version."""
        with self._lock:
            if model_id not in self._models:
                return None
            
            if version is None:
                # Get production version
                version = self._production_models.get(model_id)
                if version is None:
                    return None
            
            return self._models[model_id].get(version)
    
    def list_model_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        with self._lock:
            if model_id not in self._models:
                return []
            
            return list(self._models[model_id].values())
    
    def promote_to_production(self, model_id: str, version: str) -> bool:
        """Promote a model version to production."""
        with self._lock:
            if model_id not in self._models or version not in self._models[model_id]:
                return False
            
            model_version = self._models[model_id][version]
            model_version.status = ModelStatus.PRODUCTION
            self._production_models[model_id] = version
            
            logger.info(f"Promoted to production: {model_id}:{version}")
            return True
    
    def get_model_lineage(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get model lineage information."""
        with self._lock:
            model_key = f"{model_id}:{version}"
            
            # Find parent
            parent = None
            model_version = self.get_model(model_id, version)
            if model_version and model_version.parent_version:
                parent = f"{model_id}:{model_version.parent_version}"
            
            # Find children
            children = self._model_lineage.get(model_key, [])
            
            return {
                'model': model_key,
                'parent': parent,
                'children': [f"{model_id}:{child_version}" for child_version in children],
                'depth': self._calculate_lineage_depth(model_id, version)
            }
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model file."""
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
                return hashlib.sha256(model_data).hexdigest()
        except Exception:
            return hashlib.md5(f"{model_path}_{time.time()}".encode()).hexdigest()
    
    def _calculate_lineage_depth(self, model_id: str, version: str) -> int:
        """Calculate depth in lineage tree."""
        model_version = self.get_model(model_id, version)
        if not model_version or not model_version.parent_version:
            return 0
        
        return 1 + self._calculate_lineage_depth(model_id, model_version.parent_version)


class ABTestManager:
    """A/B testing manager for model experiments."""
    
    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self._traffic_assignments: Dict[str, str] = {}  # request_id -> treatment_name
        self._lock = threading.RLock()
    
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create a new A/B test experiment."""
        with self._lock:
            if config.experiment_id in self._experiments:
                raise ValueError(f"Experiment already exists: {config.experiment_id}")
            
            # Validate models exist
            for treatment_name, model_version in config.treatment_models.items():
                registry_model = self.model_registry.get_model(model_version.model_id, model_version.version)
                if not registry_model:
                    raise ValueError(f"Model not found: {model_version.model_id}:{model_version.version}")
            
            self._experiments[config.experiment_id] = config
            logger.info(f"Created experiment: {config.experiment_id}")
            return True
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            
            experiment = self._experiments[experiment_id]
            if experiment.status != ExperimentStatus.DRAFT:
                return False
            
            experiment.status = ExperimentStatus.RUNNING
            logger.info(f"Started experiment: {experiment_id}")
            return True
    
    def assign_treatment(self, request_id: str, experiment_id: str, 
                        user_features: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Assign a treatment for a request in an experiment."""
        with self._lock:
            if experiment_id not in self._experiments:
                return None
            
            experiment = self._experiments[experiment_id]
            if experiment.status != ExperimentStatus.RUNNING:
                return None
            
            # Simple random assignment based on traffic allocation
            # In production, this might use user features for stratified sampling
            rand_value = random.random()
            cumulative_prob = 0.0
            
            for treatment_name, allocation in experiment.traffic_allocation.items():
                cumulative_prob += allocation
                if rand_value <= cumulative_prob:
                    self._traffic_assignments[request_id] = treatment_name
                    return treatment_name
            
            # Fallback to first treatment
            first_treatment = list(experiment.treatment_models.keys())[0]
            self._traffic_assignments[request_id] = first_treatment
            return first_treatment
    
    def get_model_for_request(self, request_id: str, experiment_id: str) -> Optional[ModelVersion]:
        """Get the assigned model for a request."""
        with self._lock:
            if experiment_id not in self._experiments:
                return None
            
            treatment_name = self._traffic_assignments.get(request_id)
            if not treatment_name:
                return None
            
            experiment = self._experiments[experiment_id]
            return experiment.treatment_models.get(treatment_name)
    
    def record_metric(self, request_id: str, experiment_id: str, 
                     metric_name: str, value: float) -> None:
        """Record a metric observation for analysis."""
        with self._lock:
            if experiment_id not in self._experiments:
                return
            
            treatment_name = self._traffic_assignments.get(request_id)
            if not treatment_name:
                return
            
            # Store metric for later analysis
            # In production, this would likely go to a metrics database
            metric_key = f"{experiment_id}:{treatment_name}:{metric_name}"
            if not hasattr(self, '_metric_observations'):
                self._metric_observations = defaultdict(list)
            
            self._metric_observations[metric_key].append({
                'request_id': request_id,
                'value': value,
                'timestamp': time.time()
            })
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, List[ExperimentResult]]:
        """Analyze experiment results and determine significance."""
        with self._lock:
            if experiment_id not in self._experiments:
                return {}
            
            experiment = self._experiments[experiment_id]
            results = {}
            
            # Analyze each success metric
            for metric_name in experiment.success_metrics:
                metric_results = []
                
                # Get results for each treatment
                treatment_stats = {}
                for treatment_name in experiment.treatment_models.keys():
                    metric_key = f"{experiment_id}:{treatment_name}:{metric_name}"
                    observations = getattr(self, '_metric_observations', {}).get(metric_key, [])
                    
                    if observations:
                        values = [obs['value'] for obs in observations]
                        mean_val = sum(values) / len(values)
                        
                        # Simple std calculation
                        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                        std_dev = variance ** 0.5
                        
                        # Simple confidence interval (assuming normal distribution)
                        margin_of_error = 1.96 * (std_dev / (len(values) ** 0.5))  # 95% CI
                        ci_lower = mean_val - margin_of_error
                        ci_upper = mean_val + margin_of_error
                        
                        treatment_stats[treatment_name] = {
                            'sample_size': len(values),
                            'mean': mean_val,
                            'std': std_dev,
                            'ci': (ci_lower, ci_upper)
                        }
                        
                        metric_results.append(ExperimentResult(
                            experiment_id=experiment_id,
                            treatment_name=treatment_name,
                            metric_name=metric_name,
                            sample_size=len(values),
                            mean_value=mean_val,
                            std_deviation=std_dev,
                            confidence_interval=(ci_lower, ci_upper)
                        ))
                
                # Simple significance testing (compare first two treatments)
                if len(treatment_stats) >= 2:
                    treatments = list(treatment_stats.keys())
                    treatment_a_stats = treatment_stats[treatments[0]]
                    treatment_b_stats = treatment_stats[treatments[1]]
                    
                    # Simple t-test approximation
                    if (treatment_a_stats['sample_size'] >= experiment.minimum_sample_size and
                        treatment_b_stats['sample_size'] >= experiment.minimum_sample_size):
                        
                        # Check if confidence intervals don't overlap (simple significance test)
                        a_ci = treatment_a_stats['ci']
                        b_ci = treatment_b_stats['ci']
                        
                        is_significant = (a_ci[1] < b_ci[0] or b_ci[1] < a_ci[0])
                        
                        # Update significance in results
                        for result in metric_results:
                            result.is_significant = is_significant
                
                results[metric_name] = metric_results
            
            return results
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment status."""
        with self._lock:
            if experiment_id not in self._experiments:
                return {}
            
            experiment = self._experiments[experiment_id]
            
            # Count assignments per treatment
            treatment_counts = defaultdict(int)
            for request_id, treatment in self._traffic_assignments.items():
                treatment_counts[treatment] += 1
            
            return {
                'experiment_id': experiment_id,
                'name': experiment.name,
                'status': experiment.status.value,
                'created_at': experiment.created_at,
                'treatments': list(experiment.treatment_models.keys()),
                'traffic_allocation': experiment.traffic_allocation,
                'treatment_assignments': dict(treatment_counts),
                'total_assignments': sum(treatment_counts.values()),
                'duration_hours': (time.time() - experiment.created_at) / 3600
            }


# Demo function
async def demo_advanced_ml_pipeline():
    """Demonstrate advanced ML pipeline capabilities."""
    
    print("Advanced ML Pipeline Demo - Generation 3")
    print("=" * 50)
    
    # Create components
    feature_store = FeatureStore()
    model_registry = ModelRegistry()
    ab_test_manager = ABTestManager(feature_store, model_registry)
    
    print("✓ Created ML pipeline components")
    
    # Register features
    print("\\nRegistering feature groups...")
    user_features = {
        'age': 25,
        'income': 50000,
        'location': 'urban',
        'usage_score': 0.75
    }
    
    item_features = {
        'category': 'electronics',
        'price': 299.99,
        'rating': 4.5,
        'popularity': 0.8
    }
    
    feature_store.register_feature_group('user_features', user_features, version='1.0')
    feature_store.register_feature_group('item_features', item_features, version='1.0')
    
    print("✓ Registered feature groups")
    
    # Register model versions
    print("\\nRegistering model versions...")
    
    # Create model versions
    model_v1 = ModelVersion(
        model_id='recommendation_model',
        version='1.0',
        status=ModelStatus.PRODUCTION,
        metadata={'algorithm': 'collaborative_filtering', 'accuracy': 0.85},
        performance_metrics={'auc': 0.85, 'precision': 0.82, 'recall': 0.78}
    )
    
    model_v2 = ModelVersion(
        model_id='recommendation_model',
        version='2.0',
        status=ModelStatus.STAGING,
        metadata={'algorithm': 'neural_collaborative_filtering', 'accuracy': 0.88},
        performance_metrics={'auc': 0.88, 'precision': 0.85, 'recall': 0.81},
        parent_version='1.0'
    )
    
    model_registry.register_model(model_v1)
    model_registry.register_model(model_v2)
    model_registry.promote_to_production('recommendation_model', '1.0')
    
    print("✓ Registered model versions")
    
    # Test feature retrieval
    print("\\nTesting feature retrieval...")
    features = feature_store.get_features('user_features', ['age', 'income'], entity_id='user_123')
    print(f"Retrieved features: {features}")
    
    # Test model lineage
    print("\\nTesting model lineage...")
    lineage = model_registry.get_model_lineage('recommendation_model', '2.0')
    print(f"Model lineage: {lineage}")
    
    # Create A/B test experiment
    print("\\nCreating A/B test experiment...")
    
    experiment_config = ExperimentConfig(
        experiment_id='recommendation_ab_test_001',
        name='Neural CF vs Collaborative Filtering',
        description='Compare neural collaborative filtering with traditional CF',
        treatment_models={
            'control': model_v1,
            'treatment': model_v2
        },
        traffic_allocation={
            'control': 0.5,
            'treatment': 0.5
        },
        success_metrics=['click_through_rate', 'conversion_rate'],
        minimum_sample_size=1000
    )
    
    ab_test_manager.create_experiment(experiment_config)
    ab_test_manager.start_experiment('recommendation_ab_test_001')
    
    print("✓ Created and started A/B test")
    
    # Simulate experiment traffic
    print("\\nSimulating experiment traffic...")
    
    for i in range(100):
        request_id = f"request_{i}"
        
        # Assign treatment
        treatment = ab_test_manager.assign_treatment(
            request_id, 
            'recommendation_ab_test_001'
        )
        
        # Simulate metrics
        if treatment == 'control':
            ctr = random.gauss(0.05, 0.01)  # 5% CTR
            conversion = random.gauss(0.02, 0.005)  # 2% conversion
        else:  # treatment
            ctr = random.gauss(0.065, 0.01)  # 6.5% CTR (better)
            conversion = random.gauss(0.025, 0.005)  # 2.5% conversion (better)
        
        ab_test_manager.record_metric(request_id, 'recommendation_ab_test_001', 'click_through_rate', max(0, ctr))
        ab_test_manager.record_metric(request_id, 'recommendation_ab_test_001', 'conversion_rate', max(0, conversion))
    
    print("✓ Simulated 100 requests")
    
    # Analyze experiment
    print("\\nAnalyzing experiment results...")
    results = ab_test_manager.analyze_experiment('recommendation_ab_test_001')
    
    for metric_name, metric_results in results.items():
        print(f"\\nMetric: {metric_name}")
        for result in metric_results:
            print(f"  {result.treatment_name}: {result.mean_value:.4f} "
                  f"(n={result.sample_size}, CI: {result.confidence_interval[0]:.4f}-{result.confidence_interval[1]:.4f})")
    
    # Show experiment status
    status = ab_test_manager.get_experiment_status('recommendation_ab_test_001')
    print(f"\\nExperiment Status:")
    print(f"  Status: {status['status']}")
    print(f"  Total assignments: {status['total_assignments']}")
    print(f"  Treatment distribution: {status['treatment_assignments']}")
    print(f"  Duration: {status['duration_hours']:.2f} hours")
    
    # Show feature store stats
    feature_stats = feature_store.get_feature_statistics()
    print(f"\\nFeature Store Statistics:")
    print(f"  Feature groups: {feature_stats['total_feature_groups']}")
    print(f"  Cache utilization: {feature_stats['cache_utilization']:.2%}")
    
    print("\\n🧪 Advanced ML Pipeline Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_advanced_ml_pipeline())