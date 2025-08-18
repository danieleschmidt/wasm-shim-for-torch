"""Hyperdimensional Caching System for WASM-Torch

Advanced multi-dimensional caching with predictive algorithms, quantum-inspired
optimization, and autonomous cache management for maximum performance.
"""

import asyncio
import time
import hashlib
import pickle
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, OrderedDict
import heapq
import statistics
from abc import ABC, abstractmethod
import sys
import gc
import weakref

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_ULTRA_FAST = 1      # In-memory, ultra-fast access
    L2_FAST = 2            # In-memory, fast access
    L3_MEDIUM = 3          # Compressed in-memory
    L4_SLOW = 4            # Persistent storage
    L5_ARCHIVE = 5         # Long-term archive

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    ARC = "arc"                    # Adaptive Replacement Cache
    PREDICTIVE = "predictive"      # ML-based prediction
    QUANTUM_OPTIMAL = "quantum"    # Quantum-inspired optimization
    HYPERDIMENSIONAL = "hyperdim"  # Hyperdimensional optimization

@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: int = 5  # 1-10, 10 being highest
    access_pattern: List[float] = field(default_factory=list)
    prediction_score: float = 0.0
    quantum_coherence: float = 1.0
    dimensional_coordinates: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate memory size of cached value"""
        try:
            return sys.getsizeof(pickle.dumps(self.value))
        except:
            return sys.getsizeof(str(self.value))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return time.time() > self.last_access + self.ttl
    
    def access(self):
        """Record cache access"""
        current_time = time.time()
        self.access_count += 1
        self.access_pattern.append(current_time)
        self.last_access = current_time
        
        # Keep only recent access pattern (last 100 accesses)
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    bytes_stored: int = 0
    bytes_evicted: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate

class CachePredictor(ABC):
    """Base class for cache access predictors"""
    
    @abstractmethod
    def predict_access_probability(self, key: str, entry: CacheEntry) -> float:
        """Predict probability of cache entry being accessed"""
        pass
    
    @abstractmethod
    def update_model(self, access_patterns: Dict[str, List[float]]):
        """Update prediction model with new access patterns"""
        pass

class MLCachePredictor(CachePredictor):
    """Machine learning-based cache predictor"""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[float]] = {}
        self.model_weights: Dict[str, float] = {
            "recency": 0.4,
            "frequency": 0.3,
            "periodicity": 0.2,
            "trend": 0.1
        }
    
    def predict_access_probability(self, key: str, entry: CacheEntry) -> float:
        """Predict access probability using ML features"""
        if len(entry.access_pattern) < 2:
            return 0.5  # Default probability
        
        current_time = time.time()
        
        # Recency score (exponential decay)
        recency_score = np.exp(-(current_time - entry.last_access) / 3600)  # 1-hour half-life
        
        # Frequency score (normalized access count)
        frequency_score = min(1.0, entry.access_count / 100.0)
        
        # Periodicity score (detect periodic access patterns)
        periodicity_score = self._calculate_periodicity_score(entry.access_pattern)
        
        # Trend score (access rate trend)
        trend_score = self._calculate_trend_score(entry.access_pattern)
        
        # Weighted combination
        probability = (
            self.model_weights["recency"] * recency_score +
            self.model_weights["frequency"] * frequency_score +
            self.model_weights["periodicity"] * periodicity_score +
            self.model_weights["trend"] * trend_score
        )
        
        return min(1.0, max(0.0, probability))
    
    def _calculate_periodicity_score(self, access_pattern: List[float]) -> float:
        """Calculate periodicity in access pattern"""
        if len(access_pattern) < 5:
            return 0.0
        
        # Calculate time intervals between accesses
        intervals = [access_pattern[i] - access_pattern[i-1] 
                    for i in range(1, len(access_pattern))]
        
        if not intervals:
            return 0.0
        
        # Check for periodic patterns (coefficient of variation)
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0
        
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        cv = std_interval / mean_interval
        
        # Lower coefficient of variation indicates more periodic access
        periodicity_score = max(0.0, 1.0 - cv)
        return periodicity_score
    
    def _calculate_trend_score(self, access_pattern: List[float]) -> float:
        """Calculate access trend (increasing/decreasing frequency)"""
        if len(access_pattern) < 10:
            return 0.5
        
        # Split into two halves and compare access rates
        mid_point = len(access_pattern) // 2
        first_half = access_pattern[:mid_point]
        second_half = access_pattern[mid_point:]
        
        if not first_half or not second_half:
            return 0.5
        
        first_duration = first_half[-1] - first_half[0] if len(first_half) > 1 else 1
        second_duration = second_half[-1] - second_half[0] if len(second_half) > 1 else 1
        
        first_rate = len(first_half) / max(first_duration, 1)
        second_rate = len(second_half) / max(second_duration, 1)
        
        if first_rate == 0:
            return 0.5
        
        # Positive trend if second half has higher access rate
        trend_ratio = second_rate / first_rate
        trend_score = min(1.0, max(0.0, (trend_ratio - 0.5) * 2))
        
        return trend_score
    
    def update_model(self, access_patterns: Dict[str, List[float]]):
        """Update ML model with new access patterns"""
        self.access_patterns.update(access_patterns)
        
        # Simple adaptive weight adjustment based on prediction accuracy
        # In a real implementation, this would use proper ML algorithms

class QuantumCacheOptimizer:
    """Quantum-inspired cache optimization"""
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.quantum_state = np.random.normal(0, 1, dimension) + 1j * np.random.normal(0, 1, dimension)
        self.entanglement_matrix = np.eye(dimension, dtype=complex)
        
    def optimize_cache_layout(self, cache_entries: Dict[str, CacheEntry]) -> Dict[str, float]:
        """Use quantum-inspired optimization for cache layout"""
        if not cache_entries:
            return {}
        
        # Map cache entries to quantum state space
        entry_states = {}
        optimization_scores = {}
        
        for key, entry in cache_entries.items():
            # Create quantum state representation
            state_vector = self._create_quantum_state(entry)
            entry_states[key] = state_vector
            
            # Calculate quantum optimization score
            optimization_scores[key] = self._calculate_quantum_score(state_vector, entry)
        
        return optimization_scores
    
    def _create_quantum_state(self, entry: CacheEntry) -> np.ndarray:
        """Create quantum state representation of cache entry"""
        # Initialize state vector
        state = np.zeros(self.dimension, dtype=complex)
        
        # Encode entry properties into quantum state
        # Access frequency
        freq_component = min(1.0, entry.access_count / 100.0)
        state[:20] = freq_component * np.exp(1j * np.random.uniform(0, 2*np.pi, 20))
        
        # Recency
        current_time = time.time()
        recency_component = np.exp(-(current_time - entry.last_access) / 3600)
        state[20:40] = recency_component * np.exp(1j * np.random.uniform(0, 2*np.pi, 20))
        
        # Size (inverse - smaller is better)
        size_component = 1.0 / (1.0 + entry.size_bytes / 1024)  # Normalize by KB
        state[40:60] = size_component * np.exp(1j * np.random.uniform(0, 2*np.pi, 20))
        
        # Priority
        priority_component = entry.priority / 10.0
        state[60:80] = priority_component * np.exp(1j * np.random.uniform(0, 2*np.pi, 20))
        
        # Random quantum noise
        state[80:] = 0.1 * np.random.normal(0, 1, 20) + 1j * 0.1 * np.random.normal(0, 1, 20)
        
        return state
    
    def _calculate_quantum_score(self, state_vector: np.ndarray, entry: CacheEntry) -> float:
        """Calculate quantum optimization score"""
        # Quantum amplitude gives optimization score
        amplitude = np.abs(state_vector)
        
        # Apply quantum entanglement effects
        entangled_amplitude = np.abs(self.entanglement_matrix @ state_vector)
        
        # Calculate composite score
        score = np.mean(amplitude) + 0.3 * np.mean(entangled_amplitude)
        
        # Apply quantum coherence factor
        score *= entry.quantum_coherence
        
        return float(score)

class HyperdimensionalCache:
    """Hyperdimensional cache with multi-level hierarchy"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 512,
                 policy: CachePolicy = CachePolicy.HYPERDIMENSIONAL,
                 enable_prediction: bool = True,
                 enable_quantum_optimization: bool = True):
        
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.enable_prediction = enable_prediction
        self.enable_quantum_optimization = enable_quantum_optimization
        
        # Multi-level cache storage
        self.cache_levels: Dict[CacheLevel, Dict[str, CacheEntry]] = {
            level: {} for level in CacheLevel
        }
        
        # Cache management
        self.stats = CacheStats()
        self.predictor = MLCachePredictor() if enable_prediction else None
        self.quantum_optimizer = QuantumCacheOptimizer() if enable_quantum_optimization else None
        
        # Access tracking
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = defaultdict(int)  # For LFU
        self.access_heap: List[Tuple[float, str]] = []  # For priority-based eviction
        
        # Threading
        self._lock = threading.RLock()
        
        # Background tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_optimizing = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HyperdimensionalCache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent retrieval"""
        with self._lock:
            self.stats.total_requests += 1
            
            # Search through cache levels
            for level in CacheLevel:
                if key in self.cache_levels[level]:
                    entry = self.cache_levels[level][key]
                    
                    # Check expiration
                    if entry.is_expired():
                        await self._remove_entry(key, level)
                        continue
                    
                    # Record access
                    entry.access()
                    self._update_access_tracking(key)
                    
                    # Promote to higher cache level if beneficial
                    await self._consider_promotion(key, entry, level)
                    
                    self.stats.hits += 1
                    self.logger.debug(f"Cache hit for key {key} at level {level.name}")
                    
                    return entry.value
            
            # Cache miss
            self.stats.misses += 1
            self.logger.debug(f"Cache miss for key {key}")
            return None
    
    async def put(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[float] = None,
                  priority: int = 5) -> bool:
        """Put value in cache with intelligent placement"""
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                priority=priority
            )
            
            # Determine optimal cache level
            target_level = await self._determine_optimal_level(entry)
            
            # Check capacity and evict if necessary
            await self._ensure_capacity(target_level, entry.size_bytes)
            
            # Store entry
            self.cache_levels[target_level][key] = entry
            self._update_access_tracking(key)
            
            # Update statistics
            self.stats.bytes_stored += entry.size_bytes
            
            self.logger.debug(f"Cached key {key} at level {target_level.name}")
            return True
    
    async def _determine_optimal_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine optimal cache level for entry"""
        # High priority or small size -> L1
        if entry.priority >= 8 or entry.size_bytes < 1024:  # < 1KB
            return CacheLevel.L1_ULTRA_FAST
        
        # Medium priority and size -> L2
        if entry.priority >= 6 or entry.size_bytes < 10240:  # < 10KB
            return CacheLevel.L2_FAST
        
        # Large but frequently accessed -> L3
        if entry.priority >= 4:
            return CacheLevel.L3_MEDIUM
        
        # Default to L4
        return CacheLevel.L4_SLOW
    
    async def _consider_promotion(self, key: str, entry: CacheEntry, current_level: CacheLevel):
        """Consider promoting entry to higher cache level"""
        if current_level == CacheLevel.L1_ULTRA_FAST:
            return  # Already at highest level
        
        # Calculate promotion score
        promotion_score = self._calculate_promotion_score(entry)
        
        # Promote if score is high enough
        if promotion_score > 0.7:
            target_level = CacheLevel(current_level.value - 1)
            
            # Check capacity at target level
            target_cache = self.cache_levels[target_level]
            if len(target_cache) < self.max_size // (2 ** target_level.value):
                # Move entry to higher level
                del self.cache_levels[current_level][key]
                self.cache_levels[target_level][key] = entry
                self.logger.debug(f"Promoted key {key} from {current_level.name} to {target_level.name}")
    
    def _calculate_promotion_score(self, entry: CacheEntry) -> float:
        """Calculate score for promoting cache entry"""
        current_time = time.time()
        
        # Recency score
        recency_score = np.exp(-(current_time - entry.last_access) / 1800)  # 30-min half-life
        
        # Frequency score
        frequency_score = min(1.0, entry.access_count / 50.0)
        
        # Priority score
        priority_score = entry.priority / 10.0
        
        # Prediction score (if available)
        prediction_score = 0.0
        if self.predictor:
            prediction_score = self.predictor.predict_access_probability("", entry)
        
        # Quantum score (if available)
        quantum_score = 0.0
        if self.quantum_optimizer:
            quantum_entries = {"temp": entry}
            quantum_scores = self.quantum_optimizer.optimize_cache_layout(quantum_entries)
            quantum_score = quantum_scores.get("temp", 0.0)
        
        # Weighted combination
        total_score = (
            0.3 * recency_score +
            0.25 * frequency_score +
            0.2 * priority_score +
            0.15 * prediction_score +
            0.1 * quantum_score
        )
        
        return total_score
    
    async def _ensure_capacity(self, level: CacheLevel, required_bytes: int):
        """Ensure sufficient capacity at cache level"""
        cache = self.cache_levels[level]
        level_max_size = self.max_size // (2 ** level.value)
        level_max_bytes = self.max_memory_bytes // (2 ** level.value)
        
        # Calculate current usage
        current_bytes = sum(entry.size_bytes for entry in cache.values())
        
        # Evict entries if necessary
        while (len(cache) >= level_max_size or 
               current_bytes + required_bytes > level_max_bytes):
            
            if not cache:
                break
            
            # Select victim for eviction
            victim_key = await self._select_eviction_victim(level)
            if victim_key is None:
                break
            
            # Evict victim
            victim_entry = cache[victim_key]
            await self._remove_entry(victim_key, level)
            
            current_bytes -= victim_entry.size_bytes
            self.stats.evictions += 1
            self.stats.bytes_evicted += victim_entry.size_bytes
    
    async def _select_eviction_victim(self, level: CacheLevel) -> Optional[str]:
        """Select cache entry for eviction based on policy"""
        cache = self.cache_levels[level]
        
        if not cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            return self._select_lru_victim(cache)
        elif self.policy == CachePolicy.LFU:
            return self._select_lfu_victim(cache)
        elif self.policy == CachePolicy.PREDICTIVE and self.predictor:
            return await self._select_predictive_victim(cache)
        elif self.policy == CachePolicy.QUANTUM_OPTIMAL and self.quantum_optimizer:
            return await self._select_quantum_victim(cache)
        elif self.policy == CachePolicy.HYPERDIMENSIONAL:
            return await self._select_hyperdimensional_victim(cache)
        else:
            # Default to LRU
            return self._select_lru_victim(cache)
    
    def _select_lru_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """Select LRU victim"""
        if not cache:
            return None
        
        lru_key = min(cache.keys(), key=lambda k: cache[k].last_access)
        return lru_key
    
    def _select_lfu_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """Select LFU victim"""
        if not cache:
            return None
        
        lfu_key = min(cache.keys(), key=lambda k: cache[k].access_count)
        return lfu_key
    
    async def _select_predictive_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """Select victim using ML prediction"""
        if not cache:
            return None
        
        # Calculate prediction scores for all entries
        prediction_scores = {}
        for key, entry in cache.items():
            prediction_scores[key] = self.predictor.predict_access_probability(key, entry)
        
        # Select entry with lowest prediction score
        victim_key = min(prediction_scores.keys(), key=lambda k: prediction_scores[k])
        return victim_key
    
    async def _select_quantum_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """Select victim using quantum optimization"""
        if not cache:
            return None
        
        # Get quantum optimization scores
        quantum_scores = self.quantum_optimizer.optimize_cache_layout(cache)
        
        # Select entry with lowest quantum score
        victim_key = min(quantum_scores.keys(), key=lambda k: quantum_scores[k])
        return victim_key
    
    async def _select_hyperdimensional_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """Select victim using hyperdimensional optimization"""
        if not cache:
            return None
        
        # Combine multiple scoring methods
        composite_scores = {}
        
        for key, entry in cache.items():
            score = 0.0
            
            # Recency component
            current_time = time.time()
            recency_score = np.exp(-(current_time - entry.last_access) / 3600)
            score += 0.3 * recency_score
            
            # Frequency component
            frequency_score = min(1.0, entry.access_count / 100.0)
            score += 0.25 * frequency_score
            
            # Size component (inverse - smaller entries preferred)
            size_score = 1.0 / (1.0 + entry.size_bytes / 1024)
            score += 0.2 * size_score
            
            # Priority component
            priority_score = entry.priority / 10.0
            score += 0.15 * priority_score
            
            # Prediction component (if available)
            if self.predictor:
                prediction_score = self.predictor.predict_access_probability(key, entry)
                score += 0.1 * prediction_score
            
            composite_scores[key] = score
        
        # Select entry with lowest composite score
        victim_key = min(composite_scores.keys(), key=lambda k: composite_scores[k])
        return victim_key
    
    async def _remove_entry(self, key: str, level: CacheLevel):
        """Remove entry from cache"""
        cache = self.cache_levels[level]
        if key in cache:
            del cache[key]
        
        # Clean up tracking structures
        self.access_order.pop(key, None)
        self.access_frequency.pop(key, None)
    
    def _update_access_tracking(self, key: str):
        """Update access tracking structures"""
        # Update LRU order
        self.access_order.pop(key, None)
        self.access_order[key] = time.time()
        
        # Update frequency
        self.access_frequency[key] += 1
    
    async def start_optimization(self):
        """Start background optimization"""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Started hyperdimensional cache optimization")
    
    async def stop_optimization(self):
        """Stop background optimization"""
        self.is_optimizing = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped hyperdimensional cache optimization")
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while self.is_optimizing:
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                # Perform optimization
                await self._optimize_cache_layout()
                
                # Update prediction model
                if self.predictor:
                    await self._update_prediction_model()
                
                # Garbage collection
                await self._cleanup_expired_entries()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    async def _optimize_cache_layout(self):
        """Optimize cache layout using advanced algorithms"""
        with self._lock:
            # Collect all cache entries
            all_entries = {}
            for level, cache in self.cache_levels.items():
                all_entries.update(cache)
            
            if not all_entries:
                return
            
            # Use quantum optimization if available
            if self.quantum_optimizer:
                quantum_scores = self.quantum_optimizer.optimize_cache_layout(all_entries)
                
                # Redistribute entries based on quantum scores
                await self._redistribute_entries(all_entries, quantum_scores)
    
    async def _redistribute_entries(self, 
                                   entries: Dict[str, CacheEntry], 
                                   scores: Dict[str, float]):
        """Redistribute cache entries based on optimization scores"""
        # Sort entries by score (descending)
        sorted_entries = sorted(entries.items(), 
                              key=lambda x: scores.get(x[0], 0.0), 
                              reverse=True)
        
        # Clear current cache levels
        for level in CacheLevel:
            self.cache_levels[level].clear()
        
        # Redistribute entries
        level_capacities = {
            CacheLevel.L1_ULTRA_FAST: self.max_size // 8,
            CacheLevel.L2_FAST: self.max_size // 4,
            CacheLevel.L3_MEDIUM: self.max_size // 2,
            CacheLevel.L4_SLOW: self.max_size,
            CacheLevel.L5_ARCHIVE: self.max_size * 2
        }
        
        current_level = CacheLevel.L1_ULTRA_FAST
        current_count = 0
        
        for key, entry in sorted_entries:
            # Check if current level is full
            if current_count >= level_capacities[current_level]:
                # Move to next level
                current_level_value = current_level.value + 1
                if current_level_value <= 5:
                    current_level = CacheLevel(current_level_value)
                    current_count = 0
                else:
                    break  # No more levels
            
            # Place entry in current level
            self.cache_levels[current_level][key] = entry
            current_count += 1
    
    async def _update_prediction_model(self):
        """Update ML prediction model"""
        if not self.predictor:
            return
        
        # Collect access patterns from all cache levels
        access_patterns = {}
        for level, cache in self.cache_levels.items():
            for key, entry in cache.items():
                if len(entry.access_pattern) > 1:
                    access_patterns[key] = entry.access_pattern
        
        # Update model
        if access_patterns:
            self.predictor.update_model(access_patterns)
    
    async def _cleanup_expired_entries(self):
        """Remove expired cache entries"""
        with self._lock:
            expired_keys = []
            
            for level, cache in self.cache_levels.items():
                for key, entry in cache.items():
                    if entry.is_expired():
                        expired_keys.append((key, level))
            
            # Remove expired entries
            for key, level in expired_keys:
                await self._remove_entry(key, level)
                self.logger.debug(f"Removed expired entry: {key}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        with self._lock:
            # Calculate level-specific stats
            level_stats = {}
            total_entries = 0
            total_bytes = 0
            
            for level, cache in self.cache_levels.items():
                level_bytes = sum(entry.size_bytes for entry in cache.values())
                level_stats[level.name] = {
                    "entries": len(cache),
                    "bytes": level_bytes,
                    "avg_access_count": statistics.mean([entry.access_count for entry in cache.values()]) if cache else 0,
                    "avg_size": level_bytes / len(cache) if cache else 0
                }
                total_entries += len(cache)
                total_bytes += level_bytes
            
            return {
                "global_stats": {
                    "hit_rate": self.stats.hit_rate,
                    "miss_rate": self.stats.miss_rate,
                    "total_requests": self.stats.total_requests,
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "evictions": self.stats.evictions,
                    "bytes_stored": self.stats.bytes_stored,
                    "bytes_evicted": self.stats.bytes_evicted
                },
                "level_stats": level_stats,
                "capacity_utilization": {
                    "entries": total_entries / self.max_size if self.max_size > 0 else 0,
                    "memory": total_bytes / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
                },
                "configuration": {
                    "policy": self.policy.value,
                    "max_size": self.max_size,
                    "max_memory_mb": self.max_memory_bytes // (1024 * 1024),
                    "prediction_enabled": self.enable_prediction,
                    "quantum_optimization_enabled": self.enable_quantum_optimization
                }
            }

# Global hyperdimensional cache instance
_global_hyperdimensional_cache: Optional[HyperdimensionalCache] = None

def get_hyperdimensional_cache() -> HyperdimensionalCache:
    """Get global hyperdimensional cache instance"""
    global _global_hyperdimensional_cache
    if _global_hyperdimensional_cache is None:
        _global_hyperdimensional_cache = HyperdimensionalCache()
    return _global_hyperdimensional_cache

async def cached_inference(key: str, 
                          inference_func: Callable, 
                          *args, 
                          ttl: Optional[float] = None,
                          priority: int = 5,
                          **kwargs) -> Any:
    """Perform cached inference with hyperdimensional caching"""
    cache = get_hyperdimensional_cache()
    
    # Try to get from cache
    result = await cache.get(key)
    if result is not None:
        return result
    
    # Compute result
    if asyncio.iscoroutinefunction(inference_func):
        result = await inference_func(*args, **kwargs)
    else:
        result = inference_func(*args, **kwargs)
    
    # Store in cache
    await cache.put(key, result, ttl=ttl, priority=priority)
    
    return result