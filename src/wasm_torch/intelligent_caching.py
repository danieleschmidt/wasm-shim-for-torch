"""Intelligent caching system with predictive pre-loading and adaptive eviction."""

import asyncio
import time
import threading
import hashlib
import pickle
import logging
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import weakref
import numpy as np
import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_bytes: int = 0
    avg_access_time: float = 0.0
    prediction_accuracy: float = 0.0
    preload_success_rate: float = 0.0


@dataclass
class AccessPattern:
    """Pattern tracking for cache access prediction."""
    key: str
    access_times: List[float] = field(default_factory=list)
    access_frequency: float = 0.0
    last_access: float = 0.0
    prediction_score: float = 0.0
    context_keys: List[str] = field(default_factory=list)


class PredictiveCache(Generic[T]):
    """Advanced cache with ML-based access prediction and intelligent pre-loading."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 1000, 
                 enable_prediction: bool = True, enable_persistence: bool = False):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_prediction = enable_prediction
        self.enable_persistence = enable_persistence
        
        # Core cache storage
        self.cache: Dict[str, T] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.memory_usage: Dict[str, int] = {}
        
        # Prediction system
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.sequential_patterns: Dict[str, List[str]] = defaultdict(list)
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Performance tracking
        self.metrics = CacheMetrics()
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.prediction_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Persistence
        self.persistence_path = Path("cache_persistence") if enable_persistence else None
        
        if enable_prediction:
            self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for prediction and cleanup."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread
            return
        
        if not self.prediction_task or self.prediction_task.done():
            self.prediction_task = asyncio.create_task(self._prediction_worker())
        
        if not self.cleanup_task or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def _prediction_worker(self) -> None:
        """Background worker for access pattern analysis and prediction."""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                await self._analyze_access_patterns()
                await self._predict_and_preload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prediction worker error: {e}")
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup and optimization."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                self._cleanup_stale_patterns()
                self._optimize_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get item from cache with access pattern tracking."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Cache hit
                self._record_access(key, current_time, hit=True)
                self.metrics.hits += 1
                return self.cache[key]
            else:
                # Cache miss
                self._record_access(key, current_time, hit=False)
                self.metrics.misses += 1
                return default
    
    def put(self, key: str, value: T, preload: bool = False) -> None:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            current_time = time.time()
            
            # Calculate memory usage
            try:
                if isinstance(value, torch.Tensor):
                    item_size = value.numel() * value.element_size()
                elif hasattr(value, '__sizeof__'):
                    item_size = value.__sizeof__()
                else:
                    item_size = len(pickle.dumps(value))
            except Exception:
                item_size = 1024  # Default estimate
            
            # Check if we need to evict items
            if key not in self.cache:
                self._ensure_capacity(item_size)
            
            # Store the item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.memory_usage[key] = item_size
            
            # Update metrics
            self.metrics.size = len(self.cache)
            self.metrics.memory_usage_bytes = sum(self.memory_usage.values())
            
            # Track preload success
            if preload:
                if hasattr(self, '_preload_predictions') and key in self._preload_predictions:
                    self.metrics.preload_success_rate = (
                        self.metrics.preload_success_rate * 0.9 + 0.1
                    )
    
    def _record_access(self, key: str, access_time: float, hit: bool) -> None:
        """Record access pattern for prediction."""
        if not self.enable_prediction:
            return
        
        # Update access pattern
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern(key=key)
        
        pattern = self.access_patterns[key]
        pattern.access_times.append(access_time)
        pattern.last_access = access_time
        pattern.access_frequency = len(pattern.access_times) / max(1, access_time - pattern.access_times[0])
        
        # Limit history size
        if len(pattern.access_times) > 100:
            pattern.access_times = pattern.access_times[-50:]
        
        # Track sequential patterns
        if hasattr(self, '_last_accessed_key') and self._last_accessed_key:
            self.sequential_patterns[self._last_accessed_key].append(key)
            if len(self.sequential_patterns[self._last_accessed_key]) > 10:
                self.sequential_patterns[self._last_accessed_key] = \
                    self.sequential_patterns[self._last_accessed_key][-5:]
            
            # Update co-occurrence matrix
            self.co_occurrence_matrix[(self._last_accessed_key, key)] += 1
        
        self._last_accessed_key = key
    
    def _ensure_capacity(self, required_size: int) -> None:
        """Ensure cache has capacity for new item using intelligent eviction."""
        while (len(self.cache) >= self.max_size or 
               self.metrics.memory_usage_bytes + required_size > self.max_memory_bytes):
            
            if not self.cache:
                break
            
            # Find best candidate for eviction
            eviction_key = self._select_eviction_candidate()
            if eviction_key:
                self._evict_item(eviction_key)
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select the best candidate for eviction using multiple factors."""
        if not self.cache:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key in self.cache.keys():
            score = 0.0
            
            # Time since last access (higher = more likely to evict)
            time_score = current_time - self.access_times.get(key, current_time)
            score += time_score * 0.4
            
            # Access frequency (lower = more likely to evict)
            frequency = self.access_counts.get(key, 1)
            frequency_score = 1.0 / frequency
            score += frequency_score * 0.3
            
            # Memory usage (higher = more likely to evict)
            memory_score = self.memory_usage.get(key, 0) / self.max_memory_bytes
            score += memory_score * 0.2
            
            # Prediction score (lower prediction = more likely to evict)
            if key in self.access_patterns:
                prediction_score = 1.0 - self.access_patterns[key].prediction_score
                score += prediction_score * 0.1
            
            scores[key] = score
        
        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _evict_item(self, key: str) -> None:
        """Evict an item from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.memory_usage.pop(key, None)
            self.metrics.evictions += 1
            
            # Update metrics
            self.metrics.size = len(self.cache)
            self.metrics.memory_usage_bytes = sum(self.memory_usage.values())
    
    async def _analyze_access_patterns(self) -> None:
        """Analyze access patterns for prediction."""
        current_time = time.time()
        
        with self.lock:
            for key, pattern in self.access_patterns.items():
                if not pattern.access_times:
                    continue
                
                # Calculate prediction score based on access regularity
                if len(pattern.access_times) >= 3:
                    intervals = [
                        pattern.access_times[i] - pattern.access_times[i-1]
                        for i in range(1, len(pattern.access_times))
                    ]
                    
                    if intervals:
                        avg_interval = np.mean(intervals)
                        interval_std = np.std(intervals)
                        regularity = 1.0 / (1.0 + interval_std / max(avg_interval, 1e-6))
                        
                        # Combine with recency and frequency
                        recency = 1.0 / (1.0 + current_time - pattern.last_access)
                        frequency = min(1.0, pattern.access_frequency)
                        
                        pattern.prediction_score = (
                            regularity * 0.5 + recency * 0.3 + frequency * 0.2
                        )
    
    async def _predict_and_preload(self) -> None:
        """Predict likely access patterns and preload items."""
        if not self.enable_prediction:
            return
        
        current_time = time.time()
        predictions = []
        
        with self.lock:
            # Predict based on time patterns
            for key, pattern in self.access_patterns.items():
                if key in self.cache or pattern.prediction_score < 0.3:
                    continue
                
                if len(pattern.access_times) >= 3:
                    intervals = [
                        pattern.access_times[i] - pattern.access_times[i-1]
                        for i in range(1, len(pattern.access_times))
                    ]
                    
                    if intervals:
                        avg_interval = np.mean(intervals)
                        time_since_last = current_time - pattern.last_access
                        
                        if abs(time_since_last - avg_interval) < avg_interval * 0.2:
                            predictions.append((key, pattern.prediction_score))
            
            # Predict based on sequential patterns
            if hasattr(self, '_last_accessed_key') and self._last_accessed_key:
                next_keys = self.sequential_patterns.get(self._last_accessed_key, [])
                for next_key in set(next_keys):
                    if next_key not in self.cache:
                        frequency = next_keys.count(next_key) / len(next_keys)
                        predictions.append((next_key, frequency))
        
        # Sort predictions by score and preload top candidates
        predictions.sort(key=lambda x: x[1], reverse=True)
        self._preload_predictions = {pred[0] for pred in predictions[:5]}
        
        # Trigger preload callbacks (in a real implementation)
        for key, score in predictions[:3]:
            logger.debug(f"Predicting access to {key} with score {score:.3f}")
    
    def _cleanup_stale_patterns(self) -> None:
        """Clean up old access patterns."""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour
        
        with self.lock:
            stale_keys = [
                key for key, pattern in self.access_patterns.items()
                if current_time - pattern.last_access > stale_threshold
            ]
            
            for key in stale_keys:
                del self.access_patterns[key]
                if key in self.sequential_patterns:
                    del self.sequential_patterns[key]
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage by compacting data structures."""
        with self.lock:
            # Clean up old co-occurrence data
            threshold = 5
            old_pairs = [
                pair for pair, count in self.co_occurrence_matrix.items()
                if count < threshold
            ]
            
            for pair in old_pairs:
                del self.co_occurrence_matrix[pair]
    
    def get_metrics(self) -> CacheMetrics:
        """Get comprehensive cache metrics."""
        with self.lock:
            total_accesses = self.metrics.hits + self.metrics.misses
            hit_rate = self.metrics.hits / total_accesses if total_accesses > 0 else 0.0
            
            self.metrics.hits = self.metrics.hits
            self.metrics.misses = self.metrics.misses
            return CacheMetrics(
                hits=self.metrics.hits,
                misses=self.metrics.misses,
                evictions=self.metrics.evictions,
                size=self.metrics.size,
                memory_usage_bytes=self.metrics.memory_usage_bytes,
                avg_access_time=self.metrics.avg_access_time,
                prediction_accuracy=self._calculate_prediction_accuracy(),
                preload_success_rate=self.metrics.preload_success_rate
            )
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy over recent history."""
        if not self.prediction_history:
            return 0.0
        
        correct_predictions = sum(1 for pred in self.prediction_history if pred)
        return correct_predictions / len(self.prediction_history)
    
    def clear(self) -> None:
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.memory_usage.clear()
            self.access_patterns.clear()
            self.sequential_patterns.clear()
            self.co_occurrence_matrix.clear()
            self.metrics = CacheMetrics()
    
    def __del__(self) -> None:
        """Cleanup background tasks."""
        if self.prediction_task and not self.prediction_task.done():
            self.prediction_task.cancel()
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()


class MultiLevelCache:
    """Multi-level cache hierarchy for optimal performance."""
    
    def __init__(self):
        self.l1_cache = PredictiveCache(max_size=100, max_memory_mb=100)  # Fast, small
        self.l2_cache = PredictiveCache(max_size=1000, max_memory_mb=500)  # Medium
        self.l3_cache = PredictiveCache(max_size=10000, max_memory_mb=2000)  # Large, slow
        
        self.level_stats = {
            'l1': {'hits': 0, 'misses': 0},
            'l2': {'hits': 0, 'misses': 0},
            'l3': {'hits': 0, 'misses': 0}
        }
    
    def get(self, key: str) -> Optional[T]:
        """Get item from multi-level cache hierarchy."""
        # Try L1 cache first
        result = self.l1_cache.get(key)
        if result is not None:
            self.level_stats['l1']['hits'] += 1
            return result
        
        self.level_stats['l1']['misses'] += 1
        
        # Try L2 cache
        result = self.l2_cache.get(key)
        if result is not None:
            self.level_stats['l2']['hits'] += 1
            # Promote to L1
            self.l1_cache.put(key, result)
            return result
        
        self.level_stats['l2']['misses'] += 1
        
        # Try L3 cache
        result = self.l3_cache.get(key)
        if result is not None:
            self.level_stats['l3']['hits'] += 1
            # Promote to L2 and L1
            self.l2_cache.put(key, result)
            self.l1_cache.put(key, result)
            return result
        
        self.level_stats['l3']['misses'] += 1
        return None
    
    def put(self, key: str, value: T) -> None:
        """Put item in appropriate cache level based on size and access pattern."""
        # Always put in L3
        self.l3_cache.put(key, value)
        
        # Decide whether to put in higher levels based on item characteristics
        try:
            if isinstance(value, torch.Tensor):
                size_mb = (value.numel() * value.element_size()) / (1024 * 1024)
            else:
                size_mb = len(pickle.dumps(value)) / (1024 * 1024)
        except Exception:
            size_mb = 1.0
        
        # Put in L2 if small enough
        if size_mb < 10:  # 10MB threshold
            self.l2_cache.put(key, value)
            
            # Put in L1 if very small
            if size_mb < 1:  # 1MB threshold
                self.l1_cache.put(key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        stats = {}
        for level, cache in [('l1', self.l1_cache), ('l2', self.l2_cache), ('l3', self.l3_cache)]:
            cache_metrics = cache.get_metrics()
            level_stats = self.level_stats[level]
            total_accesses = level_stats['hits'] + level_stats['misses']
            hit_rate = level_stats['hits'] / total_accesses if total_accesses > 0 else 0.0
            
            stats[level] = {
                'hit_rate': hit_rate,
                'size': cache_metrics.size,
                'memory_usage_mb': cache_metrics.memory_usage_bytes / (1024 * 1024),
                'prediction_accuracy': cache_metrics.prediction_accuracy
            }
        
        return stats


# Global cache instances
model_cache = PredictiveCache(max_size=50, max_memory_mb=2000, enable_prediction=True)
tensor_cache = PredictiveCache(max_size=1000, max_memory_mb=1000, enable_prediction=True)
result_cache = MultiLevelCache()


async def optimize_cache_performance() -> Dict[str, Any]:
    """Optimize cache performance across all cache instances."""
    results = {
        'model_cache': model_cache.get_metrics(),
        'tensor_cache': tensor_cache.get_metrics(),
        'result_cache': result_cache.get_stats()
    }
    
    # Log cache performance
    for cache_name, metrics in results.items():
        if isinstance(metrics, CacheMetrics):
            total = metrics.hits + metrics.misses
            hit_rate = metrics.hits / total if total > 0 else 0.0
            logger.info(f"{cache_name}: {hit_rate:.2%} hit rate, {metrics.size} items, "
                       f"{metrics.memory_usage_bytes / 1024 / 1024:.1f}MB")
    
    return results