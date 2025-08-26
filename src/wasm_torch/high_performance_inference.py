"""High-performance inference engine with advanced optimization and scalability."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import weakref
import hashlib
import pickle
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    """High-performance inference job with optimization metadata."""
    job_id: str
    model_id: str
    input_data: Any
    batch_size: int = 1
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    optimization_level: str = "balanced"  # fast, balanced, accurate
    caching_enabled: bool = True
    preprocessing_pipeline: Optional[List[str]] = None
    postprocessing_pipeline: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """High-performance inference result with detailed metrics."""
    job_id: str
    output_data: Any
    latency_ms: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    cpu_utilization: float
    cache_hit: bool = False
    optimization_applied: List[str] = field(default_factory=list)
    processing_stages: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class IntelligentCachingSystem:
    """Advanced caching system with LRU, frequency-based eviction and predictive loading."""
    
    def __init__(self, max_memory_mb: int = 1024, max_entries: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
        self.total_memory_mb = 0.0
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_pressure_evictions': 0
        }
    
    def _generate_cache_key(self, model_id: str, input_hash: str, config_hash: str) -> str:
        """Generate cache key from model, input, and configuration."""
        return f"{model_id}:{input_hash}:{config_hash}"
    
    def _hash_input(self, input_data: Any) -> str:
        """Generate hash for input data."""
        if isinstance(input_data, np.ndarray):
            return hashlib.md5(input_data.tobytes()).hexdigest()[:16]
        else:
            return hashlib.md5(str(input_data).encode()).hexdigest()[:16]
    
    def get(self, model_id: str, input_data: Any, config: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available."""
        input_hash = self._hash_input(input_data)
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:16]
        cache_key = self._generate_cache_key(model_id, input_hash, config_hash)
        
        with self._lock:
            if cache_key in self.cache:
                # Update access statistics
                self.access_frequency[cache_key] += 1
                self.last_access[cache_key] = time.time()
                self.stats['hits'] += 1
                
                return self.cache[cache_key]['result']
            else:
                self.stats['misses'] += 1
                return None
    
    def put(
        self, 
        model_id: str, 
        input_data: Any, 
        config: Dict[str, Any], 
        result: Any,
        memory_mb: float = 0.0
    ) -> bool:
        """Store result in cache with intelligent eviction."""
        input_hash = self._hash_input(input_data)
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:16]
        cache_key = self._generate_cache_key(model_id, input_hash, config_hash)
        
        with self._lock:
            # Check if we need to evict entries
            self._evict_if_needed(memory_mb)
            
            # Store the result
            self.cache[cache_key] = {
                'result': result,
                'model_id': model_id,
                'stored_at': time.time()
            }
            
            self.access_frequency[cache_key] = 1
            self.last_access[cache_key] = time.time()
            self.memory_usage[cache_key] = memory_mb
            self.total_memory_mb += memory_mb
            
            return True
    
    def _evict_if_needed(self, new_entry_memory: float) -> None:
        """Evict entries if memory or count limits would be exceeded."""
        # Evict based on memory pressure
        while self.total_memory_mb + new_entry_memory > self.max_memory_mb and self.cache:
            victim_key = self._select_eviction_victim()
            self._evict_entry(victim_key)
            self.stats['memory_pressure_evictions'] += 1
        
        # Evict based on entry count
        while len(self.cache) >= self.max_entries and self.cache:
            victim_key = self._select_eviction_victim()
            self._evict_entry(victim_key)
            self.stats['evictions'] += 1
    
    def _select_eviction_victim(self) -> str:
        """Select entry to evict using LRU + frequency hybrid approach."""
        if not self.cache:
            return ""
        
        # Calculate eviction scores (lower is better)
        scores = {}
        current_time = time.time()
        
        for key in self.cache.keys():
            # Combine recency and frequency
            last_access_score = current_time - self.last_access[key]
            frequency_score = 1.0 / (self.access_frequency[key] + 1)
            memory_score = self.memory_usage[key] / 100  # Favor evicting large entries
            
            # Weighted combination
            scores[key] = (last_access_score * 0.4 + 
                          frequency_score * 0.4 + 
                          memory_score * 0.2)
        
        return max(scores, key=scores.get)
    
    def _evict_entry(self, cache_key: str) -> None:
        """Evict specific cache entry."""
        if cache_key in self.cache:
            self.total_memory_mb -= self.memory_usage.get(cache_key, 0)
            del self.cache[cache_key]
            del self.access_frequency[cache_key]
            del self.last_access[cache_key]
            del self.memory_usage[cache_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_entries': len(self.cache),
            'memory_usage_mb': self.total_memory_mb,
            'memory_utilization': self.total_memory_mb / self.max_memory_mb,
            'statistics': self.stats.copy()
        }


class AdaptiveLoadBalancer:
    """Load balancer that adapts to system performance and model characteristics."""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())
        self.worker_stats: Dict[int, Dict[str, Any]] = {
            i: {
                'active_jobs': 0,
                'total_jobs': 0,
                'avg_latency': 0.0,
                'load_factor': 0.0,
                'last_job_time': 0.0
            }
            for i in range(self.num_workers)
        }
        self.model_affinity: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def select_worker(self, job: InferenceJob) -> int:
        """Select optimal worker for job based on load and model affinity."""
        with self._lock:
            # Check for model affinity (workers that have processed this model before)
            preferred_workers = self.model_affinity.get(job.model_id, [])
            
            if preferred_workers:
                # Select least loaded worker among preferred ones
                available_preferred = [
                    w for w in preferred_workers 
                    if self.worker_stats[w]['active_jobs'] < 2
                ]
                
                if available_preferred:
                    return min(available_preferred, 
                              key=lambda w: self.worker_stats[w]['load_factor'])
            
            # Select globally least loaded worker
            return min(self.worker_stats.keys(),
                      key=lambda w: (
                          self.worker_stats[w]['active_jobs'],
                          self.worker_stats[w]['load_factor']
                      ))
    
    def record_job_start(self, worker_id: int, job: InferenceJob) -> None:
        """Record job start for load balancing."""
        with self._lock:
            stats = self.worker_stats[worker_id]
            stats['active_jobs'] += 1
            stats['last_job_time'] = time.time()
            
            # Update model affinity
            if worker_id not in self.model_affinity[job.model_id]:
                self.model_affinity[job.model_id].append(worker_id)
    
    def record_job_completion(self, worker_id: int, job: InferenceJob, latency: float) -> None:
        """Record job completion for performance tracking."""
        with self._lock:
            stats = self.worker_stats[worker_id]
            stats['active_jobs'] = max(0, stats['active_jobs'] - 1)
            stats['total_jobs'] += 1
            
            # Update average latency with exponential moving average
            alpha = 0.1  # Smoothing factor
            stats['avg_latency'] = (alpha * latency + 
                                   (1 - alpha) * stats['avg_latency'])
            
            # Calculate load factor
            stats['load_factor'] = stats['active_jobs'] + (stats['avg_latency'] / 1000.0)
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_active = sum(stats['active_jobs'] for stats in self.worker_stats.values())
            total_completed = sum(stats['total_jobs'] for stats in self.worker_stats.values())
            avg_latency = sum(stats['avg_latency'] for stats in self.worker_stats.values()) / len(self.worker_stats)
            
            return {
                'active_jobs': total_active,
                'completed_jobs': total_completed,
                'average_latency': avg_latency,
                'worker_stats': self.worker_stats.copy(),
                'model_affinity_count': len(self.model_affinity)
            }


class PerformanceOptimizer:
    """Optimizes inference performance based on model and system characteristics."""
    
    def __init__(self):
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.model_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def optimize_job(self, job: InferenceJob) -> Tuple[InferenceJob, List[str]]:
        """Apply optimizations to inference job."""
        optimizations_applied = []
        optimized_job = job
        
        with self._lock:
            # Get model profile
            profile = self.model_profiles.get(job.model_id, {})
            
            # Apply batch size optimization
            if job.optimization_level in ['balanced', 'fast']:
                optimal_batch_size = self._get_optimal_batch_size(job.model_id, profile)
                if optimal_batch_size > job.batch_size:
                    optimized_job.batch_size = optimal_batch_size
                    optimizations_applied.append('batch_size_optimization')
            
            # Apply preprocessing optimizations
            if job.optimization_level == 'fast':
                optimized_pipeline = self._optimize_preprocessing_pipeline(
                    job.preprocessing_pipeline or [], profile
                )
                if optimized_pipeline != (job.preprocessing_pipeline or []):
                    optimized_job.preprocessing_pipeline = optimized_pipeline
                    optimizations_applied.append('preprocessing_optimization')
            
            # Apply memory optimizations
            if profile.get('memory_intensive', False):
                optimized_job.caching_enabled = False
                optimizations_applied.append('memory_optimization')
        
        return optimized_job, optimizations_applied
    
    def _get_optimal_batch_size(self, model_id: str, profile: Dict[str, Any]) -> int:
        """Determine optimal batch size based on model profile."""
        base_batch_size = 8
        
        # Adjust based on model characteristics
        if profile.get('model_size_mb', 0) > 500:
            return max(1, base_batch_size // 2)  # Large models: smaller batches
        elif profile.get('avg_latency_ms', 0) < 50:
            return base_batch_size * 2  # Fast models: larger batches
        
        return base_batch_size
    
    def _optimize_preprocessing_pipeline(
        self, 
        pipeline: List[str], 
        profile: Dict[str, Any]
    ) -> List[str]:
        """Optimize preprocessing pipeline based on performance data."""
        if not pipeline:
            return pipeline
        
        # Remove expensive operations for fast mode
        fast_mode_exclusions = ['complex_normalization', 'data_augmentation']
        return [op for op in pipeline if op not in fast_mode_exclusions]
    
    def record_performance(self, job: InferenceJob, result: InferenceResult) -> None:
        """Record performance data for future optimization."""
        with self._lock:
            # Update model profile
            if job.model_id not in self.model_profiles:
                self.model_profiles[job.model_id] = {
                    'total_jobs': 0,
                    'avg_latency_ms': 0.0,
                    'avg_memory_mb': 0.0,
                    'optimal_batch_sizes': []
                }
            
            profile = self.model_profiles[job.model_id]
            profile['total_jobs'] += 1
            
            # Update averages with exponential moving average
            alpha = 0.1
            profile['avg_latency_ms'] = (alpha * result.latency_ms + 
                                        (1 - alpha) * profile['avg_latency_ms'])
            profile['avg_memory_mb'] = (alpha * result.memory_peak_mb + 
                                       (1 - alpha) * profile['avg_memory_mb'])
            
            # Track successful batch sizes
            if result.success and result.latency_ms < profile['avg_latency_ms'] * 1.2:
                profile['optimal_batch_sizes'].append(job.batch_size)
                # Keep only recent batch sizes
                profile['optimal_batch_sizes'] = profile['optimal_batch_sizes'][-20:]
            
            # Record in optimization history
            self.optimization_history[job.model_id].append({
                'timestamp': time.time(),
                'job': job,
                'result': result,
                'optimizations': result.optimization_applied
            })


class HighPerformanceInferenceEngine:
    """High-performance inference engine with advanced optimization and scalability."""
    
    def __init__(
        self,
        num_workers: int = None,
        enable_caching: bool = True,
        cache_memory_mb: int = 1024,
        enable_optimization: bool = True,
        max_concurrent_jobs: int = 100
    ):
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())
        self.enable_caching = enable_caching
        self.enable_optimization = enable_optimization
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Core components
        self.cache_system = IntelligentCachingSystem(cache_memory_mb) if enable_caching else None
        self.load_balancer = AdaptiveLoadBalancer(self.num_workers)
        self.optimizer = PerformanceOptimizer() if enable_optimization else None
        
        # Execution infrastructure
        self.thread_executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, self.num_workers // 2))
        
        # Job management
        self.active_jobs: Dict[str, InferenceJob] = {}
        self.job_results: Dict[str, InferenceResult] = {}
        self.job_queue = deque()
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_metrics = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'avg_latency_ms': 0.0,
            'total_throughput': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Background processing
        self._shutdown = False
        self._processing_task: Optional[asyncio.Task] = None
        
        logger.info(f"High-Performance Inference Engine initialized with {self.num_workers} workers")
    
    async def initialize(self) -> None:
        """Initialize the inference engine."""
        self._processing_task = asyncio.create_task(self._job_processing_loop())
        logger.info("High-Performance Inference Engine started")
    
    async def submit_job(self, job: InferenceJob) -> str:
        """Submit inference job for processing."""
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            raise RuntimeError("Maximum concurrent jobs limit exceeded")
        
        # Apply optimizations if enabled
        if self.optimizer:
            optimized_job, optimizations = self.optimizer.optimize_job(job)
            job = optimized_job
            job.metadata['optimizations'] = optimizations
        
        with self._lock:
            self.active_jobs[job.job_id] = job
            self.job_queue.append(job)
        
        return job.job_id
    
    async def get_result(self, job_id: str, timeout: float = 30.0) -> InferenceResult:
        """Get job result with timeout."""
        start_time = time.time()
        
        while job_id not in self.job_results:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(0.01)  # 10ms polling
        
        with self._lock:
            result = self.job_results.pop(job_id)
            self.active_jobs.pop(job_id, None)
        
        return result
    
    async def _job_processing_loop(self) -> None:
        """Main job processing loop."""
        while not self._shutdown:
            try:
                if self.job_queue:
                    # Process jobs in batches for efficiency
                    batch_jobs = []
                    with self._lock:
                        for _ in range(min(4, len(self.job_queue))):
                            if self.job_queue:
                                batch_jobs.append(self.job_queue.popleft())
                    
                    if batch_jobs:
                        # Process batch concurrently
                        tasks = [self._process_single_job(job) for job in batch_jobs]
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                else:
                    await asyncio.sleep(0.001)  # 1ms when idle
            
            except Exception as e:
                logger.error(f"Job processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_single_job(self, job: InferenceJob) -> None:
        """Process a single inference job."""
        start_time = time.time()
        processing_stages = {}
        cache_hit = False
        
        try:
            # Check cache first
            if self.cache_system and job.caching_enabled:
                cache_start = time.time()
                cached_result = self.cache_system.get(
                    job.model_id, 
                    job.input_data, 
                    {'batch_size': job.batch_size, 'optimization': job.optimization_level}
                )
                processing_stages['cache_lookup'] = (time.time() - cache_start) * 1000
                
                if cached_result is not None:
                    cache_hit = True
                    result = InferenceResult(
                        job_id=job.job_id,
                        output_data=cached_result,
                        latency_ms=(time.time() - start_time) * 1000,
                        throughput_ops_per_sec=job.batch_size / ((time.time() - start_time) or 0.001),
                        memory_peak_mb=0.0,  # Cache hits don't use additional memory
                        cpu_utilization=0.0,
                        cache_hit=True,
                        optimization_applied=job.metadata.get('optimizations', []),
                        processing_stages=processing_stages
                    )
                    
                    with self._lock:
                        self.job_results[job.job_id] = result
                    
                    self._update_performance_metrics(result)
                    return
            
            # Select worker for job
            worker_id = self.load_balancer.select_worker(job)
            self.load_balancer.record_job_start(worker_id, job)
            
            # Execute inference
            execution_start = time.time()
            output_data = await self._execute_inference(job, worker_id)
            execution_time = (time.time() - execution_start) * 1000
            processing_stages['inference_execution'] = execution_time
            
            total_latency = (time.time() - start_time) * 1000
            
            # Create result
            result = InferenceResult(
                job_id=job.job_id,
                output_data=output_data,
                latency_ms=total_latency,
                throughput_ops_per_sec=job.batch_size / (total_latency / 1000.0),
                memory_peak_mb=128.0,  # Mock value
                cpu_utilization=50.0,  # Mock value
                cache_hit=cache_hit,
                optimization_applied=job.metadata.get('optimizations', []),
                processing_stages=processing_stages
            )
            
            # Store in cache if enabled
            if self.cache_system and job.caching_enabled:
                cache_store_start = time.time()
                self.cache_system.put(
                    job.model_id,
                    job.input_data,
                    {'batch_size': job.batch_size, 'optimization': job.optimization_level},
                    output_data,
                    result.memory_peak_mb
                )
                processing_stages['cache_store'] = (time.time() - cache_store_start) * 1000
            
            # Record performance data
            if self.optimizer:
                self.optimizer.record_performance(job, result)
            
            self.load_balancer.record_job_completion(worker_id, job, total_latency)
            
            with self._lock:
                self.job_results[job.job_id] = result
            
            self._update_performance_metrics(result)
        
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            error_result = InferenceResult(
                job_id=job.job_id,
                output_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                throughput_ops_per_sec=0.0,
                memory_peak_mb=0.0,
                cpu_utilization=0.0,
                success=False,
                error_message=str(e),
                processing_stages=processing_stages
            )
            
            with self._lock:
                self.job_results[job.job_id] = error_result
            
            self._update_performance_metrics(error_result)
    
    async def _execute_inference(self, job: InferenceJob, worker_id: int) -> Any:
        """Execute the actual inference (mock implementation)."""
        # Simulate inference processing time based on batch size and model complexity
        base_latency = 10  # ms
        batch_factor = job.batch_size * 0.8  # Batch efficiency
        processing_time = (base_latency + batch_factor) / 1000.0
        
        await asyncio.sleep(processing_time)
        
        # Return mock result
        return f"inference_result_for_{job.model_id}_batch_{job.batch_size}"
    
    def _update_performance_metrics(self, result: InferenceResult) -> None:
        """Update system performance metrics."""
        with self._lock:
            self.performance_metrics['total_jobs'] += 1
            
            if result.success:
                self.performance_metrics['successful_jobs'] += 1
                
                # Update average latency
                alpha = 0.1
                self.performance_metrics['avg_latency_ms'] = (
                    alpha * result.latency_ms + 
                    (1 - alpha) * self.performance_metrics['avg_latency_ms']
                )
                
                self.performance_metrics['total_throughput'] += result.throughput_ops_per_sec
            else:
                self.performance_metrics['failed_jobs'] += 1
            
            # Update cache hit rate
            if self.cache_system:
                cache_stats = self.cache_system.get_statistics()
                self.performance_metrics['cache_hit_rate'] = cache_stats['hit_rate']
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        dashboard = {
            'engine_metrics': self.performance_metrics.copy(),
            'load_balancer_stats': self.load_balancer.get_load_statistics(),
            'active_jobs': len(self.active_jobs),
            'pending_jobs': len(self.job_queue)
        }
        
        if self.cache_system:
            dashboard['cache_stats'] = self.cache_system.get_statistics()
        
        return dashboard
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health assessment."""
        health_score = 100
        issues = []
        
        # Check success rate
        total_jobs = self.performance_metrics['total_jobs']
        if total_jobs > 0:
            success_rate = self.performance_metrics['successful_jobs'] / total_jobs
            if success_rate < 0.95:
                health_score -= 20
                issues.append(f"Low success rate: {success_rate:.2%}")
        
        # Check average latency
        avg_latency = self.performance_metrics['avg_latency_ms']
        if avg_latency > 500:  # 500ms threshold
            health_score -= 15
            issues.append(f"High latency: {avg_latency:.1f}ms")
        
        # Check queue length
        queue_length = len(self.job_queue)
        if queue_length > 50:
            health_score -= 10
            issues.append(f"Large queue: {queue_length} jobs")
        
        # Determine status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 50:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': health_score,
            'issues': issues,
            'recommendations': self._generate_recommendations(issues)
        }
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate performance recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if 'success rate' in issue:
                recommendations.append("Investigate job failures and improve error handling")
            elif 'latency' in issue:
                recommendations.append("Consider increasing worker count or optimizing models")
            elif 'queue' in issue:
                recommendations.append("Increase worker capacity or implement job prioritization")
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown the inference engine gracefully."""
        logger.info("Shutting down High-Performance Inference Engine")
        self._shutdown = True
        
        if self._processing_task:
            self._processing_task.cancel()
        
        # Wait for active jobs to complete (with timeout)
        timeout = 30.0
        start_time = time.time()
        while self.active_jobs and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("High-Performance Inference Engine shutdown complete")
