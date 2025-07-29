# Performance Guide

This guide covers performance optimization strategies, benchmarking, and profiling for WASM Shim for Torch.

## Overview

WASM Shim for Torch is designed to provide optimal performance for PyTorch model inference in browsers while maintaining compatibility and ease of use. This document outlines the performance characteristics, optimization strategies, and tools available for measuring and improving performance.

## Performance Targets

### Baseline Performance Goals

| Model Category | Target vs Native | Memory Usage | Load Time |
|----------------|------------------|---------------|-----------|
| Small models (<10MB) | 2-3x slower | <100MB | <2s |
| Medium models (10-100MB) | 2.5-3.5x slower | <500MB | <10s |
| Large models (>100MB) | 3-4x slower | <2GB | <30s |

### Browser Compatibility Performance

| Browser | SIMD Support | Threading | Expected Performance |
|---------|--------------|-----------|---------------------|
| Chrome 91+ | ✅ | ✅ | Baseline |
| Firefox 89+ | ✅ | ✅ | 95-100% of baseline |
| Safari 16.4+ | ✅ | ⚠️ Limited | 80-90% of baseline |
| Edge 91+ | ✅ | ✅ | 95-100% of baseline |

## Performance Optimization Strategies

### 1. Model Export Optimization

#### Operator Fusion

Combine multiple operations into single kernels to reduce memory bandwidth and improve cache efficiency:

```python
from wasm_torch.optimize import optimize_for_browser

# Enable aggressive operator fusion
optimized_model = optimize_for_browser(
    model,
    optimization_passes=[
        "fuse_conv_bn_relu",      # Fuse conv + batch norm + activation
        "fuse_linear_activation", # Fuse linear + activation
        "fuse_attention_ops",     # Fuse attention mechanisms
        "eliminate_dead_code",    # Remove unused operations
        "constant_folding"        # Pre-compute constant expressions
    ]
)
```

#### Memory Layout Optimization

Optimize tensor layouts for better SIMD utilization:

```python
# Optimize for SIMD-friendly memory layouts
export_to_wasm(
    model,
    output_path="optimized_model.wasm",
    optimization_level="O3",
    memory_layout="nhwc",  # Better for SIMD than NCHW
    use_simd=True,
    align_tensors=True     # 16-byte alignment for SIMD
)
```

### 2. Quantization

#### INT8 Quantization

Reduce model size and improve performance with quantization:

```python
from wasm_torch.optimize import quantize_for_wasm

# Dynamic quantization (easiest)
quantized_model = quantize_for_wasm(
    model,
    quantization_type="dynamic",
    preserve_accuracy=True
)

# Static quantization (best performance)
quantized_model = quantize_for_wasm(
    model,
    quantization_type="static",
    calibration_data=calibration_loader,
    target_accuracy=0.99  # Accept 1% accuracy loss for better performance
)
```

#### Performance Impact of Quantization

| Quantization Type | Size Reduction | Speed Improvement | Accuracy Loss |
|-------------------|----------------|-------------------|---------------|
| Dynamic INT8 | 2-4x smaller | 1.5-2x faster | <1% |
| Static INT8 | 3-4x smaller | 2-3x faster | 1-3% |
| Mixed Precision | 1.5-2x smaller | 1.2-1.5x faster | <0.5% |

### 3. Runtime Optimization

#### SIMD Utilization

Ensure optimal SIMD usage:

```javascript
// Initialize with SIMD support
const runtime = await WASMTorch.init({
    simd: true,
    threads: navigator.hardwareConcurrency || 4,
    memory_growth: true,
    optimize_for_speed: true
});

// Verify SIMD is active
console.log(`SIMD enabled: ${runtime.hasSIMD()}`);
```

#### Memory Management

Optimize memory usage patterns:

```javascript
// Pre-allocate memory pools
const runtime = await WASMTorch.init({
    initial_memory: 256 * 1024 * 1024,  // 256MB
    memory_pool_size: 128 * 1024 * 1024, // 128MB pool
    enable_memory_recycling: true
});

// Manage tensor lifecycle
const input = runtime.createTensor(inputData);
const output = await model.forward(input);

// Explicit cleanup for large tensors
input.dispose();
output.dispose();
```

### 4. Threading Optimization

#### Optimal Thread Configuration

```javascript
// Detect optimal thread count
const optimalThreads = Math.min(
    navigator.hardwareConcurrency || 4,
    4  // Cap at 4 threads for web context
);

const runtime = await WASMTorch.init({
    threads: optimalThreads,
    thread_affinity: true,  // Pin threads to cores
    work_stealing: true     // Enable work-stealing scheduler
});
```

#### Thread Pool Configuration

```javascript
// Configure thread pool for different workloads
const threadConfig = {
    compute_threads: Math.max(2, optimalThreads - 1),
    io_threads: 1,
    thread_stack_size: 64 * 1024,  // 64KB per thread
    enable_numa_awareness: false   // Not applicable in browser
};
```

## Benchmarking and Profiling

### 1. Built-in Benchmarking

#### Model Export Benchmarking

```python
from benchmarks.python.benchmark_export import ModelBenchmark

# Run comprehensive benchmark
benchmark = ModelBenchmark(output_dir="results/")
benchmark.run_comprehensive_benchmark()

# Custom model benchmark
results = benchmark.benchmark_model_export(
    model=your_model,
    model_name="custom_model",
    input_shape=(3, 224, 224),
    optimization_levels=["O2", "O3"]
)
```

#### Runtime Benchmarking

```javascript
// Built-in performance monitoring
const model = await runtime.loadModel('model.wasm');

// Warm-up runs
for (let i = 0; i < 5; i++) {
    await model.forward(warmupInput);
}

// Benchmark inference
const startTime = performance.now();
for (let i = 0; i < 100; i++) {
    await model.forward(input);
}
const avgTime = (performance.now() - startTime) / 100;

console.log(`Average inference time: ${avgTime.toFixed(2)}ms`);
```

### 2. Memory Profiling

#### Python Memory Analysis

```python
from memory_profiler import profile
import psutil

@profile
def export_model_with_profiling():
    model = load_large_model()
    export_to_wasm(model, "output.wasm")

# Monitor system memory
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

#### JavaScript Memory Monitoring

```javascript
// Monitor WASM memory usage
const memStats = runtime.getMemoryStats();
console.log(`WASM heap used: ${memStats.heapUsed / 1024 / 1024:.1f} MB`);
console.log(`Peak allocation: ${memStats.peakBytes / 1024 / 1024:.1f} MB`);

// Monitor JavaScript memory
if (performance.memory) {
    console.log(`JS heap: ${performance.memory.usedJSHeapSize / 1024 / 1024:.1f} MB`);
}
```

### 3. CPU Profiling

#### Python Profiling

```python
import cProfile
import pstats

# Profile model export
cProfile.run('export_to_wasm(model, "output.wasm")', 'export_profile.prof')

# Analyze results
stats = pstats.Stats('export_profile.prof')
stats.sort_stats('cumulative').print_stats(20)
```

#### Browser Profiling

```javascript
// Use browser developer tools
console.profile('Model Inference');
await model.forward(input);
console.profileEnd('Model Inference');

// Manual timing for specific operations
console.time('Tensor Creation');
const tensor = runtime.createTensor(data);
console.timeEnd('Tensor Creation');
```

## Performance Monitoring

### 1. Automated Performance Regression Detection

```python
# In your CI pipeline
import pytest
from benchmarks.python.benchmark_export import ModelBenchmark

@pytest.mark.benchmark
def test_performance_regression():
    """Ensure performance doesn't regress."""
    
    benchmark = ModelBenchmark()
    results = benchmark.benchmark_model_export(
        model=reference_model,
        model_name="regression_test",
        input_shape=(3, 224, 224)
    )
    
    # Check against baseline
    baseline_time = 0.150  # 150ms baseline
    actual_time = results["optimization_results"]["O3"]["export_time_sec"]
    
    assert actual_time < baseline_time * 1.1, \
        f"Performance regression detected: {actual_time:.3f}s > {baseline_time * 1.1:.3f}s"
```

### 2. Real-time Performance Dashboard

```javascript
// Performance metrics collection
class PerformanceMonitor {
    constructor() {
        this.metrics = [];
    }
    
    recordInference(startTime, endTime, inputSize) {
        const duration = endTime - startTime;
        const throughput = inputSize / duration * 1000; // ops/sec
        
        this.metrics.push({
            timestamp: Date.now(),
            duration,
            throughput,
            inputSize
        });
        
        // Keep only recent metrics
        if (this.metrics.length > 1000) {
            this.metrics = this.metrics.slice(-1000);
        }
    }
    
    getAveragePerformance(windowSize = 100) {
        const recent = this.metrics.slice(-windowSize);
        const avgDuration = recent.reduce((sum, m) => sum + m.duration, 0) / recent.length;
        const avgThroughput = recent.reduce((sum, m) => sum + m.throughput, 0) / recent.length;
        
        return { avgDuration, avgThroughput };
    }
}
```

## Performance Best Practices

### 1. Model Design Guidelines

- **Minimize dynamic shapes**: Use fixed input sizes when possible
- **Reduce memory allocations**: Reuse tensors where possible
- **Optimize operation sequence**: Place expensive operations early in conditional branches
- **Use appropriate precisions**: Float32 is optimal for WASM SIMD

### 2. Runtime Best Practices

- **Warm up models**: Run several inferences before timing
- **Batch operations**: Process multiple inputs together when possible
- **Memory management**: Explicitly dispose of large tensors
- **Thread management**: Don't create more threads than CPU cores

### 3. Browser-Specific Optimizations

#### Chrome/Chromium

- Enable experimental WebAssembly features in chrome://flags
- Use SharedArrayBuffer for optimal threading performance
- Leverage Chrome DevTools Performance panel

#### Firefox

- Configure `javascript.options.wasm_simd` and `javascript.options.wasm_multi_memory`
- Use Firefox Profiler for detailed analysis

#### Safari

- Work around SharedArrayBuffer limitations with fallback implementations
- Optimize for single-threaded performance
- Test on both Intel and Apple Silicon Macs

## Troubleshooting Performance Issues

### Common Performance Problems

1. **Slower than expected inference**:
   - Check SIMD is enabled: `runtime.hasSIMD()`
   - Verify threading: `runtime.getThreadCount()`
   - Profile memory usage: Look for excessive allocations

2. **High memory usage**:
   - Enable memory recycling
   - Dispose of tensors explicitly
   - Consider quantization

3. **Slow model loading**:
   - Check network performance
   - Enable compression
   - Use progressive loading for large models

### Performance Debugging Tools

```javascript
// Debug mode with detailed timing
const runtime = await WASMTorch.init({
    debug: true,
    profiling: true,
    log_level: 'verbose'
});

// Enable operation-level timing
runtime.enableOperationProfiling(true);

// Run inference with detailed logs
const result = await model.forward(input);
const profile = runtime.getProfilingData();

console.table(profile.operationTimes);
```

## Continuous Performance Monitoring

### CI/CD Integration

```yaml
# In .github/workflows/performance.yml
- name: Run Performance Benchmarks
  run: |
    python benchmarks/python/benchmark_export.py
    pytest benchmarks/ --benchmark-json=benchmark_results.json

- name: Upload Performance Results
  uses: actions/upload-artifact@v3
  with:
    name: performance-results
    path: benchmark_results.json
```

### Performance Alerting

Set up alerts for performance regressions:

```python
# In your monitoring system
def check_performance_regression(current_results, baseline_results):
    for metric in ['export_time', 'inference_time', 'memory_usage']:
        current = current_results[metric]
        baseline = baseline_results[metric]
        
        if current > baseline * 1.2:  # 20% regression threshold
            send_alert(f"Performance regression in {metric}: {current:.3f} vs {baseline:.3f}")
```

This performance guide should help you optimize WASM Shim for Torch for your specific use cases. Regular benchmarking and profiling are key to maintaining optimal performance across different models and deployment scenarios.