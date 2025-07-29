# Architecture Overview

This document provides a comprehensive overview of the WASM Shim for Torch architecture, design decisions, and implementation details.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Environment                       │
├─────────────────────────────────────────────────────────────────┤
│  JavaScript API Layer                                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Model Loader  │  │  Inference API   │  │  Memory Manager │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  WASI-NN Shim Layer (TypeScript/JavaScript)                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Tensor Bridge  │  │  Operation Maps  │  │  Thread Pool    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  WebAssembly Runtime                                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  SIMD Kernels   │  │  Memory Layout   │  │  Model Executor │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Development Environment                       │
├─────────────────────────────────────────────────────────────────┤
│  Python Export Pipeline                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ Model Analyzer  │  │  Optimizer       │  │  WASM Compiler  │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Export Pipeline

**Purpose**: Convert PyTorch models to WASM-compatible format

**Key Classes**:
- `ModelExporter`: Main export orchestrator
- `GraphAnalyzer`: Analyzes PyTorch computation graph
- `OperationMapper`: Maps PyTorch ops to WASM implementations
- `OptimizationPipeline`: Applies performance optimizations

**Flow**:
```python
# High-level export process
model = torch.load("model.pth")
exporter = ModelExporter(model)
exporter.analyze_graph()
exporter.optimize_for_wasm()
exporter.compile_to_wasm("model.wasm")
```

### 2. WASM Runtime

**Purpose**: Execute models in browser with optimal performance

**Architecture**:
```cpp
// C++ runtime structure
class WASMTorchRuntime {
private:
    MemoryManager memory_manager_;
    ThreadPool thread_pool_;
    SIMDKernelRegistry simd_kernels_;
    
public:
    void load_model(const uint8_t* model_data);
    Tensor forward(const Tensor& input);
    void cleanup();
};
```

**Memory Layout**:
```
WASM Memory (Linear, 32-bit addressing)
┌─────────────────────────────────────────────────┐
│ Stack (1MB)                                     │
├─────────────────────────────────────────────────┤
│ Model Parameters (Dynamic)                      │
├─────────────────────────────────────────────────┤
│ Activation Tensors (Dynamic)                    │
├─────────────────────────────────────────────────┤
│ SIMD Scratch Space (aligned to 16-byte)        │
├─────────────────────────────────────────────────┤
│ Thread Communication (SharedArrayBuffer)       │
└─────────────────────────────────────────────────┘
```

### 3. JavaScript Bridge

**Purpose**: Provide easy-to-use API and manage WASM interactions

**API Design**:
```javascript
class WASMTorch {
    static async init(options) {
        // Initialize runtime with SIMD/threading options
    }
    
    async loadModel(wasmBuffer) {
        // Load compiled model
    }
    
    async forward(inputTensor) {
        // Run inference
    }
    
    getMemoryStats() {
        // Return memory usage statistics
    }
}
```

## Design Decisions

### 1. WASI-NN vs Custom Interface

**Decision**: Implement WASI-NN compatible interface
**Rationale**:
- Future-proof with emerging standards
- Interoperability with other WASM ML runtimes
- Standardized tensor formats and operations

### 2. SIMD vs Scalar Operations

**Decision**: Prioritize SIMD implementations with scalar fallbacks
**Rationale**:
- 4x performance improvement for common operations
- Wide browser support (Chrome 91+, Firefox 89+)
- Graceful degradation on older browsers

**Implementation Pattern**:
```cpp
// SIMD-first implementation
void matmul_f32(const float* a, const float* b, float* c, 
                int M, int N, int K) {
    if (has_simd_support()) {
        matmul_f32_simd(a, b, c, M, N, K);
    } else {
        matmul_f32_scalar(a, b, c, M, N, K);
    }
}
```

### 3. Memory Management Strategy

**Decision**: Manual memory management with reference counting
**Rationale**:
- Predictable memory usage (important for 4GB WASM limit)
- Avoid garbage collection pauses during inference
- Better control over memory layout for cache efficiency

### 4. Threading Model

**Decision**: Web Workers with SharedArrayBuffer
**Rationale**:
- True parallelism for CPU-intensive operations
- Scales with available CPU cores
- Required COOP/COEP headers ensure security

**Threading Architecture**:
```
Main Thread              Worker Thread 1         Worker Thread N
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   JS API    │         │WASM Instance│         │WASM Instance│
│ Coordination│◄────────┤  Compute    │◄────────┤  Compute    │
│   Tensor I/O│         │  Kernels    │         │  Kernels    │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                    SharedArrayBuffer
                    (Tensor Data & Sync)
```

## Performance Optimizations

### 1. Operator Fusion

**Technique**: Combine multiple operations into single kernels
**Example**:
```cpp
// Instead of: conv2d -> batch_norm -> relu (3 passes)
// Fused: conv2d_bn_relu (1 pass)
void conv2d_bn_relu_fused(const Tensor& input, const Tensor& weight,
                          const Tensor& bn_weight, const Tensor& bn_bias,
                          Tensor& output);
```

### 2. Memory Layout Optimization

**Technique**: Arrange tensors for optimal cache performance
- NCHW → NHWC conversion for better SIMD utilization
- Padding elimination where possible
- Aligned allocations for SIMD operations

### 3. Computation Graph Optimization

**Optimizations Applied**:
- Dead code elimination
- Constant folding
- Common subexpression elimination
- Loop fusion and unrolling

### 4. Quantization Support

**INT8 Quantization Pipeline**:
```python
# Post-training quantization
def quantize_model(model, calibration_data):
    # Analyze activation ranges
    ranges = analyze_activations(model, calibration_data)
    
    # Compute quantization parameters
    scales, zero_points = compute_qparams(ranges)
    
    # Replace operations with quantized versions
    return replace_ops_with_quantized(model, scales, zero_points)
```

## Supported Operations

### Fully Implemented

| Category | Operations | SIMD | Threading |
|----------|------------|------|-----------|
| Linear | `linear`, `matmul` | ✅ | ✅ |
| Convolution | `conv1d`, `conv2d`, `conv_transpose2d` | ✅ | ✅ |
| Activation | `relu`, `gelu`, `silu`, `swish` | ✅ | ❌ |
| Normalization | `batch_norm`, `layer_norm`, `group_norm` | ✅ | ❌ |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | ✅ | ❌ |
| Tensor Ops | `add`, `mul`, `sub`, `div`, `cat`, `split` | ✅ | ✅ |

### Partially Implemented

| Operation | Status | Notes |
|-----------|--------|-------|
| `attention` | ⚠️ Partial | Basic scaled dot-product only |
| `embedding` | ⚠️ Partial | No sparse gradients |
| `lstm`/`gru` | ⚠️ Partial | Forward pass only |

### Planned Implementation

- Transformer blocks (optimized)
- Dynamic shape operations
- Advanced pooling operations
- Sparse tensor operations

## Security Considerations

### 1. Sandbox Isolation

**Browser Sandbox**: All computation runs in browser security sandbox
- No file system access
- No network access (except initial model loading)
- Memory isolation from other tabs/processes

### 2. Memory Safety

**WASM Memory Model**: Linear memory with bounds checking
- No buffer overflows possible
- Deterministic memory layout
- Controlled growth (WebAssembly.Memory.grow())

### 3. Side-Channel Protection

**Timing Attack Mitigation**:
- Constant-time operations where possible
- Reduced precision timers in some browsers
- No access to high-resolution performance counters

## Browser Compatibility

### Minimum Requirements

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebAssembly | 57+ | 52+ | 11+ | 16+ |
| WASM SIMD | 91+ | 89+ | 16.4+ | 91+ |
| SharedArrayBuffer | 68+ | 79+ | 15.2+ | 79+ |
| Web Workers | 4+ | 3.5+ | 4+ | 12+ |

### Feature Detection

```javascript
// Runtime capability detection
const capabilities = {
    simd: typeof WebAssembly.SIMD !== 'undefined',
    threads: typeof SharedArrayBuffer !== 'undefined',
    bigint: typeof BigInt !== 'undefined'
};

// Load appropriate WASM variant
const wasmUrl = capabilities.simd 
    ? 'model-simd.wasm' 
    : 'model-basic.wasm';
```

## Testing Strategy

### 1. Unit Testing

**C++ Runtime**:
- Individual kernel testing
- Memory management testing
- Thread safety verification

**Python Export**:
- Graph analysis correctness
- Optimization pass validation
- Cross-platform compilation

### 2. Integration Testing

**End-to-End Workflows**:
- PyTorch model → WASM export → Browser inference
- Performance regression testing
- Memory leak detection

### 3. Browser Testing

**Cross-Browser Compatibility**:
- Automated testing via Playwright/Selenium
- Performance benchmarking across browsers
- Feature fallback testing

## Future Roadmap

### Short Term (Q3 2025)
- WebNN backend integration
- Advanced quantization (INT4, mixed precision)
- Improved error handling and debugging

### Medium Term (Q4 2025)
- WebGPU hybrid mode for large models
- Streaming inference for video/audio
- Model zoo with pre-compiled popular models

### Long Term (2026+)
- Distributed browser training
- Federated learning support
- Advanced compression techniques

## Contributing to Architecture

### Adding New Operations

1. **Define interface** in `src/wasm_torch/ops/`
2. **Implement C++ kernel** in `runtime/kernels/`
3. **Add SIMD optimization** if applicable  
4. **Write comprehensive tests**
5. **Update operation registry**

### Performance Optimization

1. **Profile current implementation**
2. **Identify bottlenecks** (CPU, memory, cache)
3. **Implement optimized version**
4. **Benchmark against baseline**
5. **Document performance characteristics**

### Browser Support

1. **Research new browser features**
2. **Implement feature detection**
3. **Add progressive enhancement**
4. **Test across browser matrix**
5. **Update compatibility documentation**

For detailed implementation guides, see the [Development Documentation](DEVELOPMENT.md).