# ADR-0001: WASM Architecture Choice for PyTorch Inference

## Status

Accepted

## Context

We need to enable PyTorch model inference in web browsers. Several approaches are available:

1. **WebGPU-based solutions**: High performance but limited browser support
2. **ONNX.js**: Good compatibility but performance limitations
3. **WASM with WASI-NN**: Universal compatibility with reasonable performance
4. **Pure JavaScript**: Easiest to implement but very slow

Browser compatibility and universal deployment are critical requirements for this project.

## Decision

We will use WebAssembly (WASM) with WASI-NN interface for PyTorch model inference:

- **Primary runtime**: WASM with SIMD and threads support
- **Interface**: WASI-NN shim layer for neural network operations
- **Optimization**: Hand-optimized SIMD kernels for critical operations
- **Fallback**: None required due to universal WASM support

## Consequences

### Positive Consequences

- **Universal compatibility**: Runs on any WASM-capable browser
- **Predictable performance**: No GPU driver dependencies
- **Better mobile support**: Many mobile browsers lack WebGPU
- **Easier deployment**: Single WASM file distribution
- **Privacy preserving**: All computation stays in browser sandbox
- **Thread support**: Parallel execution via SharedArrayBuffer

### Negative Consequences

- **Performance overhead**: ~3x slower than native PyTorch
- **Memory limitations**: Limited to 4GB address space (WASM32)
- **Development complexity**: Requires Emscripten toolchain knowledge
- **Limited operators**: Some PyTorch operations not implementable in WASM

## Alternatives Considered

- **WebGPU**: Rejected due to limited browser support and mobile compatibility issues
- **ONNX.js**: Rejected due to performance limitations (4-5x slower than our WASM approach)
- **TensorFlow.js**: Rejected due to model compatibility requirements
- **Pure JavaScript**: Rejected due to unacceptable performance (10x+ slower)

## Implementation Notes

- Use Emscripten 3.1.61+ for SIMD and threads support
- Implement custom WASI-NN shim in TypeScript
- Hand-optimize critical kernels using WASM SIMD instructions
- Support both single-threaded and multi-threaded execution modes

## References

- [WASI-NN Specification](https://github.com/WebAssembly/wasi-nn)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [Browser WASM Support](https://caniuse.com/wasm-simd)