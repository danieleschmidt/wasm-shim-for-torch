"""Model export functionality for converting PyTorch models to WASM."""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import logging
import json
import tempfile
import subprocess
import shutil


logger = logging.getLogger(__name__)


def export_to_wasm(
    model: nn.Module,
    output_path: Union[str, Path],
    example_input: torch.Tensor,
    optimization_level: str = "O2",
    use_simd: bool = True,
    use_threads: bool = True,
    **kwargs: Any
) -> None:
    """Export PyTorch model to WASM format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save the WASM file
        example_input: Example input tensor for tracing
        optimization_level: Compilation optimization level (O0, O1, O2, O3)
        use_simd: Enable SIMD optimizations
        use_threads: Enable multi-threading support
        **kwargs: Additional export options
        
    Raises:
        ValueError: If optimization level is invalid or model tracing fails
        RuntimeError: If Emscripten toolchain is not available
    """
    output_path = Path(output_path)
    
    # Validate inputs
    _validate_export_inputs(model, example_input, optimization_level)
    
    # Set model to evaluation mode
    model.eval()
    
    logger.info(f"Starting WASM export to {output_path}")
    
    try:
        # Step 1: Trace the PyTorch model
        logger.info("Tracing PyTorch model...")
        traced_model = _trace_model(model, example_input)
        
        # Step 2: Convert to intermediate representation
        logger.info("Converting to intermediate representation...")
        ir_data = _convert_to_ir(traced_model, example_input, use_simd, use_threads)
        
        # Step 3: Generate WASM compilation units
        logger.info("Generating WASM compilation units...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            compilation_files = _generate_compilation_units(ir_data, temp_path)
            
            # Step 4: Compile to WASM
            logger.info(f"Compiling to WASM with optimization level {optimization_level}...")
            _compile_to_wasm(
                compilation_files, 
                output_path, 
                optimization_level,
                use_simd,
                use_threads,
                **kwargs
            )
            
        logger.info(f"WASM export completed successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"WASM export failed: {e}")
        raise RuntimeError(f"Failed to export model to WASM: {e}") from e


def _validate_export_inputs(
    model: nn.Module, 
    example_input: torch.Tensor, 
    optimization_level: str
) -> None:
    """Validate export function inputs."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    
    if not isinstance(example_input, torch.Tensor):
        raise ValueError("example_input must be a torch.Tensor")
    
    valid_opt_levels = {"O0", "O1", "O2", "O3"}
    if optimization_level not in valid_opt_levels:
        raise ValueError(f"optimization_level must be one of {valid_opt_levels}")


def _trace_model(model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
    """Trace PyTorch model for export."""
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        return traced_model
    except Exception as e:
        raise ValueError(f"Failed to trace model: {e}") from e


def _convert_to_ir(
    traced_model: torch.jit.ScriptModule,
    example_input: torch.Tensor,
    use_simd: bool,
    use_threads: bool
) -> Dict[str, Any]:
    """Convert traced model to intermediate representation."""
    # Extract model graph
    graph = traced_model.graph
    
    # Analyze operations and data flow
    operations = []
    for node in graph.nodes():
        op_info = {
            "kind": node.kind(),
            "inputs": [str(inp) for inp in node.inputs()],
            "outputs": [str(out) for out in node.outputs()],
            "attributes": {attr: getattr(node, attr, None) for attr in node.attributeNames()},
        }
        operations.append(op_info)
    
    # Extract model parameters
    parameters = {}
    for name, param in traced_model.named_parameters():
        parameters[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "data": param.detach().numpy().tobytes() if param.requires_grad else None,
        }
    
    # Create IR data structure
    ir_data = {
        "model_info": {
            "input_shape": list(example_input.shape),
            "input_dtype": str(example_input.dtype),
        },
        "graph": {
            "operations": operations,
            "parameters": parameters,
        },
        "optimization": {
            "use_simd": use_simd,
            "use_threads": use_threads,
        },
        "metadata": {
            "torch_version": torch.__version__,
            "export_version": "0.1.0",
        }
    }
    
    return ir_data


def _generate_compilation_units(ir_data: Dict[str, Any], temp_path: Path) -> Dict[str, Path]:
    """Generate C++ compilation units for WASM."""
    files = {}
    
    # Generate main runtime file
    runtime_cpp = temp_path / "wasm_runtime.cpp"
    _generate_runtime_cpp(ir_data, runtime_cpp)
    files["runtime"] = runtime_cpp
    
    # Generate model-specific operations
    ops_cpp = temp_path / "model_ops.cpp"
    _generate_operations_cpp(ir_data, ops_cpp)
    files["operations"] = ops_cpp
    
    # Generate header files
    header_h = temp_path / "model.h"
    _generate_header_file(ir_data, header_h)
    files["header"] = header_h
    
    # Generate JavaScript interface
    interface_js = temp_path / "interface.js"
    _generate_js_interface(ir_data, interface_js)
    files["interface"] = interface_js
    
    # Generate CMake build file
    cmake_file = temp_path / "CMakeLists.txt"
    _generate_cmake_file(ir_data, cmake_file)
    files["cmake"] = cmake_file
    
    return files


def _generate_runtime_cpp(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Generate C++ runtime code."""
    use_simd = ir_data["optimization"]["use_simd"]
    use_threads = ir_data["optimization"]["use_threads"]
    
    cpp_code = f'''// Generated WASM runtime for PyTorch model
#include <emscripten.h>
#include <emscripten/bind.h>
#include <vector>
#include <memory>
{"#include <immintrin.h>" if use_simd else ""}
{"#include <thread>" if use_threads else ""}
#include "model.h"

using namespace emscripten;

class WASMModel {{
private:
    std::vector<float> weights;
    ModelConfig config;
    
public:
    WASMModel() {{
        // Initialize model configuration
        config.use_simd = {str(use_simd).lower()};
        config.use_threads = {str(use_threads).lower()};
        
        // Load model weights (placeholder)
        init_weights();
    }}
    
    void init_weights() {{
        // Initialize model weights from embedded data
        // This would be populated with actual model parameters
    }}
    
    std::vector<float> forward(const std::vector<float>& input) {{
        if (input.empty()) {{
            throw std::runtime_error("Input tensor cannot be empty");
        }}
        
        // Perform forward pass
        return run_inference(input);
    }}
    
    std::vector<float> run_inference(const std::vector<float>& input) {{
        std::vector<float> output = input; // Placeholder implementation
        
        {"// SIMD optimizations enabled" if use_simd else "// SIMD optimizations disabled"}
        {"// Multi-threading enabled" if use_threads else "// Multi-threading disabled"}
        
        // Apply model operations
        for (const auto& op : get_model_operations()) {{
            output = apply_operation(output, op);
        }}
        
        return output;
    }}
    
    int get_memory_usage() const {{
        return weights.size() * sizeof(float);
    }}
}};

EMSCRIPTEN_BINDINGS(wasm_torch) {{
    class_<WASMModel>("WASMModel")
        .constructor<>()
        .function("forward", &WASMModel::forward)
        .function("getMemoryUsage", &WASMModel::get_memory_usage);
        
    register_vector<float>("FloatVector");
}}
'''
    
    output_path.write_text(cpp_code)


def _generate_operations_cpp(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Generate model operations C++ code."""
    operations = ir_data["graph"]["operations"]
    
    cpp_code = '''// Generated model operations
#include "model.h"
#include <vector>
#include <string>
#include <stdexcept>

std::vector<ModelOperation> get_model_operations() {
    std::vector<ModelOperation> ops;
    
'''
    
    # Generate operations based on the model graph
    for i, op in enumerate(operations):
        op_kind = op["kind"]
        cpp_code += f'''    // Operation {i}: {op_kind}
    ops.push_back(ModelOperation{{
        .type = "{op_kind}",
        .id = {i}
    }});
    
'''
    
    cpp_code += '''    return ops;
}

std::vector<float> apply_operation(
    const std::vector<float>& input, 
    const ModelOperation& op
) {
    std::vector<float> output = input;
    
    // Apply operation based on type
    if (op.type == "aten::linear") {
        // Linear layer implementation
        return apply_linear(input, op);
    } else if (op.type == "aten::relu") {
        // ReLU activation implementation
        return apply_relu(input);
    }
    // Add more operations as needed
    
    return output;
}

std::vector<float> apply_linear(
    const std::vector<float>& input,
    const ModelOperation& op
) {
    // Placeholder linear operation
    std::vector<float> output = input;
    for (auto& val : output) {
        val = val * 0.5f + 0.1f; // Simple transformation
    }
    return output;
}

std::vector<float> apply_relu(const std::vector<float>& input) {
    std::vector<float> output = input;
    for (auto& val : output) {
        val = std::max(0.0f, val);
    }
    return output;
}
'''
    
    output_path.write_text(cpp_code)


def _generate_header_file(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Generate header file."""
    header_code = '''// Generated model header
#pragma once
#include <vector>
#include <string>

struct ModelConfig {
    bool use_simd;
    bool use_threads;
};

struct ModelOperation {
    std::string type;
    int id;
};

// Function declarations
std::vector<ModelOperation> get_model_operations();
std::vector<float> apply_operation(
    const std::vector<float>& input, 
    const ModelOperation& op
);
std::vector<float> apply_linear(
    const std::vector<float>& input,
    const ModelOperation& op
);
std::vector<float> apply_relu(const std::vector<float>& input);
'''
    
    output_path.write_text(header_code)


def _generate_js_interface(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Generate JavaScript interface."""
    input_shape = ir_data["model_info"]["input_shape"]
    
    js_code = f'''// Generated JavaScript interface for WASM model
class WASMTorchModel {{
    constructor(wasmModule) {{
        this.module = wasmModule;
        this.model = new wasmModule.WASMModel();
        this.inputShape = {input_shape};
    }}
    
    async forward(inputData) {{
        // Validate input
        if (!Array.isArray(inputData)) {{
            throw new Error('Input must be an array');
        }}
        
        // Convert to Float32Array if needed
        const floatInput = new Float32Array(inputData);
        
        // Create input vector
        const inputVector = new this.module.FloatVector();
        for (let i = 0; i < floatInput.length; i++) {{
            inputVector.push_back(floatInput[i]);
        }}
        
        try {{
            // Run inference
            const outputVector = this.model.forward(inputVector);
            
            // Convert result back to JavaScript array
            const result = [];
            for (let i = 0; i < outputVector.size(); i++) {{
                result.push(outputVector.get(i));
            }}
            
            return result;
        }} finally {{
            // Clean up vectors
            inputVector.delete();
        }}
    }}
    
    getMemoryUsage() {{
        return this.model.getMemoryUsage();
    }}
}}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = WASMTorchModel;
}}
'''
    
    output_path.write_text(js_code)


def _generate_cmake_file(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Generate CMake build configuration."""
    use_simd = ir_data["optimization"]["use_simd"]
    use_threads = ir_data["optimization"]["use_threads"]
    
    cmake_code = f'''# Generated CMake configuration for WASM build
cmake_minimum_required(VERSION 3.26)
project(wasm_torch_model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emscripten-specific settings
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    
    # Optimization flags
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -O3")
    {"set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -msimd128\")" if use_simd else ""}
    {"set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -pthread\")" if use_threads else ""}
    
    # Emscripten link flags
    set(EMSCRIPTEN_LINK_FLAGS
        "--bind"
        "-s WASM=1"
        "-s ALLOW_MEMORY_GROWTH=1"
        "-s MODULARIZE=1"
        "-s EXPORT_ES6=1"
        {"'-s USE_PTHREADS=1'" if use_threads else ""}
        {"'-s SIMD=1'" if use_simd else ""}
    )
    
    string(REPLACE ";" " " EMSCRIPTEN_LINK_FLAGS_STR "${{EMSCRIPTEN_LINK_FLAGS}}")
    set(CMAKE_EXE_LINKER_FLAGS "${{CMAKE_EXE_LINKER_FLAGS}} ${{EMSCRIPTEN_LINK_FLAGS_STR}}")
endif()

# Source files
set(SOURCES
    wasm_runtime.cpp
    model_ops.cpp
)

# Create executable
add_executable(wasm_torch_model ${{SOURCES}})

# Include directories
target_include_directories(wasm_torch_model PRIVATE .)
'''
    
    output_path.write_text(cmake_code)


def _compile_to_wasm(
    compilation_files: Dict[str, Path],
    output_path: Path,
    optimization_level: str,
    use_simd: bool,
    use_threads: bool,
    **kwargs: Any
) -> None:
    """Compile generated C++ code to WASM."""
    # Check for Emscripten
    if not _check_emscripten():
        raise RuntimeError(
            "Emscripten not found. Please install Emscripten SDK:\n"
            "https://emscripten.org/docs/getting_started/downloads.html"
        )
    
    build_dir = compilation_files["cmake"].parent / "build"
    build_dir.mkdir(exist_ok=True)
    
    try:
        # Configure with CMake
        logger.info("Configuring build with CMake...")
        cmake_cmd = [
            "emcmake", "cmake", "..",
            f"-DCMAKE_BUILD_TYPE=Release"
        ]
        subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True)
        
        # Build with Make
        logger.info("Building WASM module...")
        make_cmd = ["emmake", "make", "-j4"]
        result = subprocess.run(make_cmd, cwd=build_dir, check=True, capture_output=True)
        
        # Copy output files
        wasm_file = build_dir / "wasm_torch_model.wasm"
        js_file = build_dir / "wasm_torch_model.js"
        
        if wasm_file.exists():
            shutil.copy2(wasm_file, output_path)
            # Also copy the JS loader if requested
            if output_path.suffix == ".wasm":
                js_output = output_path.with_suffix(".js")
                if js_file.exists():
                    shutil.copy2(js_file, js_output)
        else:
            raise RuntimeError("WASM compilation failed - output file not found")
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown build error"
        raise RuntimeError(f"WASM compilation failed: {error_msg}") from e


def _check_emscripten() -> bool:
    """Check if Emscripten is available."""
    try:
        result = subprocess.run(
            ["emcc", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def register_custom_op(name: str):
    """Decorator for registering custom WASM operators.
    
    Args:
        name: Name of the custom operator
        
    Returns:
        Decorator function that registers the custom operator
    """
    def decorator(func):
        # Store custom operator in global registry
        if not hasattr(register_custom_op, '_custom_ops'):
            register_custom_op._custom_ops = {}
        
        register_custom_op._custom_ops[name] = {
            'function': func,
            'name': name,
            'signature': func.__annotations__ if hasattr(func, '__annotations__') else {}
        }
        
        logger.info(f"Registered custom operator: {name}")
        return func
    
    return decorator


def get_custom_operators() -> Dict[str, Any]:
    """Get all registered custom operators.
    
    Returns:
        Dictionary of registered custom operators
    """
    return getattr(register_custom_op, '_custom_ops', {})