# CMakeLists.txt for WASM Shim for Torch
# Cross-platform build configuration for C++/WASM components

cmake_minimum_required(VERSION 3.26.0)

# Project definition
project(WASMTorch 
    VERSION 0.1.0
    DESCRIPTION "WebAssembly System Interface (WASI-NN) shim for PyTorch"
    LANGUAGES CXX C
)

# C++ standard requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(EMSCRIPTEN)
    set(PLATFORM_WASM ON)
    message(STATUS "Building for WebAssembly with Emscripten")
else()
    set(PLATFORM_NATIVE ON)
    message(STATUS "Building for native platform")
endif()

# Configuration options
option(USE_SIMD "Enable SIMD optimizations" ON)
option(USE_THREADS "Enable threading support" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(ENABLE_DEBUG_LOGGING "Enable debug logging" OFF)

# Version configuration
set(WASM_TORCH_VERSION_MAJOR 0)
set(WASM_TORCH_VERSION_MINOR 1)
set(WASM_TORCH_VERSION_PATCH 0)
set(WASM_TORCH_VERSION "${WASM_TORCH_VERSION_MAJOR}.${WASM_TORCH_VERSION_MINOR}.${WASM_TORCH_VERSION_PATCH}")

# Configure version header
configure_file(
    "${CMAKE_SOURCE_DIR}/runtime/include/version.h.in"
    "${CMAKE_BINARY_DIR}/include/wasm_torch/version.h"
    @ONLY
)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# Common compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall 
        -Wextra 
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# Platform-specific configuration
if(PLATFORM_WASM)
    # Emscripten-specific settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
    
    # SIMD support
    if(USE_SIMD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msimd128")
        add_definitions(-DWASM_TORCH_USE_SIMD=1)
    endif()
    
    # Threading support
    if(USE_THREADS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
        add_definitions(-DWASM_TORCH_USE_THREADS=1)
    endif()
    
    # Emscripten linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s WASM=1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MAXIMUM_MEMORY=4GB")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MODULARIZE=1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORT_NAME='WASMTorchModule'")
    
    if(USE_THREADS)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s USE_PTHREADS=1")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s PTHREAD_POOL_SIZE=4")
    endif()
    
else()
    # Native platform settings
    if(USE_SIMD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            add_compile_options(-msse4.1 -mavx -mavx2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            add_compile_options(-march=armv8-a+simd)
        endif()
        add_definitions(-DWASM_TORCH_USE_SIMD=1)
    endif()
    
    if(USE_THREADS)
        find_package(Threads REQUIRED)
        add_definitions(-DWASM_TORCH_USE_THREADS=1)
    endif()
endif()

# Additional definitions
if(ENABLE_DEBUG_LOGGING)
    add_definitions(-DWASM_TORCH_DEBUG_LOGGING=1)
endif()

if(ENABLE_PROFILING)
    add_definitions(-DWASM_TORCH_PROFILING=1)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_BINARY_DIR}/include
)

# Source files
set(RUNTIME_SOURCES
    runtime/src/runtime.cpp
    runtime/src/memory_manager.cpp
    runtime/src/tensor.cpp
    runtime/src/model_loader.cpp
    runtime/src/executor.cpp
    runtime/src/thread_pool.cpp
)

# Kernel sources
file(GLOB KERNEL_SOURCES "runtime/src/kernels/*.cpp")
list(APPEND RUNTIME_SOURCES ${KERNEL_SOURCES})

# SIMD kernel sources
if(USE_SIMD)
    file(GLOB SIMD_KERNEL_SOURCES "runtime/src/kernels/simd/*.cpp")
    list(APPEND RUNTIME_SOURCES ${SIMD_KERNEL_SOURCES})
endif()

# Create main library
add_library(wasm_torch_runtime ${RUNTIME_SOURCES})

# Link libraries
if(PLATFORM_NATIVE AND USE_THREADS)
    target_link_libraries(wasm_torch_runtime Threads::Threads)
endif()

# Create WASM module
if(PLATFORM_WASM)
    add_executable(wasm_torch_module
        runtime/src/wasm_bindings.cpp
        $<TARGET_OBJECTS:wasm_torch_runtime>
    )
    
    # Export C functions for JavaScript
    set_target_properties(wasm_torch_module PROPERTIES
        LINK_FLAGS "-s EXPORTED_FUNCTIONS=['_malloc','_free','_init_runtime','_load_model','_forward','_cleanup']"
    )
endif()

# Python bindings (using pybind11)
if(PLATFORM_NATIVE AND BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(wasm_torch_native
        runtime/src/python_bindings.cpp
    )
    
    target_link_libraries(wasm_torch_native PRIVATE wasm_torch_runtime)
endif()

# Tests
if(BUILD_TESTS AND PLATFORM_NATIVE)
    enable_testing()
    
    # Find Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Download and build Google Test
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50f33f9142bb2e7c8b4e9a1a1a3a2a.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Test sources
    file(GLOB TEST_SOURCES "tests/cpp/*.cpp")
    
    add_executable(wasm_torch_tests ${TEST_SOURCES})
    target_link_libraries(wasm_torch_tests 
        wasm_torch_runtime 
        GTest::gtest_main
    )
    
    # Register tests
    include(GoogleTest)
    gtest_discover_tests(wasm_torch_tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS AND PLATFORM_NATIVE)
    # Find Google Benchmark
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/v1.8.3.zip
        )
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
    
    # Benchmark sources
    file(GLOB BENCHMARK_SOURCES "benchmarks/cpp/*.cpp")
    
    add_executable(wasm_torch_benchmarks ${BENCHMARK_SOURCES})
    target_link_libraries(wasm_torch_benchmarks 
        wasm_torch_runtime 
        benchmark::benchmark
    )
endif()

# Installation rules
if(PLATFORM_NATIVE)
    install(TARGETS wasm_torch_runtime
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(DIRECTORY runtime/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )
    
    install(FILES "${CMAKE_BINARY_DIR}/include/wasm_torch/version.h"
        DESTINATION include/wasm_torch
    )
endif()

# Package configuration
if(PLATFORM_WASM)
    # Install WASM artifacts
    install(FILES 
        "${CMAKE_BINARY_DIR}/wasm_torch_module.wasm"
        "${CMAKE_BINARY_DIR}/wasm_torch_module.js"
        DESTINATION share/wasm_torch
    )
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(
            "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in"
            "${CMAKE_BINARY_DIR}/Doxyfile"
            @ONLY
        )
        
        add_custom_target(docs
            ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "WASM Torch Configuration Summary:")
message(STATUS "  Version: ${WASM_TORCH_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  SIMD Support: ${USE_SIMD}")
message(STATUS "  Threading: ${USE_THREADS}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Debug Logging: ${ENABLE_DEBUG_LOGGING}")
message(STATUS "  Profiling: ${ENABLE_PROFILING}")
if(PLATFORM_WASM)
    message(STATUS "  Emscripten: ${EMSCRIPTEN_VERSION}")
endif()
message(STATUS "")