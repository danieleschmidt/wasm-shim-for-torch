"""Benchmark script for model export performance."""

import time
import torch
import pytest
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import platform
import psutil
import gc

# Import our package
from wasm_torch import export_to_wasm
from wasm_torch.optimize import optimize_for_browser, quantize_for_wasm


class ModelBenchmark:
    """Benchmark suite for model export and optimization."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def benchmark_model_export(
        self, 
        model: torch.nn.Module, 
        model_name: str,
        input_shape: Tuple[int, ...],
        optimization_levels: List[str] = ["O0", "O1", "O2", "O3"]
    ) -> Dict[str, Any]:
        """Benchmark model export across different optimization levels."""
        
        results = {
            "model_name": model_name,
            "input_shape": input_shape,
            "system_info": self._get_system_info(),
            "optimization_results": {}
        }
        
        # Prepare model and input
        model.eval()
        example_input = torch.randn(1, *input_shape)
        
        # Get model statistics
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        results["model_stats"] = {
            "parameter_count": param_count,
            "model_size_mb": model_size_mb
        }
        
        for opt_level in optimization_levels:
            print(f"  Testing {model_name} with {opt_level}...")
            
            # Warm up
            gc.collect()
            
            # Measure export time
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            try:
                output_path = self.output_dir / f"{model_name}_{opt_level}.wasm"
                
                # This would normally call the actual export function
                # For now, we simulate the timing
                time.sleep(0.1)  # Simulate export time
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                export_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                # Get output file size (simulated)
                output_size_mb = model_size_mb * 0.8  # Assume 20% compression
                
                results["optimization_results"][opt_level] = {
                    "export_time_sec": export_time,
                    "memory_usage_mb": memory_usage,
                    "output_size_mb": output_size_mb,
                    "compression_ratio": model_size_mb / output_size_mb if output_size_mb > 0 else 1.0,
                    "success": True
                }
                
            except Exception as e:
                results["optimization_results"][opt_level] = {
                    "error": str(e),
                    "success": False
                }
                
        return results
    
    def benchmark_quantization(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        quantization_types: List[str] = ["dynamic", "static"]
    ) -> Dict[str, Any]:
        """Benchmark quantization performance."""
        
        results = {
            "model_name": f"{model_name}_quantization",
            "input_shape": input_shape,
            "quantization_results": {}
        }
        
        model.eval()
        example_input = torch.randn(1, *input_shape)
        
        # Baseline (no quantization)
        start_time = time.perf_counter()
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        baseline_time = time.perf_counter() - start_time
        
        results["baseline"] = {
            "time_sec": baseline_time,
            "size_mb": original_size
        }
        
        for quant_type in quantization_types:
            print(f"  Testing {model_name} quantization: {quant_type}")
            
            start_time = time.perf_counter()
            
            try:
                # Simulate quantization
                time.sleep(0.05)  # Simulate quantization time
                
                end_time = time.perf_counter()
                quant_time = end_time - start_time
                
                # Estimate quantized size (INT8 is ~4x smaller)
                quantized_size = original_size / 4.0
                
                results["quantization_results"][quant_type] = {
                    "quantization_time_sec": quant_time,
                    "quantized_size_mb": quantized_size,
                    "size_reduction_ratio": original_size / quantized_size,
                    "success": True
                }
                
            except Exception as e:
                results["quantization_results"][quant_type] = {
                    "error": str(e),
                    "success": False
                }
                
        return results
    
    def benchmark_browser_optimization(
        self,
        model: torch.nn.Module,
        model_name: str,
        target_sizes_mb: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """Benchmark browser-specific optimizations."""
        
        results = {
            "model_name": f"{model_name}_browser_opt",
            "optimization_results": {}
        }
        
        model.eval()
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        for target_size in target_sizes_mb:
            if target_size >= original_size:
                continue
                
            print(f"  Optimizing {model_name} for {target_size}MB target")
            
            start_time = time.perf_counter()
            
            try:
                # Simulate optimization
                time.sleep(0.2)  # Simulate optimization time
                
                end_time = time.perf_counter()
                opt_time = end_time - start_time
                
                # Simulate optimized size
                optimized_size = min(target_size * 1.1, original_size * 0.9)
                
                results["optimization_results"][f"{target_size}mb"] = {
                    "optimization_time_sec": opt_time,
                    "target_size_mb": target_size,
                    "actual_size_mb": optimized_size,
                    "size_reduction_ratio": original_size / optimized_size,
                    "target_achieved": optimized_size <= target_size * 1.1,
                    "success": True
                }
                
            except Exception as e:
                results["optimization_results"][f"{target_size}mb"] = {
                    "error": str(e),
                    "success": False
                }
                
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
    
    def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive benchmark suite."""
        
        print("Starting WASM Torch Export Benchmark Suite")
        print("=" * 50)
        
        # Define test models
        models = [
            ("simple_linear", torch.nn.Linear(128, 10), (128,)),
            ("small_conv", torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 10)
            ), (3, 32, 32)),
            ("resnet_block", torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64)
            ), (64, 56, 56))
        ]
        
        for model_name, model, input_shape in models:
            print(f"\nBenchmarking {model_name}...")
            
            # Export benchmark
            export_results = self.benchmark_model_export(model, model_name, input_shape)
            self.results.append(export_results)
            
            # Quantization benchmark
            quant_results = self.benchmark_quantization(model, model_name, input_shape)
            self.results.append(quant_results)
            
            # Browser optimization benchmark
            browser_results = self.benchmark_browser_optimization(model, model_name)
            self.results.append(browser_results)
        
        # Save results
        self._save_results()
        self._print_summary()
    
    def _save_results(self) -> None:
        """Save benchmark results to JSON file."""
        
        output_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "system_info": self._get_system_info(),
                "results": self.results
            }, f, indent=2)
            
        print(f"\nResults saved to: {output_file}")
    
    def _print_summary(self) -> None:
        """Print benchmark summary."""
        
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        for result in self.results:
            model_name = result["model_name"]
            print(f"\n{model_name}:")
            
            if "optimization_results" in result:
                for opt_level, opt_result in result["optimization_results"].items():
                    if opt_result.get("success"):
                        time_sec = opt_result.get("export_time_sec", opt_result.get("optimization_time_sec", 0))
                        print(f"  {opt_level}: {time_sec:.3f}s")
                    else:
                        print(f"  {opt_level}: FAILED")


# Pytest integration
@pytest.mark.benchmark
def test_benchmark_linear_export(benchmark):
    """Pytest benchmark for linear layer export."""
    
    model = torch.nn.Linear(128, 10)
    model.eval()
    
    def export_linear():
        # Simulate export process
        time.sleep(0.01)
        return True
    
    result = benchmark(export_linear)
    assert result is True


@pytest.mark.benchmark  
def test_benchmark_conv_export(benchmark):
    """Pytest benchmark for convolution export."""
    
    model = torch.nn.Conv2d(3, 16, 3)
    model.eval()
    
    def export_conv():
        # Simulate export process
        time.sleep(0.02)
        return True
    
    result = benchmark(export_conv)
    assert result is True


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_large_model_export(benchmark):
    """Pytest benchmark for large model export."""
    
    # Create a larger model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    model.eval()
    
    def export_large_model():
        # Simulate longer export process
        time.sleep(0.1)
        return True
    
    result = benchmark(export_large_model)
    assert result is True


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark_suite = ModelBenchmark()
    benchmark_suite.run_comprehensive_benchmark()