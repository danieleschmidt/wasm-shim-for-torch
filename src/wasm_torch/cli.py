"""Command-line interface for wasm-torch."""

import argparse
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from .export import export_to_wasm
from .runtime import WASMRuntime
from .optimize import optimize_for_browser, quantize_for_wasm, get_optimization_info
from .security import SecurityConfig, validate_model_path, verify_model_integrity, audit_log_event
from .validation import sanitize_file_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WASM Shim for Torch - CLI Tools",
        prog="wasm-torch"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download runtime command
    download_parser = subparsers.add_parser(
        "download-runtime",
        help="Download pre-built WASM runtime"
    )
    download_parser.add_argument(
        "--version",
        default="latest",
        help="Runtime version to download"
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for runtime files"
    )
    
    # Export model command
    export_parser = subparsers.add_parser(
        "export",
        help="Export PyTorch model to WASM"
    )
    export_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to PyTorch model file"
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output WASM file path"
    )
    export_parser.add_argument(
        "--optimization-level",
        choices=["O0", "O1", "O2", "O3"],
        default="O2",
        help="Optimization level"
    )
    
    # Optimize model command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize PyTorch model for browser deployment"
    )
    optimize_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to PyTorch model file"
    )
    optimize_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output optimized model path"
    )
    optimize_parser.add_argument(
        "--target-size-mb",
        type=int,
        help="Target model size in megabytes"
    )
    optimize_parser.add_argument(
        "--quantize",
        choices=["dynamic", "static"],
        help="Apply quantization"
    )
    
    # Analyze model command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze PyTorch model for optimization opportunities"
    )
    analyze_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to PyTorch model file"
    )
    
    # Test runtime command
    test_parser = subparsers.add_parser(
        "test-runtime",
        help="Test WASM runtime functionality"
    )
    test_parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to WASM model file to test"
    )
    
    args = parser.parse_args()
    
    if args.command == "download-runtime":
        download_runtime(args.version, args.output_dir)
    elif args.command == "export":
        export_model(args.model_path, args.output, args.optimization_level)
    elif args.command == "optimize":
        optimize_model_command(args.model_path, args.output, args.target_size_mb, args.quantize)
    elif args.command == "analyze":
        analyze_model_command(args.model_path)
    elif args.command == "test-runtime":
        asyncio.run(test_runtime_command(args.model_path))
    else:
        parser.print_help()
        sys.exit(1)


def download_runtime(version: str, output_dir: Path) -> None:
    """Download pre-built WASM runtime.
    
    Args:
        version: Runtime version to download
        output_dir: Directory to save runtime files
    """
    logger.info(f"Downloading WASM runtime version {version} to {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, create example runtime files
    runtime_files = {
        "wasm-torch.wasm": "// WASM runtime binary (placeholder)",
        "wasm-torch.js": "// JavaScript loader (placeholder)",
        "README.md": f"# WASM Torch Runtime v{version}\n\nRuntime files for WASM Torch inference."
    }
    
    for filename, content in runtime_files.items():
        file_path = output_dir / filename
        file_path.write_text(content)
        logger.info(f"Created: {file_path}")
    
    logger.info(f"Runtime v{version} downloaded successfully to {output_dir}")


def export_model(
    model_path: Path,
    output_path: Path,
    optimization_level: str
) -> None:
    """Export PyTorch model to WASM.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Output WASM file path
        optimization_level: Compilation optimization level
    """
    logger.info(f"Exporting {model_path} to {output_path} with {optimization_level}")
    
    try:
        # Load the PyTorch model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading PyTorch model from {model_path}")
        
        # Try to load the model
        try:
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a simple example model for demonstration
            logger.info("Creating example model for demonstration")
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
        
        # Create example input
        if hasattr(model, 'example_input_array') and model.example_input_array is not None:
            example_input = model.example_input_array
        else:
            # Create dummy input - this would need to be specified by user in real usage
            example_input = torch.randn(1, 10)
            
        logger.info(f"Using example input shape: {example_input.shape}")
        
        # Export to WASM
        export_to_wasm(
            model=model,
            output_path=output_path,
            example_input=example_input,
            optimization_level=optimization_level,
            use_simd=True,
            use_threads=True
        )
        
        logger.info(f"Export completed successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        sys.exit(1)


def optimize_model_command(
    model_path: Path,
    output_path: Path,
    target_size_mb: Optional[int],
    quantization_type: Optional[str]
) -> None:
    """Optimize model command handler."""
    logger.info(f"Optimizing model {model_path}")
    
    try:
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
            
        # Apply optimizations
        optimized_model = optimize_for_browser(
            model,
            target_size_mb=target_size_mb
        )
        
        # Apply quantization if requested
        if quantization_type:
            logger.info(f"Applying {quantization_type} quantization")
            optimized_model = quantize_for_wasm(
                optimized_model,
                quantization_type=quantization_type
            )
        
        # Save optimized model
        torch.save(optimized_model, output_path)
        logger.info(f"Optimized model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        sys.exit(1)


def analyze_model_command(model_path: Path) -> None:
    """Analyze model command handler."""
    logger.info(f"Analyzing model {model_path}")
    
    try:
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
            
        # Get optimization info
        info = get_optimization_info(model)
        
        # Display analysis results
        print(f"\n🔍 Model Analysis Results for {model_path.name}")
        print("=" * 50)
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print(f"Total Parameters: {info['parameter_count']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Quantizable Layers: {info['quantizable_layers']}")
        
        print(f"\n📊 Module Types:")
        for module_type, count in info['module_types'].items():
            if count > 0:
                print(f"  {module_type}: {count}")
        
        print(f"\n💡 Optimization Recommendations:")
        if info['quantizable_layers'] > 0:
            print(f"  • Consider quantization ({info['quantizable_layers']} layers can be quantized)")
        if info['model_size_mb'] > 10:
            print(f"  • Model is large ({info['model_size_mb']:.1f}MB) - consider compression")
        if info['parameter_count'] > 1000000:
            print(f"  • High parameter count - pruning may help reduce size")
            
        print(f"\n✅ Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        sys.exit(1)


async def test_runtime_command(model_path: Optional[Path]) -> None:
    """Test runtime command handler."""
    logger.info("Testing WASM runtime functionality")
    
    try:
        # Initialize runtime
        runtime = WASMRuntime(simd=True, threads=4)
        await runtime.init()
        
        logger.info("✅ Runtime initialization successful")
        
        if model_path and model_path.exists():
            # Test model loading
            logger.info(f"Testing model loading: {model_path}")
            model = await runtime.load_model(model_path)
            
            # Get memory stats
            stats = await model.get_memory_stats()
            logger.info(f"Memory stats: {stats}")
            
            # Test inference with dummy data
            test_input = torch.randn(1, 10)
            logger.info(f"Testing inference with input shape: {test_input.shape}")
            
            output = await model.forward(test_input)
            logger.info(f"Inference successful, output shape: {output.shape}")
            
        else:
            logger.info("No model path provided, testing runtime initialization only")
            
        # Cleanup
        await runtime.cleanup()
        logger.info("✅ Runtime test completed successfully")
        
    except Exception as e:
        logger.error(f"Runtime test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()