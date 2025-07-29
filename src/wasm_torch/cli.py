"""Command-line interface for wasm-torch."""

import argparse
import sys
from pathlib import Path
from typing import Optional


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
    
    args = parser.parse_args()
    
    if args.command == "download-runtime":
        download_runtime(args.version, args.output_dir)
    elif args.command == "export":
        export_model(args.model_path, args.output, args.optimization_level)
    else:
        parser.print_help()
        sys.exit(1)


def download_runtime(version: str, output_dir: Path) -> None:
    """Download pre-built WASM runtime.
    
    Args:
        version: Runtime version to download
        output_dir: Directory to save runtime files
    """
    print(f"Downloading WASM runtime version {version} to {output_dir}")
    raise NotImplementedError("Runtime download not yet implemented.")


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
    print(f"Exporting {model_path} to {output_path} with {optimization_level}")
    raise NotImplementedError("Model export not yet implemented.")


if __name__ == "__main__":
    main()