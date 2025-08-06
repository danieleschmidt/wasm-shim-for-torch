#!/usr/bin/env python3
"""
Basic example of exporting a PyTorch model to WASM format.

This example demonstrates the core functionality of the WASM export system,
including model tracing, compilation unit generation, and WASM compilation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from wasm_torch import export_to_wasm, register_custom_op, get_custom_operators


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@register_custom_op("scaled_attention")
def scaled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Custom scaled dot-product attention operator."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def main():
    """Main export demonstration."""
    print("🚀 WASM Torch Export Demo")
    print("=" * 40)
    
    # Create a simple model
    print("📦 Creating model...")
    model = SimpleClassifier(input_size=784, hidden_size=64, num_classes=10)
    model.eval()
    
    # Create example input (28x28 flattened image)
    print("🔢 Creating example input...")
    example_input = torch.randn(1, 784)
    
    # Set output path
    output_dir = Path("./wasm_models")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "simple_classifier.wasm"
    
    print(f"📁 Output path: {output_path}")
    
    # Export configurations to test
    export_configs = [
        {"optimization_level": "O2", "use_simd": True, "use_threads": True, "name": "optimized"},
        {"optimization_level": "O0", "use_simd": False, "use_threads": False, "name": "basic"},
    ]
    
    for config in export_configs:
        name = config.pop("name")
        config_output = output_path.with_stem(f"{output_path.stem}_{name}")
        
        print(f"\n🔄 Exporting {name} configuration...")
        print(f"   Optimization: {config['optimization_level']}")
        print(f"   SIMD: {config['use_simd']}")
        print(f"   Threads: {config['use_threads']}")
        
        try:
            export_to_wasm(
                model=model,
                output_path=config_output,
                example_input=example_input,
                **config
            )
            print(f"✅ Export successful: {config_output}")
            
            # Check output files
            if config_output.exists():
                size_mb = config_output.stat().st_size / (1024 * 1024)
                print(f"   📏 WASM file size: {size_mb:.2f} MB")
            
            js_file = config_output.with_suffix(".js")
            if js_file.exists():
                print(f"   📄 JavaScript loader: {js_file}")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    print(f"\n🎯 Custom operators registered: {list(get_custom_operators().keys())}")
    print("\n🏁 Demo completed!")


if __name__ == "__main__":
    main()