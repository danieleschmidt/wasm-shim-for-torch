"""Simple example of using WASM Torch for model export and inference."""

import asyncio
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wasm_torch import export_to_wasm, optimize_for_browser, quantize_for_wasm
from wasm_torch.runtime import WASMRuntime


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


async def main():
    """Demonstrate WASM Torch functionality."""
    print("🚀 WASM Torch Example")
    print("=" * 50)
    
    # 1. Create and train a simple model
    print("1. Creating simple model...")
    model = SimpleModel(input_size=10, hidden_size=20, output_size=1)
    
    # Generate some dummy training data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print("2. Training model (10 epochs)...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.4f}")
    
    model.eval()
    print(f"   Training completed! Final loss: {loss.item():.4f}")
    
    # 2. Optimize model for browser
    print("3. Optimizing model for browser deployment...")
    try:
        optimized_model = optimize_for_browser(
            model,
            target_size_mb=5,
            optimization_passes=["fuse_operations", "optimize_memory_layout"]
        )
        print("   ✅ Model optimization completed")
    except Exception as e:
        print(f"   ⚠️ Optimization failed (expected in demo): {e}")
        optimized_model = model
    
    # 3. Apply quantization
    print("4. Applying quantization...")
    try:
        quantized_model = quantize_for_wasm(
            optimized_model,
            quantization_type="dynamic"
        )
        print("   ✅ Quantization completed")
    except Exception as e:
        print(f"   ⚠️ Quantization failed (expected in demo): {e}")
        quantized_model = optimized_model
    
    # 4. Export to WASM
    print("5. Exporting model to WASM...")
    example_input = torch.randn(1, 10)
    output_path = Path("output/simple_model.wasm")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        export_to_wasm(
            model=quantized_model,
            output_path=output_path,
            example_input=example_input,
            optimization_level="O2",
            use_simd=True,
            use_threads=True
        )
        print(f"   ✅ Model exported to {output_path}")
    except Exception as e:
        print(f"   ⚠️ Export failed (expected without Emscripten): {e}")
        # Create dummy WASM file for demo
        output_path.write_text("// Dummy WASM content for demo")
        print(f"   📝 Created dummy WASM file for demo: {output_path}")
    
    # 5. Test WASM runtime
    print("6. Testing WASM runtime...")
    try:
        runtime = WASMRuntime(simd=True, threads=4, memory_limit_mb=512)
        await runtime.init()
        print("   ✅ Runtime initialized successfully")
        
        # Test model loading (will work with our implementation)
        if output_path.exists():
            print(f"   📁 Loading model from {output_path}")
            wasm_model = await runtime.load_model(output_path)
            print("   ✅ Model loaded successfully")
            
            # Test inference
            test_input = torch.randn(1, 10)
            print(f"   🧮 Running inference with input shape: {test_input.shape}")
            
            output = await wasm_model.forward(test_input)
            print(f"   ✅ Inference completed! Output shape: {output.shape}")
            print(f"      Output value: {output.item():.4f}")
            
            # Get memory stats
            memory_stats = await wasm_model.get_memory_stats()
            print(f"   📊 Memory usage: {memory_stats}")
            
        await runtime.cleanup()
        print("   🧹 Runtime cleaned up")
        
    except Exception as e:
        print(f"   ❌ Runtime test failed: {e}")
    
    # 6. Performance comparison
    print("7. Performance comparison...")
    
    # Native PyTorch inference
    print("   🐍 Native PyTorch inference:")
    native_input = torch.randn(1, 10)
    
    import time
    start_time = time.time()
    with torch.no_grad():
        native_output = model(native_input)
    native_time = time.time() - start_time
    
    print(f"      Time: {native_time*1000:.2f}ms")
    print(f"      Output: {native_output.item():.4f}")
    
    print("\n🎉 WASM Torch example completed!")
    print("This example demonstrates the core functionality of WASM Torch:")
    print("  • Model optimization for browser deployment")
    print("  • Quantization for size reduction")
    print("  • WASM export pipeline (requires Emscripten)")
    print("  • WASM runtime for browser inference")
    print("\nNext steps:")
    print("  • Install Emscripten for actual WASM compilation")
    print("  • Test in a browser environment")
    print("  • Optimize for your specific model architecture")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())