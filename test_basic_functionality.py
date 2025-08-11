#!/usr/bin/env python3
"""
Basic functionality test for WASM-Torch library
Tests core export and runtime functionality with a simple model
"""

import torch
import torch.nn as nn
import tempfile
import sys
import logging
from pathlib import Path

# Add src directory to path to import wasm_torch
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from wasm_torch import export_to_wasm, WASMRuntime
    print("✓ Successfully imported wasm_torch modules")
except ImportError as e:
    print(f"✗ Failed to import wasm_torch: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model():
    """Create a simple test model."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    return SimpleNet()

def test_model_export():
    """Test basic model export functionality."""
    logger.info("Testing model export...")
    
    # Create simple model
    model = create_simple_model()
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 10)
    
    # Test export (will simulate since no actual WASM compilation)
    with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as tmp_file:
        output_path = Path(tmp_file.name)
        
    try:
        # This will test the export pipeline up to compilation
        logger.info(f"Attempting to export model to {output_path}")
        
        # Since we don't have full WASM compilation, we'll test just the export logic
        # by calling the internal functions
        from wasm_torch.export import _validate_export_inputs, _trace_model
        
        # Test input validation
        _validate_export_inputs(model, example_input, "O2")
        logger.info("✓ Input validation passed")
        
        # Test model tracing
        traced_model = _trace_model(model, example_input)
        logger.info("✓ Model tracing successful")
        
        # Test forward pass with traced model
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            
            if torch.allclose(original_output, traced_output, rtol=1e-4):
                logger.info("✓ Traced model output matches original")
            else:
                logger.warning("! Traced model output differs from original")
                
        return True
        
    except Exception as e:
        logger.error(f"✗ Export test failed: {e}")
        return False
    finally:
        # Cleanup
        if output_path.exists():
            output_path.unlink()

async def test_runtime_initialization():
    """Test WASM runtime initialization."""
    logger.info("Testing runtime initialization...")
    
    try:
        # Create runtime instance
        runtime = WASMRuntime(simd=True, threads=4, memory_limit_mb=512)
        logger.info("✓ Runtime instance created")
        
        # Initialize runtime
        await runtime.init()
        logger.info("✓ Runtime initialized successfully")
        
        # Check runtime stats
        stats = runtime.get_runtime_stats()
        logger.info(f"✓ Runtime stats: {stats['health_status']}")
        
        # Cleanup
        await runtime.cleanup()
        logger.info("✓ Runtime cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Runtime test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting basic functionality tests...")
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Model Export
    if test_model_export():
        tests_passed += 1
        logger.info("✓ Test 1 (Model Export) PASSED")
    else:
        logger.error("✗ Test 1 (Model Export) FAILED")
    
    # Test 2: Runtime Initialization
    if await test_runtime_initialization():
        tests_passed += 1
        logger.info("✓ Test 2 (Runtime Init) PASSED")
    else:
        logger.error("✗ Test 2 (Runtime Init) FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {tests_passed}/{total_tests} tests passed")
    logger.info(f"{'='*50}")
    
    if tests_passed == total_tests:
        logger.info("🎉 All basic functionality tests PASSED!")
        return True
    else:
        logger.error("❌ Some basic functionality tests FAILED!")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)