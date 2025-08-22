#!/usr/bin/env python3
"""
Test our mock system to ensure it's working correctly.
"""

import sys
import os
sys.path.insert(0, '/root/repo')
sys.path.insert(0, '/root/repo/src')

def test_mock_imports():
    """Test that all our mock modules work."""
    print("Testing mock system imports...")
    
    try:
        # Test mock torch
        from src.wasm_torch.mock_torch import torch, MockTensor
        print("✓ Mock torch import successful")
        
        # Test creating tensors
        tensor = torch.randn(3, 3)
        print(f"✓ Mock tensor created: {len(tensor.data)} elements")
        
        # Test tensor operations
        tensor2 = torch.zeros(3, 3)
        result = tensor + tensor2
        print(f"✓ Mock tensor operations work")
        
    except Exception as e:
        print(f"✗ Mock torch failed: {e}")
        return False
    
    try:
        # Test mock numpy
        import numpy as np
        arr = np.zeros(5)
        print("✓ Mock numpy import successful")
        
        # Test ndarray
        test_arr = np.array([1, 2, 3])
        print(f"✓ Mock numpy array created: {test_arr.data}")
        
    except Exception as e:
        print(f"✗ Mock numpy failed: {e}")
        return False
    
    try:
        # Test main module
        import wasm_torch
        print("✓ Main wasm_torch module import successful")
        print(f"✓ Torch available: {getattr(wasm_torch, 'torch_available', 'Unknown')}")
        
        # Test basic functionality
        from wasm_torch.basic_model_loader import MockExporter, MockWASMRuntime
        exporter = MockExporter()
        runtime = MockWASMRuntime()
        print("✓ Mock implementations work")
        
    except Exception as e:
        print(f"✗ Main module failed: {e}")
        return False
    
    return True

def test_mock_functionality():
    """Test that mock functionality actually works."""
    print("\nTesting mock functionality...")
    
    try:
        from src.wasm_torch.mock_torch import torch
        
        # Create and test a mock model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        input_tensor = torch.randn(1, 10)
        
        # Test forward pass
        output = model(input_tensor)
        print("✓ Mock model forward pass works")
        
        # Test export functionality
        from wasm_torch.basic_model_loader import MockExporter
        exporter = MockExporter()
        
        wasm_path = "/tmp/test_model.wasm"
        result = exporter.export_to_wasm(model, wasm_path, input_tensor)
        
        if os.path.exists(wasm_path):
            print("✓ Mock WASM export works")
            os.remove(wasm_path)
            
            # Clean up metadata file
            metadata_path = wasm_path.replace('.wasm', '.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        else:
            print("✗ Mock WASM export failed - no file created")
            return False
            
    except Exception as e:
        print(f"✗ Mock functionality test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🔍 Testing Mock System")
    print("=" * 50)
    
    success = True
    success &= test_mock_imports()
    success &= test_mock_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All mock system tests passed!")
    else:
        print("❌ Some mock system tests failed!")
        sys.exit(1)