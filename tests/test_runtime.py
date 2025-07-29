"""Tests for WASM runtime functionality."""

import pytest
from pathlib import Path
from wasm_torch.runtime import WASMRuntime, WASMModel


@pytest.fixture
def runtime():
    """Create a WASM runtime instance."""
    return WASMRuntime(simd=True, threads=4, memory_limit_mb=512)


@pytest.mark.asyncio
async def test_runtime_init_not_implemented(runtime):
    """Test that runtime initialization raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        await runtime.init()


@pytest.mark.asyncio
async def test_load_model_not_implemented(runtime):
    """Test that model loading raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        await runtime.load_model("model.wasm")


def test_runtime_attributes(runtime):
    """Test runtime attribute initialization."""
    assert runtime.simd is True
    assert runtime.threads == 4
    assert runtime.memory_limit_mb == 512
    assert runtime._initialized is False


@pytest.mark.asyncio
async def test_wasm_model_forward_not_implemented():
    """Test that model forward pass raises NotImplementedError."""
    model = WASMModel(Path("test.wasm"))
    
    with pytest.raises(NotImplementedError):
        import torch
        await model.forward(torch.randn(1, 10))


@pytest.mark.asyncio
async def test_wasm_model_memory_stats_not_implemented():
    """Test that memory stats raises NotImplementedError."""
    model = WASMModel(Path("test.wasm"))
    
    with pytest.raises(NotImplementedError):
        await model.get_memory_stats()