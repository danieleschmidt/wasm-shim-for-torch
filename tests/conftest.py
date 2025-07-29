"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_torch_deterministic():
    """Set PyTorch to deterministic mode for reproducible tests."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def device():
    """Get the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_tensor():
    """Create a small test tensor."""
    return torch.randn(2, 3, 4)


@pytest.fixture
def large_tensor():
    """Create a larger test tensor for performance tests."""
    return torch.randn(100, 100, 100)