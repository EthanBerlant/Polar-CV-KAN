"""
Pytest fixtures for CV-KAN tests.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_length():
    """Default sequence length for tests."""
    return 16


@pytest.fixture
def d_complex():
    """Default complex dimension for tests."""
    return 32


@pytest.fixture
def sample_complex_tensor(batch_size, seq_length, d_complex):
    """Create a sample complex tensor for testing."""
    return torch.randn(batch_size, seq_length, d_complex, dtype=torch.cfloat)


@pytest.fixture
def sample_real_tensor(batch_size, seq_length, d_complex):
    """Create a sample real tensor for testing."""
    return torch.randn(batch_size, seq_length, d_complex)


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    from pathlib import Path

    return Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
