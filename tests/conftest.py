import sys
import os
from pathlib import Path
import pytest
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def test_image():
    """Create a simple test image"""
    return Image.new('RGB', (512, 512), color=(128, 128, 128))

@pytest.fixture
def small_test_image():
    """Create a small test image for quick tests"""
    return Image.new('RGB', (256, 256), color=(100, 150, 200))

@pytest.fixture
def project_root_path():
    """Return the project root path"""
    return project_root

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    # Set any necessary environment variables for testing
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    # Disable model downloads during tests unless explicitly testing downloads
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")