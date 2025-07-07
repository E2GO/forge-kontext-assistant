"""
Check Florence-2 requirements and system compatibility
"""

import sys
import subprocess
import importlib
import torch
import psutil
import GPUtil

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        return True
    else:
        print("❌ CUDA not available")
        return False

def check_memory():
    """Check system memory"""
    # RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"\n💾 System RAM: {ram_gb:.1f} GB total, {ram.available / (1024**3):.1f} GB available")
    
    # GPU VRAM
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            vram_gb = gpu.memoryTotal / 1024
            vram_used_gb = gpu.memoryUsed / 1024
            print(f"🎮 GPU {gpu.id} ({gpu.name}): {vram_gb:.1f} GB total, {vram_gb - vram_used_gb:.1f} GB free")
    except:
        print("⚠️  Could not get GPU memory info")

def check_florence2_specific():
    """Check Florence-2 specific requirements"""
    print("\n=== Florence-2 Specific Requirements ===")
    
    required_packages = [
        ("transformers", "transformers"),
        ("einops", "einops"),
        ("timm", "timm"),
        ("flash-attn", "flash_attn"),  # Optional but recommended
    ]
    
    all_ok = True
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok

def suggest_installation():
    """Suggest installation commands"""
    print("\n=== Installation Commands ===")
    print("If any packages are missing, install them with:")
    print("\n# Basic requirements:")
    print("pip install transformers>=4.36.0")
    print("pip install einops")
    print("pip install timm")
    print("\n# Optional (for faster inference):")
    print("pip install flash-attn --no-build-isolation")
    print("\n# For Florence-2:")
    print("pip install git+https://github.com/microsoft/Florence-2.git")

def main():
    """Run all checks"""
    print("=== Florence-2 Integration Requirements Check ===\n")
    
    # Basic checks
    python_ok = check_python_version()
    cuda_ok = check_cuda()
    
    print("\n=== Core Packages ===")
    torch_ok = check_package("torch")
    pil_ok = check_package("PIL", "PIL")
    numpy_ok = check_package("numpy")
    
    # Memory check
    check_memory()
    
    # Florence-2 specific
    florence_ok = check_florence2_specific()
    
    # Summary
    print("\n=== Summary ===")
    if python_ok and cuda_ok and torch_ok and florence_ok:
        print("✅ All requirements met! Ready for Florence-2 integration.")
    else:
        print("❌ Some requirements missing.")
        suggest_installation()
    
    print("\n=== Florence-2 Model Info ===")
    print("Model: microsoft/Florence-2-large")
    print("Size: ~1.5GB")
    print("VRAM Usage: ~2-3GB during inference")
    print("Recommended VRAM: 4GB+ for comfortable operation")

if __name__ == "__main__":
    main()