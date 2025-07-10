"""
Manual installation script for Liger Kernel
Use this if automatic installation fails
"""

import subprocess
import sys
import importlib.util

def check_triton():
    """Check if triton is installed"""
    spec = importlib.util.find_spec('triton')
    if spec:
        import triton
        print(f"✓ Triton found: version {getattr(triton, '__version__', 'unknown')}")
        return True
    else:
        print("✗ Triton not found")
        return False

def install_liger():
    """Try to install liger-kernel"""
    print("\nAttempting to install liger-kernel...")
    
    # Method 1: Install without dependencies
    print("\n1. Trying --no-deps install...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "liger-kernel", "--no-deps"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Liger-kernel installed successfully (no-deps)")
    else:
        print(f"✗ No-deps install failed: {result.stderr}")
        
        # Method 2: Force reinstall triton first
        print("\n2. Trying to upgrade triton first...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "triton"],
            capture_output=True
        )
        
        # Method 3: Install specific version
        print("\n3. Trying specific liger-kernel version...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "liger-kernel==0.3.0", "--no-deps"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Liger-kernel 0.3.0 installed successfully")
        else:
            print(f"✗ Version 0.3.0 install failed: {result.stderr}")

def test_liger():
    """Test if liger can be imported"""
    print("\nTesting liger-kernel import...")
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        print("✓ Liger-kernel imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    print("=== Liger Kernel Manual Installation ===")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    
    # Check triton
    has_triton = check_triton()
    
    # Try installation
    install_liger()
    
    # Test import
    success = test_liger()
    
    if success:
        print("\n✅ Liger-kernel is ready to use!")
        print("JoyCaption will run with optimization enabled.")
    else:
        print("\n❌ Liger-kernel installation failed.")
        print("JoyCaption will run without optimization.")
        print("\nPossible solutions:")
        print("1. Try installing a different triton version")
        print("2. Install liger-kernel in a fresh environment")
        print("3. Use JoyCaption without Liger optimization (still works fine)")

if __name__ == "__main__":
    main()