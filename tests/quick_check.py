#!/usr/bin/env python3
"""
Quick diagnostic check for Kontext Assistant V2
"""

import sys
import os

# Fix for Python 3.10 compatibility - MUST be first
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Check if all modules import correctly"""
    print("Checking module imports...")
    modules = [
        'ka_modules.templates',
        'ka_modules.prompt_generator', 
        'ka_modules.image_analyzer',
        'ka_modules.ui_components',
        'ka_modules.forge_integration'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    return True

def check_configs():
    """Check if config files exist"""
    print("\nChecking config files...")
    config_dir = Path(__file__).parent.parent / 'configs'
    
    if not config_dir.exists():
        print(f"✗ Config directory not found: {config_dir}")
        return False
    
    config_file = config_dir / 'task_configs.json'
    if config_file.exists():
        print(f"✓ {config_file.name}")
    else:
        print(f"✗ {config_file.name} not found")
        return False
    
    return True

def check_dependencies():
    """Check optional dependencies"""
    print("\nChecking optional dependencies...")
    
    deps = {
        'torch': 'PyTorch (required for Florence-2)',
        'transformers': 'Transformers (required for Florence-2)',
        'gradio': 'Gradio (required for Forge WebUI)'
    }
    
    for module, desc in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {desc}: {version}")
        except ImportError:
            print(f"⚠ {desc}: Not installed")

def main():
    print("="*50)
    print("Kontext Assistant V2 - Quick Diagnostic")
    print("="*50)
    print()
    
    # Check imports
    imports_ok = check_imports()
    
    # Check configs
    configs_ok = check_configs()
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "="*50)
    if imports_ok and configs_ok:
        print("✅ Core components OK!")
        print("\nNext steps:")
        print("1. Run: python test_florence2_simple.py")
        print("2. Or start Forge WebUI to test the extension")
    else:
        print("❌ Some components failed!")
        print("\nPlease fix the errors above.")
    print("="*50)

if __name__ == "__main__":
    main()