#!/usr/bin/env python
"""
Universal Cache Cleaner for Kontext Assistant
Works across all platforms (Windows, Linux, macOS)
Part of the Kontext Assistant extension
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path


def print_header():
    print("=" * 60)
    print("Kontext Assistant - Universal Cache Cleaner")
    print("=" * 60)
    print()


def find_webui_root():
    """Find the WebUI root directory by searching upward for markers"""
    current_dir = Path(__file__).resolve().parent
    
    # Try to find WebUI root by looking for characteristic files
    for _ in range(5):  # Look up to 5 levels
        if (current_dir / "webui.py").exists() or (current_dir / "launch.py").exists():
            return current_dir
        if (current_dir / "extensions").exists() and current_dir.name != "extensions":
            return current_dir
        current_dir = current_dir.parent
    
    # If not found, assume we're in extensions/forge-kontext-assistant
    return Path(__file__).resolve().parent.parent.parent


def clear_pycache(root_dir):
    """Clear all __pycache__ directories"""
    print("1. Clearing Python __pycache__ directories...")
    count = 0
    
    for pycache_dir in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"   Removed: {pycache_dir}")
            count += 1
        except Exception as e:
            print(f"   Failed to remove {pycache_dir}: {e}")
    
    print(f"   Cleared {count} __pycache__ directories\n")


def clear_pip_cache():
    """Clear pip cache"""
    print("2. Clearing pip cache...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True, text=True)
        print("   Pip cache cleared successfully\n")
    except Exception as e:
        print(f"   Failed to clear pip cache: {e}\n")


def clear_gradio_temp():
    """Clear Gradio temporary files"""
    print("3. Clearing Gradio temporary files...")
    temp_dir = Path.home() / ".cache" / "gradio" if platform.system() != "Windows" else Path(os.environ.get("TEMP", "")) / "gradio"
    
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"   Removed: {temp_dir}")
        except Exception as e:
            print(f"   Failed to remove {temp_dir}: {e}")
    else:
        print("   No Gradio temp files found")
    print()


def clear_kontext_cache():
    """Clear Kontext Assistant specific cache"""
    print("4. Clearing Kontext Assistant cache...")
    cache_dir = Path.home() / ".cache" / "kontext_assistant"
    
    if cache_dir.exists():
        print(f"   Found cache directory: {cache_dir}")
        response = input("   Do you want to remove AI model cache? This will require re-downloading models (y/N): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(cache_dir)
                print("   Kontext Assistant cache cleared")
            except Exception as e:
                print(f"   Failed to clear cache: {e}")
        else:
            print("   Skipping model cache")
    else:
        print("   No Kontext Assistant cache found")
    print()


def fix_h11_issue():
    """Reinstall h11 to fix potential issues"""
    print("5. Fixing h11 module (common issue)...")
    try:
        # Uninstall h11
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "h11"], 
                      capture_output=True, text=True)
        # Reinstall h11
        subprocess.run([sys.executable, "-m", "pip", "install", "h11==0.12.0", "--no-cache-dir"], 
                      capture_output=True, text=True)
        print("   h11 module reinstalled successfully\n")
    except Exception as e:
        print(f"   Failed to reinstall h11: {e}\n")


def print_browser_cache_instructions():
    """Print instructions for clearing browser cache"""
    print("6. Browser Cache (manual action required)")
    print("   Please clear your browser cache for the WebUI:")
    print("   - Chrome/Edge/Firefox: Ctrl+Shift+Delete (Cmd+Shift+Delete on Mac)")
    print("   - Safari: Cmd+Option+E")
    print("   - Clear cached data for localhost:7860 or your WebUI address\n")


def print_completion_instructions():
    """Print final instructions"""
    print("=" * 60)
    print("Cache clearing complete!")
    print("\nNext steps:")
    print("1. Close WebUI completely if it's running")
    print("2. Clear your browser cache (see instructions above)")
    print("3. Restart WebUI")
    print("4. Reload the browser page (Ctrl+F5 or Cmd+Shift+R)")
    print("=" * 60)


def main():
    print_header()
    
    # Find WebUI root
    webui_root = find_webui_root()
    print(f"WebUI root directory: {webui_root}\n")
    
    # Change to WebUI root for operations
    original_dir = os.getcwd()
    os.chdir(webui_root)
    
    try:
        # Execute cleaning operations
        clear_pycache(webui_root)
        clear_pip_cache()
        clear_gradio_temp()
        clear_kontext_cache()
        fix_h11_issue()
        print_browser_cache_instructions()
        print_completion_instructions()
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    # Wait for user input before closing
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()