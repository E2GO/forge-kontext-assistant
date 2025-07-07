"""
Check if kontext.py has the shared state updates
"""

from pathlib import Path
import re

def check_kontext_updates():
    kontext_file = Path("scripts/kontext.py")
    if not kontext_file.exists():
        print("❌ scripts/kontext.py not found!")
        return
    
    content = kontext_file.read_text(encoding='utf-8')
    
    print("=== Checking kontext.py for shared state integration ===\n")
    
    # Check 1: Import of shared_state
    if "from ka_modules.shared_state import shared_state" in content:
        print("✓ Import of shared_state found")
    else:
        print("❌ Missing import of shared_state")
        print("  Add near the top after other imports:")
        print("  try:")
        print("      from ka_modules.shared_state import shared_state")
        print("      KA_AVAILABLE = True")
        print("  except ImportError:")
        print("      KA_AVAILABLE = False")
        print("      shared_state = None")
    
    # Check 2: Update in store_kontext_images
    if "shared_state.set_images(images)" in content:
        print("✓ shared_state update in store_kontext_images found")
    else:
        print("❌ Missing shared_state update in store_kontext_images")
        print("  In the store_kontext_images function, add:")
        print("  if KA_AVAILABLE and shared_state:")
        print("      shared_state.set_images(images)")
    
    # Check 3: Update in process_before_every_sampling
    pattern = r'kontext_images = script_args\[1:4\].*?if KA_AVAILABLE and shared_state:\s*shared_state\.set_images\(kontext_images\)'
    if re.search(pattern, content, re.DOTALL):
        print("✓ shared_state update in process_before_every_sampling found")
    else:
        # Check if at least the kontext_images parsing exists
        if "kontext_images = script_args[1:4]" in content:
            print("⚠️ kontext_images parsing found but shared_state update might be missing")
            print("  After the line: kontext_images = script_args[1:4]")
            print("  Add:")
            print("  if KA_AVAILABLE and shared_state:")
            print("      shared_state.set_images(kontext_images)")
        else:
            print("❌ Missing shared_state update in process_before_every_sampling")
    
    # Check 4: Look for the specific line numbers
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "KA_AVAILABLE = True" in line:
            print(f"\n✓ Found KA_AVAILABLE at line {i+1}")
        if "shared_state.set_images" in line:
            print(f"✓ Found shared_state.set_images at line {i+1}")
    
    print("\n=== Check complete ===")
    print("\nIf any ❌ items above, you need to update kontext.py")

if __name__ == "__main__":
    check_kontext_updates()