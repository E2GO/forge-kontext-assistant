#!/usr/bin/env python3
"""
Add collections compatibility patch to kontext_assistant.py
"""

from pathlib import Path
import shutil
from datetime import datetime

def add_patch():
    print("🔧 Adding Python 3.10+ compatibility patch to kontext_assistant.py\n")
    
    # Patch code to add
    patch_code = '''# Python 3.10+ compatibility patch for huggingface_hub
import sys
import collections.abc
if sys.version_info >= (3, 10):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Iterable = collections.abc.Iterable
    collections.MutableSet = collections.abc.MutableSet
    collections.Callable = collections.abc.Callable

'''
    
    # Path to kontext_assistant.py
    assistant_path = Path.cwd() / 'scripts' / 'kontext_assistant.py'
    
    if not assistant_path.exists():
        print("❌ kontext_assistant.py not found!")
        return False
    
    # Backup
    backup_name = f'kontext_assistant_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    backup_path = assistant_path.parent / backup_name
    shutil.copy(assistant_path, backup_path)
    print(f"📁 Backed up to {backup_name}")
    
    # Read current content
    with open(assistant_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if patch already exists
    if 'collections.Mapping = collections.abc.Mapping' in content:
        print("✅ Patch already present!")
        return True
    
    # Find where to insert (after initial docstring and before other imports)
    lines = content.split('\n')
    insert_index = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        # Skip initial docstring
        if i < 5 and '"""' in line:
            in_docstring = not in_docstring
            continue
        
        # Find first import statement after docstring
        if not in_docstring and (line.startswith('import ') or line.startswith('from ')):
            insert_index = i
            break
    
    # Insert patch before first import
    lines.insert(insert_index, patch_code)
    
    # Write updated content
    updated_content = '\n'.join(lines)
    with open(assistant_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Patch added successfully!")
    
    # Test the import
    print("\n🧪 Testing import after patch...")
    try:
        import sys
        sys.path.insert(0, str(Path.cwd() / 'scripts'))
        
        # Clear any cached imports
        if 'kontext_assistant' in sys.modules:
            del sys.modules['kontext_assistant']
        
        import kontext_assistant
        print("✅ kontext_assistant imports successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    if add_patch():
        print("\n✨ All done! The compatibility patch is now permanent.")
        print("\n📋 Final test - run this to verify:")
        print("   python test_basic.py")
        print("\nThen restart Forge WebUI to use the extension!")
    else:
        print("\n❌ Failed to add patch")

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")
