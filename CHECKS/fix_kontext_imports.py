#!/usr/bin/env python3
"""
Fix incorrect imports in kontext.py
"""

from pathlib import Path
import re

def fix_kontext_imports():
    print("🔧 Fixing imports in kontext.py\n")
    
    kontext_path = Path.cwd() / 'scripts' / 'kontext.py'
    
    if not kontext_path.exists():
        print("❌ kontext.py not found!")
        return False
    
    # Read content
    with open(kontext_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📋 Current imports found:")
    
    # Find problematic imports
    if 'from ka_modules import scripts, shared' in content:
        print("❌ Found incorrect: from ka_modules import scripts, shared")
        content = content.replace(
            'from ka_modules import scripts, shared',
            'from modules import scripts, shared'
        )
        print("✅ Fixed to: from modules import scripts, shared")
    
    # Check for other wrong replacements
    wrong_patterns = [
        ('from ka_modules.ui_components import', 'from modules.ui_components import'),
        ('from ka_modules.processing import', 'from modules.processing import'),
        ('from ka_modules.sd_samplers', 'from modules.sd_samplers'),
    ]
    
    for wrong, correct in wrong_patterns:
        if wrong in content:
            print(f"❌ Found incorrect: {wrong}")
            content = content.replace(wrong, correct)
            print(f"✅ Fixed to: {correct}")
    
    # Write back
    with open(kontext_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ kontext.py imports fixed!")
    
    # Also check kontext_assistant.py
    print("\n📋 Checking kontext_assistant.py imports...")
    
    assistant_path = Path.cwd() / 'scripts' / 'kontext_assistant.py'
    if assistant_path.exists():
        with open(assistant_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # kontext_assistant should import from Forge modules, not ka_modules
        if 'from ka_modules import scripts' in content:
            print("❌ Found incorrect import in kontext_assistant.py")
            content = content.replace(
                'from ka_modules import scripts, shared',
                'from modules import scripts, shared'
            )
            content = content.replace(
                'from ka_modules.ui_components import InputAccordion',
                'from modules.ui_components import InputAccordion'
            )
            
            with open(assistant_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed kontext_assistant.py imports")
    
    return True

def check_all_imports():
    """Quick check of all imports"""
    print("\n🔍 Checking all import statements...\n")
    
    files_to_check = [
        'scripts/kontext.py',
        'scripts/kontext_assistant.py',
        'ka_modules/prompt_generator.py',
        'ka_modules/image_analyzer.py'
    ]
    
    for file_path in files_to_check:
        path = Path.cwd() / file_path
        if not path.exists():
            continue
            
        print(f"📄 {file_path}:")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all imports
        imports = re.findall(r'^(?:from|import)\s+.*$', content, re.MULTILINE)
        
        for imp in imports[:5]:  # Show first 5
            if 'ka_modules' in imp and file_path.startswith('scripts/'):
                # Scripts should only import their own ka_modules, not Forge modules as ka_modules
                if any(forge_module in imp for forge_module in ['scripts', 'shared', 'ui_components', 'processing']):
                    print(f"   ❌ {imp}")
                else:
                    print(f"   ✅ {imp}")
            elif 'modules' in imp and not file_path.startswith('scripts/'):
                # ka_modules files shouldn't import from Forge modules
                print(f"   ⚠️  {imp}")
            else:
                print(f"   ✅ {imp}")
        print()

def main():
    print("🔧 FIXING KONTEXT IMPORT ISSUES\n")
    
    if fix_kontext_imports():
        check_all_imports()
        print("\n✅ Import issues should be fixed!")
        print("\n📋 Next steps:")
        print("1. Restart Forge WebUI")
        print("2. Check if both extensions load without errors")
    else:
        print("\n❌ Failed to fix imports")

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")
