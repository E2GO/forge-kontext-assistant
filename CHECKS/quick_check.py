#!/usr/bin/env python3
"""
Quick visual diagnostic for forge-kontext-assistant
Shows current state with colors and actionable fixes
"""

import os
import sys
from pathlib import Path
import json

# ANSI color codes for terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    WHITE = '\033[97m'
    END = '\033[0m'

def print_colored(text, color=None, bold=False):
    """Print with color"""
    if color:
        text = f"{color}{text}{Colors.END}"
    if bold:
        text = f"{Colors.BOLD}{text}{Colors.END}"
    print(text)

def check_project_state():
    """Quick check of project state"""
    root = Path(__file__).parent
    
    print_colored("\n🔍 FORGE-KONTEXT-ASSISTANT QUICK CHECK", Colors.CYAN, bold=True)
    print_colored("=" * 50, Colors.CYAN)
    
    # Check 1: Directory naming
    print_colored("\n1️⃣  Directory Structure:", Colors.BLUE, bold=True)
    
    has_modules = (root / 'modules').exists()
    has_ka_modules = (root / 'ka_modules').exists()
    
    if has_modules and not has_ka_modules:
        print_colored("   ❌ 'modules' directory exists (WRONG)", Colors.RED)
        print_colored("   ↳ Must be renamed to 'ka_modules'", Colors.YELLOW)
        status = "NEEDS_RENAME"
    elif has_ka_modules and not has_modules:
        print_colored("   ✅ 'ka_modules' directory exists (CORRECT)", Colors.GREEN)
        status = "STRUCTURE_OK"
    elif has_modules and has_ka_modules:
        print_colored("   ⚠️  Both 'modules' and 'ka_modules' exist", Colors.YELLOW)
        print_colored("   ↳ Remove 'modules' directory", Colors.YELLOW)
        status = "DUPLICATE_DIRS"
    else:
        print_colored("   ❌ Neither directory exists", Colors.RED)
        status = "MISSING_DIRS"
    
    # Check 2: Key files
    print_colored("\n2️⃣  Key Files:", Colors.BLUE, bold=True)
    
    key_files = {
        'scripts/kontext.py': 'Original Kontext',
        'scripts/kontext_assistant.py': 'Smart Assistant',
        f'{has_ka_modules and "ka_modules" or "modules"}/templates.py': 'Templates',
        f'{has_ka_modules and "ka_modules" or "modules"}/prompt_generator.py': 'Generator',
    }
    
    missing_files = []
    for file, desc in key_files.items():
        path = root / file
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                print_colored(f"   ✅ {file} ({size:,} bytes)", Colors.GREEN)
            else:
                print_colored(f"   ⚠️  {file} (empty file)", Colors.YELLOW)
                missing_files.append(file)
        else:
            print_colored(f"   ❌ {file} (missing)", Colors.RED)
            missing_files.append(file)
    
    # Check 3: Import status
    print_colored("\n3️⃣  Import Check:", Colors.BLUE, bold=True)
    
    if has_ka_modules:
        try:
            # Check if a file has wrong imports
            assistant_path = root / 'scripts' / 'kontext_assistant.py'
            if assistant_path.exists():
                with open(assistant_path, 'r') as f:
                    content = f.read()
                
                if 'from modules import' in content or 'from modules.' in content:
                    print_colored("   ❌ Wrong imports found (still using 'modules')", Colors.RED)
                    imports_ok = False
                elif 'from ka_modules' in content:
                    print_colored("   ✅ Imports are correct (using 'ka_modules')", Colors.GREEN)
                    imports_ok = True
                else:
                    print_colored("   ⚠️  No module imports found", Colors.YELLOW)
                    imports_ok = True
            else:
                print_colored("   ❌ Can't check - kontext_assistant.py missing", Colors.RED)
                imports_ok = False
        except:
            print_colored("   ❌ Error checking imports", Colors.RED)
            imports_ok = False
    else:
        print_colored("   ⏭️  Skipped (fix directory structure first)", Colors.YELLOW)
        imports_ok = False
    
    # Check 4: Utils
    print_colored("\n4️⃣  Utilities:", Colors.BLUE, bold=True)
    
    utils_files = ['utils/cache.py', 'utils/validators.py']
    utils_ok = True
    for file in utils_files:
        path = root / file
        if path.exists() and path.stat().st_size > 0:
            print_colored(f"   ✅ {file}", Colors.GREEN)
        else:
            print_colored(f"   ❌ {file} (missing or empty)", Colors.RED)
            utils_ok = False
    
    # Summary and recommendations
    print_colored("\n" + "=" * 50, Colors.CYAN)
    print_colored("📊 SUMMARY:", Colors.CYAN, bold=True)
    
    if status == "NEEDS_RENAME":
        print_colored("\n🔧 IMMEDIATE ACTION REQUIRED:", Colors.RED, bold=True)
        print_colored("\n   Run these commands:", Colors.YELLOW)
        print_colored(f"   cd {root}", Colors.WHITE)
        print_colored("   mv modules ka_modules", Colors.WHITE)
        print_colored("   python fix_imports.py", Colors.WHITE)
        
    elif status == "STRUCTURE_OK" and imports_ok and utils_ok and not missing_files:
        print_colored("\n✅ PROJECT IS READY!", Colors.GREEN, bold=True)
        print_colored("\n   Next steps:", Colors.GREEN)
        print_colored("   1. Restart Forge WebUI", Colors.WHITE)
        print_colored("   2. Check the Extensions tab", Colors.WHITE)
        print_colored("   3. Load FluxKontext model and test", Colors.WHITE)
        
    else:
        print_colored("\n⚠️  PROJECT NEEDS FIXES:", Colors.YELLOW, bold=True)
        
        if not imports_ok:
            print_colored("\n   Fix imports:", Colors.YELLOW)
            print_colored("   python fix_imports.py", Colors.WHITE)
        
        if not utils_ok:
            print_colored("\n   Create utilities:", Colors.YELLOW)
            print_colored("   python setup_project.py", Colors.WHITE)
        
        if missing_files:
            print_colored(f"\n   Missing {len(missing_files)} files", Colors.YELLOW)
            print_colored("   Run: python setup_project.py", Colors.WHITE)
    
    # Test import
    print_colored("\n5️⃣  Import Test:", Colors.BLUE, bold=True)
    
    if status == "STRUCTURE_OK":
        sys.path.insert(0, str(root))
        try:
            from ka_modules.templates import PromptTemplates
            print_colored("   ✅ Import test PASSED!", Colors.GREEN)
        except ImportError as e:
            print_colored(f"   ❌ Import test FAILED: {e}", Colors.RED)
        except Exception as e:
            print_colored(f"   ❌ Unexpected error: {e}", Colors.RED)
    else:
        print_colored("   ⏭️  Skipped (fix structure first)", Colors.YELLOW)
    
    print_colored("\n" + "=" * 50 + "\n", Colors.CYAN)

if __name__ == "__main__":
    check_project_state()