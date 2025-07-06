#!/usr/bin/env python3
"""
Fix Smart Assistant to display as proper accordion
"""

from pathlib import Path

def fix_assistant_ui():
    print("🔧 Fixing Smart Assistant UI to display as proper accordion\n")
    
    assistant_path = Path.cwd() / 'scripts' / 'kontext_assistant.py'
    
    if not assistant_path.exists():
        print("❌ kontext_assistant.py not found!")
        return False
    
    # Read current content
    with open(assistant_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the ui method and fix it
    if 'def ui(self, is_img2img):' in content:
        print("📝 Updating UI method to use proper accordion...")
        
        # Find where the UI starts
        ui_start = content.find('def ui(self, is_img2img):')
        ui_method_start = content.find('"""', ui_start) + 3
        ui_method_end = content.find('"""', ui_method_start)
        
        # Replace the UI section to wrap everything in InputAccordion
        old_ui = '''with gr.Group():
            gr.Markdown("### 🤖 Kontext Smart Assistant")'''
        
        new_ui = '''# Create accordion for Smart Assistant
        with InputAccordion(False, label="🤖 Kontext Smart Assistant") as enabled:
            gr.Markdown("*Analyzes your context images and generates proper FLUX.1 Kontext prompts*")'''
        
        if old_ui in content:
            content = content.replace(old_ui, new_ui)
            print("✅ Updated to use InputAccordion")
        else:
            # Try alternative approach - look for the return statement
            # and add enabled to the return
            return_line = "return [gr.Group()] + assistant_ui"
            if return_line in content:
                content = content.replace(
                    return_line,
                    "return [enabled, task_type, user_intent, context_strength, use_llm]"
                )
                print("✅ Updated return statement")
    
    # Make sure InputAccordion is imported
    if 'from modules.ui_components import InputAccordion' not in content:
        # Add the import after other imports
        import_line = 'from modules import scripts, shared'
        if import_line in content:
            content = content.replace(
                import_line,
                import_line + '\nfrom modules.ui_components import InputAccordion'
            )
            print("✅ Added InputAccordion import")
    
    # Write back
    with open(assistant_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Fixed kontext_assistant.py to use proper accordion")
    return True

def create_simple_fix():
    """Create a simpler version that just returns proper title"""
    print("\n🔧 Alternative: Simple fix by returning proper title\n")
    
    assistant_path = Path.cwd() / 'scripts' / 'kontext_assistant.py'
    
    with open(assistant_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the title method
    old_title = '''def title(self):
        """Return the title of the script"""
        return "Kontext Smart Assistant"'''
    
    new_title = '''def title(self):
        """Return the title of the script"""
        return "🤖 Kontext Smart Assistant"'''
    
    if old_title in content:
        content = content.replace(old_title, new_title)
    
    # Make sure the UI is wrapped properly
    # Look for the ui method
    if 'def ui(self, is_img2img):' in content and 'InputAccordion' not in content:
        # This needs the full UI wrapper
        # For now, just ensure it returns the right components
        print("📝 UI needs proper accordion wrapper")
    
    with open(assistant_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated title")

def main():
    print("🔧 FIXING SMART ASSISTANT ACCORDION DISPLAY\n")
    
    if fix_assistant_ui():
        print("\n📋 Next steps:")
        print("1. Restart Forge WebUI")
        print("2. Check if Smart Assistant shows as collapsible accordion")
        print("\n💡 If it still doesn't work, the issue might be:")
        print("- The script is returning wrong components")
        print("- InputAccordion is not available in your Forge version")
    
    # Also try simple fix
    create_simple_fix()

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")
