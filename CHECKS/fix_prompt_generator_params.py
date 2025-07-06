#!/usr/bin/env python3
"""
Fix prompt_generator.py to properly pass parameters to template methods
"""

from pathlib import Path
import shutil
from datetime import datetime

def fix_prompt_generator():
    print("🔧 Fixing prompt_generator.py parameter passing\n")
    
    # Path to prompt_generator.py
    generator_path = Path.cwd() / 'ka_modules' / 'prompt_generator.py'
    
    if not generator_path.exists():
        print("❌ prompt_generator.py not found!")
        return False
    
    # Backup
    backup_name = f'prompt_generator_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    shutil.copy(generator_path, generator_path.parent / backup_name)
    print(f"📁 Backed up to {backup_name}")
    
    # Read current content
    with open(generator_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the generate method and fix it
    print("📝 Updating generate method to properly extract parameters...")
    
    # Replace the template method calls section
    old_pattern = '''# Generate prompt using template
        base_prompt = template_method(
            target=parsed_intent.get('target', 'the main subject'),
            action=parsed_intent.get('action', user_intent),
            context=context
        )'''
    
    new_pattern = '''# Generate prompt using template with proper parameters
        # Extract parameters based on task type
        template_params = self._extract_template_params(task_type, parsed_intent, user_intent)
        
        # Call template method with unpacked parameters
        base_prompt = template_method(**template_params)'''
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ Updated template method call")
    
    # Add the new method to extract parameters
    extract_method = '''
    def _extract_template_params(self, task_type: str, parsed_intent: Dict[str, str], 
                                user_intent: str) -> Dict[str, Any]:
        """Extract proper parameters for each template type"""
        
        params = {}
        
        if task_type == "object_color":
            params = {
                'target': parsed_intent.get('target', 'object'),
                'current_color': parsed_intent.get('current_color', 'current color'),
                'new_color': parsed_intent.get('new_color', 'new color')
            }
        
        elif task_type == "object_state":
            params = {
                'target': parsed_intent.get('target', 'object'),
                'new_state': parsed_intent.get('new_state', user_intent)
            }
        
        elif task_type == "style_transfer":
            params = {
                'style': parsed_intent.get('style', user_intent)
            }
        
        elif task_type == "environment_change":
            params = {
                'new_environment': parsed_intent.get('target_env', user_intent)
            }
        
        elif task_type == "element_combination":
            # Parse elements from intent
            elements = parsed_intent.get('elements', [])
            if not elements and 'and' in user_intent:
                elements = [part.strip() for part in user_intent.split('and')]
            elif not elements:
                elements = [user_intent]
            params = {
                'elements': elements
            }
        
        elif task_type == "state_changes":
            params = {
                'subject': parsed_intent.get('target', 'subject'),
                'start_state': parsed_intent.get('start_state', 'original'),
                'end_state': parsed_intent.get('end_state', user_intent)
            }
        
        elif task_type == "outpainting":
            # Extract direction from intent
            directions = ['left', 'right', 'up', 'down', 'top', 'bottom']
            direction = 'outward'
            for d in directions:
                if d in user_intent.lower():
                    direction = d
                    break
            params = {
                'direction': parsed_intent.get('direction', direction)
            }
        
        return params
'''
    
    # Find where to insert the method (after _parse_intent)
    parse_intent_end = content.find('return parsed')
    if parse_intent_end > 0:
        # Find the next method definition
        next_method = content.find('\n    def ', parse_intent_end)
        if next_method > 0:
            # Insert before the next method
            content = content[:next_method] + extract_method + content[next_method:]
            print("✅ Added _extract_template_params method")
    
    # Fix the _parse_intent method to better extract parameters
    print("📝 Improving _parse_intent method...")
    
    # Update the object_state parsing
    old_state_parsing = '''# Generic parsing for other types
        if not parsed:
            parsed['action'] = intent
            parsed['target'] = 'the subject' '''
    
    new_state_parsing = '''# Object state parsing
        elif task_type == "object_state":
            # Pattern: "make/turn the X Y" or "X to Y"
            state_patterns = [
                r"(?:make|turn|change)\s+(?:the\s+)?(\w+)\s+(.+)",
                r"(\w+)\s+to\s+(.+)",
                r"(\w+)\s+(.+)"  # Fallback
            ]
            for pattern in state_patterns:
                match = re.search(pattern, intent_lower)
                if match:
                    parsed['target'] = match.group(1)
                    parsed['new_state'] = match.group(2)
                    break
        
        # Generic parsing for other types
        if not parsed:
            parsed['action'] = intent
            parsed['target'] = 'the subject' '''
    
    if old_state_parsing in content:
        content = content.replace(old_state_parsing, new_state_parsing)
        print("✅ Improved state parsing")
    
    # Write the fixed content
    with open(generator_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Fixed prompt_generator.py!")
    return True

def test_fix():
    """Quick test of the fix"""
    print("\n🧪 Testing the fix...")
    
    try:
        import sys
        sys.path.insert(0, str(Path.cwd()))
        
        from ka_modules.prompt_generator import PromptGenerator
        from ka_modules.templates import PromptTemplates
        
        generator = PromptGenerator()
        
        # Test different task types
        tests = [
            ("object_color", "make the car blue"),
            ("object_state", "make the door open"),
            ("style_transfer", "impressionist style"),
            ("environment_change", "beach background"),
            ("outpainting", "extend to the right"),
        ]
        
        for task_type, intent in tests:
            try:
                prompt = generator.generate(
                    task_type=task_type,
                    user_intent=intent,
                    context_strength=0.7
                )
                print(f"✅ {task_type}: {prompt[:50]}...")
            except Exception as e:
                print(f"❌ {task_type}: {str(e)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    if fix_prompt_generator():
        test_fix()
        print("\n✅ All done! Restart Forge WebUI and try generating prompts again.")
    else:
        print("\n❌ Failed to fix prompt_generator.py")

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")
