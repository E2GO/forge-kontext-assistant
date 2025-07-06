#!/usr/bin/env python3
"""
Create a fresh working templates.py file
"""

from pathlib import Path
import shutil
from datetime import datetime

def main():
    print("🔧 Creating fresh templates.py\n")
    
    root = Path.cwd()
    templates_path = root / 'ka_modules' / 'templates.py'
    
    # Backup existing if needed
    if templates_path.exists():
        backup_name = f'templates_broken_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
        backup_path = templates_path.parent / backup_name
        shutil.move(str(templates_path), str(backup_path))
        print(f"📁 Moved broken file to {backup_name}")
    
    # Create new content
    content = '''"""
Prompt templates for different FLUX.1 Kontext task types.
"""

from typing import Dict, List, Optional, Any

class PromptTemplates:
    """Templates for FLUX.1 Kontext prompt generation"""
    
    def __init__(self):
        self.templates = {
            'object_color': {
                'base': "Change the {object} from {current_color} to {new_color}",
                'preserve': ["shadows", "reflections", "texture", "position"]
            },
            'style_transfer': {
                'base': "Transform the image into {style} style",
                'preserve': ["composition", "subject identity", "scene layout"]
            },
            'environment_change': {
                'base': "Change the background to {new_environment}",
                'preserve': ["subject", "pose", "lighting on subject"]
            },
            'object_state': {
                'base': "Change the {object} to appear {new_state}",
                'preserve': ["position", "size", "surrounding elements"]
            },
            'element_combination': {
                'base': "Combine {elements} into a cohesive scene",
                'preserve': ["individual element qualities", "relative sizes"]
            },
            'state_changes': {
                'base': "Show the {subject} transitioning from {start_state} to {end_state}",
                'preserve': ["identity", "location", "key features"]
            },
            'outpainting': {
                'base': "Extend the image {direction} to show more of the scene",
                'preserve': ["existing content", "style", "perspective"]
            }
        }
    
    def get_object_color_template(self, target: str, current_color: str = None, 
                                 new_color: str = None, **kwargs) -> str:
        """Generate object color change prompt"""
        base = self.templates['object_color']['base']
        preserve = self.templates['object_color']['preserve']
        
        # Fill in the template
        prompt = base.format(
            object=target,
            current_color=current_color or "current color",
            new_color=new_color or "new color"
        )
        
        # Add preservation rules
        prompt += f". Maintain the {', '.join(preserve)}"
        
        return prompt
    
    def get_style_transfer_template(self, style: str, subject: str = None, **kwargs) -> str:
        """Generate style transfer prompt"""
        base = "Transform this "
        if subject:
            base += f"{subject} "
        base += f"into {style} style"
        
        preserve = self.templates['style_transfer']['preserve']
        base += f" while preserving {', '.join(preserve)}"
        
        return base
    
    def get_environment_change_template(self, new_environment: str, 
                                      subject: str = None, **kwargs) -> str:
        """Generate environment change prompt"""
        base = f"Change the background to {new_environment}"
        
        if subject:
            base += f" while keeping the {subject} unchanged"
        
        preserve = self.templates['environment_change']['preserve']
        base += f". Preserve the {', '.join(preserve)}"
        
        return base
    
    def get_object_state_template(self, target: str, new_state: str, **kwargs) -> str:
        """Generate object state change prompt"""
        prompt = f"Modify the {target} to appear {new_state}"
        preserve = self.templates['object_state']['preserve']
        prompt += f" while maintaining {', '.join(preserve)}"
        return prompt
    
    def get_element_combination_template(self, elements: List[str], 
                                       arrangement: str = None, **kwargs) -> str:
        """Generate element combination prompt"""
        elements_str = ", ".join(elements) if isinstance(elements, list) else elements
        prompt = f"Combine {elements_str} into a unified composition"
        
        if arrangement:
            prompt += f" with {arrangement}"
        
        preserve = self.templates['element_combination']['preserve']
        prompt += f". Maintain {', '.join(preserve)}"
        
        return prompt
    
    def get_state_changes_template(self, subject: str, start_state: str, 
                                 end_state: str, **kwargs) -> str:
        """Generate state change prompt"""
        prompt = f"Transform the {subject} from {start_state} to {end_state}"
        preserve = self.templates['state_changes']['preserve']
        prompt += f" while preserving {', '.join(preserve)}"
        return prompt
    
    def get_outpainting_template(self, direction: str, content_hint: str = None, **kwargs) -> str:
        """Generate outpainting prompt"""
        prompt = f"Extend the image {direction}"
        
        if content_hint:
            prompt += f" to reveal {content_hint}"
        else:
            prompt += " naturally continuing the scene"
        
        preserve = self.templates['outpainting']['preserve']
        prompt += f". Maintain {', '.join(preserve)}"
        
        return prompt
'''
    
    # Write the file
    with open(templates_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Created new templates.py")
    
    # Test it
    print("\n🧪 Testing import...")
    import sys
    sys.path.insert(0, str(root))
    
    try:
        from ka_modules.templates import PromptTemplates
        print("✅ Import successful!")
        
        # Test usage
        templates = PromptTemplates()
        test_prompt = templates.get_object_color_template("car", "red", "blue")
        print(f"✅ Test prompt: '{test_prompt}'")
        
        print("\n✨ templates.py is now working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if main():
        print("\n📋 Next step: Run 'python test_basic.py' to verify everything works")
    input("\nPress Enter to continue...")
