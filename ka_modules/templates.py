"""
Prompt templates for FLUX.1 Kontext
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
            'object_state': {
                'base': "Change the {object} to appear {new_state}",
                'preserve': ["position", "size", "surrounding elements"]
            },
            'style_transfer': {
                'base': "Transform the image into {style} style",
                'preserve': ["composition", "subject identity", "scene layout"]
            },
            'environment_change': {
                'base': "Change the background to {new_environment}",
                'preserve': ["subject", "pose", "lighting on subject"]
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
        template = self.templates['object_color']
        base = template['base']
        preserve = template['preserve']
        
        prompt = base.format(
            object=target,
            current_color=current_color or "current color",
            new_color=new_color or "new color"
        )
        
        prompt += f". Maintain the {', '.join(preserve)}"
        return prompt
    
    def get_style_transfer_template(self, style: str, **kwargs) -> str:
        """Generate style transfer prompt"""
        template = self.templates['style_transfer']
        prompt = template['base'].format(style=style)
        prompt += f" while preserving {', '.join(template['preserve'])}"
        return prompt
    
    def get_environment_change_template(self, new_environment: str, **kwargs) -> str:
        """Generate environment change prompt"""
        template = self.templates['environment_change']
        prompt = template['base'].format(new_environment=new_environment)
        prompt += f". Preserve the {', '.join(template['preserve'])}"
        return prompt
    
    def get_object_state_template(self, target: str, new_state: str, **kwargs) -> str:
        """Generate object state change prompt"""
        template = self.templates['object_state']
        prompt = template['base'].format(object=target, new_state=new_state)
        prompt += f" while maintaining {', '.join(template['preserve'])}"
        return prompt
    
    def get_element_combination_template(self, elements: List[str], **kwargs) -> str:
        """Generate element combination prompt"""
        template = self.templates['element_combination']
        elements_str = ", ".join(elements) if isinstance(elements, list) else elements
        prompt = template['base'].format(elements=elements_str)
        prompt += f". Maintain {', '.join(template['preserve'])}"
        return prompt
    
    def get_state_changes_template(self, subject: str, start_state: str, 
                                 end_state: str, **kwargs) -> str:
        """Generate state change prompt"""
        template = self.templates['state_changes']
        prompt = template['base'].format(
            subject=subject, 
            start_state=start_state, 
            end_state=end_state
        )
        prompt += f" while preserving {', '.join(template['preserve'])}"
        return prompt
    
    def get_outpainting_template(self, direction: str, **kwargs) -> str:
        """Generate outpainting prompt"""
        template = self.templates['outpainting']
        prompt = template['base'].format(direction=direction)
        prompt += f". Maintain {', '.join(template['preserve'])}"
        return prompt
