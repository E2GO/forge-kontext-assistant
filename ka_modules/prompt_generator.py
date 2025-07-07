"""
Prompt Generator Module for Kontext Smart Assistant
Generates instructional prompts for FLUX.1 Kontext based on user intent
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

# Fix imports for Forge environment
try:
    # Try absolute import first
    from ka_modules.templates import PromptTemplates
except ImportError:
    try:
        # Try relative import
        from .templates import PromptTemplates
    except ImportError:
        # Fallback to direct import if in same directory
        from templates import PromptTemplates

# For Python 3.10 compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


class PromptGenerator:
    """Generate contextual prompts for FLUX.1 Kontext"""
    
    def __init__(self, templates: Optional[PromptTemplates] = None):
        """Initialize with template system"""
        self.templates = templates or PromptTemplates()
        self.last_generated = None
        self.generation_history = []
        
    def generate(self, task_type: str, user_intent: str, 
                 image_analysis: Optional[Dict] = None,
                 advanced_mode: bool = False) -> str:
        """
        Generate a FLUX.1 Kontext prompt based on inputs
        
        Args:
            task_type: Type of editing task
            user_intent: User's description of desired change
            image_analysis: Optional analysis results from Florence-2
            advanced_mode: Whether to use more complex generation
            
        Returns:
            Generated instructional prompt
        """
        # Validate inputs
        if not task_type or not user_intent:
            return "Please specify both task type and what you want to change"
            
        # Clean and normalize user intent
        user_intent = user_intent.strip().lower()
        
        # Get base template
        template = self.templates.get_template(task_type)
        if not template:
            # Fallback to generic template
            template = "Change {user_intent}. Preserve all other elements."
        
        # Extract parameters from intent
        params = self._extract_parameters(user_intent, task_type, image_analysis)
        
        # Fill template
        try:
            # First, try to format with extracted params
            prompt = template.format(**params)
        except KeyError as e:
            # If some keys are missing, use a simpler approach
            prompt = self._simple_fill(template, params)
        
        # Add preservation rules based on task
        prompt = self._add_preservation_rules(prompt, task_type, params)
        
        # Add context from image analysis
        if image_analysis:
            prompt = self._enhance_with_context(prompt, image_analysis)
        
        # Store in history
        self.last_generated = prompt
        self.generation_history.append({
            'task_type': task_type,
            'user_intent': user_intent,
            'prompt': prompt
        })
        
        return prompt
    
    def _extract_parameters(self, user_intent: str, task_type: str, 
                           image_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract relevant parameters from user intent"""
        params = {
            'user_intent': user_intent,
            'original_intent': user_intent
        }
        
        # Task-specific extraction
        if task_type == 'object_manipulation':
            params.update(self._extract_object_params(user_intent))
        elif task_type == 'style_transfer':
            params.update(self._extract_style_params(user_intent))
        elif task_type == 'environment_change':
            params.update(self._extract_environment_params(user_intent))
        elif task_type == 'lighting_adjustment':
            params.update(self._extract_lighting_params(user_intent))
        elif task_type == 'state_change':
            params.update(self._extract_state_params(user_intent))
        elif task_type == 'outpainting':
            params.update(self._extract_outpainting_params(user_intent))
        
        # Add context from image analysis
        if image_analysis:
            params.update(self._extract_from_analysis(image_analysis))
        
        return params
    
    def _extract_object_params(self, intent: str) -> Dict[str, str]:
        """Extract object-related parameters"""
        params = {}
        
        # Color patterns
        color_match = re.search(r'(red|blue|green|yellow|black|white|purple|orange|pink|brown|gray|grey)\s+(\w+)', intent)
        if color_match:
            params['target_color'] = color_match.group(1)
            params['object'] = color_match.group(2)
        
        # Action patterns
        if 'remove' in intent:
            params['action'] = 'remove'
        elif 'add' in intent:
            params['action'] = 'add'
        elif 'change' in intent or 'make' in intent:
            params['action'] = 'change'
        
        # Extract object if not already found
        if 'object' not in params:
            # Common objects
            objects = ['car', 'person', 'building', 'tree', 'sky', 'ground', 'wall', 'door', 'window']
            for obj in objects:
                if obj in intent:
                    params['object'] = obj
                    break
        
        return params
    
    def _extract_style_params(self, intent: str) -> Dict[str, str]:
        """Extract style-related parameters"""
        params = {}
        
        # Art styles
        styles = {
            'anime': 'anime art style with characteristic features',
            'oil painting': 'oil painting with visible brushstrokes',
            'watercolor': 'watercolor painting with fluid colors',
            'sketch': 'pencil sketch with detailed linework',
            'cartoon': 'cartoon style with bold outlines',
            'photorealistic': 'photorealistic rendering',
            'cyberpunk': 'cyberpunk style with neon and tech',
            'vintage': 'vintage photography style',
            'minimalist': 'minimalist artistic style'
        }
        
        for style, description in styles.items():
            if style in intent:
                params['target_style'] = style
                params['style_description'] = description
                break
        
        return params
    
    def _extract_environment_params(self, intent: str) -> Dict[str, str]:
        """Extract environment-related parameters"""
        params = {}
        
        # Environments
        environments = ['beach', 'mountain', 'city', 'forest', 'desert', 'space', 
                       'underwater', 'indoor', 'outdoor', 'street', 'park']
        
        for env in environments:
            if env in intent:
                params['target_environment'] = env
                break
        
        # Time of day
        times = ['sunset', 'sunrise', 'night', 'day', 'evening', 'morning', 'afternoon']
        for time in times:
            if time in intent:
                params['time_of_day'] = time
                break
        
        # Weather
        weather = ['sunny', 'rainy', 'cloudy', 'snowy', 'foggy', 'stormy']
        for w in weather:
            if w in intent:
                params['weather'] = w
                break
        
        return params
    
    def _extract_lighting_params(self, intent: str) -> Dict[str, str]:
        """Extract lighting-related parameters"""
        params = {}
        
        # Lighting conditions
        if 'bright' in intent:
            params['lighting_intensity'] = 'bright'
        elif 'dark' in intent:
            params['lighting_intensity'] = 'dark'
        elif 'dim' in intent:
            params['lighting_intensity'] = 'dim'
        
        # Lighting types
        lighting_types = ['natural', 'artificial', 'neon', 'candlelight', 'moonlight', 'sunlight']
        for light in lighting_types:
            if light in intent:
                params['lighting_type'] = light
                break
        
        return params
    
    def _extract_state_params(self, intent: str) -> Dict[str, str]:
        """Extract state change parameters"""
        params = {}
        
        # States
        if 'broken' in intent:
            params['target_state'] = 'broken'
        elif 'old' in intent or 'aged' in intent:
            params['target_state'] = 'aged'
        elif 'new' in intent:
            params['target_state'] = 'new'
        elif 'wet' in intent:
            params['target_state'] = 'wet'
        elif 'dry' in intent:
            params['target_state'] = 'dry'
        
        return params
    
    def _extract_outpainting_params(self, intent: str) -> Dict[str, str]:
        """Extract outpainting parameters"""
        params = {}
        
        # Directions
        directions = ['left', 'right', 'top', 'bottom', 'all sides']
        for direction in directions:
            if direction in intent:
                params['direction'] = direction
                break
        
        # Amount
        if '2x' in intent or 'double' in intent:
            params['expansion'] = '2x'
        elif '3x' in intent or 'triple' in intent:
            params['expansion'] = '3x'
        
        return params
    
    def _extract_from_analysis(self, analysis: Dict) -> Dict[str, Any]:
        """Extract parameters from image analysis"""
        params = {}
        
        # This will be enhanced when Florence-2 is integrated
        if 'objects' in analysis:
            params['detected_objects'] = ', '.join(analysis['objects'])
        if 'style' in analysis:
            params['current_style'] = analysis['style']
        
        return params
    
    def _simple_fill(self, template: str, params: Dict[str, Any]) -> str:
        """Simple template filling when format() fails"""
        result = template
        
        # Replace placeholders manually
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        # Remove any remaining placeholders
        result = re.sub(r'\{[^}]+\}', '', result)
        
        return result.strip()
    
    def _add_preservation_rules(self, prompt: str, task_type: str, 
                               params: Dict[str, Any]) -> str:
        """Add appropriate preservation rules based on task type"""
        preservation = []
        
        # Get preservation rules from config
        rules = self.templates.get_preservation_rules(task_type)
        if rules:
            preservation.extend(rules)
        
        # Add dynamic rules based on parameters
        if params.get('object') and task_type != 'object_manipulation':
            preservation.append(f"the {params['object']}")
        
        if preservation and "preserve" not in prompt.lower():
            preservation_text = f" Preserve {', '.join(preservation)}."
            prompt += preservation_text
        
        return prompt
    
    def _enhance_with_context(self, prompt: str, analysis: Dict) -> str:
        """Enhance prompt with context from image analysis"""
        # This will be expanded when Florence-2 is integrated
        # For now, just ensure we have context awareness
        if analysis and not prompt.startswith("In the image"):
            context_prefix = "In the analyzed image, "
            return context_prefix + prompt[0].lower() + prompt[1:]
        return prompt
    
    def _extract_template_params(self, template: str) -> List[str]:
        """Extract parameter names from a template string"""
        # Find all {param_name} patterns
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, template)
        return list(set(matches))  # Remove duplicates
    
    def get_history(self) -> List[Dict]:
        """Get generation history"""
        return self.generation_history.copy()
    
    def clear_history(self) -> None:
        """Clear generation history"""
        self.generation_history.clear()
        self.last_generated = None
