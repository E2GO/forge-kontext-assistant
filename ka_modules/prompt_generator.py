"""
Prompt generation engine with Florence-2 integration.
Generates FLUX.1 Kontext-compatible instructional prompts.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Compatibility fix
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from .templates import PromptTemplates

logger = logging.getLogger("KontextAssistant.PromptGenerator")


@dataclass
class GenerationContext:
    """Context for prompt generation."""
    task_type: str
    subtype: str
    user_intent: str
    image_analysis: Optional[Dict[str, Any]] = None
    preserve_strength: float = 0.7
    custom_params: Optional[Dict[str, Any]] = None


class PromptGenerator:
    """Generates instructional prompts for FLUX.1 Kontext."""
    
    def __init__(self, templates: PromptTemplates):
        self.templates = templates
        self.complexity_threshold = 0.7
        
    def generate(
        self,
        task_type: str,
        user_intent: str,
        image_analysis: Optional[Dict[str, Any]] = None,
        subtype: Optional[str] = None,
        preserve_strength: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a FLUX.1 Kontext prompt based on inputs.
        
        Args:
            task_type: Type of editing task
            user_intent: User's description of desired change
            image_analysis: Analysis data from Florence-2
            subtype: Specific subtype of the task
            preserve_strength: How much to preserve (0-1)
            **kwargs: Additional parameters
            
        Returns:
            Generated instructional prompt
        """
        try:
            # Create generation context
            context = GenerationContext(
                task_type=task_type,
                subtype=subtype or "default",
                user_intent=user_intent,
                image_analysis=image_analysis or {},
                preserve_strength=preserve_strength,
                custom_params=kwargs
            )
            
            # Extract key information from user intent
            intent_params = self._parse_user_intent(context)
            
            # Get appropriate template
            template = self._select_template(context, intent_params)
            
            # Extract template parameters from analysis
            template_params = self._extract_template_params(
                template, 
                context, 
                intent_params
            )
            
            # Generate base prompt
            prompt = self._fill_template(template, template_params)
            
            # Add preservation rules
            prompt = self._add_preservation_rules(prompt, context)
            
            # Optimize for FLUX.1 Kontext
            prompt = self._optimize_for_kontext(prompt, context)
            
            logger.info(f"Generated prompt for {task_type}/{subtype}")
            return prompt
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            # Fallback to simple generation
            return self._generate_fallback(user_intent)
    
    def _parse_user_intent(self, context: GenerationContext) -> Dict[str, Any]:
        """Parse user intent to extract key information."""
        intent = context.user_intent.lower()
        params = {}
        
        # Extract colors
        color_pattern = r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|brown|golden|silver)\b'
        colors = re.findall(color_pattern, intent)
        if colors:
            params['target_color'] = colors[-1]  # Last mentioned color is usually target
            if len(colors) > 1:
                params['source_color'] = colors[0]
        
        # Extract objects
        # Common objects in prompts
        object_words = [
            'car', 'person', 'building', 'tree', 'sky', 'road', 'wall',
            'dress', 'shirt', 'hair', 'eyes', 'background', 'object'
        ]
        for obj in object_words:
            if obj in intent:
                params['target_object'] = obj
                break
        
        # Extract actions
        action_patterns = {
            'add': r'\b(add|insert|place|put)\b',
            'remove': r'\b(remove|delete|erase|eliminate)\b',
            'change': r'\b(change|transform|convert|make|turn)\b',
            'replace': r'\b(replace|swap|substitute)\b'
        }
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, intent):
                params['action'] = action
                break
        
        # Extract styles
        style_words = [
            'realistic', 'cartoon', 'anime', 'painting', 'sketch',
            'vintage', 'modern', 'retro', 'cyberpunk', 'steampunk'
        ]
        for style in style_words:
            if style in intent:
                params['target_style'] = style
                break
        
        # Extract time/weather
        time_words = {
            'morning': 'morning',
            'sunset': 'evening', 
            'sunrise': 'dawn',
            'night': 'night',
            'evening': 'evening',
            'noon': 'midday'
        }
        for word, time in time_words.items():
            if word in intent:
                params['target_time'] = time
                break
        
        weather_words = ['rain', 'snow', 'fog', 'sunny', 'cloudy', 'storm']
        for weather in weather_words:
            if weather in intent:
                params['target_weather'] = weather
                break
        
        return params
    
    def _select_template(self, context: GenerationContext, intent_params: Dict) -> str:
        """Select the most appropriate template."""
        task_type = context.task_type
        subtype = context.subtype
        
        # Try to get specific template
        template = self.templates.get_template(task_type, subtype)
        if template:
            return template
        
        # Fallback to default for task type
        template = self.templates.get_template(task_type, "default")
        if template:
            return template
        
        # Ultimate fallback
        return "Change {target} to {description}. Keep everything else unchanged."
    
    def _extract_template_params(
        self, 
        template: str, 
        context: GenerationContext,
        intent_params: Dict
    ) -> Dict[str, str]:
        """Extract parameters needed for template from context and analysis."""
        # Find all template variables
        variables = re.findall(r'\{(\w+)\}', template)
        params = {}
        
        # Get data from image analysis if available
        analysis = context.image_analysis
        
        for var in variables:
            # First check intent params
            if var in intent_params:
                params[var] = intent_params[var]
                continue
            
            # Then check analysis data
            if analysis:
                # Map template variables to analysis data
                if var == 'object' and 'objects' in analysis:
                    objects = analysis.get('objects', [])
                    if objects:
                        params[var] = objects[0] if isinstance(objects, list) else "object"
                
                elif var == 'current_style' and 'styles' in analysis:
                    styles = analysis.get('styles', [])
                    if styles:
                        params[var] = styles[0] if isinstance(styles, list) else "current style"
                
                elif var == 'setting' and 'environment' in analysis:
                    params[var] = analysis.get('environment', {}).get('setting', 'scene')
                
                elif var == 'current_time' and 'environment' in analysis:
                    params[var] = analysis.get('environment', {}).get('time_of_day', 'current time')
            
            # Default values for common variables
            if var not in params:
                defaults = {
                    'object': 'subject',
                    'target': 'element',
                    'description': context.user_intent,
                    'style': 'style',
                    'current_style': 'current style',
                    'new_style': intent_params.get('target_style', 'new style'),
                    'setting': 'environment',
                    'atmosphere': 'atmosphere',
                    'current_time': 'current time',
                    'new_time': intent_params.get('target_time', 'new time'),
                    'preservation_rules': self._get_preservation_rules(context)
                }
                params[var] = defaults.get(var, var)
        
        return params
    
    def _fill_template(self, template: str, params: Dict[str, str]) -> str:
        """Fill template with parameters."""
        prompt = template
        for key, value in params.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
    
    def _add_preservation_rules(self, prompt: str, context: GenerationContext) -> str:
        """Add preservation rules based on context."""
        if context.preserve_strength < 0.3:
            # Low preservation - minimal rules
            return prompt
        
        # Check if preservation rules already in prompt
        if "preserve" in prompt.lower() or "maintain" in prompt.lower():
            return prompt
        
        # Add preservation based on task type
        preservation_rules = self._get_preservation_rules(context)
        
        if preservation_rules and context.preserve_strength > 0.5:
            prompt += f" {preservation_rules}"
        
        return prompt
    
    def _get_preservation_rules(self, context: GenerationContext) -> str:
        """Get preservation rules for the task type."""
        rules = {
            "object_manipulation": "Preserve all other objects, background, lighting, and composition exactly as they are.",
            "style_transfer": "Maintain the exact same composition, objects, and their positions while only changing the artistic style.",
            "environment_change": "Keep all subjects in their exact positions and poses, only modify the environment around them.",
            "element_combination": "Preserve the individual characteristics of each element while blending them naturally.",
            "state_change": "Maintain object identity and position while showing the transformation.",
            "outpainting": "Seamlessly extend the scene while maintaining consistency with the original image."
        }
        
        base_rule = rules.get(context.task_type, "Keep all unmodified elements exactly as they are.")
        
        # Adjust based on preservation strength
        if context.preserve_strength > 0.8:
            base_rule = "Strictly " + base_rule.lower()
        elif context.preserve_strength < 0.5:
            base_rule = base_rule.replace("exactly", "generally")
        
        return base_rule
    
    def _optimize_for_kontext(self, prompt: str, context: GenerationContext) -> str:
        """Optimize prompt for FLUX.1 Kontext's instruction-following."""
        # Remove vague descriptors
        vague_words = ['maybe', 'perhaps', 'somewhat', 'kind of', 'sort of']
        for word in vague_words:
            prompt = prompt.replace(word, '')
        
        # Ensure action verbs are clear
        if not any(verb in prompt.lower() for verb in ['change', 'add', 'remove', 'replace', 'transform']):
            # Add explicit action verb
            prompt = f"Change to: {prompt}"
        
        # Remove redundant spaces
        prompt = ' '.join(prompt.split())
        
        # Ensure it ends with a period
        if not prompt.endswith('.'):
            prompt += '.'
        
        return prompt
    
    def _generate_fallback(self, user_intent: str) -> str:
        """Generate a simple fallback prompt."""
        # Clean up intent
        intent = user_intent.strip()
        if not intent:
            return "Make adjustments to the image as needed."
        
        # Ensure it's instructional
        if not any(word in intent.lower() for word in ['change', 'add', 'remove', 'make']):
            intent = f"Change to: {intent}"
        
        # Add basic preservation
        return f"{intent}. Maintain all other elements unchanged."
    
    def assess_complexity(self, user_intent: str, image_analysis: Optional[Dict] = None) -> float:
        """
        Assess the complexity of the user's request.
        
        Returns:
            float: Complexity score (0-1)
        """
        score = 0.0
        
        # Length complexity
        word_count = len(user_intent.split())
        if word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1
        
        # Multiple operations
        operations = ['and', 'then', 'also', 'plus', 'with']
        op_count = sum(1 for op in operations if op in user_intent.lower())
        score += min(op_count * 0.15, 0.3)
        
        # Cultural/artistic references
        cultural_terms = [
            'style of', 'inspired by', 'reminiscent of', 'like a',
            'as if', 'similar to', 'in the manner of'
        ]
        if any(term in user_intent.lower() for term in cultural_terms):
            score += 0.2
        
        # Ambiguous terms
        ambiguous = ['artistic', 'beautiful', 'better', 'improve', 'enhance']
        amb_count = sum(1 for term in ambiguous if term in user_intent.lower())
        score += min(amb_count * 0.1, 0.2)
        
        # Complex state changes
        complex_changes = ['aging', 'decay', 'transformation', 'metamorphosis']
        if any(term in user_intent.lower() for term in complex_changes):
            score += 0.2
        
        return min(score, 1.0)