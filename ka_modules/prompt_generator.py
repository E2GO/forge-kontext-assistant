"""
Prompt Generator Module - Intelligent prompt generation for FLUX.1 Kontext
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import templates from ka_modules
from ka_modules.templates import PromptTemplates

logger = logging.getLogger(__name__)

class PromptGenerator:
    """Generates FLUX.1 Kontext instructional prompts from user intent"""
    
    def __init__(self):
        self.templates = PromptTemplates()
        self.task_configs = self._load_task_configs()
        
    def _load_task_configs(self) -> Dict[str, Any]:
        """Load task configurations"""
        config_path = Path(__file__).parent.parent / "configs" / "task_configs.json"
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load task configs: {e}")
        
        # Fallback config
        return {
            "tasks": {
                "object_color": {
                    "display_name": "Change Object Color",
                    "complexity": "simple",
                    "required_analysis": ["objects", "lighting"],
                    "auto_preserve": ["shadows", "reflections", "texture", "position"]
                },
                "style_transfer": {
                    "display_name": "Apply Artistic Style", 
                    "complexity": "medium",
                    "required_analysis": ["style", "objects", "composition"],
                    "auto_preserve": ["object_identity", "scene_layout"]
                }
            }
        }
    
    def generate(self, task_type: str, user_intent: str, 
                 image_analyses: List[Dict[str, Any]] = None,
                 context_strength: float = 0.7) -> str:
        """Generate instructional prompt for FLUX.1 Kontext"""
        
        # Parse user intent
        parsed_intent = self._parse_intent(user_intent, task_type)
        
        # Get base template
        template_method = getattr(self.templates, f"get_{task_type}_template", None)
        if not template_method:
            logger.warning(f"No template for task type: {task_type}")
            return self._generate_generic_prompt(user_intent, image_analyses)
        
        # Extract context from analyses
        context = self._extract_context(image_analyses) if image_analyses else {}
        
        # Generate prompt using template
        base_prompt = template_method(
            target=parsed_intent.get('target', 'the main subject'),
            action=parsed_intent.get('action', user_intent),
            context=context
        )
        
        # Add preservation rules based on context strength
        preservation_rules = self._get_preservation_rules(task_type, context_strength)
        
        # Combine prompt with preservation rules
        final_prompt = f"{base_prompt} {preservation_rules}".strip()
        
        return final_prompt
    
    def _parse_intent(self, intent: str, task_type: str) -> Dict[str, str]:
        """Parse user intent into structured components"""
        intent_lower = intent.lower()
        parsed = {}
        
        # Object color parsing
        if task_type == "object_color":
            # Pattern: "make/change the X (to) Y"
            color_pattern = r"(?:make|change|turn|paint)\s+(?:the\s+)?(\w+)\s+(?:to\s+)?(\w+)"
            match = re.search(color_pattern, intent_lower)
            if match:
                parsed['target'] = match.group(1)
                parsed['new_color'] = match.group(2)
                parsed['action'] = f"change color to {match.group(2)}"
        
        # Style transfer parsing  
        elif task_type == "style_transfer":
            # Extract style references
            style_keywords = ["style", "like", "aesthetic", "look", "appearance"]
            for keyword in style_keywords:
                if keyword in intent_lower:
                    parsed['style'] = intent
                    parsed['action'] = f"apply {intent} style"
                    break
        
        # Environment change parsing
        elif task_type == "environment_change":
            location_keywords = ["beach", "forest", "city", "mountain", "desert", "indoor", "outdoor"]
            for location in location_keywords:
                if location in intent_lower:
                    parsed['target_env'] = location
                    parsed['action'] = f"change location to {location}"
                    break
        
        # Generic parsing for other types
        if not parsed:
            parsed['action'] = intent
            parsed['target'] = 'the subject'
        
        return parsed
    
    def _extract_context(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant context from image analyses"""
        context = {
            'objects': [],
            'main_subject': None,
            'style': None,
            'environment': None,
            'lighting': None
        }
        
        if not analyses:
            return context
        
        # Aggregate from all analyses
        for analysis in analyses:
            if 'objects' in analysis:
                context['objects'].extend(analysis['objects'].get('main', []))
                if not context['main_subject'] and analysis['objects'].get('main'):
                    context['main_subject'] = analysis['objects']['main'][0]
            
            if 'style' in analysis and not context['style']:
                context['style'] = analysis['style'].get('artistic', 'photorealistic')
            
            if 'environment' in analysis and not context['environment']:
                context['environment'] = analysis['environment'].get('setting', 'unknown')
            
            if 'lighting' in analysis and not context['lighting']:
                context['lighting'] = analysis['lighting']
        
        return context
    
    def _get_preservation_rules(self, task_type: str, strength: float) -> str:
        """Generate preservation rules based on task and strength"""
        
        task_config = self.task_configs.get('tasks', {}).get(task_type, {})
        auto_preserve = task_config.get('auto_preserve', [])
        
        if strength > 0.8:
            # Strong preservation
            rules = "Maintain exact composition, lighting, shadows, reflections, and all details"
        elif strength > 0.5:
            # Medium preservation  
            preserve_items = auto_preserve[:3] if auto_preserve else ["composition", "lighting", "style"]
            rules = f"Preserve {', '.join(preserve_items)}"
        else:
            # Minimal preservation
            rules = "Keep the general scene intact"
        
        # Add specific rules for task type
        if task_type == "object_color":
            rules += " while only changing the specified color"
        elif task_type == "style_transfer":
            rules += " while maintaining recognizable subjects"
        elif task_type == "environment_change":
            rules += " while keeping the subject in the exact same pose and position"
        
        return rules
    
    def _generate_generic_prompt(self, intent: str, analyses: List[Dict[str, Any]]) -> str:
        """Fallback generic prompt generation"""
        
        # Extract main subject from analyses
        main_subject = "the subject"
        if analyses:
            for analysis in analyses:
                if 'objects' in analysis and analysis['objects'].get('main'):
                    main_subject = analysis['objects']['main'][0]
                    break
        
        # Build generic instructional prompt
        prompt = f"Modify {main_subject} to {intent}. "
        prompt += "Maintain the overall composition and quality of the image."
        
        return prompt
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validate generated prompt for FLUX.1 Kontext compatibility"""
        
        # Check length
        if len(prompt) < 10:
            return False, "Prompt too short"
        if len(prompt) > 500:
            return False, "Prompt too long (max 500 characters)"
        
        # Check for instructional language
        instruction_keywords = ["change", "modify", "add", "remove", "replace", "convert", "transform", "apply", "make"]
        has_instruction = any(keyword in prompt.lower() for keyword in instruction_keywords)
        
        if not has_instruction:
            return False, "Prompt should contain clear instructions (change, modify, add, etc.)"
        
        # Check for preservation clauses
        preservation_keywords = ["maintain", "preserve", "keep", "while", "except"]
        has_preservation = any(keyword in prompt.lower() for keyword in preservation_keywords)
        
        if not has_preservation:
            logger.warning("Prompt lacks preservation clauses - may cause unwanted changes")
        
        return True, "Valid"