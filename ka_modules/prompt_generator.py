"""
Prompt generation logic for FLUX.1 Kontext
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PromptGenerator")

@dataclass
class GenerationContext:
    """Context for prompt generation"""
    task_type: str
    user_intent: str
    image_analysis: Optional[Dict[str, Any]] = None
    preservation_strength: float = 0.8
    
class PromptGenerator:
    """Generates FLUX.1 Kontext prompts from user intent"""
    
    def __init__(self, templates):
        self.templates = templates
        self.intent_patterns = self._compile_intent_patterns()
        
    def _compile_intent_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for intent detection"""
        return {
            "color_change": [
                re.compile(r"make\s+(?:the\s+)?(\w+)\s+(\w+)", re.I),
                re.compile(r"change\s+(?:the\s+)?(\w+)(?:\s+color)?\s+to\s+(\w+)", re.I),
                re.compile(r"(\w+)\s+to\s+(\w+)", re.I),
            ],
            "style_transfer": [
                re.compile(r"(?:in|as|like)\s+(\w+)\s+style", re.I),
                re.compile(r"make\s+it\s+look\s+(\w+)", re.I),
                re.compile(r"(\w+)\s+(?:painting|art|aesthetic)", re.I),
            ],
            "environment": [
                re.compile(r"(?:at|in|to)\s+(?:the\s+)?(\w+)", re.I),
                re.compile(r"change\s+(?:background|setting)\s+to\s+(\w+)", re.I),
            ],
            "removal": [
                re.compile(r"remove\s+(?:the\s+)?(\w+)", re.I),
                re.compile(r"delete\s+(?:the\s+)?(\w+)", re.I),
                re.compile(r"get\s+rid\s+of\s+(?:the\s+)?(\w+)", re.I),
            ],
            "addition": [
                re.compile(r"add\s+(?:a\s+)?(\w+)", re.I),
                re.compile(r"put\s+(?:a\s+)?(\w+)", re.I),
                re.compile(r"place\s+(?:a\s+)?(\w+)", re.I),
            ]
        }
    
    def generate(self, task_type: str, user_intent: str, 
                 image_analysis: Optional[Dict[str, Any]] = None,
                 preservation_strength: float = 0.8) -> str:
        """Generate FLUX.1 Kontext prompt"""
        
        context = GenerationContext(
            task_type=task_type,
            user_intent=user_intent,
            image_analysis=image_analysis,
            preservation_strength=preservation_strength
        )
        
        # Determine subtype from user intent
        subtype = self._detect_subtype(context)
        
        # Special handling for lighting with direction
        if task_type == "lighting_adjustment" and "from" in user_intent.lower():
            # Force direction subtype when "from" is present
            subtype = "direction"
        
        # Get base template
        template = self.templates.get_template(task_type, subtype)
        
        # Extract parameters from intent and analysis
        params = self._extract_template_params(context, template)
        
        # Fill template
        prompt = self._fill_template(template, params)
        
        # Add preservation rules
        preservation_rules = self.templates.get_preservation_rules(task_type)
        preservation_clause = self.templates.format_preservation_clause(preservation_rules)
        
        # Combine with preservation
        if preservation_clause:
            prompt = f"{prompt}. {preservation_clause}"
        
        # Add context from image analysis if available
        if image_analysis:
            prompt = self._enhance_with_analysis(prompt, image_analysis, task_type)
        
        return prompt
    
    def _detect_subtype(self, context: GenerationContext) -> Optional[str]:
        """Detect task subtype from user intent"""
        intent_lower = context.user_intent.lower()
        
        # Object manipulation subtypes
        if context.task_type == "object_manipulation":
            if any(word in intent_lower for word in ["color", "red", "blue", "green", "yellow", "black", "white"]):
                return "color"
            elif any(word in intent_lower for word in ["remove", "delete", "erase"]):
                return "removal"
            elif any(word in intent_lower for word in ["add", "put", "place"]):
                return "addition"
            else:
                return "state"
        
        # Style transfer subtypes
        elif context.task_type == "style_transfer":
            if any(word in intent_lower for word in ["painting", "impressionist", "abstract", "realistic", "oil", "watercolor"]):
                return "artistic"
            elif any(word in intent_lower for word in ["vintage", "retro", "modern", "futuristic"]):
                return "temporal"
            elif any(word in intent_lower for word in ["anime", "manga", "western", "eastern"]):
                return "cultural"
        
        # Environment change subtypes
        elif context.task_type == "environment_change":
            if any(word in intent_lower for word in ["weather", "rain", "snow", "sunny", "cloudy"]):
                return "weather"
            elif any(word in intent_lower for word in ["morning", "evening", "night", "sunset", "sunrise"]):
                return "time"
            else:
                return "location"
        
        # Lighting adjustment subtypes
        elif context.task_type == "lighting_adjustment":
            if any(word in intent_lower for word in ["soft", "hard", "dramatic", "subtle"]):
                return "quality"
            elif any(word in intent_lower for word in ["warm", "cool", "cold", "neutral"]):
                return "color"
            elif any(word in intent_lower for word in ["left", "right", "top", "bottom", "front", "back", "side"]):
                return "direction"
            else:
                return "direction"  # Default to direction
        
        # State change subtypes
        elif context.task_type == "state_change":
            if any(word in intent_lower for word in ["old", "aged", "weathered", "worn"]):
                return "age"
            elif any(word in intent_lower for word in ["broken", "damaged", "cracked", "torn"]):
                return "damage"
            elif any(word in intent_lower for word in ["season", "winter", "summer", "spring", "fall"]):
                return "seasonal"
        
        # Outpainting subtypes
        elif context.task_type == "outpainting":
            if any(word in intent_lower for word in ["zoom", "wider", "more"]):
                return "zoom"
            else:
                return "extend"
        
        return None
    
    def _extract_template_params(self, context: GenerationContext, template: str) -> Dict[str, str]:
        """Extract parameters needed for template from context"""
        params = {}
        intent = context.user_intent
        
        # Find all template variables
        template_vars = re.findall(r'\{(\w+)\}', template)
        
        for var in template_vars:
            if var == "object":
                # Extract object from intent or analysis
                params[var] = self._extract_object(intent, context.image_analysis)
            elif var == "current_color":
                params[var] = self._extract_current_color(context.image_analysis)
            elif var == "new_color":
                params[var] = self._extract_new_color(intent)
            elif var == "style":
                params[var] = self._extract_style(intent)
            elif var == "new_location":
                params[var] = self._extract_location(intent)
            elif var == "weather_condition":
                params[var] = self._extract_weather(intent)
            elif var == "time":
                params[var] = self._extract_time(intent)
            elif var == "direction":
                params[var] = self._extract_direction(intent)
            elif var == "quality":
                params[var] = self._extract_quality(intent)
            elif var == "temperature":
                params[var] = self._extract_temperature(intent)
            elif var == "season":
                params[var] = self._extract_season(intent)
            elif var == "amount":
                # For aging, extract amount or use default
                params[var] = self._extract_amount(intent)
            elif var == "damage_type":
                params[var] = self._extract_damage_type(intent)
            elif var == "current_state":
                params[var] = self._extract_current_state(context.image_analysis)
            elif var == "new_state":
                params[var] = self._extract_new_state(intent)
            elif var == "new_object":
                params[var] = self._extract_new_object(intent)
            elif var == "position":
                params[var] = self._extract_position(intent)
            elif var == "time_period":
                params[var] = self._extract_time_period(intent)
            elif var == "culture":
                params[var] = self._extract_culture(intent)
            else:
                # Generic extraction
                params[var] = self._extract_generic(intent, var)
        
        return params
    
    def _fill_template(self, template: str, params: Dict[str, str]) -> str:
        """Fill template with parameters"""
        prompt = template
        for key, value in params.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt
    
    def _enhance_with_analysis(self, prompt: str, analysis: Dict[str, Any], task_type: str) -> str:
        """Enhance prompt with image analysis context"""
        if not analysis:
            return prompt
        
        # Add specific context based on task type
        enhancements = []
        
        if task_type == "style_transfer" and "style" in analysis:
            current_style = analysis["style"].get("artistic", "current")
            enhancements.append(f"maintaining composition from {current_style} style")
        
        if task_type == "environment_change" and "environment" in analysis:
            current_env = analysis["environment"].get("setting", "current setting")
            enhancements.append(f"transitioning from {current_env}")
        
        if task_type == "lighting_adjustment" and "lighting" in analysis:
            current_light = analysis["lighting"].get("primary_source", "current lighting")
            enhancements.append(f"adjusting from {current_light}")
        
        if enhancements:
            prompt += " " + ", ".join(enhancements)
        
        return prompt
    
    # Extraction helper methods
    def _extract_object(self, intent: str, analysis: Optional[Dict]) -> str:
        """Extract object from intent or analysis"""
        # Skip "it" and "look" as objects
        skip_words = ["it", "them", "this", "that", "look", "make"]
        
        # Common objects to look for
        object_words = ["car", "person", "building", "tree", "sky", "road", "house", "dog", "cat", "chair", "table", "flower", "mountain", "water"]
        intent_lower = intent.lower()
        
        # First check for explicit objects
        for obj in object_words:
            if obj in intent_lower:
                return obj
        
        # Then try patterns
        patterns = [
            re.compile(r"(?:the\s+)?(\w+)\s+(?:to|from|blue|red|green|yellow|old|new)", re.I),
            re.compile(r"change\s+(?:the\s+)?(\w+)", re.I),
            re.compile(r"make\s+(?:the\s+)?(\w+)\s+(?:look|appear)", re.I),
        ]
        
        for pattern in patterns:
            match = pattern.search(intent)
            if match and match.group(1).lower() not in skip_words:
                return match.group(1)
        
        # For state changes, if no object found, look at analysis
        if analysis and "objects" in analysis:
            main_objects = analysis["objects"].get("main", [])
            if main_objects:
                return main_objects[0]
        
        return "subject"
    
    def _extract_current_color(self, analysis: Optional[Dict]) -> str:
        """Extract current color from analysis"""
        if analysis and "style" in analysis:
            palette = analysis["style"].get("color_palette", {})
            dominant = palette.get("dominant", [])
            if dominant:
                return dominant[0]
        return "current color"
    
    def _extract_new_color(self, intent: str) -> str:
        """Extract target color from intent"""
        color_words = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown"]
        intent_lower = intent.lower()
        
        for color in color_words:
            if color in intent_lower:
                return color
        
        return "new color"
    
    def _extract_style(self, intent: str) -> str:
        """Extract style from intent"""
        # First check for specific style keywords
        style_keywords = {
            "oil painting": "oil painting",
            "watercolor": "watercolor",
            "pencil": "pencil sketch",
            "charcoal": "charcoal drawing",
            "digital": "digital art",
            "impressionist": "impressionist",
            "abstract": "abstract",
            "photorealistic": "photorealistic",
            "anime": "anime",
            "manga": "manga"
        }
        
        intent_lower = intent.lower()
        for keyword, style in style_keywords.items():
            if keyword in intent_lower:
                return style
        
        # Then try patterns
        style_patterns = [
            re.compile(r"(\w+\s+\w+)\s+style", re.I),  # Two-word styles
            re.compile(r"(\w+)\s+style", re.I),
            re.compile(r"like\s+(\w+)", re.I),
            re.compile(r"(\w+)\s+painting", re.I),
        ]
        
        for pattern in style_patterns:
            match = pattern.search(intent)
            if match:
                return match.group(1)
        
        return "artistic"
    
    def _extract_location(self, intent: str) -> str:
        """Extract location from intent"""
        location_words = ["beach", "mountain", "city", "forest", "desert", "space", "underwater", "sky"]
        intent_lower = intent.lower()
        
        for loc in location_words:
            if loc in intent_lower:
                return loc
        
        return "new location"
    
    def _extract_weather(self, intent: str) -> str:
        """Extract weather condition from intent"""
        weather_words = ["rain", "snow", "sunny", "cloudy", "storm", "fog", "clear"]
        intent_lower = intent.lower()
        
        for weather in weather_words:
            if weather in intent_lower:
                return weather
        
        return "clear"
    
    def _extract_time(self, intent: str) -> str:
        """Extract time of day from intent"""
        time_words = ["morning", "evening", "night", "sunset", "sunrise", "noon", "midnight"]
        intent_lower = intent.lower()
        
        for time in time_words:
            if time in intent_lower:
                return time
        
        return "evening"
    
    def _extract_direction(self, intent: str) -> str:
        """Extract direction from intent"""
        direction_words = ["left", "right", "top", "bottom", "above", "below"]
        intent_lower = intent.lower()
        
        for direction in direction_words:
            if direction in intent_lower:
                return direction
        
        return "right"
    
    def _extract_quality(self, intent: str) -> str:
        """Extract lighting quality from intent"""
        quality_words = ["soft", "hard", "dramatic", "subtle", "natural"]
        intent_lower = intent.lower()
        
        for quality in quality_words:
            if quality in intent_lower:
                return quality
        
        return "natural"
    
    def _extract_temperature(self, intent: str) -> str:
        """Extract color temperature from intent"""
        temp_words = ["warm", "cool", "cold", "neutral"]
        intent_lower = intent.lower()
        
        for temp in temp_words:
            if temp in intent_lower:
                return temp
        
        return "neutral"
    
    def _extract_season(self, intent: str) -> str:
        """Extract season from intent"""
        season_words = ["winter", "summer", "spring", "fall", "autumn"]
        intent_lower = intent.lower()
        
        for season in season_words:
            if season in intent_lower:
                return season
        
        return "summer"
    
    def _extract_generic(self, intent: str, var_name: str) -> str:
        """Generic parameter extraction"""
        # Simple word extraction after common prepositions
        patterns = [
            re.compile(rf"to\s+(\w+)", re.I),
            re.compile(rf"from\s+(\w+)", re.I),
            re.compile(rf"into\s+(\w+)", re.I),
        ]
        
        for pattern in patterns:
            match = pattern.search(intent)
            if match:
                return match.group(1)
        
        return var_name.replace("_", " ")
    
    # Additional extraction methods
    def _extract_amount(self, intent: str) -> str:
        """Extract amount for aging"""
        # Look for numbers
        number_match = re.search(r"(\d+)\s*(?:years?|decades?)", intent)
        if number_match:
            return number_match.group(1)
        
        # Look for descriptive amounts
        if "slightly" in intent.lower():
            return "5"
        elif "very" in intent.lower() or "extremely" in intent.lower():
            return "30"
        else:
            return "10"
    
    def _extract_damage_type(self, intent: str) -> str:
        """Extract damage type"""
        damage_words = ["broken", "cracked", "torn", "rusted", "weathered", "damaged"]
        intent_lower = intent.lower()
        
        for damage in damage_words:
            if damage in intent_lower:
                return damage
        
        return "weathered"
    
    def _extract_current_state(self, analysis: Optional[Dict]) -> str:
        """Extract current state from analysis"""
        if analysis and "objects" in analysis:
            # Mock: assume objects are in good condition
            return "pristine"
        return "current"
    
    def _extract_new_state(self, intent: str) -> str:
        """Extract target state from intent"""
        # Extended state words including variations
        state_mapping = {
            "old": "old and worn",
            "aged": "aged",
            "weathered": "weathered",
            "new": "new",
            "broken": "broken",
            "fixed": "fixed",
            "clean": "clean",
            "dirty": "dirty",
            "worn": "worn",
            "damaged": "damaged",
            "pristine": "pristine",
            "abandoned": "abandoned"
        }
        
        intent_lower = intent.lower()
        
        # Check for compound states like "old and weathered"
        if "old and weathered" in intent_lower:
            return "old and weathered"
        elif "old and abandoned" in intent_lower:
            return "old and abandoned"
        
        # Check individual state words
        for state, value in state_mapping.items():
            if state in intent_lower:
                return value
        
        return "modified"
    
    def _extract_new_object(self, intent: str) -> str:
        """Extract object to add"""
        patterns = [
            re.compile(r"add\s+(?:a\s+)?(\w+)", re.I),
            re.compile(r"put\s+(?:a\s+)?(\w+)", re.I),
            re.compile(r"place\s+(?:a\s+)?(\w+)", re.I),
        ]
        
        for pattern in patterns:
            match = pattern.search(intent)
            if match:
                return match.group(1)
        
        return "object"
    
    def _extract_position(self, intent: str) -> str:
        """Extract position for object placement"""
        position_words = ["center", "left", "right", "top", "bottom", "corner", "foreground", "background"]
        intent_lower = intent.lower()
        
        for pos in position_words:
            if pos in intent_lower:
                return pos
        
        return "in the scene"
    
    def _extract_time_period(self, intent: str) -> str:
        """Extract time period for temporal style"""
        period_words = ["vintage", "retro", "modern", "futuristic", "ancient", "medieval"]
        intent_lower = intent.lower()
        
        for period in period_words:
            if period in intent_lower:
                return period
        
        return "different era"
    
    def _extract_culture(self, intent: str) -> str:
        """Extract cultural style"""
        culture_words = ["anime", "manga", "western", "eastern", "japanese", "chinese"]
        intent_lower = intent.lower()
        
        for culture in culture_words:
            if culture in intent_lower:
                return culture
        
        return "stylized"