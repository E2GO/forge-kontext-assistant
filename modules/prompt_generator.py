"""
Template-based prompt generation for FLUX.1 Kontext.

This module handles parsing user intent, extracting parameters,
and generating instructional prompts using templates.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from .templates import get_template, TemplateBuilder, PromptTemplate

logger = logging.getLogger("KontextAssistant.PromptGenerator")


@dataclass
class ParsedIntent:
    """Parsed user intent with extracted parameters"""
    action: str
    target: str
    parameters: Dict[str, str]
    confidence: float = 1.0


class IntentParser:
    """Parse natural language intent into structured parameters"""
    
    # Color mappings for common variations
    COLOR_SYNONYMS = {
        "red": ["crimson", "scarlet", "ruby", "cherry", "maroon"],
        "blue": ["azure", "navy", "cobalt", "sapphire", "cerulean", "sky"],
        "green": ["emerald", "jade", "olive", "lime", "forest"],
        "yellow": ["gold", "golden", "amber", "lemon", "mustard"],
        "purple": ["violet", "lavender", "plum", "magenta"],
        "orange": ["tangerine", "coral", "peach", "apricot"],
        "black": ["ebony", "onyx", "charcoal", "midnight"],
        "white": ["ivory", "cream", "pearl", "snow"],
        "gray": ["grey", "silver", "ash", "slate"],
        "brown": ["chocolate", "coffee", "tan", "beige", "bronze"]
    }
    
    # State change mappings
    STATE_CHANGES = {
        "open_close": {
            "patterns": ["open", "close", "shut", "opening", "closing"],
            "states": {"open": "closed", "closed": "open", "shut": "open"}
        },
        "on_off": {
            "patterns": ["turn on", "turn off", "switch on", "switch off", "lights on", "lights off"],
            "states": {"on": "off", "off": "on"}
        },
        "full_empty": {
            "patterns": ["fill", "empty", "drain", "pour"],
            "states": {"full": "empty", "empty": "full", "filled": "empty"}
        },
        "wet_dry": {
            "patterns": ["wet", "dry", "soak", "drench"],
            "states": {"wet": "dry", "dry": "wet", "soaked": "dry"}
        }
    }
    
    # Style mappings
    STYLE_MAPPINGS = {
        "artistic": {
            "impressionist": ["monet", "impressionism", "impressionistic"],
            "cubist": ["picasso", "cubism", "geometric art"],
            "surreal": ["dali", "surrealism", "dreamlike"],
            "watercolor": ["watercolour", "aquarelle"],
            "oil_painting": ["oil paint", "oils"],
            "pencil_sketch": ["pencil drawing", "graphite", "sketch"]
        },
        "temporal": {
            "vintage": ["retro", "old-fashioned", "antique", "classic"],
            "retro_80s": ["80s", "eighties", "synthwave", "vaporwave"],
            "futuristic": ["sci-fi", "cyberpunk", "future", "high-tech"],
            "noir": ["film noir", "black and white", "monochrome"]
        }
    }
    
    @classmethod
    def normalize_color(cls, color_text: str) -> str:
        """Normalize color text to base color"""
        color_lower = color_text.lower().strip()
        
        # Check direct match
        if color_lower in cls.COLOR_SYNONYMS:
            return color_lower
        
        # Check synonyms
        for base_color, synonyms in cls.COLOR_SYNONYMS.items():
            if color_lower in synonyms:
                return base_color
        
        # Return as-is if not found
        return color_lower
    
    @classmethod
    def parse_color_change(cls, intent: str, context: Optional[Dict] = None) -> Optional[ParsedIntent]:
        """Parse color change intent"""
        # Patterns for color changes
        patterns = [
            r"(?:make|change|paint|color|turn)\s+(?:the\s+)?(\w+)\s+(?:to\s+)?(\w+)",
            r"(\w+)\s+(?:car|dress|shirt|object)\s+(?:to\s+)?(\w+)",
            r"(\w+)\s+to\s+(\w+)",
            r"(?:from\s+)?(\w+)\s+(?:to\s+)?(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, intent.lower())
            if match:
                groups = match.groups()
                
                # Determine what is object and what is color
                obj, new_color = cls._identify_object_and_color(groups, context)
                
                if obj and new_color:
                    # Try to find current color from context
                    current_color = cls._find_current_color(obj, context)
                    
                    return ParsedIntent(
                        action="change_color",
                        target=obj,
                        parameters={
                            "object": obj,
                            "current_color": current_color or "current",
                            "new_color": cls.normalize_color(new_color)
                        }
                    )
        
        return None
    
    @classmethod
    def _identify_object_and_color(cls, groups: Tuple[str, ...], 
                                  context: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """Identify which part is object and which is color"""
        word1, word2 = groups[0], groups[1]
        
        # Check if either word is a known color
        color1_normalized = cls.normalize_color(word1)
        color2_normalized = cls.normalize_color(word2)
        
        is_color1 = color1_normalized in cls.COLOR_SYNONYMS or any(
            color1_normalized in syns for syns in cls.COLOR_SYNONYMS.values()
        )
        is_color2 = color2_normalized in cls.COLOR_SYNONYMS or any(
            color2_normalized in syns for syns in cls.COLOR_SYNONYMS.values()
        )
        
        # Logic to determine object and color
        if is_color2 and not is_color1:
            return word1, word2
        elif is_color1 and not is_color2:
            # This might be "red car" where we want to change red to something
            return word2, word1
        elif context and "objects" in context:
            # Use context to determine
            objects = context["objects"].get("main", [])
            for obj in objects:
                if word1 in obj.lower():
                    return word1, word2
                elif word2 in obj.lower():
                    return word2, word1
        
        # Default assumption: first word is object, second is color
        return word1, word2
    
    @classmethod
    def _find_current_color(cls, object_name: str, context: Optional[Dict] = None) -> Optional[str]:
        """Try to find current color from context"""
        if not context or "objects" not in context:
            return None
        
        # Look for color in object descriptions
        objects = context["objects"].get("main", [])
        for obj in objects:
            if object_name in obj.lower():
                # Extract color if present (e.g., "red car")
                words = obj.lower().split()
                for word in words:
                    normalized = cls.normalize_color(word)
                    if normalized in cls.COLOR_SYNONYMS:
                        return normalized
        
        return None
    
    @classmethod
    def parse_state_change(cls, intent: str, context: Optional[Dict] = None) -> Optional[ParsedIntent]:
        """Parse state change intent"""
        intent_lower = intent.lower()
        
        for change_type, info in cls.STATE_CHANGES.items():
            for pattern in info["patterns"]:
                if pattern in intent_lower:
                    # Find the object
                    obj = cls._extract_object_for_state(intent_lower, pattern, context)
                    if obj:
                        # Determine current and new states
                        current, new = cls._determine_states(pattern, info["states"])
                        
                        return ParsedIntent(
                            action="change_state",
                            target=obj,
                            parameters={
                                "object": obj,
                                "current_state": current,
                                "new_state": new
                            }
                        )
        
        return None
    
    @classmethod
    def _extract_object_for_state(cls, intent: str, pattern: str, 
                                 context: Optional[Dict] = None) -> Optional[str]:
        """Extract object for state change"""
        # Common patterns
        patterns = [
            rf"{pattern}\s+(?:the\s+)?(\w+)",
            rf"(\w+)\s+{pattern}"
        ]
        
        for p in patterns:
            match = re.search(p, intent)
            if match:
                return match.group(1)
        
        # If no explicit object, try to infer from context
        if context and "objects" in context:
            objects = context["objects"].get("main", [])
            if len(objects) == 1:
                return objects[0].split()[-1]  # Last word of object description
        
        return "object"
    
    @classmethod
    def _determine_states(cls, pattern: str, state_map: Dict[str, str]) -> Tuple[str, str]:
        """Determine current and new states"""
        for state in state_map:
            if state in pattern:
                return state, state_map[state]
        
        # Default
        first_state = list(state_map.keys())[0]
        return first_state, state_map[first_state]
    
    @classmethod
    def parse_style_transfer(cls, intent: str, context: Optional[Dict] = None) -> Optional[ParsedIntent]:
        """Parse style transfer intent"""
        intent_lower = intent.lower()
        
        # Check for artistic styles
        for category, styles in cls.STYLE_MAPPINGS.items():
            for style, synonyms in styles.items():
                all_terms = [style] + synonyms
                for term in all_terms:
                    if term in intent_lower:
                        return ParsedIntent(
                            action=f"style_{category}",
                            target="image",
                            parameters={
                                "art_style" if category == "artistic" else "time_period": style
                            }
                        )
        
        return None
    
    @classmethod
    def parse_environment_change(cls, intent: str, context: Optional[Dict] = None) -> Optional[ParsedIntent]:
        """Parse environment change intent"""
        intent_lower = intent.lower()
        
        # Weather patterns
        weather_terms = {
            "rainy": ["rain", "raining", "rainy", "wet weather"],
            "snowy": ["snow", "snowing", "snowy", "winter"],
            "sunny": ["sun", "sunny", "sunshine", "clear"],
            "foggy": ["fog", "foggy", "mist", "misty"]
        }
        
        for weather, terms in weather_terms.items():
            if any(term in intent_lower for term in terms):
                return ParsedIntent(
                    action="environment_weather",
                    target="weather",
                    parameters={"weather_condition": weather}
                )
        
        # Time of day patterns
        time_terms = {
            "sunset": ["sunset", "golden hour", "dusk"],
            "night": ["night", "nighttime", "evening", "dark"],
            "dawn": ["dawn", "sunrise", "morning"]
        }
        
        for time, terms in time_terms.items():
            if any(term in intent_lower for term in terms):
                return ParsedIntent(
                    action="environment_time",
                    target="time",
                    parameters={"time": time}
                )
        
        # Location patterns
        location_keywords = ["background", "location", "setting", "place", "scene"]
        if any(keyword in intent_lower for keyword in location_keywords):
            # Extract location description
            location = cls._extract_location(intent_lower)
            if location:
                return ParsedIntent(
                    action="environment_location",
                    target="background",
                    parameters={"location": location}
                )
        
        return None
    
    @classmethod
    def _extract_location(cls, intent: str) -> Optional[str]:
        """Extract location description from intent"""
        patterns = [
            r"(?:to|in|at)\s+(?:a\s+)?([^,\.]+)",
            r"(\w+\s*\w*)\s+(?:background|scene|setting)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, intent)
            if match:
                return match.group(1).strip()
        
        return None


class PromptGenerator:
    """Main prompt generation class"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir
        self.task_config = self._load_task_config()
        self.parser = IntentParser()
        
    def _load_task_config(self) -> Dict:
        """Load task configuration from JSON"""
        if not self.config_dir:
            return {}
        
        config_path = self.config_dir / "task_configs.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load task config: {e}")
        
        return {}
    
    def generate(self, task_type: str, user_intent: str, 
                image_analyses: Optional[List[Dict]] = None) -> str:
        """
        Generate a FLUX.1 Kontext prompt based on task and intent.
        
        Args:
            task_type: Type of task (e.g., "object_color")
            user_intent: Natural language user intent
            image_analyses: Optional list of image analysis results
            
        Returns:
            Generated instructional prompt
        """
        try:
            # Get template for task type
            template = get_template(task_type)
            if not template:
                logger.warning(f"No template found for task type: {task_type}")
                return self._generate_fallback_prompt(task_type, user_intent)
            
            # Merge analyses into single context
            context = self._merge_analyses(image_analyses) if image_analyses else None
            
            # Parse user intent based on task type
            parsed_intent = self._parse_intent_for_task(task_type, user_intent, context)
            
            if not parsed_intent:
                logger.warning(f"Could not parse intent: {user_intent}")
                return self._generate_simple_prompt(template, user_intent, context)
            
            # Build prompt using template
            prompt = TemplateBuilder.fill_template(
                template=template,
                params=parsed_intent.parameters,
                context=context
            )
            
            # Add task-specific enhancements
            prompt = self._enhance_prompt_for_task(prompt, task_type, parsed_intent, context)
            
            logger.info(f"Generated prompt: {prompt[:100]}...")
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return f"Change {user_intent} while preserving all other aspects."
    
    def _merge_analyses(self, analyses: List[Dict]) -> Dict:
        """Merge multiple image analyses into single context"""
        merged = {
            "objects": {"main": [], "secondary": []},
            "style": {},
            "environment": {},
            "lighting": {},
            "composition": {}
        }
        
        for analysis in analyses:
            if "objects" in analysis:
                merged["objects"]["main"].extend(
                    analysis["objects"].get("main", [])
                )
                merged["objects"]["secondary"].extend(
                    analysis["objects"].get("secondary", [])
                )
            
            # Merge other aspects (take first non-empty)
            for key in ["style", "environment", "lighting", "composition"]:
                if key in analysis and analysis[key]:
                    if not merged[key]:
                        merged[key] = analysis[key]
        
        # Remove duplicates
        merged["objects"]["main"] = list(set(merged["objects"]["main"]))
        merged["objects"]["secondary"] = list(set(merged["objects"]["secondary"]))
        
        return merged
    
    def _parse_intent_for_task(self, task_type: str, user_intent: str, 
                               context: Optional[Dict] = None) -> Optional[ParsedIntent]:
        """Parse intent based on task type"""
        
        # Map task types to parser methods
        parser_map = {
            "object_color": self.parser.parse_color_change,
            "object_state": self.parser.parse_state_change,
            "style_artistic": self.parser.parse_style_transfer,
            "style_temporal": self.parser.parse_style_transfer,
            "environment_weather": self.parser.parse_environment_change,
            "environment_time": self.parser.parse_environment_change,
            "environment_location": self.parser.parse_environment_change
        }
        
        parser_func = parser_map.get(task_type)
        if parser_func:
            return parser_func(user_intent, context)
        
        # For other task types, create simple parsed intent
        return self._create_simple_parsed_intent(task_type, user_intent, context)
    
    def _create_simple_parsed_intent(self, task_type: str, user_intent: str, 
                                    context: Optional[Dict] = None) -> ParsedIntent:
        """Create simple parsed intent for uncovered task types"""
        params = {}
        
        if task_type == "add_object":
            params = {
                "object": user_intent.split()[-1],  # Last word as object
                "location_description": "in the scene"
            }
        elif task_type == "remove_object":
            params = {"object": user_intent.split()[-1]}
        elif task_type == "outpainting":
            # Extract direction
            direction = "all sides"
            for d in ["left", "right", "top", "bottom"]:
                if d in user_intent.lower():
                    direction = d
                    break
            params = {
                "direction": direction,
                "description": "natural continuation of the scene"
            }
        
        return ParsedIntent(
            action=task_type,
            target=user_intent.split()[0] if user_intent else "object",
            parameters=params
        )
    
    def _generate_simple_prompt(self, template: PromptTemplate, user_intent: str, 
                               context: Optional[Dict] = None) -> str:
        """Generate simple prompt when parsing fails"""
        # Use template base with minimal parameters
        base = template.base.replace("{", "").replace("}", "")
        preservation = TemplateBuilder.build_preservation_clause(template.preserve)
        
        return f"{user_intent} {preservation}."
    
    def _generate_fallback_prompt(self, task_type: str, user_intent: str) -> str:
        """Generate fallback prompt when no template exists"""
        return f"Execute the following task: {user_intent}. Preserve all other aspects of the image unchanged."
    
    def _enhance_prompt_for_task(self, prompt: str, task_type: str, 
                                parsed_intent: ParsedIntent, 
                                context: Optional[Dict] = None) -> str:
        """Add task-specific enhancements to prompt"""
        
        # Add context-aware enhancements
        if context and "lighting" in context:
            lighting = context["lighting"]
            if "direction" in lighting:
                prompt = prompt.rstrip(".") + f", maintaining the {lighting['direction']} lighting."
        
        # Task-specific additions
        if task_type == "style_artistic" and context:
            if "objects" in context and context["objects"]["main"]:
                main_subject = context["objects"]["main"][0]
                prompt = prompt.rstrip(".") + f". Ensure the {main_subject} remains the focal point."
        
        return prompt


# Example usage
if __name__ == "__main__":
    # Test the generator
    generator = PromptGenerator()
    
    # Test color change
    prompt1 = generator.generate(
        task_type="object_color",
        user_intent="make the car blue",
        image_analyses=[{
            "objects": {"main": ["red car", "street"]},
            "lighting": {"direction": "top-left sunlight"}
        }]
    )
    print("Color change prompt:")
    print(prompt1)
    print()
    
    # Test style transfer
    prompt2 = generator.generate(
        task_type="style_artistic",
        user_intent="impressionist style like monet",
        image_analyses=[{"objects": {"main": ["landscape", "lake"]}}]
    )
    print("Style transfer prompt:")
    print(prompt2)