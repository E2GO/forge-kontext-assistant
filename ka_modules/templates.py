"""
Prompt templates for FLUX.1 Kontext
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

class PromptTemplates:
    """Manages prompt templates for different task types"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
        self.task_configs = self._load_task_configs()
        
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default prompt templates"""
        return {
            # Object manipulation templates
            "object_color": "Change the {object} from {current_color} to {new_color} while maintaining all other properties",
            "object_state": "Transform the {object} from {current_state} to {new_state}, preserving its identity",
            "object_removal": "Remove the {object} from the scene and naturally fill the space",
            "object_addition": "Add a {new_object} to the scene at {position}, matching the lighting and style",
            
            # Style transfer templates
            "style_artistic": "Convert the entire image to {style} style while preserving the scene composition",
            "style_temporal": "Transform the scene to appear as if from {time_period}, maintaining all subjects",
            "style_cultural": "Apply {culture} aesthetic to the image while keeping the core elements",
            
            # Environment templates
            "env_location": "Change the background setting to {new_location} while keeping all foreground elements identical",
            "env_weather": "Modify the weather to {weather_condition}, adjusting lighting naturally",
            "env_time": "Change the time of day to {time}, updating shadows and lighting accordingly",
            
            # Lighting templates
            "light_direction": "Adjust the {quality} lighting to come from the {direction} side",
            "light_quality": "Change the lighting quality to {quality}",
            "light_color": "Shift the color temperature to {temperature} lighting",
            
            # State change templates
            "state_age": "Make the {object} look {new_state}, showing natural aging effects",
            "state_damage": "Show the {object} as {damage_type}",
            "state_seasonal": "Transform the scene to {season} season",
            
            # Outpainting templates
            "outpaint_extend": "Extend the image {direction} by naturally continuing the scene",
            "outpaint_zoom": "Zoom out to reveal more of the surrounding environment"
        }
    
    def _load_task_configs(self) -> Dict[str, Any]:
        """Load task configurations"""
        # Try to load from file, fallback to defaults
        config_path = Path(__file__).parent.parent / "configs" / "task_configs.json"
        
        default_configs = {
            "object_manipulation": {
                "subtypes": ["color", "state", "removal", "addition"],
                "requires_object": True,
                "preservation_rules": ["shadows", "reflections", "lighting", "composition"]
            },
            "style_transfer": {
                "subtypes": ["artistic", "temporal", "cultural"],
                "requires_style": True,
                "preservation_rules": ["subject_identity", "scene_layout", "object_positions"]
            },
            "environment_change": {
                "subtypes": ["location", "weather", "time"],
                "requires_environment": True,
                "preservation_rules": ["foreground_objects", "subject_poses", "object_relationships"]
            },
            "lighting_adjustment": {
                "subtypes": ["direction", "quality", "color"],
                "requires_analysis": True,
                "preservation_rules": ["object_colors", "scene_mood", "material_properties"]
            },
            "state_change": {
                "subtypes": ["age", "damage", "seasonal"],
                "requires_state": True,
                "preservation_rules": ["object_identity", "scene_context", "spatial_relationships"]
            },
            "outpainting": {
                "subtypes": ["extend", "zoom"],
                "requires_direction": True,
                "preservation_rules": ["style_consistency", "perspective", "lighting_continuity"]
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f).get("tasks", default_configs)
            except:
                pass
                
        return default_configs
    
    def get_template(self, task_type: str, subtype: Optional[str] = None) -> str:
        """Get template for specific task with validation"""
        # Validate task_type
        if task_type not in self.task_configs:
            logger.warning(f"Unknown task type: {task_type}, using default")
            task_type = "object_manipulation"
        
        # Map task types to template keys
        template_map = {
            "object_manipulation": {
                "color": "object_color",
                "state": "object_state",
                "removal": "object_removal",
                "addition": "object_addition"
            },
            "style_transfer": {
                "artistic": "style_artistic",
                "temporal": "style_temporal",
                "cultural": "style_cultural"
            },
            "environment_change": {
                "location": "env_location",
                "weather": "env_weather",
                "time": "env_time"
            },
            "lighting_adjustment": {
                "direction": "light_direction",
                "quality": "light_quality",
                "color": "light_color"
            },
            "state_change": {
                "age": "state_age",
                "damage": "state_damage",
                "seasonal": "state_seasonal"
            },
            "outpainting": {
                "extend": "outpaint_extend",
                "zoom": "outpaint_zoom"
            }
        }
        
        if task_type in template_map:
            if subtype and subtype in template_map[task_type]:
                template_key = template_map[task_type][subtype]
                return self.templates.get(template_key, self.templates["object_color"])
            else:
                # Return first template for the task type
                first_subtype = list(template_map[task_type].keys())[0]
                template_key = template_map[task_type][first_subtype]
                return self.templates.get(template_key, self.templates["object_color"])
        
        return self.templates["object_color"]  # Default fallback
    
    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """Get configuration for task type"""
        return self.task_configs.get(task_type, {})
    
    def get_preservation_rules(self, task_type: str) -> List[str]:
        """Get preservation rules for task type"""
        config = self.get_task_config(task_type)
        return config.get("preservation_rules", ["lighting", "shadows", "composition"])
    
    def get_all_task_types(self) -> List[str]:
        """Get list of all available task types"""
        return list(self.task_configs.keys())
    
    def format_preservation_clause(self, rules: List[str]) -> str:
        """Format preservation rules into a clause"""
        if not rules:
            return ""
        
        rule_descriptions = {
            "shadows": "shadows and reflections",
            "reflections": "reflections and highlights",
            "lighting": "lighting conditions and color temperature",
            "composition": "overall composition and framing",
            "subject_identity": "exact facial features and identifying characteristics",
            "scene_layout": "spatial arrangement and perspective",
            "object_positions": "positions and orientations of all objects",
            "foreground_objects": "all foreground elements unchanged",
            "subject_poses": "exact poses and expressions",
            "object_relationships": "relative positions between objects",
            "object_colors": "original colors under new lighting",
            "scene_mood": "emotional atmosphere of the scene",
            "material_properties": "textures and material appearances",
            "object_identity": "recognizable features of the object",
            "scene_context": "surrounding environment and context",
            "spatial_relationships": "distances and proportions",
            "style_consistency": "artistic style and rendering",
            "perspective": "viewing angle and depth",
            "lighting_continuity": "consistent light sources and shadows"
        }
        
        preserved_items = []
        for rule in rules:
            if rule in rule_descriptions:
                preserved_items.append(rule_descriptions[rule])
            else:
                preserved_items.append(rule.replace("_", " "))
        
        if len(preserved_items) == 1:
            return f"Preserve {preserved_items[0]}."
        elif len(preserved_items) == 2:
            return f"Preserve {preserved_items[0]} and {preserved_items[1]}."
        else:
            return f"Preserve {', '.join(preserved_items[:-1])}, and {preserved_items[-1]}."