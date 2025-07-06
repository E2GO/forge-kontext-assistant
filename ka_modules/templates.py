"""
Enhanced prompt templates for FLUX.1 Kontext with new task types.
Version 2.0 with expanded template library.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import logging

# Compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

logger = logging.getLogger("KontextAssistant.Templates")


class PromptTemplates:
    """Manages instructional prompt templates for various tasks."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional custom config path."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "task_configs.json"
        
        self.config_path = config_path
        self.templates = self._load_base_templates()
        self.custom_templates = {}
        
        # Load from config if exists
        if config_path.exists():
            self._load_config_templates()
        else:
            logger.warning(f"Config file not found: {config_path}")
    
    def _load_base_templates(self) -> Dict[str, Dict[str, str]]:
        """Load hardcoded base templates."""
        return {
            # Object Manipulation Templates
            "object_manipulation": {
                "default": "Change {target} to {description}. Preserve all other elements exactly.",
                "color_change": "Change the {object} color from {current_color} to {target_color}. Maintain exact shadows, reflections, and material properties.",
                "add_element": "Add {element} to the {location} of the image. Ensure proper lighting, shadows, and perspective integration.",
                "remove_element": "Remove the {object} completely from the scene. Fill the area naturally with appropriate background.",
                "modify_attribute": "Change the {object}'s {attribute} to {new_value}. Keep all other properties unchanged.",
                "resize": "Resize the {object} to be {size}. Maintain proportions and adjust shadows accordingly.",
                "reposition": "Move the {object} to {new_position}. Update shadows and reflections for the new position."
            },
            
            # Style Transfer Templates
            "style_transfer": {
                "default": "Transform into {style} style. Maintain composition and recognizable elements.",
                "artistic_style": "Convert the entire image to {style} artistic style. Preserve all subjects, their positions, and the scene composition while applying the artistic treatment.",
                "time_period": "Transform the scene to appear as if from {era}. Update clothing, technology, and atmosphere while keeping people and locations recognizable.",
                "medium_change": "Render the image as if created with {medium}. Maintain all content while applying the characteristics of the chosen medium.",
                "mood_change": "Adjust the overall mood to be {mood} through color grading, lighting, and atmospheric effects. Keep all objects and people unchanged.",
                "genre_change": "Transform into {genre} genre style while preserving the core scene elements."
            },
            
            # Environment Change Templates
            "environment_change": {
                "default": "Change the environment to {setting}. Keep all subjects in exact positions.",
                "location": "Transport everything to {new_location}. Maintain all subject poses, expressions, and relative positions perfectly.",
                "weather": "Change weather conditions to {weather_condition}. Apply appropriate atmospheric effects while preserving all structures and objects.",
                "time_of_day": "Shift the time to {time}. Adjust lighting, shadows, and sky naturally while keeping all objects unchanged.",
                "season": "Transform the scene to {season}. Change foliage, weather, and seasonal elements while preserving all permanent features.",
                "indoor_outdoor": "Move the scene from {from_setting} to {to_setting}. Adapt lighting and background appropriately."
            },
            
            # Element Combination Templates
            "element_combination": {
                "default": "Combine {element1} with {element2} seamlessly.",
                "merge_scenes": "Blend {scene1} and {scene2} into a cohesive composition. Create smooth transitions between elements.",
                "composite_objects": "Merge {object1} and {object2} into a single unified object. Combine their features naturally.",
                "blend_styles": "Apply both {style1} and {style2} throughout the image. Create a harmonious blend of both artistic styles.",
                "fusion": "Fuse {elements} together while maintaining recognizable features of each."
            },
            
            # State Change Templates
            "state_change": {
                "default": "Transform {object} from {current_state} to {target_state}.",
                "age_progression": "Age {subject} by {years} years naturally. Show realistic aging while maintaining identity and setting.",
                "damage_repair": "Change {object} from {damage_state} condition to {repaired_state}. Show realistic transformation.",
                "transformation": "Transform {source} into {target} while maintaining the same spatial position and scale.",
                "material_state": "Change {object} from {material1} to {material2}. Update texture and reflective properties.",
                "decay_growth": "Show {object} in {state} state. Apply natural progression effects."
            },
            
            # Outpainting Templates
            "outpainting": {
                "default": "Extend the image {direction}. Continue the scene naturally.",
                "extend_scene": "Expand the canvas {direction} by {amount}. Continue existing patterns, textures, and scene elements naturally.",
                "add_context": "Reveal more of the {context_type} around the current view. Maintain consistent style and lighting.",
                "expand_canvas": "Enlarge the image {direction} and fill with contextually appropriate content that matches the existing scene.",
                "panoramic": "Extend into a panoramic view showing more of the surrounding environment."
            },
            
            # Lighting Adjustment Templates
            "lighting_adjustment": {
                "default": "Adjust lighting to {description}. Maintain object colors and positions.",
                "direction_change": "Change the primary light source to come from {direction}. Update all shadows and highlights accordingly.",
                "intensity_change": "Make the lighting {intensity}. Adjust exposure and contrast throughout the image.",
                "add_light_source": "Add {light_type} lighting from {position}. Blend naturally with existing illumination.",
                "color_temperature": "Shift color temperature to {temperature}. Apply consistent color grading.",
                "dramatic_lighting": "Create {mood} dramatic lighting. Enhance contrast and directional shadows."
            },
            
            # Texture Change Templates
            "texture_change": {
                "default": "Change {object} texture to {new_texture}.",
                "material_swap": "Replace {object}'s {current_material} material with {new_material}. Update reflections and surface properties.",
                "surface_treatment": "Apply {treatment} to {surface}. Show realistic material transformation.",
                "pattern_application": "Add {pattern} pattern to {object}. Wrap naturally following object contours.",
                "finish_change": "Change surface finish from {current_finish} to {new_finish}."
            },
            
            # Perspective Shift Templates
            "perspective_shift": {
                "default": "Change viewpoint to {angle}. Show the same scene from a different perspective.",
                "angle_change": "Shift camera angle to {angle} view. Maintain all scene elements while changing perspective.",
                "zoom_adjust": "Adjust to {zoom_level} view. {zoom_direction} while keeping the main subject in frame.",
                "rotation": "Rotate view {degrees} degrees {direction}. Show the scene from the rotated perspective.",
                "focal_shift": "Shift focus to {new_focus}. Apply appropriate depth of field effects."
            },
            
            # Seasonal Change Templates (Extended)
            "seasonal_change": {
                "default": "Transform scene to {season}. Update all seasonal elements.",
                "spring": "Convert to spring season. Add blooming flowers, fresh green foliage, and bright atmosphere.",
                "summer": "Transform to summer. Show lush greenery, bright sunlight, and clear skies.",
                "autumn": "Change to autumn/fall. Add colorful falling leaves, golden light, and seasonal atmosphere.",
                "winter": "Convert to winter. Add snow, frost, bare trees, and cold atmosphere where appropriate."
            }
        }
    
    def _load_config_templates(self) -> None:
        """Load templates from configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract templates from config
            for task_type, task_data in config.get('task_types', {}).items():
                if task_type not in self.templates:
                    self.templates[task_type] = {}
                
                for subtype, subtype_data in task_data.get('subtypes', {}).items():
                    if 'template' in subtype_data:
                        self.templates[task_type][subtype] = subtype_data['template']
            
            logger.info(f"Loaded templates from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config templates: {e}")
    
    def get_template(self, task_type: str, subtype: str = "default") -> Optional[str]:
        """
        Get a specific template.
        
        Args:
            task_type: Main task category
            subtype: Specific subtype (default: "default")
            
        Returns:
            Template string or None if not found
        """
        # Check custom templates first
        custom_key = f"{task_type}.{subtype}"
        if custom_key in self.custom_templates:
            return self.custom_templates[custom_key]
        
        # Then check loaded templates
        if task_type in self.templates:
            if subtype in self.templates[task_type]:
                return self.templates[task_type][subtype]
            elif "default" in self.templates[task_type]:
                return self.templates[task_type]["default"]
        
        # Ultimate fallback
        return None
    
    def add_custom_template(self, task_type: str, subtype: str, template: str) -> None:
        """Add a custom template."""
        key = f"{task_type}.{subtype}"
        self.custom_templates[key] = template
        logger.info(f"Added custom template: {key}")
    
    def get_task_types(self) -> List[str]:
        """Get list of available task types."""
        return list(self.templates.keys())
    
    def get_subtypes(self, task_type: str) -> List[str]:
        """Get subtypes for a task type."""
        if task_type in self.templates:
            return list(self.templates[task_type].keys())
        return []
    
    def get_all_templates(self) -> Dict[str, Dict[str, str]]:
        """Get all templates including custom ones."""
        all_templates = self.templates.copy()
        
        # Add custom templates
        for key, template in self.custom_templates.items():
            task_type, subtype = key.split('.', 1)
            if task_type not in all_templates:
                all_templates[task_type] = {}
            all_templates[task_type][subtype] = template
        
        return all_templates
    
    def get_template_variables(self, template: str) -> List[str]:
        """Extract variable names from a template."""
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def validate_template(self, template: str) -> bool:
        """Check if template is valid."""
        try:
            # Check for balanced braces
            open_braces = template.count('{')
            close_braces = template.count('}')
            if open_braces != close_braces:
                return False
            
            # Check for valid variable names
            variables = self.get_template_variables(template)
            for var in variables:
                if not var.isidentifier():
                    return False
            
            return True
        except:
            return False
    
    def save_custom_templates(self, filepath: Path) -> None:
        """Save custom templates to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.custom_templates, f, indent=2)
            logger.info(f"Saved custom templates to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save custom templates: {e}")
    
    def load_custom_templates(self, filepath: Path) -> None:
        """Load custom templates from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.custom_templates = json.load(f)
            logger.info(f"Loaded custom templates from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load custom templates: {e}")