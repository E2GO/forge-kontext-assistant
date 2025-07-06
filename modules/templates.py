"""
Prompt templates for different FLUX.1 Kontext task types.

This module contains structured templates for generating instructional prompts
based on the task type and user intent. Each template includes preservation
rules to maintain important aspects of the original image.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Structure for a prompt template"""
    base: str
    preserve: List[str]
    context_rules: Optional[Dict[str, List[str]]] = None
    examples: Optional[List[str]] = None


class KontextTemplates:
    """Collection of templates for FLUX.1 Kontext prompt generation"""
    
    # Object manipulation templates
    OBJECT_COLOR = PromptTemplate(
        base="Change the {object} from {current_color} to {new_color}",
        preserve=[
            "exact same shape and form",
            "shadows and lighting",
            "reflections",
            "texture and material properties",
            "position and orientation",
            "surrounding elements"
        ],
        context_rules={
            "car": ["metallic reflections", "brand logos", "wheel design"],
            "clothing": ["fabric folds", "fit on person", "wrinkles and creases"],
            "furniture": ["wood grain", "cushion indentations", "structural details"],
            "nature": ["leaf patterns", "natural variations", "organic shapes"]
        }
    )
    
    OBJECT_STATE = PromptTemplate(
        base="Transform the {object} from {current_state} to {new_state}",
        preserve=[
            "object identity and recognizability",
            "position in scene",
            "relative scale",
            "surrounding context"
        ],
        examples=[
            "closed door to open door",
            "empty glass to full glass",
            "lights off to lights on"
        ]
    )
    
    ADD_OBJECT = PromptTemplate(
        base="Add {object} to the scene {location_description}",
        preserve=[
            "all existing elements",
            "original composition",
            "lighting consistency",
            "perspective and scale",
            "style and aesthetic"
        ]
    )
    
    REMOVE_OBJECT = PromptTemplate(
        base="Remove the {object} from the image",
        preserve=[
            "background continuity",
            "natural appearance where object was",
            "all other elements unchanged",
            "lighting and shadows of remaining objects"
        ]
    )
    
    # Style transfer templates
    STYLE_ARTISTIC = PromptTemplate(
        base="Transform the image into {art_style} style",
        preserve=[
            "scene composition",
            "recognizable objects and people",
            "spatial relationships",
            "main subject prominence"
        ],
        context_rules={
            "impressionist": ["with soft brushstrokes, vibrant colors, and emphasis on light"],
            "cubist": ["with geometric shapes, multiple perspectives, and fragmented forms"],
            "watercolor": ["with translucent washes, soft edges, and paper texture visible"],
            "oil_painting": ["with thick brushstrokes, rich colors, and canvas texture"],
            "pencil_sketch": ["with detailed line work, shading, and paper grain"]
        }
    )
    
    STYLE_TEMPORAL = PromptTemplate(
        base="Convert the image to {time_period} aesthetic",
        preserve=[
            "people's identities and poses",
            "scene layout",
            "main actions or interactions"
        ],
        context_rules={
            "vintage": ["with sepia tones, film grain, aged appearance, and period-appropriate details"],
            "retro_80s": ["with neon colors, VHS artifacts, synthwave elements"],
            "futuristic": ["with holographic elements, sleek surfaces, advanced technology"],
            "noir": ["with high contrast black and white, dramatic shadows, 1940s atmosphere"]
        }
    )
    
    # Environment modification templates
    ENVIRONMENT_WEATHER = PromptTemplate(
        base="Change the weather to {weather_condition}",
        preserve=[
            "all people and objects",
            "scene composition",
            "architectural elements",
            "relative positions"
        ],
        context_rules={
            "rainy": ["add rain drops, wet surfaces, reflections, cloudy sky"],
            "snowy": ["add falling snow, snow accumulation, winter atmosphere"],
            "sunny": ["bright lighting, clear blue sky, defined shadows"],
            "foggy": ["reduced visibility, soft lighting, atmospheric haze"]
        }
    )
    
    ENVIRONMENT_TIME = PromptTemplate(
        base="Change the time of day to {time}",
        preserve=[
            "all objects and people",
            "scene content",
            "positions and poses"
        ],
        context_rules={
            "sunset": ["golden hour lighting, long shadows, warm orange sky"],
            "night": ["dark sky, artificial lighting, stars or moon visible"],
            "dawn": ["soft morning light, pink/purple sky, early morning atmosphere"]
        }
    )
    
    ENVIRONMENT_LOCATION = PromptTemplate(
        base="Change the background to {location}",
        preserve=[
            "all foreground subjects exactly as they are",
            "poses and expressions",
            "clothing and accessories",
            "relative positions",
            "lighting on subjects"
        ],
        examples=[
            "beach with ocean waves",
            "mountain landscape", 
            "urban cityscape",
            "indoor office setting"
        ]
    )
    
    # Complex templates
    OUTPAINTING = PromptTemplate(
        base="Extend the image {direction} with {description}",
        preserve=[
            "existing image content unchanged",
            "consistent style and quality",
            "natural continuation of perspective",
            "matching lighting and colors"
        ]
    )


class TemplateBuilder:
    """Builds complete prompts from templates and parameters"""
    
    @staticmethod
    def build_preservation_clause(preserve_list: List[str]) -> str:
        """Build the preservation part of the prompt"""
        if not preserve_list:
            return ""
        
        # Create natural language list
        if len(preserve_list) == 1:
            return f" while maintaining {preserve_list[0]}"
        elif len(preserve_list) == 2:
            return f" while maintaining {preserve_list[0]} and {preserve_list[1]}"
        else:
            items = ", ".join(preserve_list[:-1])
            return f" while maintaining {items}, and {preserve_list[-1]}"
    
    @staticmethod
    def fill_template(template: PromptTemplate, params: Dict[str, str], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Fill a template with parameters and build complete prompt.
        
        Args:
            template: The prompt template to use
            params: Dictionary of parameters to fill in template
            context: Optional context from image analysis
            
        Returns:
            Complete instructional prompt
        """
        # Fill base template
        base_prompt = template.base.format(**params)
        
        # Add context-specific details if available
        if template.context_rules and context:
            for key, value in params.items():
                if value in template.context_rules:
                    context_detail = template.context_rules[value]
                    if isinstance(context_detail, list):
                        context_detail = context_detail[0]
                    base_prompt += f" {context_detail}"
        
        # Add preservation clause
        preservation = TemplateBuilder.build_preservation_clause(template.preserve)
        
        # Combine into final prompt
        final_prompt = base_prompt + preservation
        
        # Add specific preservation based on detected objects
        if context and "objects" in context:
            main_objects = context["objects"].get("main", [])
            if main_objects:
                final_prompt += f". Keep the {', '.join(main_objects[:2])} exactly the same except for the requested changes"
        
        return final_prompt + "."


# Convenience function to get template by task type
def get_template(task_type: str) -> Optional[PromptTemplate]:
    """Get template by task type string"""
    template_map = {
        "object_color": KontextTemplates.OBJECT_COLOR,
        "object_state": KontextTemplates.OBJECT_STATE,
        "add_object": KontextTemplates.ADD_OBJECT,
        "remove_object": KontextTemplates.REMOVE_OBJECT,
        "style_artistic": KontextTemplates.STYLE_ARTISTIC,
        "style_temporal": KontextTemplates.STYLE_TEMPORAL,
        "environment_weather": KontextTemplates.ENVIRONMENT_WEATHER,
        "environment_time": KontextTemplates.ENVIRONMENT_TIME,
        "environment_location": KontextTemplates.ENVIRONMENT_LOCATION,
        "outpainting": KontextTemplates.OUTPAINTING
    }
    
    return template_map.get(task_type)


# Example usage for testing
if __name__ == "__main__":
    # Test object color change
    template = KontextTemplates.OBJECT_COLOR
    params = {
        "object": "car",
        "current_color": "red",
        "new_color": "blue"
    }
    context = {
        "objects": {
            "main": ["car", "road"],
            "secondary": ["trees", "buildings"]
        }
    }
    
    prompt = TemplateBuilder.fill_template(template, params, context)
    print("Generated prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Test style transfer
    style_template = KontextTemplates.STYLE_ARTISTIC
    style_params = {"art_style": "impressionist"}
    style_prompt = TemplateBuilder.fill_template(style_template, style_params)
    print("Style transfer prompt:")
    print(style_prompt)