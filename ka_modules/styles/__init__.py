"""
Kontext Assistant Style Library
Organized collection of all style presets
"""
import logging
import time
from typing import Dict, List, Optional

from .base import StylePreset, StyleCategory

# Import all style collections
from .cartoon_styles import CARTOON_STYLES
from .game_styles import GAME_STYLES
from .anime_manga_styles import ANIME_MANGA_STYLES
from .comic_styles import COMIC_STYLES
from .movie_styles import MOVIE_STYLES
from .traditional_art_styles import TRADITIONAL_ART_STYLES
from .digital_art_styles import DIGITAL_ART_STYLES
from .famous_artists_styles import FAMOUS_ARTISTS_STYLES
from .art_movements_styles import ART_MOVEMENTS_STYLES
from .cultural_styles import CULTURAL_STYLES
from .concept_art_styles import CONCEPT_ART_STYLES
from .photography_styles import PHOTOGRAPHY_STYLES
from .material_transform_styles import MATERIAL_TRANSFORM_STYLES
from .environment_transform_styles import ENVIRONMENT_TRANSFORM_STYLES

logger = logging.getLogger(__name__)

# Combine all styles into a single dictionary
ALL_STYLES = {}
ALL_STYLES.update(CARTOON_STYLES)
ALL_STYLES.update(GAME_STYLES)
ALL_STYLES.update(ANIME_MANGA_STYLES)
ALL_STYLES.update(COMIC_STYLES)
ALL_STYLES.update(MOVIE_STYLES)
ALL_STYLES.update(TRADITIONAL_ART_STYLES)
ALL_STYLES.update(DIGITAL_ART_STYLES)
ALL_STYLES.update(FAMOUS_ARTISTS_STYLES)
ALL_STYLES.update(ART_MOVEMENTS_STYLES)
ALL_STYLES.update(CULTURAL_STYLES)
ALL_STYLES.update(CONCEPT_ART_STYLES)
ALL_STYLES.update(PHOTOGRAPHY_STYLES)
ALL_STYLES.update(MATERIAL_TRANSFORM_STYLES)
ALL_STYLES.update(ENVIRONMENT_TRANSFORM_STYLES)


class StyleLibrary:
    """Main class for managing style presets - maintains compatibility with original API"""
    
    styles = ALL_STYLES
    
    def __init__(self):
        """Initialize style library with all available styles"""
        self.styles = ALL_STYLES
        self.style_mixers = self._initialize_mixers()
        self.quick_modifiers = self._initialize_modifiers()
        # Custom styles loading is handled by the main app
        logger.info(f"Initialized StyleLibrary with {len(self.styles)} styles")
        
    def _initialize_mixers(self) -> Dict:
        """Initialize style mixing helpers"""
        return {
            "traditional_digital": ["oil_painting", "digital_painting"],
            "east_west": ["anime_modern", "comic_book"],
            "retro_modern": ["pixel_art", "digital_art"],
            "fine_pop": ["renaissance", "pop_art"],
        }
    
    def _initialize_modifiers(self) -> Dict:
        """Initialize quick style modifiers"""
        return {
            "dramatic": "dramatic lighting, high contrast, moody atmosphere",
            "soft": "soft lighting, gentle colors, peaceful mood",
            "vibrant": "vibrant saturated colors, high energy, dynamic",
            "muted": "muted colors, subtle tones, understated",
            "detailed": "highly detailed, intricate, complex textures",
            "minimalist": "minimalist, simple, clean composition",
        }
    
    @classmethod
    def get_styles_by_category(cls, category: StyleCategory) -> List[StylePreset]:
        """Get all styles in a specific category"""
        return [v for k, v in cls.styles.items() if v.category == category]
    
    @classmethod
    def get_style(cls, style_id: str) -> Optional[StylePreset]:
        """Get a specific style by ID"""
        return cls.styles.get(style_id)
    
    @classmethod
    def get_all_style_ids(cls) -> List[str]:
        """Get list of all available style IDs"""
        return list(cls.styles.keys())
    
    @classmethod
    def get_categories(cls) -> List[StyleCategory]:
        """Get list of all categories that have styles"""
        return list(set(style.category for style in cls.styles.values()))
    
    def build_style_prompt(self, 
                          style_ids: List[str],
                          modifiers: Optional[Dict[str, str]] = None,
                          preserve_elements: Optional[List[str]] = None) -> str:
        """
        Build a complete style prompt from style IDs and modifiers
        
        Args:
            style_ids: List of style preset IDs to combine
            modifiers: Dictionary of modifier types and values
            preserve_elements: List of elements to preserve from original
            
        Returns:
            Complete style transformation prompt
        """
        if not style_ids:
            return ""
        
        # Get primary style
        primary_style = self.styles.get(style_ids[0])
        if not primary_style:
            logger.warning(f"Unknown style ID: {style_ids[0]} (available: {len(self.styles)} styles)")
            # Log the first few available style IDs for debugging
            available_ids = list(self.styles.keys())[:5]
            logger.debug(f"Available style IDs (first 5): {available_ids}")
            return ""
        
        # Start with primary style description
        prompt_parts = [primary_style.example_prompt]
        
        # Add additional styles if mixing
        if len(style_ids) > 1:
            for style_id in style_ids[1:]:
                style = self.styles.get(style_id)
                if style:
                    # Extract key elements from secondary styles
                    prompt_parts.append(f"with {style.name.lower()} influences")
        
        # Add modifiers
        if modifiers:
            for mod_type, mod_value in modifiers.items():
                if mod_value:
                    prompt_parts.append(mod_value)
        
        # Add preservation instructions
        if preserve_elements:
            preserve_text = "while maintaining " + ", ".join(preserve_elements)
            prompt_parts.append(preserve_text)
        
        return ", ".join(prompt_parts)
    
    def get_styles_by_category(self, category: StyleCategory) -> List[StylePreset]:
        """Get all styles in a category - instance method for compatibility"""
        return [v for k, v in self.styles.items() if v.category == category]
    
    def get_style(self, style_id: str) -> Optional[StylePreset]:
        """Get a specific style preset - instance method for compatibility"""
        return self.styles.get(style_id)
    
    def get_compatible_styles(self, style_id: str) -> List[StylePreset]:
        """Get styles that work well with the given style"""
        style = self.styles.get(style_id)
        if not style:
            return []
        
        compatible_styles = []
        for comp_id in style.compatible_with:
            comp_style = self.styles.get(comp_id)
            if comp_style:
                compatible_styles.append(comp_style)
        
        return compatible_styles
    
    def get_style_elements(self, style_id: str) -> Dict[str, List[str]]:
        """Get detailed elements of a style for UI display"""
        style = self.styles.get(style_id)
        if not style:
            return {}
        
        return {
            "visual": style.visual_elements,
            "color": style.color_characteristics,
            "technique": style.technique_details
        }
    
    def suggest_styles_for_content(self, content_type: str) -> List[str]:
        """Suggest appropriate styles based on content type"""
        suggestions = {
            "portrait": ["portrait_studio", "oil_painting", "anime_modern", "watercolor"],
            "landscape": ["oil_painting", "watercolor", "studio_ghibli", "cinematic"],
            "character": ["anime_modern", "genshin_impact", "digital_art", "pixel_art"],
            "urban": ["cyberpunk_game", "cinematic", "digital_art", "pixel_art"],
            "fantasy": ["studio_ghibli", "genshin_impact", "watercolor", "digital_art"]
        }
        
        return suggestions.get(content_type, ["digital_art", "oil_painting", "cinematic"])
    
    def save_custom_style(self, name: str, prompt: str, description: str = "") -> str:
        """Save a custom style prompt
        
        Args:
            name: Display name for the style
            prompt: The actual style prompt
            description: Optional description
            
        Returns:
            Style ID for the saved style
        """
        import json
        import os
        import time
        from pathlib import Path
        
        # Create custom styles directory if it doesn't exist
        extension_dir = Path(__file__).parent.parent.parent
        custom_styles_dir = extension_dir / "custom_styles"
        custom_styles_dir.mkdir(exist_ok=True)
        
        # Generate unique ID
        style_id = f"custom_{name.lower().replace(' ', '_')}"
        
        # Create style data
        style_data = {
            "id": style_id,
            "name": name,
            "prompt": prompt,
            "description": description,
            "created_at": time.time()
        }
        
        # Save to JSON file
        style_file = custom_styles_dir / f"{style_id}.json"
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(style_data, f, indent=2, ensure_ascii=False)
        
        # Create StylePreset and add to library
        custom_style = StylePreset(
            id=style_id,
            name=name,
            category=StyleCategory.CUSTOM,
            description=description or f"Custom style: {name}",
            visual_elements=[prompt],  # Store full prompt as single element
            color_characteristics=[],
            technique_details=[],
            example_prompt=prompt,
            compatible_with=[],
            tips=[f"This is a custom saved style"]
        )
        
        self.styles[style_id] = custom_style
        logger.info(f"Saved custom style: {name} (ID: {style_id})")
        
        return style_id
    
    def load_custom_styles(self):
        """Load all saved custom styles from disk"""
        import json
        import time
        from pathlib import Path
        
        extension_dir = Path(__file__).parent.parent.parent
        custom_styles_dir = extension_dir / "custom_styles"
        
        if not custom_styles_dir.exists():
            return
        
        loaded_count = 0
        for style_file in custom_styles_dir.glob("*.json"):
            try:
                with open(style_file, 'r', encoding='utf-8') as f:
                    style_data = json.load(f)
                
                # Create StylePreset from saved data
                custom_style = StylePreset(
                    id=style_data["id"],
                    name=style_data["name"],
                    category=StyleCategory.CUSTOM,
                    description=style_data.get("description", ""),
                    visual_elements=[style_data["prompt"]],
                    color_characteristics=[],
                    technique_details=[],
                    example_prompt=style_data["prompt"],
                    compatible_with=[],
                    tips=[f"Custom style created on {time.strftime('%Y-%m-%d', time.localtime(style_data.get('created_at', 0)))}"]
                )
                
                self.styles[style_data["id"]] = custom_style
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load custom style from {style_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} custom styles")
    
    def delete_custom_style(self, style_id: str) -> bool:
        """Delete a custom style
        
        Args:
            style_id: ID of the style to delete
            
        Returns:
            True if deleted, False if not found or not custom
        """
        style = self.styles.get(style_id)
        if not style or style.category != StyleCategory.CUSTOM:
            return False
        
        # Remove from memory
        del self.styles[style_id]
        
        # Delete file
        from pathlib import Path
        extension_dir = Path(__file__).parent.parent.parent
        style_file = extension_dir / "custom_styles" / f"{style_id}.json"
        
        if style_file.exists():
            style_file.unlink()
            logger.info(f"Deleted custom style: {style_id}")
            return True
        
        return False
    
    def get_custom_style_prompt(self, style_id: str) -> Optional[str]:
        """Get the raw prompt from a custom style"""
        style = self.styles.get(style_id)
        if style and style.category == StyleCategory.CUSTOM and style.visual_elements:
            return style.visual_elements[0]  # Custom styles store prompt as first element
        return None


# Export for backward compatibility
__all__ = ['StyleLibrary', 'StylePreset', 'StyleCategory', 'ALL_STYLES']