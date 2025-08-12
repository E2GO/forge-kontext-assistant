"""
Base classes and enums for the style system
"""
from dataclasses import dataclass
from enum import Enum
from typing import List


class StyleCategory(Enum):
    """Categories of styles"""
    # Traditional Art Mediums
    TRADITIONAL_ART = "traditional_art"  # Oil, watercolor, gouache, charcoal, etc.
    DIGITAL_ART = "digital_art"  # Digital painting, 3D render, vector art, etc.
    
    # Entertainment Media
    GAME = "game"  # Video game styles
    ANIME_MANGA = "anime_manga"  # Japanese animation and manga
    CARTOON = "cartoon"  # Western animation
    COMIC = "comic"  # Comic books and graphic novels
    MOVIE = "movie"  # Cinematic and film styles
    
    # Professional Art
    CONCEPT_ART = "concept_art"  # Character, environment, vehicle concepts
    FAMOUS_ARTISTS = "famous_artists"  # Styles of renowned artists
    
    # Historical and Cultural
    ART_MOVEMENTS = "art_movements"  # Impressionism, Art Nouveau, etc.
    CULTURAL = "cultural"  # Cultural and regional styles
    
    # Transformations
    MATERIAL_TRANSFORM = "material_transform"  # Transform to different materials
    ENVIRONMENT_TRANSFORM = "environment_transform"  # Season, time, setting changes
    
    # Photography and Effects
    PHOTOGRAPHY = "photography"  # Photography styles and techniques
    
    # Reference-based
    FROM_REFERENCE = "from_reference"  # Use loaded image as style reference
    
    # User Created
    CUSTOM = "custom"  # User saved styles


@dataclass 
class StylePreset:
    """Predefined style with all necessary information"""
    id: str
    name: str
    category: StyleCategory
    description: str
    visual_elements: List[str]
    color_characteristics: List[str]
    technique_details: List[str]
    example_prompt: str
    compatible_with: List[str]  # Other style IDs that mix well
    tips: List[str]