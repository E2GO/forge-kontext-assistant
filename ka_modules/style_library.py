"""
Style Library for Kontext Assistant
Contains predefined styles, templates, and style-related utilities
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


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


class StyleLibrary:
    """Main class for managing style presets and generation"""
    
    def __init__(self):
        self.styles = self._initialize_styles()
        self.style_mixers = self._initialize_mixers()
        self.quick_modifiers = self._initialize_modifiers()
        # Load any saved custom styles
        self.load_custom_styles()
        
    def _initialize_styles(self) -> Dict[str, StylePreset]:
        """Initialize the style preset library"""
        styles = {
            # Artistic Styles
            "oil_painting": StylePreset(
                id="oil_painting",
                name="Oil Painting",
                category=StyleCategory.TRADITIONAL_ART,
                description="Classic oil painting with visible brushstrokes",
                visual_elements=["thick brushstrokes", "textured canvas", "paint buildup", "rich impasto"],
                color_characteristics=["deep saturated colors", "subtle color mixing", "warm undertones"],
                technique_details=["layered paint application", "visible brush texture", "traditional painting techniques"],
                example_prompt="convert to traditional oil painting with thick impasto brushstrokes, rich saturated colors, and visible canvas texture",
                compatible_with=["impressionist", "renaissance", "portrait"],
                tips=["Works best with portraits and landscapes", "Specify brushstroke size for different effects"]
            ),
            
            "watercolor": StylePreset(
                id="watercolor",
                name="Watercolor",
                category=StyleCategory.TRADITIONAL_ART,
                description="Soft watercolor painting with translucent washes",
                visual_elements=["soft edges", "paint bleeds", "paper texture", "translucent layers"],
                color_characteristics=["muted colors", "gentle gradients", "white paper showing through"],
                technique_details=["wet-on-wet technique", "color bleeding", "spontaneous flows"],
                example_prompt="convert to watercolor artwork with translucent washes, soft color bleeds, and visible paper texture",
                compatible_with=["sketch", "minimalist", "botanical"],
                tips=["Great for dreamy, soft atmospheres", "Mention paper texture for authenticity"]
            ),
            
            "gouache_illustration": StylePreset(
                id="gouache_illustration",
                name="Gouache Illustration",
                category=StyleCategory.TRADITIONAL_ART,
                description="Opaque water-based paint with matte finish popular in illustration",
                visual_elements=["flat color areas", "matte surface", "crisp edges", "layered opacity"],
                color_characteristics=["vibrant opaque colors", "no transparency", "uniform coverage", "matte finish"],
                technique_details=["opaque layering", "flat color blocks", "precise edges", "minimal blending"],
                example_prompt="convert to gouache illustration style with opaque flat colors, matte finish, crisp edges between color areas, vibrant pigments with uniform coverage, traditional illustration technique with layered opacity",
                compatible_with=["editorial_illustration", "children_book", "poster_art"],
                tips=["Keep colors flat and opaque", "Use precise edges", "Layer from dark to light"]
            ),
            
            "risograph_print": StylePreset(
                id="risograph_print",
                name="Risograph Print Style",
                category=StyleCategory.DIGITAL_ART,
                description="Trendy duplicator printing aesthetic with limited colors and grain",
                visual_elements=["color separation", "grain texture", "misregistration", "limited palette"],
                color_characteristics=["fluorescent inks", "spot colors", "overlapping transparencies", "vibrant limited palette"],
                technique_details=["screen printing effect", "halftone patterns", "intentional imperfection", "layer offset"],
                example_prompt="convert to risograph print style with limited fluorescent color palette, visible grain texture, slight misregistration between color layers, halftone dot patterns, trendy duplicator printing aesthetic with intentional imperfections",
                compatible_with=["zine_art", "indie_poster", "screen_print"],
                tips=["Use 2-4 colors max", "Embrace misregistration", "Add grain texture"]
            ),
            
            "cyanotype": StylePreset(
                id="cyanotype",
                name="Cyanotype Photography",
                category=StyleCategory.PHOTOGRAPHY,
                description="Historic photographic printing process with distinctive Prussian blue tones",
                visual_elements=["blueprint aesthetic", "botanical silhouettes", "sun print quality", "negative space"],
                color_characteristics=["Prussian blue monochrome", "white highlights", "cyan gradients", "deep blue shadows"],
                technique_details=["photogram process", "contact printing", "UV exposure", "chemical development"],
                example_prompt="convert to cyanotype photography style, Prussian blue monochromatic print, sun-exposed photogram aesthetic, botanical blueprint quality, white silhouettes on deep cyan background, historic alternative photography process",
                compatible_with=["botanical_art", "vintage_photo", "blueprint"],
                tips=["Use only blue tones", "High contrast works best", "Include botanical elements"]
            ),
            
            "brutalist_graphic": StylePreset(
                id="brutalist_graphic",
                name="Brutalist Graphic Design",
                category=StyleCategory.DIGITAL_ART,
                description="Raw, unpolished graphic design aesthetic with harsh typography",
                visual_elements=["raw textures", "harsh typography", "concrete aesthetics", "anti-design elements"],
                color_characteristics=["monochromatic", "concrete gray", "warning colors", "high contrast"],
                technique_details=["distressed textures", "overlapping elements", "broken grid", "digital artifacts"],
                example_prompt="convert to brutalist graphic design style, raw concrete textures, harsh bold typography, anti-design aesthetic with intentional ugliness, monochromatic palette with warning color accents, distressed digital artifacts, broken grid layout",
                compatible_with=["constructivism", "punk_zine", "industrial"],
                tips=["Embrace ugliness", "Use harsh contrasts", "Break design rules"]
            ),
            
            "vaporwave_aesthetic": StylePreset(
                id="vaporwave_aesthetic",
                name="Vaporwave Aesthetic",
                category=StyleCategory.DIGITAL_ART,
                description="Nostalgic 80s/90s internet art with surreal corporate imagery",
                visual_elements=["Greek statues", "palm trees", "corporate logos", "glitch effects"],
                color_characteristics=["pink and purple gradients", "cyan highlights", "sunset colors", "neon glow"],
                technique_details=["digital collage", "VHS distortion", "3D primitive shapes", "Windows 95 aesthetic"],
                example_prompt="convert to vaporwave aesthetic, pink and purple gradient background, Greek statue with glitch effects, palm trees silhouettes, VHS scan lines, Windows 95 interface elements, nostalgic digital collage, neon sunset colors, surreal corporate imagery",
                compatible_with=["synthwave", "retrofuturism", "glitch_art"],
                tips=["Add Greek statues", "Use pink/purple/cyan", "Include retro tech"]
            ),
            
            "digital_art": StylePreset(
                id="digital_art",
                name="Digital Art",
                category=StyleCategory.DIGITAL_ART,
                description="Professional digital 2D illustration and concept art",
                visual_elements=["clean lines", "smooth gradients", "perfect shapes", "digital brushes"],
                color_characteristics=["vibrant colors", "high contrast", "perfect color transitions"],
                technique_details=["digital painting", "vector-like elements", "layer effects"],
                example_prompt="convert to professional digital illustration with clean vector-like lines, vibrant colors, and polished rendering",
                compatible_with=["anime", "concept_art", "cyberpunk"],
                tips=["Specify software style (Photoshop, Procreate)", "Works well with fantasy subjects"]
            ),
            
            # Photographic Styles
            "cinematic": StylePreset(
                id="cinematic",
                name="Cinematic",
                category=StyleCategory.PHOTOGRAPHY,
                description="Movie-like cinematography with dramatic lighting",
                visual_elements=["wide aspect ratio", "depth of field", "lens flares", "atmospheric haze"],
                color_characteristics=["color grading", "teal and orange", "desaturated tones"],
                technique_details=["anamorphic lens", "dramatic lighting", "film grain"],
                example_prompt="make as cinematic shot with dramatic lighting, shallow depth of field, anamorphic lens characteristics, and film color grading",
                compatible_with=["noir", "documentary", "vintage"],
                tips=["Add specific film references for accuracy", "Mention lighting setup"]
            ),
            
            "portrait_studio": StylePreset(
                id="portrait_studio",
                name="Studio Portrait",
                category=StyleCategory.PHOTOGRAPHY,
                description="Professional studio portrait photography",
                visual_elements=["soft lighting", "clean background", "sharp focus", "catchlights"],
                color_characteristics=["natural skin tones", "balanced exposure", "subtle highlights"],
                technique_details=["three-point lighting", "beauty dish", "professional retouching"],
                example_prompt="make as professional studio portrait with three-point lighting, seamless backdrop, natural skin tones, and sharp focus",
                compatible_with=["fashion", "commercial", "headshot"],
                tips=["Specify lighting setup for best results", "Mention background color/type"]
            ),
            
            # Game Styles
            "cyberpunk_game": StylePreset(
                id="cyberpunk_game",
                name="Cyberpunk 2077 Style",
                category=StyleCategory.GAME,
                description="Neon-lit futuristic game aesthetic",
                visual_elements=["neon lights", "holographic displays", "urban decay", "tech implants"],
                color_characteristics=["neon pink and blue", "high contrast", "glowing elements"],
                technique_details=["ray-traced reflections", "volumetric fog", "chromatic aberration"],
                example_prompt="convert to Cyberpunk 2077 3D game style with neon lights, futuristic technology, urban dystopian atmosphere, and ray-traced reflections",
                compatible_with=["synthwave", "noir", "scifi"],
                tips=["Great for night scenes", "Add specific tech elements"]
            ),
            
            "genshin_impact": StylePreset(
                id="genshin_impact",
                name="Genshin Impact Style",
                category=StyleCategory.GAME,
                description="Anime-inspired cel-shaded game art",
                visual_elements=["cel shading", "anime features", "fantasy elements", "particle effects"],
                color_characteristics=["bright saturated colors", "gradient skies", "soft shadows"],
                technique_details=["toon shading", "outline rendering", "stylized proportions"],
                example_prompt="convert to Genshin Impact 3D cel-shaded anime game style with stylized characters, vibrant fantasy colors, and painterly environments",
                compatible_with=["anime", "fantasy", "breath_of_wild"],
                tips=["Works best with character art", "Specify element type for effects"]
            ),
            
            "pixel_art": StylePreset(
                id="pixel_art",
                name="Pixel Art",
                category=StyleCategory.GAME,
                description="Retro pixel art game style",
                visual_elements=["visible pixels", "limited palette", "dithering", "blocky shapes"],
                color_characteristics=["restricted color palette", "flat colors", "no gradients"],
                technique_details=["8-bit or 16-bit style", "sprite-like", "tile-based"],
                example_prompt="convert to pixel art with clearly defined pixels, restricted color palette, and 8-bit or 16-bit retro aesthetic",
                compatible_with=["retro", "8bit", "arcade"],
                tips=["Specify bit level (8-bit, 16-bit)", "Works best with simple subjects"]
            ),
            
            # Casual Mobile Game Styles
            "mobile_game_style": StylePreset(
                id="mobile_game_style",
                name="Mobile Game 2D Style",
                category=StyleCategory.GAME,
                description="Professional mobile game 2D illustration style with vibrant colors",
                visual_elements=["exaggerated cartoon proportions", "oversized heads", "expressive eyes", "rounded character designs", "simplified anatomy", "theatrical staging", "dynamic poses", "whimsical fantasy elements"],
                color_characteristics=["vibrant saturated palette", "purple/pink/turquoise dominant", "rich color gradients", "soft ambient lighting", "subtle rim lights", "atmospheric perspective"],
                technique_details=["smooth gradient shading", "vector-like rendering", "clean edges", "glossy surfaces", "specular highlights", "plastic-like textures", "simplified fabric folds", "chunky sculptural hair"],
                example_prompt="convert to professional mobile game 2D illustration style, vibrant digital painting with smooth gradient shading, exaggerated cartoon proportions with oversized heads and expressive eyes, rich saturated color palette with purple/pink/turquoise dominant hues, soft ambient lighting with subtle rim lights, polished vector-like rendering with clean edges, friendly rounded character designs with simplified anatomy, theatrical staging and dynamic poses, glossy surfaces with subtle specular highlights, whimsical fantasy elements, professional quality",
                compatible_with=["candy_crush", "royal_match", "casual_game"],
                tips=["Focus on exaggerated proportions", "Use purple/pink/turquoise color scheme", "Keep shapes rounded and friendly", "Add theatrical lighting"]
            ),
            
            "character_illustration_2d": StylePreset(
                id="character_illustration_2d",
                name="Character 2D Illustration",
                category=StyleCategory.DIGITAL_ART,
                description="Professional 2D character illustration style with vibrant colors",
                visual_elements=["character-focused composition", "exaggerated expressions", "oversized heads (60% of body)", "tiny feet and hands", "smooth vector shapes", "theatrical gestures", "emotional storytelling", "cute companions"],
                color_characteristics=["colorful backgrounds", "accent colors", "warm skin tones", "gradient fills", "vibrant but harmonious", "complementary color schemes", "glowing highlights"],
                technique_details=["digital vector painting", "smooth bezier curves", "gradient mesh shading", "subsurface scattering on skin", "simplified but expressive features", "professional polish", "mobile-optimized clarity"],
                example_prompt="convert to professional 2D character illustration style, digital vector painting, character with exaggerated proportions, oversized expressive eyes with highlights, stylized hands and feet, smooth gradient shading with soft edges, colorful gradient background, warm skin tones with subsurface scattering, theatrical pose with gesture, simplified facial features with expression, sculptural hair with gradient highlights, clothing with smooth vector shapes, soft ambient lighting with colored rim lights, polished quality, centered composition",
                compatible_with=["soft_render", "children_book_deluxe", "premium_illustration"],
                tips=["Make heads 60% of body size", "Use color harmony", "Exaggerate expressions", "Add cute companions"]
            ),
            
            "clash_of_clans": StylePreset(
                id="clash_of_clans",
                name="Clash of Clans Style",
                category=StyleCategory.GAME,
                description="Supercell's distinctive chunky 3D mobile game art",
                visual_elements=["chunky 3D models", "exaggerated proportions", "bold outlines", "stylized characters"],
                color_characteristics=["bright primary colors", "strong contrast", "earthy tones for buildings", "gem-like effects"],
                technique_details=["low-poly aesthetic", "hand-painted textures", "cartoonish proportions", "isometric view"],
                example_prompt="convert to Clash of Clans mobile game style with chunky 3D models, exaggerated proportions, bright colors, hand-painted textures, and Supercell's signature cartoonish aesthetic",
                compatible_with=["clash_royale", "boom_beach", "mobile_strategy"],
                tips=["Exaggerate proportions", "Use bold, simple shapes"]
            ),
            
            "royal_match": StylePreset(
                id="royal_match",
                name="Royal Match Style",
                category=StyleCategory.GAME,
                description="Elegant casual match-3 game art with royal theme",
                visual_elements=["luxurious details", "royal decorations", "golden accents", "ornate patterns"],
                color_characteristics=["rich jewel tones", "purple and gold", "royal blue", "sparkling highlights"],
                technique_details=["polished 3D renders", "baroque ornaments", "regal atmosphere", "match-3 aesthetics"],
                example_prompt="convert to Royal Match game style with luxurious royal theme, ornate golden decorations, rich jewel tone colors, polished 3D renders, and elegant match-3 game aesthetic",
                compatible_with=["candy_crush", "playrix_style", "puzzle_game"],
                tips=["Add royal decorative elements", "Use purple and gold prominently"]
            ),
            
            "candy_crush": StylePreset(
                id="candy_crush",
                name="Candy Crush Style",
                category=StyleCategory.GAME,
                description="King's iconic candy-themed match-3 game art",
                visual_elements=["candy shapes", "jelly textures", "glossy surfaces", "sweet treats"],
                color_characteristics=["candy colors", "high saturation", "rainbow palette", "shiny highlights"],
                technique_details=["3D candy renders", "translucent effects", "sugary textures", "bubbly design"],
                example_prompt="convert to Candy Crush game style with glossy 3D candy shapes, vibrant candy colors, translucent jelly effects, sugary textures, and sweet treat aesthetic",
                compatible_with=["playrix_style", "royal_match", "match3"],
                tips=["Make everything look edible", "Use candy-like transparency"]
            ),
            
            "hay_day": StylePreset(
                id="hay_day",
                name="Hay Day Style",
                category=StyleCategory.GAME,
                description="Supercell's charming farm game art style",
                visual_elements=["chunky farm assets", "friendly animals", "rustic elements", "cozy atmosphere"],
                color_characteristics=["warm earth tones", "sunny yellows", "natural greens", "soft pastels"],
                technique_details=["stylized 3D models", "hand-painted look", "wholesome design", "farm aesthetic"],
                example_prompt="convert to Hay Day farm game style with chunky 3D models, friendly cartoon animals, warm earth tone colors, hand-painted textures, and cozy farm atmosphere",
                compatible_with=["farmville", "township", "farming_game"],
                tips=["Keep it wholesome and friendly", "Use warm, natural colors"]
            ),
            
            # High-Quality Illustration Styles
            "soft_render": StylePreset(
                id="soft_render",
                name="Soft render",
                category=StyleCategory.DIGITAL_ART,
                description="High-quality illustration with soft, puffy rendering",
                visual_elements=["soft edges", "puffy textures", "ambient occlusion", "whimsical details"],
                color_characteristics=["warm pastel colors", "soft gradients", "gentle tones", "gentle highlights"],
                technique_details=["volumetric lighting", "soft shadows", "smooth edges", "painterly rendering"],
                example_prompt="convert to deluxe children's book illustration with soft puffy textures, warm pastel colors, gentle characters, whimsical storytelling details, and high-quality painterly finish, without outline strokes",
                compatible_with=["children_book", "dreamy", "whimsical"],
                tips=["Use soft brushes", "Add atmospheric effects"]
            ),
            
            "premium_illustration": StylePreset(
                id="premium_illustration",
                name="Premium Digital Illustration",
                category=StyleCategory.DIGITAL_ART,
                description="Ultra high-quality digital illustration with meticulous details",
                visual_elements=["pristine details", "perfect rendering", "professional finish", "sophisticated composition"],
                color_characteristics=["rich color depth", "perfect color harmony", "subtle gradients", "professional palette"],
                technique_details=["master-level technique", "flawless execution", "gallery quality", "commercial grade"],
                example_prompt="convert to premium digital illustration with ultra high quality, meticulous attention to detail, sophisticated color harmony, flawless rendering, and gallery-worthy professional finish",
                compatible_with=["editorial", "advertising", "luxury"],
                tips=["Focus on perfect execution", "Pay attention to every detail"]
            ),
            
            "children_book_deluxe": StylePreset(
                id="children_book_deluxe",
                name="Deluxe Children's Book Art",
                category=StyleCategory.DIGITAL_ART,
                description="High-end children's book illustration with soft, appealing render",
                visual_elements=["gentle characters", "soft textures", "whimsical details", "storytelling elements"],
                color_characteristics=["warm pastels", "cozy atmosphere", "gentle contrasts", "inviting palette"],
                technique_details=["soft digital painting", "textured brushwork", "narrative composition", "emotional warmth"],
                example_prompt="convert to deluxe children's book illustration with soft fluffy textures, warm pastel colors, gentle characters, whimsical storytelling details, and high-quality painterly finish",
                compatible_with=["soft_render_illustration", "whimsical", "storybook"],
                tips=["Create inviting atmosphere", "Use soft, rounded shapes"]
            ),
            
            "artstation_trending": StylePreset(
                id="artstation_trending",
                name="ArtStation Trending Style",
                category=StyleCategory.DIGITAL_ART,
                description="Professional concept art quality that trends on ArtStation",
                visual_elements=["dramatic lighting", "epic scale", "professional polish", "cinematic composition"],
                color_characteristics=["moody atmosphere", "complementary colors", "dramatic contrast", "volumetric effects"],
                technique_details=["industry standard", "concept art quality", "production ready", "portfolio piece"],
                example_prompt="convert to ArtStation trending concept art style with dramatic cinematic lighting, epic scale, professional polish, volumetric effects, and industry-standard quality",
                compatible_with=["concept_art", "cinematic", "epic"],
                tips=["Focus on dramatic lighting", "Create epic atmosphere"]
            ),
            
            "behance_featured": StylePreset(
                id="behance_featured",
                name="Behance Featured Illustration",
                category=StyleCategory.DIGITAL_ART,
                description="Award-winning illustration style featured on Behance",
                visual_elements=["innovative design", "unique style", "creative composition", "artistic excellence"],
                color_characteristics=["bold color choices", "harmonious palette", "trendy gradients", "sophisticated tones"],
                technique_details=["cutting-edge technique", "award-winning quality", "creative innovation", "artistic mastery"],
                example_prompt="convert to Behance featured illustration style with innovative design, bold creative choices, sophisticated color palette, cutting-edge techniques, and award-winning quality",
                compatible_with=["premium_illustration", "editorial", "advertising"],
                tips=["Be bold and innovative", "Push creative boundaries"]
            ),
            
            # Concept Art Styles - By Direction
            "environment_concept": StylePreset(
                id="environment_concept",
                name="Environment Concept Art",
                category=StyleCategory.CONCEPT_ART,
                description="Professional environment and landscape concept art for games and films",
                visual_elements=["epic vistas", "atmospheric depth", "architectural details", "environmental storytelling"],
                color_characteristics=["mood-driven palette", "atmospheric perspective", "complementary schemes", "value hierarchy"],
                technique_details=["digital matte painting", "photobashing", "3D overpainting", "compositional rules"],
                example_prompt="convert to environment concept art style, epic landscape with atmospheric perspective, professional matte painting technique, cinematic composition, detailed architectural elements, mood-driven color palette with strong value hierarchy, environmental storytelling, production-ready concept art quality",
                compatible_with=["artstation_trending", "cinematic", "matte_painting"],
                tips=["Focus on atmosphere and mood", "Use rule of thirds", "Create depth with values"]
            ),
            
            "character_concept": StylePreset(
                id="character_concept",
                name="Character Sheet",
                category=StyleCategory.CONCEPT_ART,
                description="Professional character design for games, animation, and films",
                visual_elements=["character sheets", "costume design", "prop integration", "personality expression"],
                color_characteristics=["character-defining colors", "material indication", "lighting consistency", "skin tone accuracy"],
                technique_details=["anatomical accuracy", "design iteration", "turnaround views", "detail callouts"],
                example_prompt="transform into character sheet concept art style, maintaining original character identity while presenting in professional character design sheet format with clear silhouette, detailed costume design, material indication, expressive poses showing personality, consistent lighting, anatomically correct proportions, multiple view angles, industry-standard presentation",
                compatible_with=["artstation_trending", "game_concept", "animation_design"],
                tips=["Strong silhouette is key", "Show personality through pose", "Include detail callouts"]
            ),
            
            "creature_concept": StylePreset(
                id="creature_concept",
                name="Creature Concept Art",
                category=StyleCategory.CONCEPT_ART,
                description="Fantasy and sci-fi creature design for entertainment industry",
                visual_elements=["anatomical innovation", "believable biology", "texture details", "scale comparison"],
                color_characteristics=["natural patterns", "bioluminescence", "camouflage schemes", "threat display colors"],
                technique_details=["evolutionary logic", "skeletal structure", "muscle groups", "surface materials"],
                example_prompt="transform existing subjects into detailed creature concept art, reimagine as fantastical beings with believable anatomy, add detailed texture breakdowns, natural color patterns, bioluminescent accents, evolutionary adaptations, professional presentation maintaining original pose and composition, movie-quality creature transformation",
                compatible_with=["monster_design", "alien_concept", "fantasy_beasts"],
                tips=["Ground fantasy in real biology", "Show how it moves", "Include human for scale"]
            ),
            
            "vehicle_concept": StylePreset(
                id="vehicle_concept",
                name="Vehicle/Mech Concept Art",
                category=StyleCategory.CONCEPT_ART,
                description="Futuristic vehicle and mechanical design for sci-fi productions",
                visual_elements=["mechanical details", "functional design", "panel lines", "weathering effects"],
                color_characteristics=["industrial colors", "warning markings", "metallic surfaces", "wear patterns"],
                technique_details=["hard surface modeling", "mechanical joints", "exploded views", "material callouts"],
                example_prompt="transform existing elements into futuristic vehicle or mech concept art, convert subjects to mechanical forms with detailed hard surface design, add panel lines, hydraulics, thrusters, industrial color scheme with warning markings, weathered metal surfaces, maintain original composition while mechanizing all elements, production-ready technical illustration",
                compatible_with=["mech_design", "spaceship_concept", "industrial_design"],
                tips=["Function drives form", "Show mechanical logic", "Add wear and weathering"]
            ),
            
            "prop_concept": StylePreset(
                id="prop_concept",
                name="Prop/Weapon Concept Art",
                category=StyleCategory.CONCEPT_ART,
                description="Detailed prop and weapon designs for games and films",
                visual_elements=["material breakdowns", "ornamental details", "functional mechanisms", "wear patterns"],
                color_characteristics=["material-accurate colors", "accent details", "aging effects", "surface treatments"],
                technique_details=["orthographic views", "close-up details", "material indication", "scale reference"],
                example_prompt="transform objects in image into detailed prop or weapon concept art, enhance existing items with ornamental details, material breakdowns, functional mechanisms, realistic wear patterns, cultural decorations, maintain original composition while elevating objects to game-ready prop concepts with professional presentation",
                compatible_with=["weapon_design", "artifact_concept", "item_design"],
                tips=["Show all important angles", "Detail materials clearly", "Consider manufacturing"]
            ),
            
            # Concept Art Styles - By Famous Artists
            "syd_mead": StylePreset(
                id="syd_mead",
                name="Syd Mead Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Futuristic industrial design in the style of legendary concept artist Syd Mead",
                visual_elements=["streamlined forms", "chrome surfaces", "retro-futurism", "architectural integration"],
                color_characteristics=["chrome and glass", "neon accents", "sunset gradients", "reflective surfaces"],
                technique_details=["marker rendering", "airbrushed gradients", "technical precision", "atmospheric effects"],
                example_prompt="convert to Syd Mead concept art style, retro-futuristic vehicle design, streamlined chrome surfaces, integrated architecture, marker rendering technique with airbrushed gradients, sunset lighting with neon accents, blade runner aesthetic, visionary industrial design",
                compatible_with=["blade_runner", "retrofuturism", "vehicle_concept"],
                tips=["Emphasize reflections", "Use sunset lighting", "Integrate with architecture"]
            ),
            
            "craig_mullins": StylePreset(
                id="craig_mullins",
                name="Craig Mullins Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Digital painting mastery in the style of concept art pioneer Craig Mullins",
                visual_elements=["loose brushwork", "atmospheric depth", "light and shadow", "environmental mood"],
                color_characteristics=["limited palette", "color temperature shifts", "atmospheric haze", "dramatic lighting"],
                technique_details=["digital impressionism", "value painting", "speed painting", "photographic reference"],
                example_prompt="convert to Craig Mullins concept art style, loose confident brushstrokes, atmospheric environment painting, dramatic light and shadow, limited color palette with temperature shifts, digital impressionism technique, cinematic mood, masterful value control",
                compatible_with=["environment_concept", "speedpainting", "cinematic"],
                tips=["Work value first", "Keep brushwork loose", "Focus on mood"]
            ),
            
            "feng_zhu": StylePreset(
                id="feng_zhu",
                name="Feng Zhu Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Entertainment design in the style of concept art educator Feng Zhu",
                visual_elements=["design clarity", "functional aesthetics", "world-building", "cultural integration"],
                color_characteristics=["cohesive palettes", "environmental logic", "material accuracy", "lighting scenarios"],
                technique_details=["design fundamentals", "perspective mastery", "efficient workflow", "industry standards"],
                example_prompt="convert to Feng Zhu concept art style, clear design communication, functional aesthetic choices, strong perspective and form, environmentally integrated elements, efficient digital painting technique, Singapore school of design approach, entertainment industry quality",
                compatible_with=["environment_concept", "industrial_design", "world_building"],
                tips=["Design with purpose", "Show environmental context", "Maintain clarity"]
            ),
            
            "sparth": StylePreset(
                id="sparth",
                name="Sparth (Nicolas Bouvier) Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Geometric sci-fi environments in the style of Halo concept artist Sparth",
                visual_elements=["geometric abstraction", "mega structures", "minimalist forms", "epic scale"],
                color_characteristics=["monochromatic schemes", "atmospheric gradients", "subtle color accents", "fog effects"],
                technique_details=["structural design", "atmospheric perspective", "geometric composition", "scale contrast"],
                example_prompt="convert to Sparth concept art style, geometric mega-structures, minimalist sci-fi architecture, monochromatic color scheme with subtle accents, massive scale with tiny human figures, atmospheric fog and depth, Halo-inspired environmental design, structural abstraction",
                compatible_with=["scifi_architecture", "brutalist", "megastructure"],
                tips=["Emphasize scale", "Use geometric shapes", "Keep colors minimal"]
            ),
            
            "james_gurney": StylePreset(
                id="james_gurney",
                name="James Gurney Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Imaginative realism in the style of Dinotopia creator James Gurney",
                visual_elements=["naturalistic detail", "fantasy integration", "plein air quality", "storytelling elements"],
                color_characteristics=["natural light", "color harmony", "atmospheric effects", "warm undertones"],
                technique_details=["traditional painting", "observational accuracy", "imaginative realism", "narrative composition"],
                example_prompt="convert to James Gurney concept art style, imaginative realism combining fantasy with naturalistic detail, plein air lighting quality, Dinotopia-inspired world-building, traditional painting techniques, warm natural color harmony, narrative storytelling through environment",
                compatible_with=["fantasy_realism", "creature_concept", "world_building"],
                tips=["Ground fantasy in reality", "Use natural lighting", "Tell a story"]
            ),
            
            "ralph_mcquarrie": StylePreset(
                id="ralph_mcquarrie",
                name="Ralph McQuarrie Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Classic Star Wars concept art in the style of Ralph McQuarrie",
                visual_elements=["iconic designs", "painted quality", "dramatic scenes", "heroic proportions"],
                color_characteristics=["muted tones", "atmospheric haze", "dramatic skies", "painterly texture"],
                technique_details=["traditional painting", "matte painting", "production design", "cinematic framing"],
                example_prompt="convert to Ralph McQuarrie concept art style, Star Wars production design aesthetic, traditional painted quality, muted color palette with atmospheric haze, dramatic sky compositions, heroic character proportions, classic sci-fi matte painting technique",
                compatible_with=["star_wars", "space_opera", "retro_scifi"],
                tips=["Use muted colors", "Add atmospheric haze", "Frame cinematically"]
            ),
            
            "jean_giraud_moebius": StylePreset(
                id="jean_giraud_moebius",
                name="Moebius Concept Art Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Surreal sci-fi concept art in the style of Jean Giraud (Moebius)",
                visual_elements=["organic forms", "surreal landscapes", "flowing lines", "dreamlike quality"],
                color_characteristics=["desert palettes", "psychedelic colors", "gradient skies", "crystal clarity"],
                technique_details=["ligne claire", "organic architecture", "detailed linework", "European comics influence"],
                example_prompt="convert to Moebius concept art style, surreal science fiction landscape, organic flowing architecture, crystal clear ligne claire technique, desert color palette with psychedelic accents, dreamlike atmospheric quality, European bande dessinée aesthetic",
                compatible_with=["french_comics", "surreal_scifi", "desert_punk"],
                tips=["Embrace the surreal", "Use flowing organic forms", "Clear line technique"]
            ),
            
            # Anime/Manga Styles
            "anime_modern": StylePreset(
                id="anime_modern",
                name="Modern Anime",
                category=StyleCategory.ANIME_MANGA,
                description="Contemporary anime art style",
                visual_elements=["large eyes", "simplified features", "dynamic hair", "expressive faces"],
                color_characteristics=["bright colors", "cel shading", "gradient hair"],
                technique_details=["clean lineart", "flat coloring", "screen tone effects"],
                example_prompt="convert to modern anime artwork with large expressive eyes, clean black lineart, and vibrant flat cel-shaded colors",
                compatible_with=["manga", "chibi", "shoujo"],
                tips=["Specify anime genre for accuracy", "Great for character illustrations"]
            ),
            
            "studio_ghibli": StylePreset(
                id="studio_ghibli",
                name="Studio Ghibli Style",
                category=StyleCategory.ANIME_MANGA,
                description="Hayao Miyazaki's distinctive animation style",
                visual_elements=["soft backgrounds", "detailed environments", "whimsical elements", "hand-drawn feel"],
                color_characteristics=["natural palette", "soft pastels", "atmospheric perspective"],
                technique_details=["traditional animation", "painted backgrounds", "fluid movement"],
                example_prompt="convert to Studio Ghibli animation style with detailed hand-painted backgrounds, whimsical character design, and soft natural palette",
                compatible_with=["watercolor", "fantasy", "nature"],
                tips=["Excellent for landscapes", "Add magical elements for authenticity"]
            ),
            
            # Additional Game Styles
            "fortnite": StylePreset(
                id="fortnite",
                name="Fortnite Style",
                category=StyleCategory.GAME,
                description="Colorful cartoonish battle royale game aesthetic",
                visual_elements=["exaggerated proportions", "smooth surfaces", "stylized characters", "vibrant environment"],
                color_characteristics=["bright saturated colors", "high contrast", "cartoon shading"],
                technique_details=["simplified geometry", "bold outlines", "stylized textures"],
                example_prompt="convert to Fortnite 3D cartoon battle royale style with exaggerated character proportions, vibrant colors, and building mechanics aesthetic",
                compatible_with=["cartoon", "comic", "pixar"],
                tips=["Great for fun, energetic scenes", "Works well with action poses"]
            ),
            
            "red_dead": StylePreset(
                id="red_dead",
                name="Red Dead Redemption Style",
                category=StyleCategory.GAME,
                description="Realistic western game cinematography",
                visual_elements=["photorealistic textures", "dust particles", "atmospheric haze", "detailed environments"],
                color_characteristics=["warm desert tones", "golden hour lighting", "dusty atmosphere"],
                technique_details=["realistic rendering", "volumetric lighting", "cinematic camera angles"],
                example_prompt="make as Red Dead Redemption 2 photorealistic 3D western game with dusty atmosphere, volumetric lighting, and cinematic detail",
                compatible_with=["cinematic", "western", "realistic"],
                tips=["Perfect for outdoor scenes", "Add dust and atmospheric effects"]
            ),
            
            "ps1_graphics": StylePreset(
                id="ps1_graphics",
                name="PS1 Graphics",
                category=StyleCategory.GAME,
                description="Low-poly 3D retro PlayStation 1 style",
                visual_elements=["low polygon count", "texture warping", "vertex colors", "jagged edges"],
                color_characteristics=["limited color palette", "dithering effects", "flat shading"],
                technique_details=["affine texture mapping", "no anti-aliasing", "simple lighting"],
                example_prompt="convert to PS1 graphics style with extreme low-poly 3D aesthetic, vertex snapping, affine texture warping, no texture filtering, limited 15-bit color depth, gouraud shading, polygon jitter, z-fighting artifacts, and authentic 1990s PlayStation visual limitations",
                compatible_with=["retro", "pixel_art", "vaporwave", "low_poly"],
                tips=["Embrace the limitations", "Add CRT screen effects for authenticity"]
            ),
            
            "hollow_knight": StylePreset(
                id="hollow_knight",
                name="Hand-drawn 2D Style",
                category=StyleCategory.GAME,
                description="Beautiful hand-drawn 2D game art style",
                visual_elements=["hand-drawn lines", "organic shapes", "atmospheric depth", "silhouettes"],
                color_characteristics=["muted palette", "atmospheric blues", "selective bright accents"],
                technique_details=["traditional 2D animation", "parallax layers", "painted backgrounds"],
                example_prompt="convert to hand-drawn 2D style with organic shapes, atmospheric depth, and muted color palette like Hollow Knight",
                compatible_with=["watercolor", "gothic", "dark_fantasy", "indie_game"],
                tips=["Focus on atmosphere", "Use silhouettes for impact"]
            ),
            
            "isometric": StylePreset(
                id="isometric",
                name="Isometric Style",
                category=StyleCategory.GAME,
                description="Classic isometric RPG perspective",
                visual_elements=["30-degree angle view", "tile-based design", "detailed 2D sprites or 3D pre-renders", "architectural focus"],
                color_characteristics=["clear readable colors", "distinct object separation", "ambient lighting"],
                technique_details=["fixed perspective", "no perspective distortion", "grid-based layout"],
                example_prompt="transform entire scene to isometric game style with perfect 30-degree dimetric projection, convert all elements including foreground and background to isometric perspective, tile-based grid alignment, no perspective distortion, equal depth for all objects, classic RPG game aesthetics with clear readable details",
                compatible_with=["pixel_art", "strategy", "dungeon"],
                tips=["Maintain consistent angle", "Great for environments and buildings"]
            ),
            
            # Movie/Film Styles
            "noir": StylePreset(
                id="noir",
                name="Film Noir",
                category=StyleCategory.MOVIE,
                description="Classic black and white detective film style",
                visual_elements=["high contrast", "dramatic shadows", "venetian blind lighting", "smoke effects"],
                color_characteristics=["black and white", "deep blacks", "stark whites", "grayscale"],
                technique_details=["chiaroscuro lighting", "dutch angles", "atmospheric fog"],
                example_prompt="convert to classic film noir cinematography with high contrast black and white, venetian blind shadows, and 1940s detective atmosphere",
                compatible_with=["cinematic", "vintage", "mystery"],
                tips=["Use dramatic lighting", "Add smoke or fog for atmosphere"]
            ),
            
            "blade_runner": StylePreset(
                id="blade_runner",
                name="Blade Runner Style",
                category=StyleCategory.MOVIE,
                description="Dystopian sci-fi cinematography",
                visual_elements=["neon lights", "rain effects", "urban decay", "flying vehicles"],
                color_characteristics=["cyan and orange", "neon accents", "dark backgrounds", "atmospheric haze"],
                technique_details=["neo-noir lighting", "lens flares", "depth of field"],
                example_prompt="make as Blade Runner 2049 cinematography with neon-lit dystopian cityscape, constant rain, fog, and neo-noir color palette",
                compatible_with=["cyberpunk_game", "noir", "scifi"],
                tips=["Essential: rain and neon", "Use atmospheric depth"]
            ),
            
            # Additional Artistic Styles
            "impressionist": StylePreset(
                id="impressionist",
                name="Impressionist",
                category=StyleCategory.ART_MOVEMENTS,
                description="French impressionist painting style",
                visual_elements=["visible brushstrokes", "light effects", "color patches", "atmospheric scenes"],
                color_characteristics=["pure colors", "light and shadow play", "complementary colors"],
                technique_details=["broken color technique", "plein air feeling", "movement and light"],
                example_prompt="convert to French impressionist painting with broken color technique, visible brushstrokes, and captured light moments",
                compatible_with=["oil_painting", "watercolor", "landscape"],
                tips=["Focus on light and atmosphere", "Use broken color technique"]
            ),
            
            "art_nouveau": StylePreset(
                id="art_nouveau",
                name="Art Nouveau",
                category=StyleCategory.ART_MOVEMENTS,
                description="Decorative art style with natural forms",
                visual_elements=["flowing lines", "floral motifs", "ornamental borders", "organic shapes"],
                color_characteristics=["muted earth tones", "gold accents", "pastel colors"],
                technique_details=["decorative patterns", "stylized nature", "elegant curves"],
                example_prompt="convert to Art Nouveau decorative style with flowing organic lines, nature-inspired motifs, and ornamental borders",
                compatible_with=["vintage", "poster", "decorative"],
                tips=["Add ornamental frames", "Use nature-inspired patterns"]
            ),
            
            "synthwave": StylePreset(
                id="synthwave",
                name="Synthwave",
                category=StyleCategory.DIGITAL_ART,
                description="80s retro-futuristic aesthetic",
                visual_elements=["grid patterns", "palm trees", "sports cars", "sunset backdrop"],
                color_characteristics=["neon pink and purple", "cyan highlights", "dark backgrounds"],
                technique_details=["digital grid", "chrome effects", "VHS artifacts"],
                example_prompt="convert to synthwave retro-futuristic aesthetic with neon pink and cyan colors, wireframe grid floor, palm trees, and gradient sunset",
                compatible_with=["cyberpunk_game", "vaporwave", "outrun", "retrowave"],
                tips=["Essential: neon grid and sunset", "Add VHS effects"]
            ),
            
            # Cartoon Styles
            "pixar": StylePreset(
                id="pixar",
                name="Pixar 3D Style",
                category=StyleCategory.CARTOON,
                description="Pixar's signature 3D animation style",
                visual_elements=["smooth 3D rendering", "expressive characters", "stylized proportions", "detailed textures"],
                color_characteristics=["vibrant colors", "warm lighting", "rich saturation", "ambient occlusion"],
                technique_details=["subsurface scattering", "advanced shading", "particle effects"],
                example_prompt="convert to Pixar 3D CGI animation with subsurface scattering, expressive character acting, vibrant colors, and photorealistic rendering",
                compatible_with=["disney", "dreamworks", "3d_animation"],
                tips=["Focus on character expressiveness", "Add warm, inviting lighting"]
            ),
            
            "disney_classic": StylePreset(
                id="disney_classic",
                name="Classic Disney",
                category=StyleCategory.CARTOON,
                description="Traditional Disney 2D animation style",
                visual_elements=["fluid animation", "rounded shapes", "expressive eyes", "graceful movements"],
                color_characteristics=["bright primary colors", "soft shadows", "warm tones"],
                technique_details=["hand-drawn appearance", "12 principles of animation", "squash and stretch"],
                example_prompt="convert to classic Disney 2D hand-drawn animation with fluid character movement, expressive faces, and bright primary colors",
                compatible_with=["fairytale", "musical", "vintage"],
                tips=["Emphasize graceful movements", "Use round, friendly shapes"]
            ),
            
            "simpsons": StylePreset(
                id="simpsons",
                name="The Simpsons Style",
                category=StyleCategory.CARTOON,
                description="Matt Groening's iconic 2D TV cartoon style",
                visual_elements=["yellow skin tones", "bulging eyes", "overbite", "simple lines"],
                color_characteristics=["flat colors", "yellow dominant", "bright primaries"],
                technique_details=["thick outlines", "minimal shading", "exaggerated features"],
                example_prompt="convert to The Simpsons 2D cartoon style with yellow skin tones, overbite character design, thick black outlines, and flat colors",
                compatible_with=["futurama", "adult_cartoon", "satire"],
                tips=["Keep it simple and flat", "Exaggerate facial features"]
            ),
            
            "hanna_barbera": StylePreset(
                id="hanna_barbera",
                name="Hanna-Barbera Style",
                category=StyleCategory.CARTOON,
                description="Classic 2D TV cartoon style from the 1960s-70s",
                visual_elements=["limited animation", "thick outlines", "simple shapes", "repeating backgrounds"],
                color_characteristics=["flat colors", "limited palette", "bold contrasts"],
                technique_details=["economical animation", "strong silhouettes", "minimal detail"],
                example_prompt="convert to Hanna-Barbera 2D cartoon with thick outlines, flat colors, limited animation techniques, and 1960s TV character designs",
                compatible_with=["retro", "vintage", "saturday_morning"],
                tips=["Use bold, simple shapes", "Limit color palette"]
            ),
            
            "adventure_time": StylePreset(
                id="adventure_time",
                name="Adventure Time Style",
                category=StyleCategory.CARTOON,
                description="Pendleton Ward's unique 2D cartoon aesthetic",
                visual_elements=["noodle arms", "simple faces", "organic shapes", "whimsical designs"],
                color_characteristics=["pastel colors", "soft gradients", "candy-like palette"],
                technique_details=["loose linework", "minimal detail", "expressive simplicity"],
                example_prompt="convert to Adventure Time 2D cartoon with noodle limbs, simple dot eyes, organic shapes, and pastel candy colors",
                compatible_with=["steven_universe", "modern_cartoon", "whimsical"],
                tips=["Keep it loose and fun", "Use candy-inspired colors"]
            ),
            
            "looney_tunes": StylePreset(
                id="looney_tunes",
                name="Looney Tunes Style",
                category=StyleCategory.CARTOON,
                description="Classic Warner Bros 2D theatrical cartoon style",
                visual_elements=["exaggerated expressions", "slapstick poses", "dynamic action", "rubber hose limbs"],
                color_characteristics=["bright saturated colors", "high contrast", "bold primaries"],
                technique_details=["squash and stretch", "motion lines", "impact frames"],
                example_prompt="convert to Looney Tunes 2D classic cartoon with rubber hose animation, exaggerated expressions, slapstick action poses, and theatrical timing",
                compatible_with=["tex_avery", "classic_cartoon", "slapstick"],
                tips=["Exaggerate everything", "Focus on dynamic action"]
            ),
            
            # Extended Anime/Manga Styles
            "manga": StylePreset(
                id="manga",
                name="Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Traditional Japanese manga art",
                visual_elements=["screentones", "speed lines", "dramatic angles", "detailed eyes"],
                color_characteristics=["black and white", "halftone patterns", "high contrast"],
                technique_details=["ink drawing", "panel composition", "emotional symbols"],
                example_prompt="convert to Japanese manga artwork with black ink lineart, screentone shading, dramatic panel composition, and speed lines",
                compatible_with=["anime_modern", "shoujo", "shounen"],
                tips=["Use screentones for shading", "Add speed lines for action"]
            ),
            
            "shoujo": StylePreset(
                id="shoujo",
                name="Shoujo Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Romantic manga style for young women",
                visual_elements=["sparkly eyes", "flowing hair", "flower backgrounds", "delicate features"],
                color_characteristics=["soft pastels", "pink tones", "dreamy atmosphere"],
                technique_details=["detailed eyes", "romantic bubbles", "emotional close-ups"],
                example_prompt="convert to shoujo manga style with large sparkly eyes, flowing hair with highlights, decorative flower backgrounds, and soft romantic mood",
                compatible_with=["manga", "romance", "magical_girl"],
                tips=["Add sparkles and flowers", "Focus on emotions"]
            ),
            
            "shounen": StylePreset(
                id="shounen",
                name="Shounen Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Action manga style for young men",
                visual_elements=["dynamic action", "power effects", "determined expressions", "battle scenes"],
                color_characteristics=["bold colors", "energy effects", "impact flashes"],
                technique_details=["action lines", "impact frames", "power auras"],
                example_prompt="convert to shounen manga action style with dynamic fighting poses, energy auras, impact effects, and intense facial expressions",
                compatible_with=["manga", "action", "battle"],
                tips=["Emphasize power and movement", "Add energy effects"]
            ),
            
            "chibi": StylePreset(
                id="chibi",
                name="Chibi Style",
                category=StyleCategory.ANIME_MANGA,
                description="Super-deformed cute character style",
                visual_elements=["large heads", "tiny bodies", "huge eyes", "simplified features"],
                color_characteristics=["bright colors", "soft shading", "kawaii palette"],
                technique_details=["2-3 head tall proportions", "minimal detail", "cute expressions"],
                example_prompt="convert to chibi super-deformed style with 2-3 head tall proportions, oversized head, huge eyes, simplified features, and kawaii aesthetic",
                compatible_with=["anime_modern", "kawaii", "cute"],
                tips=["Make heads 1/3 of total height", "Simplify all details"]
            ),
            
            "mecha": StylePreset(
                id="mecha",
                name="Mecha Anime Style",
                category=StyleCategory.ANIME_MANGA,
                description="Giant robot 2D anime aesthetic with mechanical detail",
                visual_elements=["detailed machinery", "panel lines", "mechanical joints", "pilot cockpits"],
                color_characteristics=["metallic surfaces", "warning colors", "glowing elements"],
                technique_details=["technical detail", "perspective shots", "scale contrast"],
                example_prompt="transform into mecha anime style, enhance existing subjects with detailed mechanical armor and robotic augmentations, add complex panel lines, rivets, hydraulics, thrusters, metallic surfaces with weathering, glowing power cores, mechanical joint details, scale indicators, warning decals, and dramatic mecha aesthetic while maintaining original composition",
                compatible_with=["gundam", "evangelion", "scifi"],
                tips=["Add panel lines and rivets", "Show scale with humans"]
            ),
            
            # Extended Movie Styles
            "wes_anderson": StylePreset(
                id="wes_anderson",
                name="Wes Anderson Style",
                category=StyleCategory.MOVIE,
                description="Symmetrical, pastel-colored cinematography",
                visual_elements=["perfect symmetry", "centered composition", "vintage props", "quirky details"],
                color_characteristics=["pastel palette", "color coordination", "muted tones"],
                technique_details=["planimetric shots", "tableau framing", "dollhouse views"],
                example_prompt="make as Wes Anderson style with perfect symmetry, pastel colors, and whimsical vintage aesthetics",
                compatible_with=["vintage", "quirky", "artistic"],
                tips=["Center everything", "Use coordinated pastels"]
            ),
            
            "hitchcock": StylePreset(
                id="hitchcock",
                name="Hitchcock Style",
                category=StyleCategory.MOVIE,
                description="Suspenseful classic thriller cinematography",
                visual_elements=["dramatic angles", "shadow play", "voyeuristic framing", "suspense elements"],
                color_characteristics=["high contrast", "dramatic lighting", "noir influences"],
                technique_details=["dolly zoom", "POV shots", "tension building"],
                example_prompt="make as Hitchcock thriller style with dramatic angles, suspenseful lighting, and voyeuristic framing",
                compatible_with=["noir", "thriller", "classic"],
                tips=["Create visual tension", "Use unusual angles"]
            ),
            
            "star_wars": StylePreset(
                id="star_wars",
                name="Star Wars Style",
                category=StyleCategory.MOVIE,
                description="Epic space opera cinematography (Original/Prequel/Sequel era styles)",
                visual_elements=["vast spaces", "alien worlds", "lightsabers", "spaceships"],
                color_characteristics=["contrasting lights", "laser colors", "atmospheric hues"],
                technique_details=["epic scale", "practical effects", "hero shots"],
                example_prompt="make as Star Wars cinematic style with epic space opera visuals, alien worlds, lightsaber glow effects, and dramatic hero lighting",
                compatible_with=["scifi", "space", "epic"],
                tips=["Show epic scale", "Add atmospheric effects"]
            ),
            
            "matrix": StylePreset(
                id="matrix",
                name="Matrix Style",
                category=StyleCategory.MOVIE,
                description="Digital rain and bullet-time aesthetics with distinctive green tint",
                visual_elements=["green code", "bullet time", "leather and sunglasses", "urban decay"],
                color_characteristics=["green tint", "high contrast", "desaturated reality"],
                technique_details=["slow motion", "wire-fu action", "digital effects"],
                example_prompt="make as Matrix cinematography with green-tinted digital rain, bullet-time slow motion, and cyberpunk noir aesthetics",
                compatible_with=["cyberpunk_game", "noir", "action"],
                tips=["Add green tint", "Use high contrast"]
            ),
            
            "fallout_poster": StylePreset(
                id="fallout_poster",
                name="Fallout Poster Style",
                category=StyleCategory.GAME,
                description="Post-apocalyptic retro-futuristic poster art from Fallout universe",
                visual_elements=["vault boy mascot", "atomic age design", "radiation symbols", "pip-boy interface", "retro propaganda"],
                color_characteristics=["muted yellows and greens", "aged paper texture", "faded colors", "nuclear glow effects"],
                technique_details=["1950s advertising art", "screen printing aesthetic", "weathered and torn edges", "vintage typography"],
                example_prompt="convert to Fallout poster style, retro-futuristic 1950s propaganda poster, atomic age design with vault boy aesthetic, muted yellow and green palette, aged paper texture with tears and stains, radiation warning symbols, pip-boy interface elements, optimistic nuclear family messaging with dark undertones, weathered vintage advertisement look",
                compatible_with=["retro", "post_apocalyptic", "propaganda_poster"],
                tips=["Add vault boy thumbs up", "Use atomic age optimism", "Weather and age the poster"]
            ),
            
            "gta_poster": StylePreset(
                id="gta_poster",
                name="GTA Poster Style",
                category=StyleCategory.GAME,
                description="Grand Theft Auto's iconic illustrated poster art style",
                visual_elements=["comic book inking", "action scenes", "multiple characters", "urban setting", "vehicles and weapons"],
                color_characteristics=["high contrast colors", "bold outlines", "saturated primaries", "dramatic lighting"],
                technique_details=["digital illustration", "thick black outlines", "dynamic composition", "overlapping vignettes"],
                example_prompt="convert to GTA poster style, bold comic book illustration with thick black outlines, high contrast saturated colors, multiple character vignettes, action-packed urban scene composition, weapons and vehicles prominently featured, dramatic lighting with strong shadows, V-shaped composition, graffiti elements, crime drama aesthetic",
                compatible_with=["comic_book", "action_poster", "urban_art"],
                tips=["Use thick black outlines", "Create action vignettes", "Bold saturated colors"]
            ),
            
            # Material Transformation Styles
            "plush_toy": StylePreset(
                id="plush_toy",
                name="Plush Toy Transformation",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform subjects into soft stuffed plush toys",
                visual_elements=["soft fabric texture", "button eyes", "stitched seams", "fuzzy materials", "simplified shapes"],
                color_characteristics=["soft pastels", "fabric-like colors", "gentle shadows", "cozy warmth"],
                technique_details=["visible stitching", "fabric folds", "stuffing bulges", "sewn details"],
                example_prompt="transform into soft plush toy version, fuzzy fabric textures, visible stitched seams, button eyes, simplified cuddly proportions, stuffed animal aesthetic with cotton filling, sewn-on details, felt accessories, toy-like charm, handmade quality with slight imperfections",
                compatible_with=["cute", "kawaii", "toy_style"],
                tips=["Add visible seams", "Simplify shapes", "Use fabric textures"]
            ),
            
            "marble_statue": StylePreset(
                id="marble_statue",
                name="Marble Statue Transformation",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform subjects into classical marble sculptures",
                visual_elements=["white marble", "classical poses", "draped fabric", "pedestal base", "carved details"],
                color_characteristics=["pure white", "subtle grey veining", "stone shadows", "museum lighting"],
                technique_details=["carved surfaces", "polished finish", "classical proportions", "sculptural forms"],
                example_prompt="transform into classical marble statue, pure white Carrara marble with grey veining, polished stone surface, classical Greek/Roman sculptural style, draped fabric carved in stone, heroic proportions, museum pedestal display, dramatic lighting emphasizing form, timeless artistic beauty",
                compatible_with=["classical", "museum_art", "sculpture"],
                tips=["Use classical poses", "Add marble veining", "Dramatic museum lighting"]
            ),
            
            "clay_sculpture": StylePreset(
                id="clay_sculpture",
                name="Clay Sculpture Transformation",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform into handmade clay or pottery artwork",
                visual_elements=["clay texture", "fingerprint marks", "tool marks", "earthy surface", "handcrafted feel"],
                color_characteristics=["terracotta orange", "natural clay colors", "matte finish", "earthy tones"],
                technique_details=["hand-sculpted", "visible tool marks", "organic shapes", "pottery techniques"],
                example_prompt="transform into handmade clay sculpture, terracotta or ceramic material, visible fingerprints and tool marks, earthy natural clay colors, matte unglazed surface, artisanal pottery aesthetic, organic handcrafted shapes, workshop setting, artistic imperfections showing human touch",
                compatible_with=["pottery", "ceramic_art", "handmade"],
                tips=["Show tool marks", "Use earthy colors", "Keep it organic"]
            ),
            
            "origami_paper": StylePreset(
                id="origami_paper",
                name="Origami Paper Art",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform into folded paper origami art",
                visual_elements=["paper folds", "geometric shapes", "crisp edges", "angular forms", "paper texture"],
                color_characteristics=["bright paper colors", "clean whites", "origami patterns", "solid colors"],
                technique_details=["precise folding", "geometric construction", "paper engineering", "modular design"],
                example_prompt="transform into origami paper art, precisely folded colored paper, sharp geometric edges and creases, traditional Japanese paper folding aesthetic, clean angular forms, visible paper texture, modular construction, delicate paper engineering, cast shadows showing depth",
                compatible_with=["geometric", "japanese_art", "paper_craft"],
                tips=["Emphasize fold lines", "Use solid colors", "Show paper texture"]
            ),
            
            "lego_brick": StylePreset(
                id="lego_brick",
                name="LEGO Brick Style",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform into LEGO brick construction",
                visual_elements=["plastic bricks", "studs on top", "modular construction", "primary colors", "blocky shapes"],
                color_characteristics=["LEGO color palette", "plastic shine", "primary colors", "clean surfaces"],
                technique_details=["brick-by-brick building", "modular assembly", "technical construction", "minifig scale"],
                example_prompt="transform into LEGO brick construction, colorful plastic building blocks with visible studs, modular brick-by-brick assembly, official LEGO color palette, glossy plastic surfaces, blocky geometric interpretation, technical building techniques, playful toy aesthetic, clean Danish design",
                compatible_with=["toy_style", "geometric", "modular_design"],
                tips=["Show individual bricks", "Use LEGO colors", "Include studs detail"]
            ),
            
            "stained_glass": StylePreset(
                id="stained_glass",
                name="Stained Glass Window",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform into illuminated stained glass artwork",
                visual_elements=["lead lines", "colored glass", "light transmission", "religious motifs", "geometric patterns"],
                color_characteristics=["jewel tones", "translucent colors", "backlit glow", "rich saturation"],
                technique_details=["leaded glass", "light refraction", "traditional craftsmanship", "window design"],
                example_prompt="transform into stained glass window art, brilliant jewel-toned colored glass panels, black lead lines separating segments, translucent light shining through, cathedral window aesthetic, religious or decorative motifs, traditional glasswork craftsmanship, luminous backlit effect",
                compatible_with=["religious_art", "gothic", "decorative"],
                tips=["Bold lead lines", "Jewel tone colors", "Backlit effect"]
            ),
            
            "ice_sculpture": StylePreset(
                id="ice_sculpture",
                name="Ice Sculpture",
                category=StyleCategory.MATERIAL_TRANSFORM,
                description="Transform into crystal-clear ice sculpture",
                visual_elements=["transparent ice", "carved details", "frozen texture", "reflections", "cool lighting"],
                color_characteristics=["crystal clear", "blue tints", "frozen whites", "transparent"],
                technique_details=["chainsaw carving", "detailed chiseling", "polished surfaces", "structural support"],
                example_prompt="transform into crystal-clear ice sculpture, transparent frozen water with blue tints, expertly carved details with chainsaw and chisel marks, polished glassy surfaces, internal reflections and refractions, cool blue lighting, winter festival display, melting drops for realism",
                compatible_with=["winter", "festival_art", "transparent"],
                tips=["Show transparency", "Add blue tints", "Include reflections"]
            ),
            
            # Environmental State Changes
            "season_winter": StylePreset(
                id="season_winter",
                name="Winter Transformation",
                category=StyleCategory.ENVIRONMENT_TRANSFORM,
                description="Transform scene to winter with snow and ice",
                visual_elements=["snow coverage", "icicles", "frost", "bare trees", "winter clothing"],
                color_characteristics=["white snow", "cool blues", "grey skies", "muted colors"],
                technique_details=["snow accumulation", "frozen surfaces", "winter atmosphere", "cold lighting"],
                example_prompt="transform scene to deep winter, heavy snow coverage on all surfaces, icicles hanging from structures, frost on windows, bare deciduous trees, people in winter coats and scarves, frozen water features, grey overcast sky, cold blue lighting, visible breath in cold air",
                compatible_with=["seasonal", "weather", "atmospheric"],
                tips=["Cover everything in snow", "Cool color temperature", "Add winter details"]
            ),
            
            "post_apocalyptic": StylePreset(
                id="post_apocalyptic",
                name="Post-Apocalyptic World",
                category=StyleCategory.ENVIRONMENT_TRANSFORM,
                description="Transform to post-apocalyptic wasteland",
                visual_elements=["ruins", "overgrowth", "decay", "makeshift repairs", "survival gear"],
                color_characteristics=["dusty browns", "rust orange", "faded colors", "grey skies"],
                technique_details=["environmental destruction", "nature reclaiming", "weathering effects", "abandoned feel"],
                example_prompt="transform to post-apocalyptic wasteland, crumbling buildings with exposed rebar, nature reclaiming urban spaces, rust and decay on all metal, makeshift survivor camps, toxic sky with dust particles, abandoned vehicles, overgrown vegetation breaking through concrete, scavenged materials, desolate atmosphere",
                compatible_with=["dystopian", "survival", "wasteland"],
                tips=["Add decay and rust", "Nature reclaiming cities", "Dusty atmosphere"]
            ),
            
            "cyberpunk_future": StylePreset(
                id="cyberpunk_future",
                name="Cyberpunk Future",
                category=StyleCategory.ENVIRONMENT_TRANSFORM,
                description="Transform to neon-lit cyberpunk future",
                visual_elements=["neon signs", "holograms", "flying vehicles", "tech implants", "urban density"],
                color_characteristics=["neon pink/blue", "dark nights", "holographic effects", "LED lighting"],
                technique_details=["high-tech low-life", "vertical cities", "augmented reality", "corporate dystopia"],
                example_prompt="transform to cyberpunk future cityscape, neon signs in Japanese and English, holographic advertisements, flying vehicles between buildings, cybernetic augmentations on people, rain-slicked streets reflecting neon, vertical urban sprawl, corporate megastructures, dark atmospheric nights, tech-noir aesthetic",
                compatible_with=["sci-fi", "noir", "futuristic"],
                tips=["Neon everywhere", "Add holograms", "Rain for reflections"]
            ),
            
            "steampunk_era": StylePreset(
                id="steampunk_era",
                name="Steampunk Transformation",
                category=StyleCategory.ENVIRONMENT_TRANSFORM,
                description="Transform to Victorian-era steampunk world",
                visual_elements=["brass gears", "steam pipes", "clockwork", "airships", "goggles"],
                color_characteristics=["brass and copper", "sepia tones", "industrial browns", "steam fog"],
                technique_details=["Victorian retrofuturism", "mechanical complexity", "industrial revolution", "anachronistic tech"],
                example_prompt="transform to steampunk alternate history, brass gears and copper pipes everywhere, steam-powered machinery, Victorian architecture with industrial modifications, airships in sky, people wearing goggles and top hats, clockwork mechanisms visible, sepia-toned atmosphere, coal smoke and steam",
                compatible_with=["victorian", "industrial", "retrofuturistic"],
                tips=["Add gears and pipes", "Sepia color grading", "Victorian clothing"]
            ),
            
            "underwater_submerged": StylePreset(
                id="underwater_submerged",
                name="Underwater World",
                category=StyleCategory.ENVIRONMENT_TRANSFORM,
                description="Transform scene to underwater environment",
                visual_elements=["water caustics", "floating debris", "sea life", "coral growth", "bubbles"],
                color_characteristics=["blue-green tint", "filtered sunlight", "murky depths", "bioluminescence"],
                technique_details=["underwater physics", "light refraction", "aquatic atmosphere", "depth fog"],
                example_prompt="transform to underwater submerged environment, blue-green water tint throughout, caustic light patterns, floating particles and debris, coral growing on structures, schools of fish, seaweed and kelp, bubbles rising, filtered sunlight from above, underwater fog limiting visibility",
                compatible_with=["ocean", "aquatic", "submarine"],
                tips=["Blue-green tint everything", "Add caustic lighting", "Include sea life"]
            ),
            
            # Cultural Transformation Styles
            "ancient_slavic": StylePreset(
                id="ancient_slavic",
                name="Ancient Slavic Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Slavic/Russian folklore aesthetic",
                visual_elements=["wooden architecture", "folk patterns", "traditional clothing", "pagan symbols", "birch forests"],
                color_characteristics=["rich reds", "deep blues", "gold accents", "natural wood tones", "white birch"],
                technique_details=["ornate wood carving", "folk art patterns", "traditional embroidery", "onion domes"],
                example_prompt="transform to ancient Slavic cultural setting, wooden log architecture with carved details, people in traditional vyshyvanka embroidered clothing, kokoshnik headdresses, folk patterns and symbols, birch forest surroundings, Orthodox church domes in distance, rich red and gold color scheme, firebird and folkloric elements",
                compatible_with=["medieval", "fantasy", "folklore"],
                tips=["Add folk patterns", "Use red and gold", "Include traditional elements"]
            ),
            
            "medieval_european": StylePreset(
                id="medieval_european",
                name="Medieval European",
                category=StyleCategory.CULTURAL,
                description="Transform to medieval European castle and village life",
                visual_elements=["stone castles", "knight armor", "medieval clothing", "heraldry", "market squares"],
                color_characteristics=["stone greys", "royal purples", "heraldic colors", "torch lighting"],
                technique_details=["Gothic architecture", "illuminated manuscripts", "tapestry style", "medieval crafts"],
                example_prompt="transform to medieval European setting, stone castle architecture with Gothic elements, people in period-accurate tunics and dresses, knights in armor, heraldic banners and shields, cobblestone streets, market stalls, thatched roof houses, torch and candlelight illumination, medieval atmosphere",
                compatible_with=["fantasy", "historical", "castle"],
                tips=["Add heraldry", "Stone and wood materials", "Period lighting"]
            ),
            
            "fantasy_elven": StylePreset(
                id="fantasy_elven",
                name="Elven Fantasy Realm",
                category=StyleCategory.CULTURAL,
                description="Transform to elegant elven civilization aesthetic",
                visual_elements=["organic architecture", "living trees", "elegant clothing", "glowing crystals", "nature integration"],
                color_characteristics=["forest greens", "silver and white", "ethereal glows", "natural lights"],
                technique_details=["flowing organic lines", "nature-integrated design", "delicate details", "magical elements"],
                example_prompt="transform to elven fantasy realm, organic architecture grown from living trees, elegant flowing robes with silver embroidery, pointed ears on characters, glowing magical crystals, delicate filigree metalwork, integration with nature, ethereal lighting, mystical forest setting, elvish script decorations",
                compatible_with=["fantasy", "magical", "forest"],
                tips=["Integrate with nature", "Add ethereal glows", "Elegant flowing designs"]
            ),
            
            "ancient_egyptian": StylePreset(
                id="ancient_egyptian",
                name="Ancient Egyptian",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Egyptian palace and temple setting",
                visual_elements=["pyramids", "hieroglyphs", "pharaoh regalia", "columns", "sphinx statues"],
                color_characteristics=["gold and lapis blue", "sandstone", "turquoise", "desert tones"],
                technique_details=["hieroglyphic decorations", "profile art style", "monumental scale", "ritual objects"],
                example_prompt="transform to ancient Egyptian setting, massive stone columns with hieroglyphs, people in linen clothing and golden jewelry, pharaoh headdresses, Eye of Horus symbols, pyramid structures, sphinx statues, papyrus scrolls, golden sarcophagi, desert sand backdrop, monumental architecture scale",
                compatible_with=["ancient", "desert", "monumental"],
                tips=["Add hieroglyphs everywhere", "Gold and blue palette", "Monumental scale"]
            ),
            
            "japanese_edo": StylePreset(
                id="japanese_edo",
                name="Japanese Edo Period",
                category=StyleCategory.CULTURAL,
                description="Transform to traditional Japanese Edo period setting",
                visual_elements=["pagodas", "cherry blossoms", "kimono", "paper lanterns", "wooden bridges"],
                color_characteristics=["sakura pink", "indigo blue", "red torii", "natural wood"],
                technique_details=["ukiyo-e influence", "architectural precision", "zen aesthetics", "seasonal elements"],
                example_prompt="transform to Japanese Edo period, traditional wooden architecture with curved roofs, people in elaborate kimono with obi, cherry blossom trees, red torii gates, paper lanterns, koi ponds with wooden bridges, mount fuji in distance, ukiyo-e art influence, tea ceremony elements, zen garden features",
                compatible_with=["asian", "historical", "zen"],
                tips=["Add cherry blossoms", "Traditional architecture", "Seasonal elements"]
            ),
            
            "aztec_mayan": StylePreset(
                id="aztec_mayan",
                name="Aztec-Mayan Civilization",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Mesoamerican pyramid city",
                visual_elements=["step pyramids", "jade ornaments", "feathered headdresses", "stone carvings", "jungle setting"],
                color_characteristics=["jade green", "gold", "terracotta", "bright feathers", "jungle greens"],
                technique_details=["geometric patterns", "ritual masks", "calendar stones", "relief carvings"],
                example_prompt="transform to Aztec-Mayan setting, massive step pyramids with steep stairs, people in feathered headdresses and jade jewelry, intricate stone carvings, serpent motifs, ritual masks, jungle vegetation, colorful murals, obsidian and gold ornaments, hieroglyphic calendar stones, ceremonial plaza",
                compatible_with=["ancient", "jungle", "ceremonial"],
                tips=["Geometric patterns", "Feathered decorations", "Jungle integration"]
            ),
            
            "norse_viking": StylePreset(
                id="norse_viking",
                name="Norse Viking Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to Norse Viking settlement and mythology",
                visual_elements=["longhouses", "rune stones", "viking ships", "fur clothing", "norse symbols"],
                color_characteristics=["iron grey", "fur browns", "ice blue", "blood red", "gold metal"],
                technique_details=["runic inscriptions", "wood carving", "metalwork", "woven patterns"],
                example_prompt="transform to Norse Viking setting, wooden longhouses with dragon head decorations, warriors in fur cloaks and iron helmets, rune stone monuments, viking longships, Thor's hammer symbols, mead halls with carved pillars, northern lights sky, fjord landscape, norse mythology elements",
                compatible_with=["medieval", "warrior", "mythology"],
                tips=["Add runes", "Fur and iron materials", "Dragon motifs"]
            ),
            
            "arabian_nights": StylePreset(
                id="arabian_nights",
                name="Arabian Nights Fantasy",
                category=StyleCategory.CULTURAL,
                description="Transform to magical Middle Eastern palace setting",
                visual_elements=["minarets", "geometric patterns", "flying carpets", "oil lamps", "desert palace"],
                color_characteristics=["rich jewel tones", "gold trim", "turquoise", "sunset oranges", "deep purples"],
                technique_details=["islamic geometry", "arabesque patterns", "calligraphy", "ornate details"],
                example_prompt="transform to Arabian Nights setting, ornate palace with geometric tile patterns, minarets and onion domes, people in flowing robes and turbans, magic carpets, brass oil lamps, intricate arabesque decorations, fountain courtyards, silk curtains, golden treasures, desert oasis backdrop",
                compatible_with=["fantasy", "desert", "magical"],
                tips=["Geometric patterns", "Rich jewel colors", "Ornate details"]
            ),
            
            "celtic_druid": StylePreset(
                id="celtic_druid",
                name="Celtic Druid Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Celtic mystical setting",
                visual_elements=["stone circles", "celtic knots", "druid robes", "ancient oaks", "mist"],
                color_characteristics=["forest green", "stone grey", "mystic blue", "earth browns"],
                technique_details=["knotwork patterns", "stone carving", "natural magic", "circular designs"],
                example_prompt="transform to Celtic druid setting, ancient stone circles like Stonehenge, people in hooded druid robes, celtic knotwork patterns, sacred oak groves, mystical mist, carved standing stones with spiral designs, torcs and bronze jewelry, woad face paint, ritual fires, connection to nature magic",
                compatible_with=["fantasy", "mystical", "ancient"],
                tips=["Celtic knot patterns", "Stone circles", "Mystical atmosphere"]
            ),
            
            "chinese_imperial": StylePreset(
                id="chinese_imperial",
                name="Chinese Imperial Dynasty",
                category=StyleCategory.CULTURAL,
                description="Transform to Chinese imperial palace setting",
                visual_elements=["forbidden city", "dragon motifs", "silk robes", "jade ornaments", "pagoda roofs"],
                color_characteristics=["imperial yellow", "jade green", "lacquer red", "gold accents"],
                technique_details=["dragon symbolism", "calligraphy", "porcelain", "silk painting"],
                example_prompt="transform to Chinese imperial setting, Forbidden City architecture with golden roofs, people in silk hanfu robes with dragon embroidery, jade ornaments, red lacquered pillars, imperial throne room, porcelain vases, chinese calligraphy scrolls, guardian lion statues, traditional garden with koi pond",
                compatible_with=["asian", "royal", "traditional"],
                tips=["Dragon motifs", "Imperial colors", "Hierarchical details"]
            ),
            
            # Technique Styles
            "double_exposure": StylePreset(
                id="double_exposure",
                name="Double Exposure",
                category=StyleCategory.PHOTOGRAPHY,
                description="Multiple images blended into one",
                visual_elements=["overlapping images", "transparency effects", "silhouettes", "layered compositions"],
                color_characteristics=["blended tones", "gradient overlays", "ethereal colors"],
                technique_details=["image layering", "masking", "blend modes"],
                example_prompt="create double exposure photography technique with two overlapping transparent images, silhouette masking, and ethereal blended composition",
                compatible_with=["artistic", "portrait", "surreal"],
                tips=["Use contrasting images", "Focus on silhouettes"]
            ),
            
            "tilt_shift": StylePreset(
                id="tilt_shift",
                name="Tilt-Shift Photography",
                category=StyleCategory.PHOTOGRAPHY,
                description="Miniature world effect",
                visual_elements=["selective focus", "miniature appearance", "toy-like quality", "aerial view"],
                color_characteristics=["saturated colors", "high contrast", "vivid tones"],
                technique_details=["depth of field manipulation", "perspective control", "blur gradients"],
                example_prompt="create tilt-shift photography with miniature diorama effect, selective focus blur, high saturation, and toy-like appearance",
                compatible_with=["architectural", "cityscape", "whimsical"],
                tips=["Shoot from above", "Increase color saturation"]
            ),
            
            "long_exposure": StylePreset(
                id="long_exposure",
                name="Long Exposure",
                category=StyleCategory.PHOTOGRAPHY,
                description="Motion blur and light trails",
                visual_elements=["motion blur", "light trails", "smooth water", "star trails"],
                color_characteristics=["ethereal glows", "streaked lights", "smooth gradients"],
                technique_details=["time accumulation", "motion capture", "light painting"],
                example_prompt="create long exposure photography with smooth motion blur, streaming light trails, silky water, and time-lapse effects",
                compatible_with=["night", "landscape", "urban"],
                tips=["Capture movement", "Use for water/clouds"]
            ),
            
            # Period Styles
            "renaissance": StylePreset(
                id="renaissance",
                name="Renaissance Art",
                category=StyleCategory.ART_MOVEMENTS,
                description="15th-16th century European art",
                visual_elements=["religious themes", "classical poses", "detailed drapery", "architectural elements"],
                color_characteristics=["rich earth tones", "gold accents", "deep shadows"],
                technique_details=["sfumato", "chiaroscuro", "perspective mastery"],
                example_prompt="convert to Italian Renaissance painting with classical triangular composition, sfumato technique, rich earth tones, and chiaroscuro lighting",
                compatible_with=["baroque", "classical", "religious"],
                tips=["Use classical composition", "Add religious symbolism"]
            ),
            
            "art_nouveau": StylePreset(
                id="art_nouveau",
                name="Art Nouveau",
                category=StyleCategory.ART_MOVEMENTS,
                description="Elegant 1890s-1910s decorative art with organic forms",
                visual_elements=["flowing organic lines", "natural motifs", "female figures", "decorative borders"],
                color_characteristics=["muted earth tones", "gold accents", "pastel highlights", "natural palette"],
                technique_details=["sinuous lines", "floral patterns", "typography integration", "poster art"],
                example_prompt="convert to Art Nouveau style with flowing organic lines, natural floral motifs, elegant female figures, decorative borders with gold accents, Alphonse Mucha inspired composition, muted earth tone palette with pastel highlights",
                compatible_with=["mucha_style", "vintage_poster", "decorative_art"],
                tips=["Use flowing S-curves", "Include floral elements", "Add decorative borders"]
            ),
            
            "art_deco": StylePreset(
                id="art_deco",
                name="Art Deco",
                category=StyleCategory.ART_MOVEMENTS,
                description="1920s-30s decorative art and architecture style",
                visual_elements=["geometric patterns", "metallic elements", "sunburst motifs", "zigzag designs"],
                color_characteristics=["gold and black", "jewel tones", "metallic accents"],
                technique_details=["symmetrical design", "luxurious materials", "streamlined forms"],
                example_prompt="convert to Art Deco 1920s style with geometric patterns, metallic gold accents, symmetrical design, sunburst motifs, zigzag patterns, luxurious materials, Great Gatsby aesthetic",
                compatible_with=["vintage", "luxury", "gatsby"],
                tips=["Use geometric patterns", "Add metallic elements"]
            ),
            
            "bauhaus": StylePreset(
                id="bauhaus",
                name="Bauhaus Design",
                category=StyleCategory.ART_MOVEMENTS,
                description="1920s German design school emphasizing function and geometric forms",
                visual_elements=["primary shapes", "grid layouts", "sans-serif typography", "minimal ornamentation"],
                color_characteristics=["primary colors", "black and white", "red yellow blue", "geometric color blocks"],
                technique_details=["form follows function", "geometric abstraction", "industrial materials", "unified design"],
                example_prompt="convert to Bauhaus design style with primary geometric shapes, red yellow blue color scheme, grid-based composition, sans-serif typography, form follows function aesthetic, minimal German design school approach",
                compatible_with=["constructivism", "minimalist", "swiss_design"],
                tips=["Use primary colors only", "Stick to basic shapes", "Function over decoration"]
            ),
            
            "victorian": StylePreset(
                id="victorian",
                name="Victorian Era",
                category=StyleCategory.ART_MOVEMENTS,
                description="19th century ornate style",
                visual_elements=["ornate details", "floral patterns", "complex textures", "layered clothing"],
                color_characteristics=["deep jewel tones", "burgundy and gold", "rich fabrics"],
                technique_details=["intricate ornamentation", "pattern mixing", "formal composition"],
                example_prompt="convert to Victorian era 1890s with ornate decorative details, layered rich textures, dark jewel tones, and formal composition",
                compatible_with=["steampunk", "gothic", "formal"],
                tips=["Layer details", "Use rich, dark colors"]
            ),
            
            # Famous Artist Styles
            "van_gogh": StylePreset(
                id="van_gogh",
                name="Van Gogh Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Post-impressionist style of Vincent van Gogh with expressive brushwork",
                visual_elements=["swirling brushstrokes", "thick impasto", "dynamic movement", "emotional intensity"],
                color_characteristics=["vibrant yellows", "deep blues", "complementary contrasts", "luminous colors"],
                technique_details=["impasto technique", "directional brushwork", "expressive distortion", "visible texture"],
                example_prompt="convert to Van Gogh style painting with swirling expressive brushstrokes, thick impasto texture, vibrant yellows and deep blues, dynamic movement in sky and landscapes, post-impressionist emotional intensity, visible paint texture",
                compatible_with=["impressionism", "expressionism", "fauvism"],
                tips=["Emphasize brushstroke movement", "Use thick paint texture", "Vibrant color contrasts"]
            ),
            
            "monet": StylePreset(
                id="monet",
                name="Monet Impressionism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Claude Monet's impressionist style capturing light and atmosphere",
                visual_elements=["broken color", "light effects", "atmospheric perspective", "loose brushwork"],
                color_characteristics=["light pastels", "color reflections", "atmospheric blues", "sunrise/sunset tones"],
                technique_details=["plein air painting", "color mixing on canvas", "capturing moments", "light studies"],
                example_prompt="convert to Claude Monet impressionist style, broken color technique, capturing fleeting light effects, loose brushwork with visible strokes, atmospheric perspective, water reflections, garden scenes with dappled sunlight",
                compatible_with=["impressionism", "landscape", "garden_scenes"],
                tips=["Focus on light effects", "Use broken color", "Capture atmosphere"]
            ),
            
            "picasso_cubism": StylePreset(
                id="picasso_cubism",
                name="Picasso Cubism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Pablo Picasso's revolutionary cubist style with geometric fragmentation",
                visual_elements=["geometric fragmentation", "multiple viewpoints", "angular shapes", "abstracted forms"],
                color_characteristics=["muted earth tones", "analytical palette", "browns and grays", "limited colors"],
                technique_details=["analytical cubism", "synthetic cubism", "collage elements", "flattened perspective"],
                example_prompt="convert to Picasso cubist style with geometric fragmentation, multiple simultaneous viewpoints, angular abstracted forms, muted earth tone palette, analytical cubism technique, flattened perspective planes",
                compatible_with=["abstract", "modernism", "avant_garde"],
                tips=["Fragment into geometric shapes", "Show multiple viewpoints", "Use earth tones"]
            ),
            
            "dali_surrealism": StylePreset(
                id="dali_surrealism",
                name="Dalí Surrealism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Salvador Dalí's surrealist style with melting forms and dreamlike imagery",
                visual_elements=["melting objects", "impossible physics", "dreamlike scenes", "precise detail"],
                color_characteristics=["desert yellows", "sky blues", "stark contrasts", "luminous highlights"],
                technique_details=["paranoiac-critical method", "double images", "meticulous realism", "optical illusions"],
                example_prompt="convert to Salvador Dalí surrealist style, melting clocks and objects, dreamlike desert landscape, impossible physics, meticulous photorealistic detail within surreal context, paranoiac-critical double images",
                compatible_with=["surrealism", "dream_art", "fantasy"],
                tips=["Combine realistic detail with surreal elements", "Add melting objects", "Use desert landscapes"]
            ),
            
            "hokusai": StylePreset(
                id="hokusai",
                name="Hokusai Ukiyo-e",
                category=StyleCategory.CULTURAL,
                description="Katsushika Hokusai's iconic Japanese woodblock print style",
                visual_elements=["wave patterns", "Mount Fuji", "dynamic composition", "nature forces"],
                color_characteristics=["Prussian blue", "limited palette", "gradient skies", "white foam"],
                technique_details=["woodblock printing", "bold outlines", "flat color areas", "dynamic movement"],
                example_prompt="convert to Hokusai ukiyo-e style, bold woodblock print aesthetic, Great Wave composition, Prussian blue dominant, Mount Fuji in background, dynamic natural forces, traditional Japanese art technique",
                compatible_with=["ukiyo_e", "japanese_art", "woodblock"],
                tips=["Use Prussian blue prominently", "Add dynamic wave patterns", "Include Mount Fuji"]
            ),
            
            "klimt": StylePreset(
                id="klimt",
                name="Gustav Klimt Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Art Nouveau master Gustav Klimt's golden decorative style",
                visual_elements=["gold leaf", "decorative patterns", "symbolic elements", "sensual figures"],
                color_characteristics=["gold dominant", "rich jewel tones", "metallic accents", "warm palette"],
                technique_details=["gold leaf application", "Byzantine influence", "decorative symbolism", "pattern mixing"],
                example_prompt="convert to Gustav Klimt style with gold leaf patterns, decorative Byzantine motifs, sensual Art Nouveau figures, rich jewel tone accents, symbolic elements, mosaic-like backgrounds, Vienna Secession aesthetic",
                compatible_with=["art_nouveau", "symbolism", "decorative_art"],
                tips=["Use gold extensively", "Add decorative patterns", "Mix geometric and organic"]
            ),
            
            # Cultural Styles
            "ukiyo_e": StylePreset(
                id="ukiyo_e",
                name="Ukiyo-e",
                category=StyleCategory.CULTURAL,
                description="Japanese woodblock print style",
                visual_elements=["flat colors", "bold outlines", "wave patterns", "nature scenes"],
                color_characteristics=["limited palette", "indigo blues", "natural tones"],
                technique_details=["woodblock texture", "layered printing", "precise lines"],
                example_prompt="convert to Ukiyo-e Japanese woodblock print with flat color areas, bold black outlines, traditional wave patterns, and limited color palette",
                compatible_with=["japanese", "traditional", "nature"],
                tips=["Use limited colors", "Add wave patterns"]
            ),
            
            "aboriginal": StylePreset(
                id="aboriginal",
                name="Aboriginal Dot Painting",
                category=StyleCategory.CULTURAL,
                description="Australian indigenous art style",
                visual_elements=["dot patterns", "dreamtime symbols", "earth connections", "circular motifs"],
                color_characteristics=["earth tones", "ochre colors", "natural pigments"],
                technique_details=["dot application", "symbolic patterns", "storytelling elements"],
                example_prompt="convert to Aboriginal dot painting with intricate circular patterns, ochre earth tones, and dreamtime symbolism",
                compatible_with=["tribal", "spiritual", "earth"],
                tips=["Use dots for texture", "Include symbolic elements"]
            ),
            
            "aztec": StylePreset(
                id="aztec",
                name="Aztec Art",
                category=StyleCategory.CULTURAL,
                description="Pre-Columbian Mesoamerican style",
                visual_elements=["geometric patterns", "feathered serpents", "sun symbols", "stepped pyramids"],
                color_characteristics=["turquoise and gold", "jade green", "blood red"],
                technique_details=["stone carving style", "symbolic imagery", "hierarchical scale"],
                example_prompt="convert to Aztec art with stepped geometric patterns, turquoise and gold colors, symbolic imagery, and pre-Columbian stone carving aesthetics",
                compatible_with=["mayan", "ancient", "symbolic"],
                tips=["Use stepped patterns", "Add symbolic animals"]
            ),
            
            # Additional Game Styles
            "zelda_botw": StylePreset(
                id="zelda_botw",
                name="Zelda: Breath of the Wild",
                category=StyleCategory.GAME,
                description="3D cel-shaded adventure game with painterly aesthetics",
                visual_elements=["cel shading", "watercolor textures", "soft edges", "environmental storytelling"],
                color_characteristics=["natural palette", "soft pastels", "atmospheric haze", "vibrant accents"],
                technique_details=["painterly rendering", "soft shadows", "impressionistic backgrounds"],
                example_prompt="convert to Zelda Breath of the Wild 3D cel-shaded style with watercolor textures, painterly atmosphere, and stylized proportions",
                compatible_with=["genshin_impact", "studio_ghibli", "watercolor"],
                tips=["Great for landscapes", "Add atmospheric effects"]
            ),
            
            "overwatch": StylePreset(
                id="overwatch",
                name="Overwatch Style",
                category=StyleCategory.GAME,
                description="Stylized 3D hero shooter with Pixar-like character design",
                visual_elements=["chunky proportions", "vibrant colors", "stylized characters", "clean shapes"],
                color_characteristics=["bright saturated colors", "team color coding", "high contrast"],
                technique_details=["smooth rendering", "exaggerated features", "heroic poses"],
                example_prompt="convert to Overwatch 3D hero shooter style with Pixar-inspired character design, vibrant team colors, and smooth rendering",
                compatible_with=["pixar", "fortnite", "team_fortress"],
                tips=["Emphasize character personality", "Use bold colors"]
            ),
            
            "minecraft": StylePreset(
                id="minecraft",
                name="Minecraft Style",
                category=StyleCategory.GAME,
                description="Blocky 3D voxel art style",
                visual_elements=["cubic blocks", "pixelated textures", "voxel construction", "simple geometry"],
                color_characteristics=["flat colors", "pixel texture", "limited palette per block"],
                technique_details=["voxel art", "block-based construction", "orthogonal shapes"],
                example_prompt="convert to Minecraft 3D blocky voxel style with cubic shapes, 16x16 pixelated textures, and orthogonal geometry",
                compatible_with=["pixel_art", "lego", "voxel"],
                tips=["Everything must be cubic", "Use 16x16 texture resolution"]
            ),
            
            "persona5": StylePreset(
                id="persona5",
                name="Persona 5 Style",
                category=StyleCategory.GAME,
                description="Stylish JRPG with bold UI-inspired art and high contrast aesthetics",
                visual_elements=["high contrast", "red black white palette", "stylish UI elements", "dynamic angles"],
                color_characteristics=["red and black dominant", "stark white accents", "bold shadows"],
                technique_details=["graphic design influence", "angular compositions", "pop art elements"],
                example_prompt="convert to Persona 5 UI-inspired art with high contrast red/black/white palette, stylish graphic design elements, and dynamic angular compositions",
                compatible_with=["anime_modern", "noir", "graphic_design"],
                tips=["Use red/black/white palette", "Add dynamic angles"]
            ),
            
            "league_of_legends": StylePreset(
                id="league_of_legends",
                name="League of Legends Style",
                category=StyleCategory.GAME,
                description="MOBA game 2D splash art illustration style",
                visual_elements=["dynamic poses", "magical effects", "detailed armor", "fantasy elements"],
                color_characteristics=["vibrant magic colors", "glowing effects", "rich details"],
                technique_details=["splash art composition", "painterly style", "epic scale"],
                example_prompt="convert to League of Legends splash art illustration with dynamic action pose, magical particle effects, and painterly digital rendering",
                compatible_with=["fantasy", "digital_art", "magic"],
                tips=["Focus on dynamic action", "Add magical particle effects"]
            ),
            
            # Additional Artistic Styles with Famous Artists
            "van_gogh": StylePreset(
                id="van_gogh",
                name="Van Gogh Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Post-impressionist with swirling brushstrokes",
                visual_elements=["swirling brushstrokes", "thick impasto", "emotional intensity", "dynamic movement"],
                color_characteristics=["vibrant yellows and blues", "complementary colors", "emotional palette"],
                technique_details=["visible brushwork", "expressive technique", "textured paint"],
                example_prompt="convert to Van Gogh post-impressionist painting with dynamic swirling brushstrokes like Starry Night, vibrant yellows and blues, thick impasto texture",
                compatible_with=["impressionist", "oil_painting", "expressive"],
                tips=["Emphasize movement and emotion", "Use thick paint texture"]
            ),
            
            "monet": StylePreset(
                id="monet",
                name="Monet Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Impressionist master of light and color",
                visual_elements=["light reflections", "water lilies", "atmospheric effects", "broken color"],
                color_characteristics=["soft pastels", "light effects", "natural colors", "atmospheric perspective"],
                technique_details=["plein air painting", "color patches", "light studies"],
                example_prompt="convert to Claude Monet impressionist painting with soft natural light like Water Lilies series, broken color patches, and atmospheric perspective",
                compatible_with=["impressionist", "watercolor", "landscape"],
                tips=["Focus on light and atmosphere", "Use broken color technique"]
            ),
            
            "picasso": StylePreset(
                id="picasso",
                name="Picasso Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Cubist art with geometric abstraction",
                visual_elements=["geometric shapes", "multiple perspectives", "fragmented forms", "angular faces"],
                color_characteristics=["earthy tones", "limited palette", "bold contrasts"],
                technique_details=["cubist fragmentation", "analytical approach", "geometric abstraction"],
                example_prompt="convert to Picasso analytical cubist painting with geometric fragmentation, multiple simultaneous viewpoints, and muted earth tones",
                compatible_with=["abstract", "modern_art", "geometric"],
                tips=["Fragment the subject", "Show multiple viewpoints"]
            ),
            
            "dali": StylePreset(
                id="dali",
                name="Salvador Dali Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Surrealist art with melting reality",
                visual_elements=["melting objects", "impossible physics", "dreamlike scenes", "precise detail"],
                color_characteristics=["desert tones", "stark contrasts", "surreal lighting"],
                technique_details=["photorealistic surrealism", "impossible combinations", "symbolic imagery"],
                example_prompt="convert to Salvador Dali surrealist painting with melting clocks like The Persistence of Memory, Catalonian desert landscapes, dreamlike imagery, and photorealistic detail",
                compatible_with=["surreal", "dreamlike", "symbolic"],
                tips=["Combine realistic rendering with impossible elements", "Add symbolic objects"]
            ),
            
            "banksy": StylePreset(
                id="banksy",
                name="Banksy Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="British street art with stencil technique and social commentary",
                visual_elements=["stencil art", "high contrast", "urban setting", "political message"],
                color_characteristics=["black and white base", "selective color", "spray paint effect"],
                technique_details=["stencil technique", "graffiti style", "minimalist approach"],
                example_prompt="convert to Banksy street art with sharp stencil technique, high contrast black and white, selective red accents, and political urban graffiti aesthetic",
                compatible_with=["graffiti", "urban", "minimalist"],
                tips=["Use stencil-like shapes", "Add social commentary elements"]
            ),
            
            # Comic and Graphic Novel Styles
            "comic_book": StylePreset(
                id="comic_book",
                name="Comic Book Style",
                category=StyleCategory.COMIC,
                description="Classic American comic book art with bold lines and Ben Day dots",
                visual_elements=["bold outlines", "speech bubbles", "action lines", "Ben Day dots"],
                color_characteristics=["primary colors", "high contrast", "flat colors", "halftone patterns"],
                technique_details=["ink and color separation", "panel layout", "dynamic poses"],
                example_prompt="convert to comic book style with bold black outlines, Ben Day dot shading, primary colors, and dynamic action poses",
                compatible_with=["pop_art", "superhero", "vintage"],
                tips=["Add speech bubbles for authenticity", "Use action lines for movement"]
            ),
            
            "manga_noir": StylePreset(
                id="manga_noir",
                name="Manga Noir",
                category=StyleCategory.COMIC,
                description="Dark atmospheric manga with heavy black ink and dramatic shadows",
                visual_elements=["heavy shadows", "stark contrast", "cross-hatching", "atmospheric fog"],
                color_characteristics=["black and white only", "deep blacks", "minimal grays"],
                technique_details=["dramatic inking", "noir atmosphere", "psychological tension"],
                example_prompt="convert to noir manga style with heavy black shadows, stark contrast, cross-hatching techniques, and atmospheric tension",
                compatible_with=["noir", "manga", "horror"],
                tips=["Use shadows to create mood", "Focus on dramatic angles"]
            ),
            
            "franco_belgian": StylePreset(
                id="franco_belgian",
                name="Franco-Belgian Comics",
                category=StyleCategory.COMIC,
                description="European comic style like Tintin with clear lines and detailed backgrounds",
                visual_elements=["ligne claire", "detailed backgrounds", "clean lines", "realistic proportions"],
                color_characteristics=["flat colors", "muted palette", "precise coloring"],
                technique_details=["clear line technique", "architectural precision", "European aesthetic"],
                example_prompt="convert to Franco-Belgian comic style with ligne claire technique, detailed backgrounds, and European aesthetic sensibility",
                compatible_with=["adventure", "european", "classic"],
                tips=["Focus on clean, precise lines", "Add detailed environments"]
            ),
            
            # Vintage Animation Styles
            "rubber_hose": StylePreset(
                id="rubber_hose",
                name="1920s Rubber Hose",
                category=StyleCategory.CARTOON,
                description="Early animation style with bendy limbs and bouncy movement",
                visual_elements=["noodle limbs", "pie-cut eyes", "gloves on hands", "continuous curves"],
                color_characteristics=["black and white", "simple shading", "high contrast"],
                technique_details=["fluid animation", "squash and stretch", "bouncy physics"],
                example_prompt="convert to 1920s rubber hose animation with noodle limbs, pie-cut eyes, bouncy movement, and vintage black and white aesthetic",
                compatible_with=["vintage", "classic_cartoon", "steamboat_willie"],
                tips=["Everything should be curved and bouncy", "Add gloves to character hands"]
            ),
            
            "cuphead": StylePreset(
                id="cuphead",
                name="Cuphead Style",
                category=StyleCategory.GAME,
                description="1930s cartoon aesthetic with watercolor backgrounds and rubber hose animation",
                visual_elements=["rubber hose animation", "watercolor backgrounds", "film grain", "vintage imperfections"],
                color_characteristics=["muted watercolors", "sepia tones", "hand-painted look"],
                technique_details=["hand-drawn animation", "traditional cel animation", "1930s aesthetic"],
                example_prompt="convert to Cuphead 1930s cartoon style with rubber hose animation, watercolor backgrounds, film grain, and vintage hand-drawn aesthetic",
                compatible_with=["rubber_hose", "vintage", "watercolor"],
                tips=["Add film grain and scratches", "Use muted watercolor palette"]
            ),
            
            "fleischer": StylePreset(
                id="fleischer",
                name="Fleischer Studios Style",
                category=StyleCategory.CARTOON,
                description="Betty Boop era animation with surreal elements and fluid transformations",
                visual_elements=["surreal transformations", "big eyes", "fluid motion", "impossible physics"],
                color_characteristics=["black and white", "grayscale shading", "vintage film look"],
                technique_details=["rotoscoping influence", "metamorphosis", "dream logic"],
                example_prompt="convert to Fleischer Studios 1930s style with surreal transformations, big expressive eyes, and dreamlike animation logic",
                compatible_with=["rubber_hose", "surreal", "vintage"],
                tips=["Embrace surreal transformations", "Use exaggerated expressions"]
            ),
            
            # Black and White Artistic Styles
            "ink_wash": StylePreset(
                id="ink_wash",
                name="Ink Wash Painting",
                category=StyleCategory.TRADITIONAL_ART,
                description="Traditional East Asian ink wash technique with flowing gradients",
                visual_elements=["flowing ink", "gradient washes", "negative space", "minimal strokes"],
                color_characteristics=["black ink gradients", "white space", "subtle grays"],
                technique_details=["wet brush technique", "spontaneous strokes", "zen aesthetic"],
                example_prompt="convert to traditional ink wash painting with flowing black ink gradients, expressive brushstrokes, and zen-like negative space",
                compatible_with=["minimalist", "japanese", "calligraphy"],
                tips=["Use negative space effectively", "Keep strokes minimal and expressive"]
            ),
            
            "charcoal_sketch": StylePreset(
                id="charcoal_sketch",
                name="Charcoal Drawing",
                category=StyleCategory.TRADITIONAL_ART,
                description="Expressive charcoal drawing with rich blacks and textured strokes",
                visual_elements=["smudged textures", "rough strokes", "paper texture", "dramatic shadows"],
                color_characteristics=["deep blacks", "soft grays", "white highlights"],
                technique_details=["smudging technique", "layered shading", "gestural marks"],
                example_prompt="convert to charcoal drawing with smudged textures, dramatic shadows, visible paper grain, and expressive gestural strokes",
                compatible_with=["sketch", "portrait", "dramatic"],
                tips=["Show paper texture", "Use smudging for soft shadows"]
            ),
            
            "woodcut": StylePreset(
                id="woodcut",
                name="Woodcut Print",
                category=StyleCategory.TRADITIONAL_ART,
                description="Traditional woodblock printing with carved textures and bold contrasts",
                visual_elements=["carved lines", "wood grain", "rough edges", "block printing marks"],
                color_characteristics=["high contrast", "limited colors", "ink texture"],
                technique_details=["carved wood texture", "printmaking aesthetic", "handmade quality"],
                example_prompt="convert to woodcut print style with carved line textures, high contrast black and white, visible wood grain, and handmade printing aesthetic",
                compatible_with=["ukiyo_e", "medieval", "folk_art"],
                tips=["Show carved texture in lines", "Keep shapes bold and simple"]
            ),
            
            "pen_and_ink": StylePreset(
                id="pen_and_ink",
                name="Pen and Ink Illustration",
                category=StyleCategory.TRADITIONAL_ART,
                description="Detailed pen and ink drawing with cross-hatching and stippling",
                visual_elements=["cross-hatching", "stippling", "fine lines", "intricate detail"],
                color_characteristics=["pure black and white", "no grays", "line-based shading"],
                technique_details=["hatching techniques", "line weight variation", "technical precision"],
                example_prompt="convert to pen and ink illustration with detailed cross-hatching, stippling textures, varied line weights, and technical precision",
                compatible_with=["technical", "architectural", "scientific"],
                tips=["Vary line weights for depth", "Use different hatching patterns"]
            ),
            
            # Comprehensive Comic Book Styles
            
            # Classic American Publishers
            "marvel_comics": StylePreset(
                id="marvel_comics",
                name="Marvel Comics Style",
                category=StyleCategory.COMIC,
                description="Classic Marvel superhero comic book art",
                visual_elements=["dynamic action poses", "muscular anatomy", "speed lines", "impact bursts"],
                color_characteristics=["bold primary colors", "dramatic shadows", "glossy costumes", "vibrant effects"],
                technique_details=["Kirby krackle", "dramatic foreshortening", "panel-breaking action"],
                example_prompt="convert to Marvel Comics style with dynamic superhero action, bold muscular anatomy, vibrant primary colors, and Kirby krackle energy effects",
                compatible_with=["comic_book", "superhero", "action"],
                tips=["Emphasize dramatic poses", "Add energy effects and speed lines"]
            ),
            
            "dc_comics": StylePreset(
                id="dc_comics",
                name="DC Comics Style",
                category=StyleCategory.COMIC,
                description="Classic DC superhero comic aesthetics",
                visual_elements=["heroic poses", "iconic silhouettes", "cape dynamics", "city backgrounds"],
                color_characteristics=["primary colors", "darker tones", "atmospheric lighting", "noir influences"],
                technique_details=["clean linework", "crosshatching shadows", "architectural detail"],
                example_prompt="convert to DC Comics style with heroic character poses, clean linework, atmospheric urban backgrounds, and iconic superhero aesthetics",
                compatible_with=["comic_book", "noir", "superhero"],
                tips=["Focus on heroic silhouettes", "Add urban environments"]
            ),
            
            "dark_horse": StylePreset(
                id="dark_horse",
                name="Dark Horse Comics Style",
                category=StyleCategory.COMIC,
                description="Independent comic publisher style with diverse aesthetics",
                visual_elements=["varied art styles", "creative layouts", "genre mixing", "unique character designs"],
                color_characteristics=["genre-dependent palette", "atmospheric colors", "mood-driven choices"],
                technique_details=["experimental techniques", "creator-driven styles", "innovative paneling"],
                example_prompt="convert to Dark Horse Comics style with independent comic aesthetics, creative panel layouts, and genre-blending visual elements",
                compatible_with=["indie_comic", "graphic_novel", "experimental"],
                tips=["Experiment with unique styles", "Mix genres freely"]
            ),
            
            "image_comics": StylePreset(
                id="image_comics",
                name="Image Comics Style",
                category=StyleCategory.COMIC,
                description="Modern independent comic art with creator-owned aesthetics",
                visual_elements=["detailed artwork", "cinematic panels", "mature themes", "diverse styles"],
                color_characteristics=["sophisticated palettes", "digital coloring", "gradient effects", "mood lighting"],
                technique_details=["detailed linework", "digital painting", "mixed media approaches"],
                example_prompt="convert to Image Comics style with detailed modern comic art, cinematic panel composition, sophisticated digital coloring, and mature visual themes",
                compatible_with=["modern_comic", "graphic_novel", "digital_art"],
                tips=["Focus on cinematic storytelling", "Use sophisticated color palettes"]
            ),
            
            # Modern Comic Artists
            "jim_lee": StylePreset(
                id="jim_lee",
                name="Jim Lee Style",
                category=StyleCategory.COMIC,
                description="Detailed superhero art with intricate linework",
                visual_elements=["detailed crosshatching", "dynamic poses", "intricate costumes", "flowing hair"],
                color_characteristics=["vibrant colors", "metallic sheens", "dramatic lighting", "high contrast"],
                technique_details=["precise linework", "detailed rendering", "dynamic composition"],
                example_prompt="convert to Jim Lee comic art style with intricate crosshatching, detailed superhero costumes, dynamic action poses, and precise linework",
                compatible_with=["marvel_comics", "dc_comics", "superhero"],
                tips=["Add intricate line details", "Focus on dynamic anatomy"]
            ),
            
            "alex_ross": StylePreset(
                id="alex_ross",
                name="Alex Ross Style",
                category=StyleCategory.COMIC,
                description="Photorealistic painted comic book art",
                visual_elements=["painted realism", "classical composition", "realistic anatomy", "fabric textures"],
                color_characteristics=["naturalistic colors", "painted lighting", "subtle gradients", "realistic skin tones"],
                technique_details=["gouache painting", "photorealistic rendering", "classical art influences"],
                example_prompt="convert to Alex Ross painted comic style with photorealistic superhero portraits, gouache painting technique, classical composition, and naturalistic lighting",
                compatible_with=["painted_comic", "realistic", "portrait"],
                tips=["Emphasize realistic textures", "Use classical painting techniques"]
            ),
            
            "frank_miller": StylePreset(
                id="frank_miller",
                name="Frank Miller Style",
                category=StyleCategory.COMIC,
                description="High contrast noir-influenced comic art",
                visual_elements=["stark shadows", "minimal lines", "silhouettes", "rain effects"],
                color_characteristics=["high contrast black and white", "selective color", "noir palette", "blood red accents"],
                technique_details=["bold ink work", "negative space", "film noir influences"],
                example_prompt="convert to Frank Miller comic style with stark black and white contrast, bold shadows, noir aesthetics, and minimalist but powerful linework",
                compatible_with=["noir", "sin_city", "dark_comic"],
                tips=["Use extreme contrast", "Emphasize silhouettes"]
            ),
            
            "todd_mcfarlane": StylePreset(
                id="todd_mcfarlane",
                name="Todd McFarlane Style",
                category=StyleCategory.COMIC,
                description="Detailed dark comic art with intricate designs",
                visual_elements=["flowing capes", "chains and spikes", "organic details", "horror elements"],
                color_characteristics=["dark palette", "red accents", "atmospheric shadows", "glowing effects"],
                technique_details=["intricate detail work", "organic flowing lines", "dark fantasy elements"],
                example_prompt="convert to Todd McFarlane Spawn style with intricate organic details, flowing capes and chains, dark atmospheric colors, and horror-influenced design",
                compatible_with=["dark_comic", "horror", "fantasy"],
                tips=["Add intricate organic details", "Use flowing, dynamic elements"]
            ),
            
            # International Comic Styles
            "moebius": StylePreset(
                id="moebius",
                name="Moebius Style",
                category=StyleCategory.COMIC,
                description="French sci-fi comic art with surreal landscapes",
                visual_elements=["detailed line art", "surreal environments", "organic technology", "vast landscapes"],
                color_characteristics=["desert palette", "alien colors", "atmospheric perspective", "subtle gradients"],
                technique_details=["precise linework", "architectural detail", "surrealist influences"],
                example_prompt="convert to Moebius European comic style with detailed line art, surreal sci-fi landscapes, organic alien designs, and atmospheric desert colors",
                compatible_with=["franco_belgian", "scifi", "surreal"],
                tips=["Focus on detailed environments", "Add surreal elements"]
            ),
            
            "herge_tintin": StylePreset(
                id="herge_tintin",
                name="Hergé Tintin Style",
                category=StyleCategory.COMIC,
                description="Clear line Belgian comic style",
                visual_elements=["clear lines", "no shading", "detailed backgrounds", "adventure scenes"],
                color_characteristics=["flat colors", "bright palette", "clear color separation", "no gradients"],
                technique_details=["ligne claire", "uniform line weight", "architectural precision"],
                example_prompt="convert to Hergé Tintin clear line style with uniform black outlines, flat colors, detailed backgrounds, and classic adventure comic aesthetics",
                compatible_with=["franco_belgian", "adventure", "classic_comic"],
                tips=["Keep lines uniform and clear", "Use completely flat colors"]
            ),
            
            "asterix": StylePreset(
                id="asterix",
                name="Asterix Style",
                category=StyleCategory.COMIC,
                description="Humorous Franco-Belgian comic style",
                visual_elements=["exaggerated features", "cartoonish proportions", "dynamic action", "crowd scenes"],
                color_characteristics=["bright colors", "warm palette", "simple shading", "vibrant atmosphere"],
                technique_details=["expressive characters", "comedic timing", "detailed backgrounds"],
                example_prompt="convert to Asterix comic style with exaggerated cartoonish characters, humorous expressions, dynamic action scenes, and vibrant Gaulish village atmosphere",
                compatible_with=["franco_belgian", "humor", "cartoon"],
                tips=["Exaggerate character features", "Add humorous details"]
            ),
            
            # Manga Variant Styles
            "gekiga": StylePreset(
                id="gekiga",
                name="Gekiga Style",
                category=StyleCategory.COMIC,
                description="Dramatic Japanese manga for mature audiences",
                visual_elements=["realistic proportions", "gritty details", "dramatic shadows", "urban settings"],
                color_characteristics=["dark tones", "realistic palette", "heavy blacks", "noir influences"],
                technique_details=["realistic anatomy", "cinematic angles", "psychological depth"],
                example_prompt="convert to Gekiga dramatic manga style with realistic proportions, gritty urban settings, heavy shadows, and mature psychological themes",
                compatible_with=["manga_noir", "seinen", "realistic"],
                tips=["Use realistic proportions", "Add psychological tension"]
            ),
            
            "junji_ito": StylePreset(
                id="junji_ito",
                name="Junji Ito Horror Style",
                category=StyleCategory.COMIC,
                description="Japanese horror manga with disturbing imagery",
                visual_elements=["spiral patterns", "body horror", "detailed textures", "unsettling faces"],
                color_characteristics=["black and white", "high contrast", "detailed hatching", "dark atmosphere"],
                technique_details=["intricate linework", "disturbing imagery", "psychological horror"],
                example_prompt="convert to Junji Ito horror manga style with intricate spiral patterns, disturbing body horror elements, detailed crosshatching, and psychological terror",
                compatible_with=["horror", "manga_noir", "psychological"],
                tips=["Add unsettling details", "Use spiral and organic patterns"]
            ),
            
            # Webcomic and Indie Styles
            "webcomic": StylePreset(
                id="webcomic",
                name="Modern Webcomic Style",
                category=StyleCategory.COMIC,
                description="Digital-first comic style for online publishing",
                visual_elements=["simple lines", "expressive faces", "minimal backgrounds", "reaction faces"],
                color_characteristics=["bright digital colors", "simple shading", "flat colors", "vibrant palette"],
                technique_details=["digital drawing", "simplified style", "meme-friendly expressions"],
                example_prompt="convert to modern webcomic style with simple clean lines, expressive character faces, bright digital colors, and internet-friendly aesthetics",
                compatible_with=["digital_art", "humor", "slice_of_life"],
                tips=["Keep it simple and readable", "Focus on expressions"]
            ),
            
            "graphic_novel": StylePreset(
                id="graphic_novel",
                name="Literary Graphic Novel",
                category=StyleCategory.COMIC,
                description="Sophisticated visual storytelling for adult readers",
                visual_elements=["varied panel layouts", "atmospheric art", "symbolic imagery", "detailed environments"],
                color_characteristics=["muted palette", "mood-driven colors", "watercolor effects", "subtle tones"],
                technique_details=["experimental layouts", "mixed media", "literary themes"],
                example_prompt="convert to literary graphic novel style with sophisticated panel composition, atmospheric watercolor effects, symbolic visual metaphors, and mature storytelling",
                compatible_with=["watercolor", "artistic", "narrative"],
                tips=["Focus on mood and atmosphere", "Use symbolic visual elements"]
            ),
            
            # Comic Art Techniques
            "comic_noir": StylePreset(
                id="comic_noir",
                name="Comic Noir Style",
                category=StyleCategory.COMIC,
                description="Dark crime comic aesthetics",
                visual_elements=["heavy shadows", "rain effects", "urban decay", "smoke wisps"],
                color_characteristics=["limited color palette", "noir lighting", "selective color pops", "dark atmosphere"],
                technique_details=["chiaroscuro lighting", "film noir influence", "dramatic angles"],
                example_prompt="convert to comic noir style with heavy shadow work, rain-slicked streets, limited color palette with selective highlights, and crime story atmosphere",
                compatible_with=["noir", "crime", "dark_comic"],
                tips=["Use dramatic lighting", "Add atmospheric elements like rain or smoke"]
            ),
            
            "painted_comic": StylePreset(
                id="painted_comic",
                name="Painted Comic Style",
                category=StyleCategory.COMIC,
                description="Traditional or digital painted comic art",
                visual_elements=["painterly textures", "soft edges", "realistic rendering", "atmospheric depth"],
                color_characteristics=["rich color depth", "painted gradients", "natural lighting", "artistic palette"],
                technique_details=["painting techniques", "soft rendering", "fine art influences"],
                example_prompt="convert to painted comic style with rich painterly textures, soft edge rendering, atmospheric color depth, and fine art painting techniques",
                compatible_with=["alex_ross", "watercolor", "artistic"],
                tips=["Emphasize painterly textures", "Use atmospheric color"]
            ),
            
            "digital_comic": StylePreset(
                id="digital_comic",
                name="Digital Comic Art",
                category=StyleCategory.COMIC,
                description="Modern digital comic creation techniques",
                visual_elements=["clean digital lines", "gradient effects", "digital textures", "special effects"],
                color_characteristics=["vibrant digital colors", "smooth gradients", "glowing effects", "perfect tones"],
                technique_details=["digital inking", "layer effects", "digital coloring"],
                example_prompt="convert to modern digital comic art with clean vector-like lines, smooth digital gradients, vibrant colors, and polished digital effects",
                compatible_with=["webcomic", "modern_comic", "digital_art"],
                tips=["Use digital effects wisely", "Keep lines clean and crisp"]
            ),
            
            "underground_comic": StylePreset(
                id="underground_comic",
                name="Underground Comix Style",
                category=StyleCategory.COMIC,
                description="Alternative and countercultural comic aesthetics",
                visual_elements=["rough linework", "exaggerated features", "subversive imagery", "raw energy"],
                color_characteristics=["psychedelic colors", "high contrast", "unconventional choices", "bold combinations"],
                technique_details=["loose drawing style", "experimental approaches", "DIY aesthetic"],
                example_prompt="convert to underground comix style with rough expressive linework, exaggerated countercultural imagery, psychedelic colors, and raw DIY energy",
                compatible_with=["alternative", "indie", "experimental"],
                tips=["Embrace imperfection", "Use bold, unconventional choices"]
            )
        }
        return styles
    
    def _initialize_mixers(self) -> Dict[str, List[str]]:
        """Initialize style mixing suggestions"""
        return {
            "artistic_blend": ["oil_painting", "watercolor", "digital_art"],
            "game_fusion": ["cyberpunk_game", "pixel_art", "anime_modern"],
            "photo_artistic": ["cinematic", "oil_painting", "watercolor"],
            "anime_realistic": ["anime_modern", "portrait_studio", "digital_art"]
        }
    
    def _initialize_modifiers(self) -> Dict[str, List[str]]:
        """Initialize quick style modifiers"""
        return {
            "lighting": [
                "dramatic lighting",
                "soft diffused light", 
                "golden hour",
                "neon glow",
                "rim lighting",
                "chiaroscuro"
            ],
            "atmosphere": [
                "moody",
                "ethereal",
                "gritty",
                "dreamlike",
                "nostalgic",
                "mysterious"
            ],
            "color_mood": [
                "vibrant and saturated",
                "muted and subtle",
                "monochromatic",
                "high contrast",
                "pastel tones",
                "dark and moody"
            ],
            "texture": [
                "smooth and polished",
                "rough and textured",
                "soft and blended",
                "sharp and detailed",
                "grainy",
                "glossy"
            ]
        }
    
    def get_style(self, style_id: str) -> Optional[StylePreset]:
        """Get a specific style preset"""
        return self.styles.get(style_id)
    
    def get_styles_by_category(self, category: StyleCategory) -> List[StylePreset]:
        """Get all styles in a category"""
        return [style for style in self.styles.values() if style.category == category]
    
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
        extension_dir = Path(__file__).parent.parent
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
        
        extension_dir = Path(__file__).parent.parent
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
        extension_dir = Path(__file__).parent.parent
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