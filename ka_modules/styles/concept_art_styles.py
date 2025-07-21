"""
Concept Art styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

CONCEPT_ART_STYLES = {
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
            )
        }