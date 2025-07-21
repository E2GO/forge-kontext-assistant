"""
Environment Transform styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

ENVIRONMENT_TRANSFORM_STYLES = {
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
            )
        }