"""
Movie styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

MOVIE_STYLES = {
            "blade_runner": StylePreset(
                id="blade_runner",
                name="Blade Runner Style",
                category=StyleCategory.MOVIE,
                description="Dystopian sci-fi with neon lights, perpetual rain, and atmospheric fog",
                visual_elements=["neon signs", "rain and reflections", "flying vehicles", "urban decay", "holographic ads"],
                color_characteristics=["dark blue base", "neon accents", "cyan and magenta", "atmospheric haze", "wet surface reflections"],
                technique_details=["multiple light sources", "practical lighting", "fog diffusion", "moving lights", "layered atmosphere"],
                example_prompt="convert to Blade Runner dystopian style with dark blue atmosphere drenched in neon lights, perpetual rain with wet street reflections, dense fog diffusing light, giant holographic advertisements, urban decay with Asian neon signage",
                compatible_with=["cyberpunk", "dystopia", "neo_noir"],
                tips=["Layer neon on darkness", "Add rain and fog", "Use practical light sources"]
            ),
            
            "hitchcock": StylePreset(
                id="hitchcock",
                name="Hitchcock Style",
                category=StyleCategory.MOVIE,
                description="Suspenseful thriller cinematography with psychological shadow play",
                visual_elements=["ominous shadows", "voyeuristic angles", "ordinary turned sinister", "spiral motifs", "birds or heights"],
                color_characteristics=["strategic lighting", "shadow psychology", "highlighting objects", "contrast for tension", "noir influences"],
                technique_details=["dolly zoom", "series of close-ups", "overhead shots", "roaming camera", "precise framing"],
                example_prompt="convert to Hitchcock suspense style with dramatic shadow placement revealing character psychology, voyeuristic camera angles, ordinary objects highlighted ominously, dolly zoom effect, building tension through anticipation",
                compatible_with=["thriller", "suspense", "psychological"],
                tips=["Use shadows psychologically", "Create voyeuristic angles", "Build anticipation"]
            ),
            
            "matrix": StylePreset(
                id="matrix",
                name="Matrix Style",
                category=StyleCategory.MOVIE,
                description="Digital rain aesthetic with bullet time and green-tinted cyberpunk world",
                visual_elements=["digital rain code", "bullet time", "black leather", "sunglasses", "wire fu action"],
                color_characteristics=["matrix green tint", "desaturated reality", "CRT monitor glow", "cold vs warm worlds", "digital green"],
                technique_details=["360-degree arrays", "time slice photography", "green gel lighting", "virtual cinematography", "wire work"],
                example_prompt="convert to Matrix style with green-tinted digital world, falling code rain with Japanese characters, bullet time frozen motion effect, black leather and sunglasses aesthetic, cyberpunk wire-fu action, CRT monitor green glow",
                compatible_with=["cyberpunk", "virtual_reality", "action"],
                tips=["Apply green tint", "Add digital rain", "Use bullet time effects"]
            ),
            
            "noir": StylePreset(
                id="noir",
                name="Film Noir",
                category=StyleCategory.MOVIE,
                description="Classic detective film style with chiaroscuro lighting and venetian blind shadows",
                visual_elements=["venetian blind shadows", "cigarette smoke", "fedoras and trench coats", "urban night scenes", "femme fatales"],
                color_characteristics=["high contrast B&W", "deep blacks", "stark whites", "minimal grays", "dramatic shadows"],
                technique_details=["chiaroscuro lighting", "low-key setup", "Dutch angles", "single hard light", "perpendicular framing"],
                example_prompt="convert to film noir style with high contrast black and white, venetian blind shadows creating bar patterns, cigarette smoke atmosphere, low-key lighting with single hard light source, Dutch angles, urban night setting with wet streets",
                compatible_with=["detective", "crime", "1940s"],
                tips=["Use venetian blind shadows", "Keep contrast extreme", "Add smoke atmosphere"]
            ),
            
            "star_wars": StylePreset(
                id="star_wars",
                name="Star Wars Style",
                category=StyleCategory.MOVIE,
                description="Epic space opera with used universe aesthetic and mythic scope",
                visual_elements=["lightsabers", "space battles", "alien worlds", "droids and creatures", "force effects"],
                color_characteristics=["desert tans", "space blacks", "lightsaber glows", "sunset twins", "imperial grays"],
                technique_details=["motion control", "matte paintings", "practical effects", "epic wide shots", "used universe look"],
                example_prompt="convert to Star Wars space opera style with epic wide shots of alien worlds, used lived-in aesthetic not shiny and new, lightsaber glow effects, space battles with practical model aesthetics, binary sunset lighting",
                compatible_with=["space_opera", "scifi_fantasy", "epic"],
                tips=["Make it lived-in", "Use epic scale", "Add mystical elements"]
            ),
            
            "wes_anderson": StylePreset(
                id="wes_anderson",
                name="Wes Anderson Style",
                category=StyleCategory.MOVIE,
                description="Perfectly symmetrical compositions with pastel palettes and planimetric framing",
                visual_elements=["perfect symmetry", "centered subjects", "vintage props", "quirky details", "dollhouse aesthetic"],
                color_characteristics=["distinct pastels", "complementary colors", "warm oranges with cool blues", "premeditated palettes", "faded vintage tones"],
                technique_details=["planimetric shots", "90-degree pans", "40mm anamorphic", "flat compositions", "precise choreography"],
                example_prompt="convert to Wes Anderson style with perfect symmetrical composition, centered framing, distinct pastel color palette, planimetric camera angle, vintage props and details, dollhouse-like staging, whimsical storybook aesthetic",
                compatible_with=["quirky", "vintage", "symmetrical"],
                tips=["Center everything perfectly", "Use specific pastels", "Add quirky details"]
            )
        }