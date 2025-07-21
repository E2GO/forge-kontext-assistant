"""
Material Transform styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

MATERIAL_TRANSFORM_STYLES = {
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
            )
        }