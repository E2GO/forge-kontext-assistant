"""
Traditional Art styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

TRADITIONAL_ART_STYLES = {
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
            )
        }