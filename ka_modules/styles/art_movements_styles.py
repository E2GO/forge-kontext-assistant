"""
Art Movements styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

ART_MOVEMENTS_STYLES = {
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
            )
        }