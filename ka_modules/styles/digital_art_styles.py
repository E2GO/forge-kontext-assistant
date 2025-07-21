"""
Digital Art styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

DIGITAL_ART_STYLES = {
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
            )
        }