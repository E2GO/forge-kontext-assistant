"""
Game styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

GAME_STYLES = {
            "arcane": StylePreset(
                id="arcane",
                name="Arcane Style",
                category=StyleCategory.GAME,
                description="Fortiche's revolutionary painterly 3D animation from League of Legends",
                visual_elements=["hand-painted textures", "3D with 2D aesthetic", "steampunk elements", "detailed facial features", "brushstroke visible"],
                color_characteristics=["rich saturated palette", "neon accents on dark backgrounds", "pink and purple gradients", "atmospheric lighting", "Piltover vs Zaun contrast"],
                technique_details=["painterly 3D rendering", "manual stylized lighting", "2D effects on 3D models", "projection mapping", "custom shaders", "moving painting aesthetic"],
                example_prompt="convert to Arcane game animation style with painterly 3D aesthetic, hand-painted textures with visible brushstrokes, rich atmospheric lighting with neon accents, steampunk design elements, detailed character faces with 2D-style shading on 3D forms, dramatic contrast between light and shadow, moving painting quality",
                compatible_with=["spiderverse", "painterly_3d", "steampunk"],
                tips=["Emphasize painterly textures", "Use dramatic lighting contrasts", "Add hand-painted details"]
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
            
            "clash_of_clans": StylePreset(
                id="clash_of_clans",
                name="Clash of Clans Style",
                category=StyleCategory.GAME,
                description="Supercell's prerendered 3D sprites with Pixar-esque chunky designs",
                visual_elements=["chunky 3D models", "distinctive silhouettes", "exaggerated proportions", "stylized textures", "bold outlines"],
                color_characteristics=["bright high contrast", "strategic color coding", "vibrant primary colors", "clear faction colors", "mobile-optimized palette"],
                technique_details=["3D to 2D sprites", "Substance Painter texturing", "consistent smart materials", "optimized for readability", "scalable designs"],
                example_prompt="convert to Clash of Clans game style with chunky 3D character models rendered as sprites, Pixar-esque proportions, bright high-contrast colors, distinctive silhouettes, bold stylized texturing, mobile-optimized clarity",
                compatible_with=["supercell", "mobile_strategy", "casual"],
                tips=["Make shapes chunky", "Use bright colors", "Ensure readability"]
            ),
            
            "cuphead": StylePreset(
                id="cuphead",
                name="Cuphead Style",
                category=StyleCategory.GAME,
                description="1930s rubber hose animation with hand-drawn frames and film artifacts",
                visual_elements=["rubber hose limbs", "pie-cut eyes", "white gloves", "flowing curves", "vintage imperfections"],
                color_characteristics=["limited color palette", "watercolor backgrounds", "sepia tones", "period-appropriate colors", "film aging"],
                technique_details=["24fps hand-drawn animation", "60,000+ frames", "authentic film grain", "blur and scratches", "watercolor painting"],
                example_prompt="convert to Cuphead game 1930s cartoon style with rubber hose animation, hand-drawn flowing curves, pie-cut eyes, watercolor backgrounds, authentic film grain and scratches, limited period-appropriate color palette, vintage imperfections",
                compatible_with=["rubber_hose", "vintage", "hand_drawn"],
                tips=["Add film imperfections", "Use watercolor backgrounds", "Keep it authentic to era"]
            ),
            
            "cyberpunk_game": StylePreset(
                id="cyberpunk_game",
                name="Cyberpunk 2077 Style",
                category=StyleCategory.GAME,
                description="Neon-drenched futuristic game with ray-traced reflections and volumetric fog",
                visual_elements=["holographic advertisements", "cybernetic implants", "urban decay", "neon signage", "rain-slicked streets"],
                color_characteristics=["cyan, magenta, yellow dominance", "electric blue and hot pink neons", "radioactive green accents", "stark red/black contrasts", "volumetric colored fog"],
                technique_details=["ray-traced reflections", "volumetric lighting", "atmospheric scattering", "dynamic shadows", "screen-space reflections"],
                example_prompt="convert to Cyberpunk 2077 game style with neon-drenched urban environment, cyan and magenta lighting, ray-traced reflections on wet streets, volumetric fog with colored lights, holographic advertisements, cybernetic augmentations, gritty futuristic atmosphere",
                compatible_with=["blade_runner", "deus_ex", "neon_noir"],
                tips=["Layer multiple neon colors", "Add volumetric fog", "Include tech elements"]
            ),
            
            "fallout_poster": StylePreset(
                id="fallout_poster",
                name="Fallout Poster Style",
                category=StyleCategory.GAME,
                description="Post-apocalyptic retro-futurism with 1950s Atompunk propaganda aesthetics",
                visual_elements=["Vault Boy mascot", "atomic age design", "radiation symbols", "retro-futuristic tech", "propaganda posters"],
                color_characteristics=["1950s palette", "faded vintage colors", "rust and decay", "nuclear green glow", "patriotic red white blue"],
                technique_details=["vintage poster design", "halftone printing effects", "distressed textures", "Art Deco influence", "environmental storytelling"],
                example_prompt="convert to Fallout game retro-futuristic poster style with 1950s Atompunk aesthetic, Vault Boy mascot design, atomic age propaganda art, faded vintage colors with rust and decay, nuclear green glow effects, distressed poster textures",
                compatible_with=["retrofuturism", "post_apocalyptic", "propaganda"],
                tips=["Mix 1950s with decay", "Add propaganda elements", "Use atomic age motifs"]
            ),
            
            "fortnite": StylePreset(
                id="fortnite",
                name="Fortnite Style",
                category=StyleCategory.GAME,
                description="Colorful cartoonish battle royale with clear visual hierarchy and Unreal Engine 5 rendering",
                visual_elements=["exaggerated proportions", "smooth stylized models", "clear silhouettes", "expressive animations", "building elements"],
                color_characteristics=["bold saturated colors", "bright pastels", "high contrast", "clear visual hierarchy", "colorful effects"],
                technique_details=["Unreal Engine 5 rendering", "dynamic shadows", "simplified materials", "readable at distance", "optimized for clarity"],
                example_prompt="convert to Fortnite game style with cartoonish exaggerated proportions, smooth stylized 3D models, bold saturated colors with pastel accents, clear readable silhouettes, dynamic shadows, family-friendly aesthetic",
                compatible_with=["cartoon_game", "battle_royale", "stylized_3d"],
                tips=["Prioritize readability", "Use bold colors", "Keep it family-friendly"]
            ),
            
            "genshin_impact": StylePreset(
                id="genshin_impact",
                name="Genshin Impact Style",
                category=StyleCategory.GAME,
                description="Anime-inspired cel-shaded 3D with custom lighting and soft shadows",
                visual_elements=["cel-shaded characters", "thin dark outlines", "anime facial features", "fantasy elemental effects", "detailed environments"],
                color_characteristics=["soft shadow transitions", "vibrant saturated colors", "gradient skies", "warm subsurface colors", "elemental color coding"],
                technique_details=["custom cel-shading", "0.1 hardness shadows", "anisotropic hair", "edge detection outlines", "particle effects"],
                example_prompt="convert to Genshin Impact game anime style with cel-shaded 3D characters, soft shadow transitions, thin dark outlines, vibrant fantasy colors, elemental particle effects, anime-inspired facial features, painterly environment backgrounds",
                compatible_with=["anime_game", "breath_of_wild", "fantasy_anime"],
                tips=["Use soft cel-shading", "Add elemental effects", "Keep colors vibrant"]
            ),
            
            "gta_poster": StylePreset(
                id="gta_poster",
                name="GTA Poster Style",
                category=StyleCategory.GAME,
                description="Grand Theft Auto IV/V box art style by Stephen Bliss with compartmentalized illustrations",
                visual_elements=["compartmentalized grid layout", "character montages", "urban cityscapes", "vehicles prominently featured", "action vignettes"],
                color_characteristics=["GTA IV: muted realistic tones", "GTA V: vibrant cyan-pink gradients", "warm California sunset oranges", "high contrast shadows", "Hollywood-style lighting"],
                technique_details=["Stephen Bliss illustration style", "digital painting over photos", "compartmentalized storytelling", "dramatic perspective", "cinematic composition"],
                example_prompt="convert to GTA game poster style inspired by Stephen Bliss box art, compartmentalized character montage in grid layout, vibrant Hollywood colors with cyan-pink gradients for GTA V aesthetic (or muted tones for GTA IV), urban cityscape backdrop, prominently featured vehicles, dramatic cinematic lighting, action-packed vignettes",
                compatible_with=["crime", "urban", "poster_art"],
                tips=["Use compartmentalized grid layout", "Feature vehicles prominently", "Choose between GTA IV muted or GTA V vibrant style"]
            ),
            
            "hay_day": StylePreset(
                id="hay_day",
                name="Hay Day Style",
                category=StyleCategory.GAME,
                description="Supercell's charming farm game art style",
                visual_elements=["chunky farm assets", "friendly animals", "rustic elements", "cozy atmosphere"],
                color_characteristics=["warm earth tones", "sunny yellows", "natural greens", "soft pastels"],
                technique_details=["stylized 3D models", "hand-painted look", "wholesome design", "farm aesthetic"],
                example_prompt="convert to Hay Day game farm style with chunky 3D models, friendly cartoon animals, warm earth tone colors, hand-painted textures, and cozy farm atmosphere",
                compatible_with=["farmville", "township", "farming_game"],
                tips=["Keep it wholesome and friendly", "Use warm, natural colors"]
            ),
            
            "hollow_knight": StylePreset(
                id="hollow_knight",
                name="Hollow Knight Style",
                category=StyleCategory.GAME,
                description="Hand-drawn 2D metroidvania with gothic atmosphere and painted backgrounds",
                visual_elements=["hand-drawn frame animation", "gothic architecture", "insect characters", "parallax layers", "atmospheric depth"],
                color_characteristics=["monotone dark palette", "vibrant accent colors", "moody lighting", "limited color use", "atmospheric fog"],
                technique_details=["traditional 2D animation", "painted backgrounds", "soft transparent lighting", "parallax scrolling", "particle effects"],
                example_prompt="convert to Hollow Knight game hand-drawn 2D style with gothic architecture, dark monotone palette with vibrant accents, traditional frame-by-frame animation, painted atmospheric backgrounds, moody lighting with soft shadows",
                compatible_with=["metroidvania", "dark_fantasy", "indie_game"],
                tips=["Use limited colors", "Add atmospheric depth", "Keep it moody"]
            ),
            
            "isometric": StylePreset(
                id="isometric",
                name="Isometric Style",
                category=StyleCategory.GAME,
                description="Classic dimetric projection with 2:1 pixel ratio and 30-degree angles",
                visual_elements=["2:1 pixel ratio", "30-degree angles", "tile-based construction", "no perspective distortion", "grid alignment"],
                color_characteristics=["depth through color", "atmospheric perspective", "shadow layering", "clear tile separation", "ambient occlusion"],
                technique_details=["dimetric projection", "45-degree camera", "z-order sorting", "tile optimization", "modular assets"],
                example_prompt="convert to isometric game style with 2:1 pixel ratio dimetric projection, 30-degree angle tiles, no perspective distortion, depth through atmospheric color, grid-based construction, classic RPG aesthetic",
                compatible_with=["strategy", "rpg", "city_builder"],
                tips=["Maintain 2:1 ratio", "Use atmospheric perspective", "Align to grid"]
            ),
            
            "league_of_legends": StylePreset(
                id="league_of_legends",
                name="League of Legends Style",
                category=StyleCategory.GAME,
                description="MOBA splash art with forced perspective and elegant power visualization",
                visual_elements=["dynamic action poses", "forced perspective", "magical effects", "detailed armor", "epic scale"],
                color_characteristics=["faction color coding", "magical glow effects", "atmospheric lighting", "complementary schemes", "power visualization"],
                technique_details=["6-week production pipeline", "photographic reference", "layered painting", "separate effect passes", "cinematic composition"],
                example_prompt="convert to League of Legends game splash art style with dynamic action pose, forced perspective, magical particle effects, detailed fantasy armor, atmospheric lighting, elegant visualization of champion power, epic scale composition",
                compatible_with=["moba", "fantasy", "splash_art"],
                tips=["Use forced perspective", "Add magical effects", "Show power elegantly"]
            ),
            
            "minecraft": StylePreset(
                id="minecraft",
                name="Minecraft Style",
                category=StyleCategory.GAME,
                description="Voxel-based blocky world with 16×16 pixel textures and cubic construction",
                visual_elements=["cubic voxel blocks", "16×16 textures", "blocky characters", "pixelated details", "modular world"],
                color_characteristics=["bright saturated colors", "limited palette per block", "high contrast", "texture-based variety", "simple gradients"],
                technique_details=["voxel construction", "optimized chunk rendering", "simple lighting model", "procedural generation", "block-based physics"],
                example_prompt="convert to Minecraft game voxel style with cubic blocks, 16×16 pixel textures, bright saturated colors, blocky character models, pixelated aesthetic, modular construction, simple geometric shapes",
                compatible_with=["voxel", "sandbox", "building_game"],
                tips=["Everything must be cubic", "Use 16×16 textures", "Keep it blocky"]
            ),
            
            "mobile_game_style": StylePreset(
                id="mobile_game_style",
                name="Mobile Game 2D Style",
                category=StyleCategory.GAME,
                description="Professional mobile game illustration with vector clarity and material rendering",
                visual_elements=["clean vector shapes", "oversized heads", "expressive eyes", "rounded designs", "layered compositions"],
                color_characteristics=["limited palettes", "bright primaries", "high contrast", "solid colors", "material-specific rendering"],
                technique_details=["vector scalability", "optimized for small screens", "clear silhouettes", "modular design", "efficient rendering"],
                example_prompt="convert to professional mobile game 2D style with clean vector art, oversized character heads, bright primary colors, clear readable designs optimized for small screens, material-specific rendering for gold, wood, and gems",
                compatible_with=["casual_game", "match3", "mobile"],
                tips=["Prioritize clarity", "Use vector shapes", "Optimize for mobile"]
            ),
            
            "overwatch": StylePreset(
                id="overwatch",
                name="Overwatch Style",
                category=StyleCategory.GAME,
                description="Stylized hero shooter with Pixar-like appeal and readable character design",
                visual_elements=["exaggerated proportions", "unique silhouettes", "chunky designs", "expressive faces", "hero-specific details"],
                color_characteristics=["bright optimistic palette", "character color coding", "vibrant team colors", "clean material definition", "mood lighting"],
                technique_details=["PBR workflow", "film-quality rendering", "readable in combat", "strong shape language", "animation principles"],
                example_prompt="convert to Overwatch game hero shooter style with Pixar-like character appeal, exaggerated heroic proportions, unique readable silhouettes, bright optimistic colors, PBR materials, strong shape language for instant recognition",
                compatible_with=["hero_shooter", "team_based", "pixar_style"],
                tips=["Create unique silhouettes", "Use hero color coding", "Emphasize readability"]
            ),
            
            "persona5": StylePreset(
                id="persona5",
                name="Persona 5 Style",
                category=StyleCategory.GAME,
                description="Stylish JRPG with punk aesthetics and dynamic UI-inspired visuals",
                visual_elements=["oblique angles", "papercutting aesthetic", "halftone patterns", "dynamic UI elements", "silhouette art"],
                color_characteristics=["red, black, white palette", "high contrast", "passionate red dominance", "monochrome with accents", "punk aesthetics"],
                technique_details=["angular design language", "constant UI animation", "manga panel layouts", "typography integration", "stylish transitions"],
                example_prompt="convert to Persona 5 game style with bold red, black and white color scheme, oblique angular shapes, halftone character silhouettes, dynamic UI elements, punk rebellion aesthetic, high contrast papercutting visual style",
                compatible_with=["jrpg", "anime", "stylish"],
                tips=["Use red/black/white only", "Add oblique angles", "Include halftone patterns"]
            ),
            
            "pixel_art": StylePreset(
                id="pixel_art",
                name="Pixel Art",
                category=StyleCategory.GAME,
                description="Retro pixel art with specific bit-depth limitations and dithering techniques",
                visual_elements=["visible pixel blocks", "limited color palette", "dithering patterns", "sprite-based art", "grid construction"],
                color_characteristics=["8-bit: 256 colors max", "16-bit: 65,536 colors", "indexed color palettes", "no anti-aliasing", "palette swapping"],
                technique_details=["manual pixel placement", "dithering for gradients", "sub-pixel animation", "tile-based backgrounds", "sprite limitations"],
                example_prompt="convert to pixel art game style with visible pixel blocks, limited color palette, dithering for shading, 16-bit era graphics, sprite-based character design, indexed colors without anti-aliasing, retro game aesthetic",
                compatible_with=["retro_game", "8bit", "16bit"],
                tips=["Specify bit depth", "Use dithering patterns", "Limit color count"]
            ),
            
            "ps1_graphics": StylePreset(
                id="ps1_graphics",
                name="PS1 Graphics",
                category=StyleCategory.GAME,
                description="Low-poly 3D with vertex snapping and texture warping artifacts",
                visual_elements=["~500 triangle models", "vertex color lighting", "texture warping", "polygon jittering", "jagged edges"],
                color_characteristics=["limited texture resolution", "vertex color blending", "no texture filtering", "dithered transparency", "flat shading"],
                technique_details=["affine texture mapping", "no z-buffer", "vertex snapping", "Gouraud shading", "low resolution"],
                example_prompt="convert to PS1 game low-poly graphics style with ~500 triangle character models, vertex color lighting, texture warping artifacts, polygon jittering, pixelated textures, no texture filtering, retro 3D aesthetic",
                compatible_with=["retro_3d", "horror_game", "nostalgic"],
                tips=["Keep poly count very low", "Add texture warping", "Use vertex colors"]
            ),
            
            "red_dead": StylePreset(
                id="red_dead",
                name="Red Dead Redemption Style",
                category=StyleCategory.GAME,
                description="Photorealistic western with Hudson River School landscape painting influence",
                visual_elements=["photorealistic textures", "diverse terrains", "period-accurate details", "atmospheric weather", "wildlife"],
                color_characteristics=["natural earth tones", "golden hour lighting", "dusty atmospherics", "muted color grading", "organic shadows"],
                technique_details=["photogrammetry assets", "volumetric clouds", "dynamic weather", "realistic materials", "atmospheric perspective"],
                example_prompt="convert to Red Dead Redemption game photorealistic western style, natural lighting inspired by Hudson River School paintings, dusty atmospheric effects, period-accurate details, golden hour lighting, realistic terrain with organic shadows",
                compatible_with=["western", "realistic", "period_piece"],
                tips=["Use natural lighting", "Add atmospheric dust", "Include period details"]
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
            
            "zelda_botw": StylePreset(
                id="zelda_botw",
                name="Zelda: Breath of the Wild",
                category=StyleCategory.GAME,
                description="Cel-shaded adventure with gouache painting influence and environmental storytelling",
                visual_elements=["cel-shaded models", "painterly textures", "soft edges", "environmental details", "weather effects"],
                color_characteristics=["distinct biome palettes", "orange shrine glow", "red malice effects", "natural color gradients", "atmospheric fog"],
                technique_details=["hybrid 2D/3D style", "gouache painting influence", "en plein air aesthetic", "optimized open-world rendering", "painterly post-processing"],
                example_prompt="convert to Zelda Breath of the Wild game cel-shaded style with painterly gouache textures, soft edges, distinct environmental color palettes, orange glowing shrines, atmospheric perspective, watercolor-inspired landscapes",
                compatible_with=["adventure", "open_world", "nintendo"],
                tips=["Use painterly textures", "Add environmental storytelling", "Apply soft cel-shading"]
            )
        }