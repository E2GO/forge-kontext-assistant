"""
Game styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

GAME_STYLES = {
            "arcane": StylePreset(
                id="arcane",
                name="Arcane Style", 
                category=StyleCategory.GAME,
                description="Fortiche's revolutionary mixed-media animation from League of Legends adult series",
                visual_elements=["oil painting textures on 3D models", "hand-drawn 2D effects", "rough tactile surfaces", "expressive cel-shaded faces", "living painting aesthetic"],
                color_characteristics=["Piltover: warm golds and brass tones", "Zaun: toxic greens and purples", "neon hextech blue accents", "deep atmospheric shadows", "dramatic chiaroscuro lighting"],
                technique_details=["Touche Fortiche hybrid technique", "3D base with 2D overlays", "oil painting environment art", "hand-drawn smoke/fire/explosions", "custom shaders for painterly look", "deliberate imperfections"],
                example_prompt="convert to Arcane Fortiche animation style mixing 3D models with 2D hand-drawn effects, oil painting textures creating living artwork, rough tactile quality with visible brushstrokes, Piltover golden brass tones contrasting Zaun's toxic green-purple palette, hextech blue neon accents, dramatic chiaroscuro lighting with deep shadows, cel-shaded expressive faces, hand-drawn smoke and particle effects overlaid on 3D",
                compatible_with=["spiderverse", "oil_painting", "steampunk", "adult_animation"],
                tips=["Mix 3D base with 2D overlays", "Use oil painting textures", "Add hand-drawn effects for particles", "Create rough, tactile surfaces"]
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
                description="True isometric dimetric projection with 2:1 pixel ratio and 26.565-degree camera angle",
                visual_elements=["2:1 pixel ratio dimetric projection", "26.565-degree tile angle (arctan 1/2)", "45-degree horizontal rotation", "tile-based grid construction", "no perspective distortion"],
                color_characteristics=["atmospheric perspective for depth", "darker colors receding", "clear tile edge separation", "consistent lighting angle", "subtle ambient occlusion"],
                technique_details=["camera: 45° yaw, 30° pitch", "perfect pixel alignment", "z-order depth sorting", "orthographic projection", "modular tile system"],
                example_prompt="convert to true isometric game style, camera positioned at 45-degree horizontal rotation and 30-degree downward tilt, 2:1 pixel ratio dimetric projection creating 26.565-degree angles, tile-based grid construction, orthographic view with no perspective distortion, atmospheric color gradients for depth, classic isometric RPG aesthetic maintaining original subject composition",
                compatible_with=["strategy", "rpg", "city_builder"],
                tips=["Maintain exact 2:1 pixel ratio", "Use 45° horizontal, 30° vertical camera", "Align everything to isometric grid"]
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
                description="Stylish JRPG with punk rebellion aesthetics, manga-inspired UI, and explosive visual design",
                visual_elements=["oblique angular shapes everywhere", "halftone silhouettes", "graffiti-style text explosions", "irregular UI shapes", "manga panel cutouts"],
                color_characteristics=["strict red/black/white only", "NO other colors except HP/MP", "high contrast everywhere", "black on white text mix", "red as passion/energy"],
                technique_details=["over-exaggerated typography", "lazy pulsing UI energy", "prison garb motifs", "transforming shape animations", "Pop art meets noir"],
                example_prompt="convert to Persona 5 game visual style, STRICT red/black/white color palette (no other colors), explosive graffiti-style text in irregular oblique shapes, halftone pattern character silhouettes, manga panel composition, high contrast punk rebellion aesthetic, oversized typography mixing upper/lowercase, prison ball-and-chain motifs, dynamic angular UI elements pulsing with youthful energy",
                compatible_with=["jrpg", "anime", "stylish"],
                tips=["ONLY red/black/white colors", "Make text explosive and irregular", "Add prison/rebellion symbolism"]
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
                description="Authentic PlayStation 1 low-poly 3D with technical limitations creating distinctive visual artifacts",
                visual_elements=["230-540 triangle characters like original Lara Croft", "triangular pointy chest geometry", "sharp angular edges everywhere", "vertex snapping to pixel grid", "affine texture warping", "polygon sorting errors"],
                color_characteristics=["Gouraud vertex color shading", "256-color CLUT textures", "no texture filtering (pixelated)", "15-bit color depth (32,768 colors)", "heavy dithering patterns for gradients"],
                technique_details=["no Z-buffer (depth sorting issues)", "integer vertex positions causing jitter", "2D affine texture mapping creating wobbly textures", "single global light simulation", "dynamic polygon subdivision like Tomb Raider"],
                example_prompt="convert to authentic lowpoly 3D graphics from 1996. Low primitive blocky details, visible aliasing with jagged edges, triangulated, no lighting, muted desaturated simple shading. 320x240 resolution aesthetic",
                compatible_with=["retro_3d", "tomb_raider", "nostalgic"],
                tips=["Reference original Lara Croft's angular design", "Emphasize triangular/pointy geometry", "Add wobbly texture warping", "Use heavy dithering"]
            ),
            
            "red_dead": StylePreset(
                id="red_dead",
                name="Red Dead Redemption Poster Style",
                category=StyleCategory.GAME,
                description="Vintage western movie poster art style inspired by Red Dead Redemption's promotional materials",
                visual_elements=["dramatic character poses", "layered composition", "western iconography", "vintage poster layout", "silhouettes against sunsets"],
                color_characteristics=["sepia and amber tones", "weathered parchment texture", "sunset orange gradients", "dusty desaturated palette", "aged paper yellowing"],
                technique_details=["Drew Struzan influence", "painted illustration style", "vintage printing effects", "dramatic storytelling composition", "classic western poster design"],
                example_prompt="convert to Red Dead Redemption poster art style, vintage western movie poster aesthetic, sepia-toned with weathered parchment texture, dramatic character composition like classic western films, sunset silhouettes, painted illustration technique inspired by Drew Struzan, aged paper effects with yellowing, layered narrative elements, dusty amber color grading, 1960s-70s western movie poster design",
                compatible_with=["western", "vintage", "poster_art"],
                tips=["Add weathered paper texture", "Use sunset/sepia tones", "Layer multiple story elements"]
            ),
            
            "royal_match": StylePreset(
                id="royal_match",
                name="Royal Match Style",
                category=StyleCategory.GAME,
                description="Dream Games' distinctive stylized 3D art with Disney-esque characters and smooth, polished rendering",
                visual_elements=["smooth curved forms", "no sharp edges anywhere", "Disney-like character proportions", "oversized game pieces", "castle-themed elements"],
                color_characteristics=["vibrant but balanced palette", "rich royal purples and golds", "welcoming warm tones", "sparkling jewel effects", "soft ambient lighting"],
                technique_details=["stylized 3D rendering", "retro-modern blend", "exaggerated but grounded", "handcrafted personality", "polished casual aesthetic"],
                example_prompt="convert to Royal Match game art style, smooth curved 3D forms with absolutely no sharp edges, Disney-esque character proportions (slightly exaggerated but appealing), vibrant colors balanced with real-world grounding, castle-themed decorative elements (crowns, shields, books), oversized distinctive game pieces, welcoming joyful atmosphere, polished stylized 3D rendering with handcrafted personality, retro charm with contemporary polish",
                compatible_with=["candy_crush", "playrix_style", "disney_style"],
                tips=["Keep everything smooth and curved", "Disney-like appealing proportions", "Balance stylization with realism"]
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