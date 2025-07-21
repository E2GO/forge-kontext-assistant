"""
Comic styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

COMIC_STYLES = {
            "alex_ross": StylePreset(
                id="alex_ross",
                name="Alex Ross Style",
                category=StyleCategory.COMIC,
                description="Photorealistic painted comics with gouache and classical composition",
                visual_elements=["photorealistic faces", "painted textures", "realistic lighting", "classical poses", "Norman Rockwell influence"],
                color_characteristics=["realistic colors", "natural lighting", "gouache richness", "subtle gradients", "lifelike tones"],
                technique_details=["gouache painting", "photo reference", "classical composition", "realistic rendering", "fine art approach"],
                example_prompt="convert to Alex Ross painted comic style with photorealistic gouache painting, Norman Rockwell influenced composition, realistic natural lighting, lifelike color palette, classical heroic poses, fine art painting quality",
                compatible_with=["realistic", "painted", "classical"],
                tips=["Paint realistically", "Use photo reference", "Apply classical composition"]
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
            
            "comic_book": StylePreset(
                id="comic_book",
                name="Comic Book Style",
                category=StyleCategory.COMIC,
                description="Classic American comics with bold inks and Ben Day dots",
                visual_elements=["bold black outlines", "Ben Day dots", "speech bubbles", "action lines", "dynamic poses"],
                color_characteristics=["bright primaries", "limited palette", "flat colors", "newsprint feel", "halftone effects"],
                technique_details=["varied line weights", "cross-hatching", "stippling", "dramatic perspectives", "panel layouts"],
                example_prompt="convert to classic American comic book style with bold black ink outlines, Ben Day dot shading, bright primary colors on newsprint, dynamic action poses, speech bubbles, cross-hatching for depth, dramatic superhero perspectives",
                compatible_with=["superhero", "action", "vintage"],
                tips=["Use bold inks", "Add Ben Day dots", "Keep colors primary"]
            ),
            
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
            
            "dc_comics": StylePreset(
                id="dc_comics",
                name="DC Comics Style",
                category=StyleCategory.COMIC,
                description="Darker superhero aesthetic with Gothic influences and noir lighting",
                visual_elements=["Gothic architecture", "dramatic shadows", "iconic silhouettes", "cape dynamics", "urban settings"],
                color_characteristics=["darker palette", "noir influences", "muted tones", "atmospheric lighting", "night scenes"],
                technique_details=["heavy shadows", "chiaroscuro effects", "psychological depth", "film noir lighting", "Gothic mood"],
                example_prompt="convert to DC Comics style with Gothic architectural backgrounds, heavy dramatic shadows, iconic hero silhouettes, darker muted color palette, film noir lighting influences, psychological complexity, night urban settings",
                compatible_with=["noir", "gothic", "dark_hero"],
                tips=["Use heavy shadows", "Add Gothic elements", "Keep it darker"]
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
            
            "franco_belgian": StylePreset(
                id="franco_belgian",
                name="Franco-Belgian Comics",
                category=StyleCategory.COMIC,
                description="Clear line (ligne claire) style with uniform lines and detailed backgrounds",
                visual_elements=["uniform line weight", "no hatching", "detailed backgrounds", "clear compositions", "cartoonish characters"],
                color_characteristics=["flat vibrant colors", "no gradients", "strong color blocks", "bright palette", "clear separation"],
                technique_details=["ligne claire", "equal detail throughout", "illuminated shadows", "architectural precision", "clean inking"],
                example_prompt="convert to Franco-Belgian ligne claire comic style with uniform line weights, no cross-hatching, flat vibrant color blocks, detailed realistic backgrounds, cartoonish characters, exceptional clarity and readability, Tintin-like precision",
                compatible_with=["european", "adventure", "clear_line"],
                tips=["Keep lines uniform", "Detail backgrounds equally", "Use flat colors"]
            ),
            
            "frank_miller": StylePreset(
                id="frank_miller",
                name="Frank Miller Sin City Style",
                category=StyleCategory.COMIC,
                description="Stark noir with pure black and white contrast and selective color",
                visual_elements=["stark contrasts", "angular shapes", "negative space", "rain effects", "urban decay"],
                color_characteristics=["pure black and white", "selective red/yellow/blue", "no gray tones", "high contrast", "spot color"],
                technique_details=["heavy inks", "sharp angles", "negative space storytelling", "integrated typography", "film noir aesthetics"],
                example_prompt="convert to Frank Miller Sin City style with pure black and white high contrast, no gray tones, selective spot color in red or yellow, angular sharp linework, negative space composition, film noir rain effects",
                compatible_with=["noir", "crime", "graphic"],
                tips=["Eliminate all grays", "Use spot color sparingly", "Embrace negative space"]
            ),
            
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
            
            "jim_lee": StylePreset(
                id="jim_lee",
                name="Jim Lee Style",
                category=StyleCategory.COMIC,
                description="Detailed superhero art with distinctive crosshatching and dynamic anatomy",
                visual_elements=["intricate crosshatching", "detailed costumes", "flowing hair", "dynamic poses", "idealized anatomy"],
                color_characteristics=["modern coloring", "metallic effects", "gradient shading", "atmospheric colors", "costume shine"],
                technique_details=["penned lines", "rigid anatomy", "texture detail", "action dynamics", "costume rendering"],
                example_prompt="convert to Jim Lee comic art style with intricate crosshatching technique, highly detailed costume rendering, flowing dynamic hair, idealized superhero anatomy, modern gradient coloring with metallic effects",
                compatible_with=["superhero", "detailed", "modern_comic"],
                tips=["Master crosshatching", "Detail the costumes", "Perfect the anatomy"]
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
            
            "manga_noir": StylePreset(
                id="manga_noir",
                name="Manga Noir",
                category=StyleCategory.COMIC,
                description="Dark atmospheric manga with heavy blacks and psychological depth",
                visual_elements=["heavy black fills", "white detail lines", "rain effects", "urban settings", "cigarette smoke"],
                color_characteristics=["high contrast B&W", "deep blacks", "stark whites", "no midtones", "atmospheric effects"],
                technique_details=["beta-nuri technique", "chiaroscuro lighting", "vertical horror lines", "psychological shadows", "noir atmosphere"],
                example_prompt="convert to manga noir style with heavy beta-nuri black fills, white lines for details within black areas, high contrast chiaroscuro lighting, rain-soaked urban setting, psychological shadow placement, atmospheric smoke effects",
                compatible_with=["noir", "psychological", "urban"],
                tips=["Use heavy blacks", "Add white detail lines", "Create noir atmosphere"]
            ),
            
            "marvel_comics": StylePreset(
                id="marvel_comics",
                name="Marvel Comics Style",
                category=StyleCategory.COMIC,
                description="Dynamic superhero art with Kirby Krackle and bombastic action",
                visual_elements=["dynamic poses", "Kirby Krackle", "muscular anatomy", "flowing capes", "energy effects"],
                color_characteristics=["heroic bright colors", "costume primaries", "glowing effects", "dramatic lighting", "power visualization"],
                technique_details=["ornate linework", "explosive compositions", "emotional expressions", "epic scale", "motion emphasis"],
                example_prompt="convert to Marvel Comics superhero style with dynamic action poses, Kirby Krackle energy effects, detailed muscular anatomy, bright heroic color palette, flowing cape dynamics, explosive composition with motion lines",
                compatible_with=["superhero", "action", "epic"],
                tips=["Add Kirby Krackle", "Emphasize dynamics", "Use heroic poses"]
            ),
            
            "moebius": StylePreset(
                id="moebius",
                name="Moebius Style",
                category=StyleCategory.COMIC,
                description="French sci-fi art with intricate linework and psychedelic landscapes",
                visual_elements=["intricate details", "alien landscapes", "surreal elements", "organic forms", "philosophical themes"],
                color_characteristics=["vivid dreamlike colors", "psychedelic palettes", "atmospheric gradients", "otherworldly hues", "desert tones"],
                technique_details=["detailed linework", "ligne claire influence", "trance-like creation", "expansive vistas", "surreal perspective"],
                example_prompt="convert to Moebius French sci-fi comic style with intricate detailed linework, expansive alien desert landscapes, vivid dreamlike color palette, surreal organic forms, philosophical visual themes, psychedelic atmosphere",
                compatible_with=["scifi", "surreal", "european"],
                tips=["Detail everything intricately", "Create alien landscapes", "Use dreamlike colors"]
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
            
            "underground_comic": StylePreset(
                id="underground_comic",
                name="Underground Comix Style",
                category=StyleCategory.COMIC,
                description="Raw alternative comix with heavy crosshatching and countercultural themes",
                visual_elements=["rough linework", "heavy crosshatching", "exaggerated features", "adult content", "personal expression"],
                color_characteristics=["raw colors", "unpolished look", "high contrast", "psychedelic options", "DIY aesthetic"],
                technique_details=["pen and ink", "1920s cartoon influence", "handmade lettering", "uncensored approach", "raw energy"],
                example_prompt="convert to underground comix style with rough heavily crosshatched pen and ink linework, exaggerated cartoon features, thick rounded lettering, raw DIY aesthetic, countercultural themes, 1920s cartoon influence",
                compatible_with=["alternative", "indie", "raw"],
                tips=["Keep it rough", "Heavy crosshatching", "Embrace imperfection"]
            ),
            
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
            )
        }