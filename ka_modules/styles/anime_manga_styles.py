"""
Anime Manga styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

ANIME_MANGA_STYLES = {
            "anime_modern": StylePreset(
                id="anime_modern",
                name="Modern Anime",
                category=StyleCategory.ANIME_MANGA,
                description="Contemporary digital anime with sophisticated techniques and cinematic quality",
                visual_elements=["hyper-realistic proportions", "stylized faces", "natural hair colors", "clean linework", "detailed eyes"],
                color_characteristics=["saturated neon hues", "intense contrasts", "soft shading", "3D-like presence", "atmospheric effects"],
                technique_details=["digital integration", "CGI backgrounds", "soft-shaded forms", "cinematic angles", "particle effects"],
                example_prompt="convert to modern anime style with hyper-realistic body proportions, stylized facial features, saturated colors with neon accents, soft 3D-like shading, clean digital linework, cinematic quality backgrounds, atmospheric lighting effects",
                compatible_with=["shounen", "seinen", "digital_art"],
                tips=["Use soft shading", "Add atmospheric effects", "Keep proportions realistic"]
            ),
            
            "attack_on_titan": StylePreset(
                id="attack_on_titan",
                name="Attack on Titan Style",
                category=StyleCategory.ANIME_MANGA,
                description="Dark and gritty realistic anime style",
                visual_elements=["realistic proportions", "military uniforms", "3D maneuver gear", "intense expressions", "survey corps wings"],
                color_characteristics=["muted desaturated colors", "browns and grays", "military greens", "blood red accents"],
                technique_details=["cross-hatching shading", "realistic anatomy", "gothic horror aesthetic", "detailed backgrounds", "dramatic lighting"],
                example_prompt="convert to Attack on Titan anime style with realistic proportions, military uniform with survey corps emblem, muted desaturated colors, cross-hatching shading technique, gritty dark atmosphere, intense facial expressions, gothic horror aesthetic",
                compatible_with=["seinen", "dark_anime", "military"],
                tips=["Use realistic proportions", "Add cross-hatching", "Keep colors muted"]
            ),
            
            "chibi": StylePreset(
                id="chibi",
                name="Chibi Style",
                category=StyleCategory.ANIME_MANGA,
                description="Super-deformed cute style with 1:1 to 4:1 head-body ratio",
                visual_elements=["oversized heads", "tiny bodies", "huge eyes", "stubby limbs", "simplified features"],
                color_characteristics=["bright colors", "simple shading", "cute palette", "pastel options", "clear colors"],
                technique_details=["2-4 heads tall", "minimal detail", "exaggerated features", "comedy emphasis", "merchandising friendly"],
                example_prompt="convert to chibi super-deformed style with oversized head in 2:1 ratio to tiny body, huge expressive eyes, stubby simplified limbs, minimal facial features, bright cute colors, comedy-friendly proportions",
                compatible_with=["cute", "comedy", "mascot"],
                tips=["Make head 50% of height", "Simplify everything", "Emphasize cuteness"]
            ),
            
            "jujutsu_kaisen": StylePreset(
                id="jujutsu_kaisen",
                name="Jujutsu Kaisen Style",
                category=StyleCategory.ANIME_MANGA,
                description="Modern shounen with cursed energy effects by MAPPA",
                visual_elements=["cursed energy auras", "black uniforms", "domain expansions", "hand signs", "modern designs"],
                color_characteristics=["blue cursed energy", "purple and red effects", "darker palette", "neon accents"],
                technique_details=["fluid animation", "dynamic fights", "special effects", "impact frames", "modern shounen style"],
                example_prompt="convert to Jujutsu Kaisen anime style with cursed energy aura effects in blue and purple, black school uniform, modern shounen character design, dynamic action pose, MAPPA animation quality with fluid movements, supernatural battle aesthetic",
                compatible_with=["modern_shounen", "supernatural", "action_anime"],
                tips=["Add cursed energy effects", "Use blue/purple glows", "Show dynamic action"]
            ),
            
            "manga": StylePreset(
                id="manga",
                name="Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Traditional Japanese manga with screentones and beta-nuri techniques",
                visual_elements=["clean ink lines", "screentone shading", "speed lines", "emotional iconography", "beta-nuri fills"],
                color_characteristics=["black and white", "screentone patterns", "spot blacks", "white highlights", "grayscale values"],
                technique_details=["adhesive screentones", "cross-hatching", "white ink details", "panel layouts", "motion effects"],
                example_prompt="convert to traditional manga style with clean black ink lines, screentone shading patterns, beta-nuri solid black fills, speed lines for motion, emotional iconography, classic black and white aesthetic with spot whites",
                compatible_with=["black_white", "comic", "japanese"],
                tips=["Use screentone patterns", "Add speed lines", "Keep it black and white"]
            ),
            
            "mecha": StylePreset(
                id="mecha",
                name="Mecha Anime Style",
                category=StyleCategory.ANIME_MANGA,
                description="Giant robot 2D anime aesthetic with mechanical detail",
                visual_elements=["detailed machinery", "panel lines", "mechanical joints", "pilot cockpits"],
                color_characteristics=["metallic surfaces", "warning colors", "glowing elements"],
                technique_details=["technical detail", "perspective shots", "scale contrast"],
                example_prompt="transform into mecha anime style, enhance existing subjects with detailed mechanical armor and robotic augmentations, add complex panel lines, rivets, hydraulics, thrusters, metallic surfaces with weathering, glowing power cores, mechanical joint details, scale indicators, warning decals, and dramatic mecha aesthetic while maintaining original composition",
                compatible_with=["gundam", "evangelion", "scifi"],
                tips=["Add panel lines and rivets", "Show scale with humans"]
            ),
            
            "my_hero_academia": StylePreset(
                id="my_hero_academia",
                name="My Hero Academia Style",
                category=StyleCategory.ANIME_MANGA,
                description="Superhero shounen anime with colorful quirks",
                visual_elements=["hero costumes", "quirk effects", "muscular builds", "comic book influence", "varied body types"],
                color_characteristics=["bright primary colors", "superhero palette", "character-specific colors", "vibrant effects"],
                technique_details=["comic book aesthetic", "bold lines", "impact frames", "western influence", "dynamic poses"],
                example_prompt="convert to My Hero Academia anime style with superhero costume design, colorful quirk power effects, comic book aesthetic with bold lines, bright primary colors, muscular hero proportions, dynamic action pose with impact effects",
                compatible_with=["superhero", "shounen", "comic_style"],
                tips=["Design unique costumes", "Add quirk visual effects", "Use superhero colors"]
            ),
            
            "naruto": StylePreset(
                id="naruto",
                name="Naruto Style",
                category=StyleCategory.ANIME_MANGA,
                description="Masashi Kishimoto's ninja-themed shounen anime",
                visual_elements=["ninja headbands", "whisker marks", "sharp features", "dynamic poses", "hand signs"],
                color_characteristics=["orange and blue scheme", "vibrant chakra colors", "black ninja attire", "red cloud patterns"],
                technique_details=["2D animation", "dynamic action lines", "martial arts choreography", "special effects", "speed lines"],
                example_prompt="convert to Naruto anime style with ninja headband, whisker marks on cheeks, orange and blue color scheme, dynamic action pose, sharp anime features, chakra energy effects, shounen manga aesthetic with martial arts action",
                compatible_with=["shounen", "ninja", "action_anime"],
                tips=["Add ninja accessories", "Use orange prominently", "Show dynamic action"]
            ),
            
            "one_piece": StylePreset(
                id="one_piece",
                name="One Piece Style",
                category=StyleCategory.ANIME_MANGA,
                description="Eiichiro Oda's wildly exaggerated pirate adventure",
                visual_elements=["extreme proportions", "unique body shapes", "straw hat", "devil fruit powers", "pirate themes"],
                color_characteristics=["bright saturated colors", "tropical palette", "primary colors", "bold contrasts"],
                technique_details=["exaggerated expressions", "cartoonish style", "dynamic movement", "no rules design", "strong silhouettes"],
                example_prompt="convert to One Piece anime style with wildly exaggerated proportions, unique character shapes, bright saturated colors, cartoonish expressions, pirate adventure theme, Eiichiro Oda's no-rules character design, extreme body variations",
                compatible_with=["shounen", "pirate", "adventure_anime"],
                tips=["Exaggerate proportions wildly", "Use unique body shapes", "Make expressions extreme"]
            ),
            
            "ponyo": StylePreset(
                id="ponyo",
                name="Ponyo Style",
                category=StyleCategory.ANIME_MANGA,
                description="Miyazaki's watercolor-inspired ocean tale",
                visual_elements=["fluid water forms", "goldfish character", "childlike designs", "ocean themes", "magical transformations"],
                color_characteristics=["bright ocean blues", "warm rainbow tones", "watercolor effects", "vibrant yellows", "pastel backgrounds"],
                technique_details=["watercolor aesthetic", "hand-drawn animation", "fluid movements", "no CGI approach", "organic shapes"],
                example_prompt="convert to Ponyo Studio Ghibli style with watercolor aesthetic, bright ocean colors, fluid organic shapes, childlike character designs, hand-drawn animation with no CGI, magical underwater atmosphere, warm rainbow color palette",
                compatible_with=["ghibli", "watercolor", "children_anime"],
                tips=["Use watercolor effects", "Make movements fluid", "Keep designs childlike"]
            ),
            
            "princess_mononoke": StylePreset(
                id="princess_mononoke",
                name="Princess Mononoke Style",
                category=StyleCategory.ANIME_MANGA,
                description="Hayao Miyazaki's epic fantasy with nature spirits",
                visual_elements=["nature spirits", "traditional Japanese clothing", "detailed forests", "mystical creatures", "transformation effects"],
                color_characteristics=["earth tones", "forest greens", "natural browns", "mystical glows", "traditional palette"],
                technique_details=["hand-drawn animation", "detailed backgrounds", "traditional Japanese art influence", "atmospheric depth", "spiritual elements"],
                example_prompt="convert to Princess Mononoke Studio Ghibli style with detailed forest backgrounds, nature spirit designs, earth tone color palette, traditional Japanese aesthetic, hand-drawn animation quality, mystical atmosphere with spiritual elements",
                compatible_with=["ghibli", "fantasy", "traditional_anime"],
                tips=["Detail the nature elements", "Use earth tone palette", "Add mystical atmosphere"]
            ),
            
            "sailor_moon": StylePreset(
                id="sailor_moon",
                name="Sailor Moon Style",
                category=StyleCategory.ANIME_MANGA,
                description="Classic 90s shoujo manga style by Naoko Takeuchi",
                visual_elements=["huge expressive eyes", "elongated limbs", "flowing hair", "sailor uniforms", "crescent moon symbols"],
                color_characteristics=["pastel palette", "pink ribbons", "sparkle effects", "star and heart motifs", "hexagonal backgrounds"],
                technique_details=["cel animation style", "transformation sequences", "shoujo manga aesthetic", "detailed sparkles", "ethereal effects"],
                example_prompt="convert to Sailor Moon 90s shoujo anime style with huge expressive eyes, elongated elegant proportions, flowing hair, sailor uniform, pastel colors with pink ribbons, sparkles and star effects, transformation sequence aesthetic, classic magical girl design",
                compatible_with=["magical_girl", "shoujo", "90s_anime"],
                tips=["Make eyes huge and sparkly", "Elongate the proportions", "Add ribbons and moon motifs"]
            ),
            
            "shoujo": StylePreset(
                id="shoujo",
                name="Shoujo Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Romantic manga style with sparkles, flowers, and elegant proportions",
                visual_elements=["huge sparkly eyes", "flowing hair", "floral backgrounds", "elegant proportions", "decorative elements"],
                color_characteristics=["pastel palette", "soft pinks", "dreamy atmosphere", "sparkle effects", "romantic lighting"],
                technique_details=["detailed eyes", "flower screentones", "bubble effects", "ribbon decorations", "emotional atmosphere"],
                example_prompt="convert to shoujo manga style with huge sparkling eyes with galaxy reflections, flowing elegant hair, decorative flower and rose backgrounds, pastel color scheme, romantic bubble effects, lanky graceful proportions, dreamy atmosphere",
                compatible_with=["romance", "magical_girl", "feminine"],
                tips=["Make eyes huge and sparkly", "Add flowers everywhere", "Use pastel colors"]
            ),
            
            "shounen": StylePreset(
                id="shounen",
                name="Shounen Manga Style",
                category=StyleCategory.ANIME_MANGA,
                description="Action manga with dynamic motion and impact effects",
                visual_elements=["dynamic action poses", "speed lines", "impact effects", "muscular builds", "determined expressions"],
                color_characteristics=["bold colors", "dark tones", "high contrast", "power auras", "explosive effects"],
                technique_details=["motion blur", "trajectory lines", "multiple limbs effect", "size contrast", "battle damage"],
                example_prompt="convert to shounen manga style with dynamic action poses, speed lines showing motion trajectory, impact effects with angular shapes, muscular character builds, determined facial expressions, high contrast with bold colors, power aura effects",
                compatible_with=["action", "battle", "superhero"],
                tips=["Show dynamic motion", "Add impact effects", "Use bold contrasts"]
            ),
            
            "studio_ghibli": StylePreset(
                id="studio_ghibli",
                name="Studio Ghibli Style",
                category=StyleCategory.ANIME_MANGA,
                description="Miyazaki's hand-painted animation with watercolor backgrounds and nature themes",
                visual_elements=["clean character lines", "detailed nature", "expressive movement", "European influences", "fantastical elements"],
                color_characteristics=["watercolor palette", "nature greens", "soft transitions", "organic colors", "atmospheric perspective"],
                technique_details=["Nicker Poster Color", "wet-into-wet painting", "hand-drawn animation", "painterly backgrounds", "traditional techniques"],
                example_prompt="convert to Studio Ghibli style with hand-painted watercolor backgrounds, clean character lines with subtle expressions, dominant nature greens, soft organic color transitions, detailed environmental storytelling, traditional animation aesthetic",
                compatible_with=["fantasy", "nature", "traditional_anime"],
                tips=["Emphasize nature", "Use watercolor textures", "Add environmental detail"]
            )
        }