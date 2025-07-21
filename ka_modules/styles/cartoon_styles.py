"""
Cartoon styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

CARTOON_STYLES = {
            "adventure_time": StylePreset(
                id="adventure_time",
                name="Adventure Time Style",
                category=StyleCategory.CARTOON,
                description="Pendleton Ward's unique 2D cartoon aesthetic",
                visual_elements=["noodle arms", "simple faces", "organic shapes", "whimsical designs"],
                color_characteristics=["pastel colors", "soft gradients", "candy-like palette"],
                technique_details=["loose linework", "minimal detail", "expressive simplicity"],
                example_prompt="convert to Adventure Time 2D cartoon with noodle limbs, simple dot eyes, organic shapes, and pastel candy colors",
                compatible_with=["steven_universe", "modern_cartoon", "whimsical"],
                tips=["Keep it loose and fun", "Use candy-inspired colors"]
            ),
            
            "disney_70s": StylePreset(
                id="disney_70s",
                name="Disney Xerography Era (60s-70s)",
                category=StyleCategory.CARTOON,
                description="Disney's xerography period featuring rough sketchy lines and medieval/adventure themes",
                visual_elements=["xeroxed rough lines", "visible construction lines", "angular character designs", "Milt Kahl style", "sketchy backgrounds"],
                color_characteristics=["muted earthy tones", "medieval browns and grays", "forest greens", "limited color palette", "no gradients"],
                technique_details=["xerography process", "no ink department", "rough animator drawings", "black outline only", "hastily-sketched backgrounds"],
                example_prompt="convert to Disney 1970s xerography animation style with rough sketchy black outlines, visible construction lines, angular Ken Anderson character designs, muted earthy medieval color palette, xeroxed lines that don't align perfectly with colors, Milt Kahl angular drawing style like Sword in the Stone or Robin Hood",
                compatible_with=["medieval", "adventure", "vintage_disney"],
                tips=["Keep lines rough and sketchy", "Use muted medieval colors", "Show construction lines"]
            ),
            
            "disney_classic": StylePreset(
                id="disney_classic",
                name="Disney Renaissance (90s)",
                category=StyleCategory.CARTOON,
                description="Disney Renaissance era 2D animation (1989-1999) with CAPS digital ink",
                visual_elements=["12 principles of animation", "flowing linework", "baby face proportions", "painted backgrounds", "rotoscoped movement"],
                color_characteristics=["rich Technicolor palette", "watercolor backgrounds", "warm flesh tones", "three-strip process colors", "romantic atmospheric lighting"],
                technique_details=["multiplane camera", "hand-painted cels", "pencil tests", "clean-up animation", "effects animation"],
                example_prompt="convert to classic Disney 2D animation style with flowing hand-drawn lines, Technicolor three-strip color process, watercolor painted backgrounds, baby face character proportions, traditional cel animation quality, romantic golden age aesthetic",
                compatible_with=["fairytale", "musical", "hand_drawn"],
                tips=["Use flowing, graceful lines", "Apply watercolor-style backgrounds", "Emphasize appeal and charm"]
            ),
            
            "ernest_celestine": StylePreset(
                id="ernest_celestine",
                name="Ernest & Celestine Style",
                category=StyleCategory.CARTOON,
                description="French watercolor animation with soft storybook aesthetic",
                visual_elements=["soft edges", "bear and mouse characters", "dissolving boundaries", "storybook quality", "gentle expressions"],
                color_characteristics=["muted watercolors", "soft pastels", "warm earth tones", "cozy atmosphere", "gentle palette"],
                technique_details=["watercolor backgrounds", "impressionistic style", "hand-drawn feeling", "minimalist approach", "children's book aesthetic"],
                example_prompt="convert to Ernest & Celestine French animation style with soft watercolor aesthetic, dissolving edges, muted pastel colors, children's book illustration quality, gentle storybook atmosphere, impressionistic backgrounds with minimalist design",
                compatible_with=["watercolor", "children_book", "french_animation"],
                tips=["Soften all edges", "Use muted watercolors", "Create cozy atmosphere"]
            ),
            
            "family_guy": StylePreset(
                id="family_guy",
                name="Family Guy Style",
                category=StyleCategory.CARTOON,
                description="Seth MacFarlane's distinctive adult cartoon with round character designs",
                visual_elements=["large round heads", "dot pupils", "prominent chins", "simplified anatomy", "tiny bodies relative to heads"],
                color_characteristics=["flat colors", "suburban color palette", "warm skin tones", "no gradients"],
                technique_details=["thick black outlines", "minimal shading", "simple backgrounds", "character-focused compositions"],
                example_prompt="convert to Family Guy 2D cartoon style with oversized round heads, small dot pupils, prominent square chins, thick black outlines, flat coloring without shading, simplified character anatomy with tiny bodies, suburban setting elements",
                compatible_with=["american_dad", "simpsons", "adult_cartoon"],
                tips=["Make heads oversized and round", "Keep pupils as simple dots", "Emphasize chin shapes"]
            ),
            
            "fleischer": StylePreset(
                id="fleischer",
                name="Fleischer Studios Style",
                category=StyleCategory.CARTOON,
                description="Betty Boop era animation with rotoscoping and surreal transformations",
                visual_elements=["rotoscoped realistic movement", "surreal transformations", "urban settings", "sentient objects", "dream logic"],
                color_characteristics=["often black and white", "muted Depression-era palette when colored", "German Expressionist shadows", "stark contrasts"],
                technique_details=["rotoscope process", "3D stereoptical backgrounds", "post-synced dialogue", "jazz synchronization", "adult themes"],
                example_prompt="convert to Fleischer Studios animation style with rotoscoped fluid movement, surreal transformations, urban 1930s setting, German Expressionist shadows, dream logic with sentient objects, jazz-age synchronized animation",
                compatible_with=["betty_boop", "popeye", "surreal"],
                tips=["Use rotoscoped movement", "Add surreal elements", "Include urban backgrounds"]
            ),
            
            "futurama": StylePreset(
                id="futurama",
                name="Futurama Style",
                category=StyleCategory.CARTOON,
                description="Retro-futuristic sci-fi with 2½-D CGI backgrounds and saturated tertiary colors",
                visual_elements=["realistic skin tones", "3D CGI spaceships", "Googie architecture", "atomic age motifs", "sleek mechanical designs"],
                color_characteristics=["saturated tertiaries palette", "olive/gold/taupe highlights", "aqua/cyan technology", "brick red buildings", "NO primary colors"],
                technique_details=["2½-D hybrid animation", "CGI backgrounds textured as 2D", "on-model consistency", "Populuxe/Googie design", "smooth integrated CGI"],
                example_prompt="convert to Futurama animation style with 2½-D hybrid technique, CGI spaceships and buildings textured to look 2D, saturated tertiary color palette featuring olive/gold/taupe as main colors NOT primaries, aqua/cyan for technology, brick red architecture, Googie retro-futuristic design with atomic age boomerang shapes, realistic human skin tones, sleek mechanical designs",
                compatible_with=["retro_futurism", "googie", "scifi_comedy"],
                tips=["Use tertiary colors not primaries", "Add CGI elements textured as 2D", "Include Googie architecture"]
            ),
            
            "gravity_falls": StylePreset(
                id="gravity_falls",
                name="Gravity Falls Style",
                category=StyleCategory.CARTOON,
                description="Alex Hirsch's mystery cartoon with geometric character designs",
                visual_elements=["geometric character shapes", "large expressive eyes", "triangle motifs everywhere", "pine tree symbols", "hidden cryptograms"],
                color_characteristics=["forest palette", "warm browns and oranges", "mystery purples", "sunset lighting"],
                technique_details=["clean vector lines", "detailed backgrounds with secrets", "subtle textures", "silhouette recognition"],
                example_prompt="convert to Gravity Falls 2D cartoon style with geometric character construction, large expressive eyes, clean vector outlines, forest color palette with warm browns and mystery purples, detailed backgrounds with hidden symbols, triangle shapes integrated throughout",
                compatible_with=["steven_universe", "disney_channel", "mystery_cartoon"],
                tips=["Use geometric shapes for characters", "Hide triangles in backgrounds", "Add mysterious symbols"]
            ),
            
            "gumball": StylePreset(
                id="gumball",
                name="The Amazing World of Gumball Style", 
                category=StyleCategory.CARTOON,
                description="Mixed-media cartoon combining multiple animation styles",
                visual_elements=["2D characters on real photos", "mixed animation styles", "simple character designs", "blue cat protagonist", "varied art techniques"],
                color_characteristics=["bright saturated colors", "high contrast", "vivid character colors against realistic backgrounds"],
                technique_details=["2D animation over live-action backgrounds", "multiple animation styles in one scene", "CGI mixed with claymation", "stylistic disunity"],
                example_prompt="convert to Amazing World of Gumball mixed-media style with simple 2D cartoon characters overlaid on photographic backgrounds, bright saturated character colors, multiple animation techniques combined, blue cat-like protagonist, intentional stylistic contrast between elements",
                compatible_with=["modern_cartoon", "experimental", "mixed_media"],
                tips=["Mix real photos with 2D characters", "Use multiple art styles", "Create visual contrast"]
            ),
            
            "hanna_barbera": StylePreset(
                id="hanna_barbera",
                name="Hanna-Barbera Style",
                category=StyleCategory.CARTOON,
                description="1960s-70s TV limited animation with cost-saving techniques visible as style",
                visual_elements=["geometric character shapes", "thick uniform outlines", "static bodies with moving parts", "wraparound backgrounds", "repeated walk cycles"],
                color_characteristics=["flat primary colors", "candy-like brightness", "3-4 colors per character", "simple color separation", "no gradients"],
                technique_details=["limited animation", "held cels", "mouth shapes only", "recycled animation", "economical design"],
                example_prompt="convert to Hanna-Barbera 1960s TV cartoon style with limited animation aesthetic, geometric character shapes, thick uniform black outlines, flat candy-bright primary colors, static poses with moving mouth only, wraparound repeating backgrounds",
                compatible_with=["saturday_morning", "retro_cartoon", "tv_animation"],
                tips=["Simplify to geometric shapes", "Use flat, bright colors", "Minimize movement"]
            ),
            
            "looney_tunes": StylePreset(
                id="looney_tunes",
                name="Looney Tunes Style",
                category=StyleCategory.CARTOON,
                description="Warner Bros theatrical cartoons with extreme takes and impossible physics",
                visual_elements=["extreme squash and stretch", "wild takes with eye pops", "smear frames", "impossible physics", "slapstick props"],
                color_characteristics=["bright Technicolor palette", "high contrast", "theatrical lighting", "bold primaries", "explosive effects"],
                technique_details=["Tex Avery innovations", "Chuck Jones timing", "multiple gags per second", "fourth wall breaks", "anvil and dynamite gags"],
                example_prompt="convert to Looney Tunes theatrical cartoon style with extreme squash and stretch, wild eye-popping takes, smear frames for fast motion, bright Technicolor palette, slapstick physics with anvils and dynamite, Tex Avery-style exaggeration",
                compatible_with=["slapstick", "comedy", "theatrical_short"],
                tips=["Exaggerate everything", "Use impossible physics", "Add slapstick props"]
            ),
            
            "nightmare_before_christmas": StylePreset(
                id="nightmare_before_christmas",
                name="The Nightmare Before Christmas Style",
                category=StyleCategory.CARTOON,
                description="Henry Selick's stop-motion puppet animation with gothic expressionism and handcrafted textures",
                visual_elements=["physical puppet construction", "replacement animation heads", "metal armature poses", "hand-drawn suit stripes", "engraved texture details"],
                color_characteristics=["limited grayed-down palette", "gothic halloween colors", "dramatic three-point lighting", "textured painted surfaces", "expressionistic shadows"],
                technique_details=["stop-motion puppet animation", "polyurethane resin heads", "latex foam construction", "miniature set pieces", "motion control camera"],
                example_prompt="convert to Nightmare Before Christmas stop-motion puppet style with physical puppet construction visible, replacement animation aesthetic, elongated gothic proportions with metal armature poses, hand-painted textures and engraved details, limited grayed-down color palette, dramatic three-point lighting creating expressionistic shadows, miniature set with full dimensional depth",
                compatible_with=["stop_motion", "gothic", "puppet_animation", "tim_burton"],
                tips=["Show puppet construction seams", "Use dimensional lighting", "Add handcrafted textures"]
            ),
            
            "peppa_pig": StylePreset(
                id="peppa_pig",
                name="Peppa Pig Style",
                category=StyleCategory.CARTOON,
                description="British preschool cartoon with distinctive minimalist design",
                visual_elements=["sideways profile faces", "simple geometric shapes", "minimalist lines", "both eyes on one side", "picasso-like hairdryer shape"],
                color_characteristics=["flat solid colors", "bright primary palette", "pink pig characters", "simple color fills", "no gradients or shading"],
                technique_details=["2D animation software", "smooth clean lines", "extremely simple shapes", "no texture or detail", "profile view aesthetic"],
                example_prompt="convert to Peppa Pig 2D cartoon style with sideways profile view, both eyes visible on one side of face, extremely simple geometric shapes, flat solid colors with no shading, minimalist British preschool animation aesthetic, picasso-like abstract character design",
                compatible_with=["preschool", "minimalist", "british_cartoon"],
                tips=["Keep shapes extremely simple", "Show characters in profile", "Use flat bright colors"]
            ),
            
            "phineas_ferb": StylePreset(
                id="phineas_ferb",
                name="Phineas and Ferb Style",
                category=StyleCategory.CARTOON,
                description="Dan Povenmire's geometric cartoon with triangle-headed characters",
                visual_elements=["triangle head shapes", "rectangle heads", "geometric character construction", "simple recognizable silhouettes", "inventive gadgets"],
                color_characteristics=["bright summer colors", "orange hair", "blue skies", "vibrant primary colors"],
                technique_details=["geometric shapes in everything", "clean angular lines", "flat colors", "Tex Avery influence", "triangles hidden in backgrounds"],
                example_prompt="convert to Phineas and Ferb 2D cartoon style with extreme geometric character designs, triangle-shaped heads, rectangle shapes, bright summer colors with orange and blue dominance, clean angular lines, geometric shapes integrated into backgrounds, inventive mechanical elements",
                compatible_with=["gravity_falls", "disney_channel", "tex_avery"],
                tips=["Use triangles everywhere", "Make characters from basic shapes", "Hide geometric easter eggs"]
            ),
            
            "pixar": StylePreset(
                id="pixar",
                name="Pixar 3D Style",
                category=StyleCategory.CARTOON,
                description="Pixar's signature 3D animation with RenderMan rendering and subsurface scattering",
                visual_elements=["smooth rounded characters", "subsurface scattering on skin", "complex iris patterns", "detailed hair simulation", "expressive squash and stretch"],
                color_characteristics=["strategic color scripts", "warm cinematic lighting", "vibrant naturalistic palettes", "atmospheric perspective", "complementary color schemes"],
                technique_details=["RenderMan rendering", "physically-based shading", "ray-traced reflections", "volumetric lighting", "cloth and particle simulation"],
                example_prompt="convert to Pixar 3D animation style with smooth rounded characters, subsurface scattering on skin, warm cinematic lighting, RenderMan quality rendering, expressive character animation with squash and stretch, detailed hair and cloth simulation",
                compatible_with=["disney_3d", "dreamworks", "cgi_animation"],
                tips=["Emphasize round, appealing shapes", "Use warm, inviting lighting", "Add subsurface scattering"]
            ),
            
            "rick_morty": StylePreset(
                id="rick_morty",
                name="Rick and Morty Style",
                category=StyleCategory.CARTOON,
                description="Justin Roiland's surreal sci-fi cartoon with distinctive squiggly pupils",
                visual_elements=["squiggly wavy pupils", "oversized white orb eyes", "drooling mouths", "unibrows", "bulging eyes", "grotesque features"],
                color_characteristics=["muted color palette", "sci-fi greens and purples", "portal green effects", "desaturated tones"],
                technique_details=["wobbly rough linework", "imperfect circles", "intentionally crude details", "chaotic compositions"],
                example_prompt="convert to Rick and Morty 2D cartoon style with oversized white orb eyes, squiggly wavy pupils, wobbly rough outlines, drooling characters, muted sci-fi color palette with portal greens, crude imperfect linework, grotesque character features",
                compatible_with=["futurama", "adult_swim", "scifi_cartoon"],
                tips=["Make pupils squiggly and wobbly", "Keep lines intentionally rough", "Add drool and grotesque details"]
            ),
            
            "rubber_hose": StylePreset(
                id="rubber_hose",
                name="1920s Rubber Hose",
                category=StyleCategory.CARTOON,
                description="Early animation style where everything moves like flexible rubber tubes",
                visual_elements=["noodle limbs without joints", "pie-cut eyes", "white gloves on hands", "constant bouncing motion", "synchronized to music"],
                color_characteristics=["black and white only", "high contrast", "film grain", "scratches and dust", "vignetting"],
                technique_details=["rubber hose physics", "rhythmic animation", "no skeletal structure", "continuous curves", "dance-like movement"],
                example_prompt="convert to 1920s rubber hose animation style with noodle-like boneless limbs, pie-cut eyes, white gloves, constant bouncing synchronized to rhythm, black and white with film grain, vintage film scratches and dust",
                compatible_with=["vintage", "silent_era", "musical"],
                tips=["Make everything rubbery", "Add constant bounce", "Sync to rhythm"]
            ),
            
            "shaun_the_sheep": StylePreset(
                id="shaun_the_sheep",
                name="Shaun the Sheep Style",
                category=StyleCategory.CARTOON,
                description="Aardman's stop-motion claymation with woolly textures",
                visual_elements=["plasticine characters", "woolly texture", "fingerprint marks", "black face and legs", "expressive eyes"],
                color_characteristics=["natural wool colors", "earthy farm palette", "cream and black sheep", "colorful backgrounds"],
                technique_details=["stop-motion animation", "claymation technique", "modelling tool textures", "12-24 frames per second", "handmade quality"],
                example_prompt="convert to Shaun the Sheep stop-motion claymation style with plasticine texture, woolly sheep characters with black faces and legs, visible fingerprints and modelling marks, Aardman animation aesthetic, handcrafted stop-motion quality with textured surfaces",
                compatible_with=["wallace_gromit", "stop_motion", "british_animation"],
                tips=["Add woolly texture details", "Show plasticine fingerprints", "Use silent comedy expressions"]
            ),
            
            "simpsons": StylePreset(
                id="simpsons",
                name="The Simpsons Style",
                category=StyleCategory.CARTOON,
                description="Matt Groening's iconic 2D TV cartoon style",
                visual_elements=["yellow skin tones", "bulging eyes", "overbite", "simple lines"],
                color_characteristics=["flat colors", "yellow dominant", "bright primaries"],
                technique_details=["thick outlines", "minimal shading", "exaggerated features"],
                example_prompt="convert to The Simpsons 2D cartoon style with yellow skin tones, overbite character design, thick black outlines, and flat colors",
                compatible_with=["futurama", "adult_cartoon", "satire"],
                tips=["Keep it simple and flat", "Exaggerate facial features"]
            ),
            
            "spongebob": StylePreset(
                id="spongebob",
                name="SpongeBob SquarePants Style",
                category=StyleCategory.CARTOON,
                description="Stephen Hillenburg's underwater cartoon with cel animation aesthetic",
                visual_elements=["square sponge character", "underwater themes", "porous textures", "sea creatures", "sky flowers as clouds", "Tiki influences"],
                color_characteristics=["bright yellows", "deep ocean blues", "tropical palette", "vibrant saturated colors", "Hawaiian shirt patterns"],
                technique_details=["traditional cel animation style", "bold black outlines", "exaggerated expressions", "bubble effects", "flower patterns"],
                example_prompt="convert to SpongeBob SquarePants 2D cartoon style with underwater theme, bright yellow sponge character with porous texture, deep ocean blue backgrounds, bold black outlines, exaggerated facial expressions, bubble effects, tropical colors with Hawaiian Tiki design influences",
                compatible_with=["nickelodeon", "kids_cartoon", "underwater_theme"],
                tips=["Add underwater elements and bubbles", "Use bright yellow prominently", "Include Tiki/Hawaiian patterns"]
            ),
            
            "steven_universe": StylePreset(
                id="steven_universe",
                name="Steven Universe Style",
                category=StyleCategory.CARTOON,
                description="Rebecca Sugar's soft aesthetic with unique character silhouettes",
                visual_elements=["soft rounded shapes", "unique body proportions", "gem designs", "star motifs", "flowing hair", "diverse body types"],
                color_characteristics=["soft pastel palette", "gem-inspired colors", "warm lighting", "gentle gradients", "pink and blue dominance"],
                technique_details=["smooth clean lines", "minimal details", "expressive animation", "distinctive silhouettes for each character"],
                example_prompt="convert to Steven Universe 2D cartoon style with soft rounded character shapes, unique body proportions and silhouettes, pastel color palette with gem-inspired hues, smooth flowing lines, star and gem motifs, warm atmospheric lighting, diverse character designs",
                compatible_with=["adventure_time", "gravity_falls", "wholesome_cartoon"],
                tips=["Vary body shapes dramatically", "Use soft pastels", "Make each silhouette unique"]
            ),
            
            "winx_club": StylePreset(
                id="winx_club",
                name="Winx Club Style",
                category=StyleCategory.CARTOON,
                description="Italian magical girl animation with fashion and sparkles",
                visual_elements=["fairy wings", "fashion outfits", "long legs", "sparkle effects", "transformation sequences"],
                color_characteristics=["bright saturated colors", "glittering effects", "eye-popping hues", "pink and purple dominance", "rainbow palette"],
                technique_details=["2D/3D hybrid animation", "anime influence", "fashion designer input", "glowing auras", "magical effects"],
                example_prompt="convert to Winx Club magical girl style with fairy wings, fashionable outfits, elongated proportions with long legs, bright saturated colors, sparkle and glitter effects everywhere, Italian animation with anime influence, transformation magic aesthetic",
                compatible_with=["sailor_moon", "magical_girl", "fashion_anime"],
                tips=["Emphasize fashion elements", "Add lots of sparkles", "Use bright saturated colors"]
            )
        }