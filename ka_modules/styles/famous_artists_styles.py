"""
Famous Artists styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

FAMOUS_ARTISTS_STYLES = {
            "banksy": StylePreset(
                id="banksy",
                name="Banksy Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="British street art with stencil technique and social commentary",
                visual_elements=["stencil art", "high contrast", "urban setting", "political message"],
                color_characteristics=["black and white base", "selective color", "spray paint effect"],
                technique_details=["stencil technique", "graffiti style", "minimalist approach"],
                example_prompt="convert to Banksy street art with sharp stencil technique, high contrast black and white, selective red accents, and political urban graffiti aesthetic",
                compatible_with=["graffiti", "urban", "minimalist"],
                tips=["Use stencil-like shapes", "Add social commentary elements"]
            ),
            
            "craig_mullins": StylePreset(
                id="craig_mullins",
                name="Craig Mullins Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Digital painting mastery in the style of concept art pioneer Craig Mullins",
                visual_elements=["loose brushwork", "atmospheric depth", "light and shadow", "environmental mood"],
                color_characteristics=["limited palette", "color temperature shifts", "atmospheric haze", "dramatic lighting"],
                technique_details=["digital impressionism", "value painting", "speed painting", "photographic reference"],
                example_prompt="convert to Craig Mullins concept art style, loose confident brushstrokes, atmospheric environment painting, dramatic light and shadow, limited color palette with temperature shifts, digital impressionism technique, cinematic mood, masterful value control",
                compatible_with=["environment_concept", "speedpainting", "cinematic"],
                tips=["Work value first", "Keep brushwork loose", "Focus on mood"]
            ),
            
            "dali": StylePreset(
                id="dali",
                name="Salvador Dali Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Surrealist art with melting reality",
                visual_elements=["melting objects", "impossible physics", "dreamlike scenes", "precise detail"],
                color_characteristics=["desert tones", "stark contrasts", "surreal lighting"],
                technique_details=["photorealistic surrealism", "impossible combinations", "symbolic imagery"],
                example_prompt="convert to Salvador Dali surrealist painting with melting clocks like The Persistence of Memory, Catalonian desert landscapes, dreamlike imagery, and photorealistic detail",
                compatible_with=["surreal", "dreamlike", "symbolic"],
                tips=["Combine realistic rendering with impossible elements", "Add symbolic objects"]
            ),
            
            "dali_surrealism": StylePreset(
                id="dali_surrealism",
                name="Dalí Surrealism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Salvador Dalí's surrealist style with melting forms and dreamlike imagery",
                visual_elements=["melting objects", "impossible physics", "dreamlike scenes", "precise detail"],
                color_characteristics=["desert yellows", "sky blues", "stark contrasts", "luminous highlights"],
                technique_details=["paranoiac-critical method", "double images", "meticulous realism", "optical illusions"],
                example_prompt="convert to Salvador Dalí surrealist style, melting clocks and objects, dreamlike desert landscape, impossible physics, meticulous photorealistic detail within surreal context, paranoiac-critical double images",
                compatible_with=["surrealism", "dream_art", "fantasy"],
                tips=["Combine realistic detail with surreal elements", "Add melting objects", "Use desert landscapes"]
            ),
            
            "feng_zhu": StylePreset(
                id="feng_zhu",
                name="Feng Zhu Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Entertainment design in the style of concept art educator Feng Zhu",
                visual_elements=["design clarity", "functional aesthetics", "world-building", "cultural integration"],
                color_characteristics=["cohesive palettes", "environmental logic", "material accuracy", "lighting scenarios"],
                technique_details=["design fundamentals", "perspective mastery", "efficient workflow", "industry standards"],
                example_prompt="convert to Feng Zhu concept art style, clear design communication, functional aesthetic choices, strong perspective and form, environmentally integrated elements, efficient digital painting technique, Singapore school of design approach, entertainment industry quality",
                compatible_with=["environment_concept", "industrial_design", "world_building"],
                tips=["Design with purpose", "Show environmental context", "Maintain clarity"]
            ),
            
            "james_gurney": StylePreset(
                id="james_gurney",
                name="James Gurney Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Imaginative realism in the style of Dinotopia creator James Gurney",
                visual_elements=["naturalistic detail", "fantasy integration", "plein air quality", "storytelling elements"],
                color_characteristics=["natural light", "color harmony", "atmospheric effects", "warm undertones"],
                technique_details=["traditional painting", "observational accuracy", "imaginative realism", "narrative composition"],
                example_prompt="convert to James Gurney concept art style, imaginative realism combining fantasy with naturalistic detail, plein air lighting quality, Dinotopia-inspired world-building, traditional painting techniques, warm natural color harmony, narrative storytelling through environment",
                compatible_with=["fantasy_realism", "creature_concept", "world_building"],
                tips=["Ground fantasy in reality", "Use natural lighting", "Tell a story"]
            ),
            
            "jean_giraud_moebius": StylePreset(
                id="jean_giraud_moebius",
                name="Moebius Concept Art Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Surreal sci-fi concept art in the style of Jean Giraud (Moebius)",
                visual_elements=["organic forms", "surreal landscapes", "flowing lines", "dreamlike quality"],
                color_characteristics=["desert palettes", "psychedelic colors", "gradient skies", "crystal clarity"],
                technique_details=["ligne claire", "organic architecture", "detailed linework", "European comics influence"],
                example_prompt="convert to Moebius concept art style, surreal science fiction landscape, organic flowing architecture, crystal clear ligne claire technique, desert color palette with psychedelic accents, dreamlike atmospheric quality, European bande dessinée aesthetic",
                compatible_with=["french_comics", "surreal_scifi", "desert_punk"],
                tips=["Embrace the surreal", "Use flowing organic forms", "Clear line technique"]
            ),
            
            "klimt": StylePreset(
                id="klimt",
                name="Gustav Klimt Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Art Nouveau master Gustav Klimt's golden decorative style",
                visual_elements=["gold leaf", "decorative patterns", "symbolic elements", "sensual figures"],
                color_characteristics=["gold dominant", "rich jewel tones", "metallic accents", "warm palette"],
                technique_details=["gold leaf application", "Byzantine influence", "decorative symbolism", "pattern mixing"],
                example_prompt="convert to Gustav Klimt style with gold leaf patterns, decorative Byzantine motifs, sensual Art Nouveau figures, rich jewel tone accents, symbolic elements, mosaic-like backgrounds, Vienna Secession aesthetic",
                compatible_with=["art_nouveau", "symbolism", "decorative_art"],
                tips=["Use gold extensively", "Add decorative patterns", "Mix geometric and organic"]
            ),
            
            "monet": StylePreset(
                id="monet",
                name="Monet Impressionism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Claude Monet's impressionist style capturing light and atmosphere",
                visual_elements=["broken color", "light effects", "atmospheric perspective", "loose brushwork"],
                color_characteristics=["light pastels", "color reflections", "atmospheric blues", "sunrise/sunset tones"],
                technique_details=["plein air painting", "color mixing on canvas", "capturing moments", "light studies"],
                example_prompt="convert to Claude Monet impressionist style, broken color technique, capturing fleeting light effects, loose brushwork with visible strokes, atmospheric perspective, water reflections, garden scenes with dappled sunlight",
                compatible_with=["impressionism", "landscape", "garden_scenes"],
                tips=["Focus on light effects", "Use broken color", "Capture atmosphere"]
            ),
            
            "monet": StylePreset(
                id="monet",
                name="Monet Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Impressionist master of light and color",
                visual_elements=["light reflections", "water lilies", "atmospheric effects", "broken color"],
                color_characteristics=["soft pastels", "light effects", "natural colors", "atmospheric perspective"],
                technique_details=["plein air painting", "color patches", "light studies"],
                example_prompt="convert to Claude Monet impressionist painting with soft natural light like Water Lilies series, broken color patches, and atmospheric perspective",
                compatible_with=["impressionist", "watercolor", "landscape"],
                tips=["Focus on light and atmosphere", "Use broken color technique"]
            ),
            
            "picasso": StylePreset(
                id="picasso",
                name="Picasso Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Cubist art with geometric abstraction",
                visual_elements=["geometric shapes", "multiple perspectives", "fragmented forms", "angular faces"],
                color_characteristics=["earthy tones", "limited palette", "bold contrasts"],
                technique_details=["cubist fragmentation", "analytical approach", "geometric abstraction"],
                example_prompt="convert to Picasso analytical cubist painting with geometric fragmentation, multiple simultaneous viewpoints, and muted earth tones",
                compatible_with=["abstract", "modern_art", "geometric"],
                tips=["Fragment the subject", "Show multiple viewpoints"]
            ),
            
            "picasso_cubism": StylePreset(
                id="picasso_cubism",
                name="Picasso Cubism",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Pablo Picasso's revolutionary cubist style with geometric fragmentation",
                visual_elements=["geometric fragmentation", "multiple viewpoints", "angular shapes", "abstracted forms"],
                color_characteristics=["muted earth tones", "analytical palette", "browns and grays", "limited colors"],
                technique_details=["analytical cubism", "synthetic cubism", "collage elements", "flattened perspective"],
                example_prompt="convert to Picasso cubist style with geometric fragmentation, multiple simultaneous viewpoints, angular abstracted forms, muted earth tone palette, analytical cubism technique, flattened perspective planes",
                compatible_with=["abstract", "modernism", "avant_garde"],
                tips=["Fragment into geometric shapes", "Show multiple viewpoints", "Use earth tones"]
            ),
            
            "ralph_mcquarrie": StylePreset(
                id="ralph_mcquarrie",
                name="Ralph McQuarrie Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Classic Star Wars concept art in the style of Ralph McQuarrie",
                visual_elements=["iconic designs", "painted quality", "dramatic scenes", "heroic proportions"],
                color_characteristics=["muted tones", "atmospheric haze", "dramatic skies", "painterly texture"],
                technique_details=["traditional painting", "matte painting", "production design", "cinematic framing"],
                example_prompt="convert to Ralph McQuarrie concept art style, Star Wars production design aesthetic, traditional painted quality, muted color palette with atmospheric haze, dramatic sky compositions, heroic character proportions, classic sci-fi matte painting technique",
                compatible_with=["star_wars", "space_opera", "retro_scifi"],
                tips=["Use muted colors", "Add atmospheric haze", "Frame cinematically"]
            ),
            
            "sparth": StylePreset(
                id="sparth",
                name="Sparth (Nicolas Bouvier) Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Geometric sci-fi environments in the style of Halo concept artist Sparth",
                visual_elements=["geometric abstraction", "mega structures", "minimalist forms", "epic scale"],
                color_characteristics=["monochromatic schemes", "atmospheric gradients", "subtle color accents", "fog effects"],
                technique_details=["structural design", "atmospheric perspective", "geometric composition", "scale contrast"],
                example_prompt="convert to Sparth concept art style, geometric mega-structures, minimalist sci-fi architecture, monochromatic color scheme with subtle accents, massive scale with tiny human figures, atmospheric fog and depth, Halo-inspired environmental design, structural abstraction",
                compatible_with=["scifi_architecture", "brutalist", "megastructure"],
                tips=["Emphasize scale", "Use geometric shapes", "Keep colors minimal"]
            ),
            
            "syd_mead": StylePreset(
                id="syd_mead",
                name="Syd Mead Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Futuristic industrial design in the style of legendary concept artist Syd Mead",
                visual_elements=["streamlined forms", "chrome surfaces", "retro-futurism", "architectural integration"],
                color_characteristics=["chrome and glass", "neon accents", "sunset gradients", "reflective surfaces"],
                technique_details=["marker rendering", "airbrushed gradients", "technical precision", "atmospheric effects"],
                example_prompt="convert to Syd Mead concept art style, retro-futuristic vehicle design, streamlined chrome surfaces, integrated architecture, marker rendering technique with airbrushed gradients, sunset lighting with neon accents, blade runner aesthetic, visionary industrial design",
                compatible_with=["blade_runner", "retrofuturism", "vehicle_concept"],
                tips=["Emphasize reflections", "Use sunset lighting", "Integrate with architecture"]
            ),
            
            "van_gogh": StylePreset(
                id="van_gogh",
                name="Van Gogh Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Post-impressionist style of Vincent van Gogh with expressive brushwork",
                visual_elements=["swirling brushstrokes", "thick impasto", "dynamic movement", "emotional intensity"],
                color_characteristics=["vibrant yellows", "deep blues", "complementary contrasts", "luminous colors"],
                technique_details=["impasto technique", "directional brushwork", "expressive distortion", "visible texture"],
                example_prompt="convert to Van Gogh style painting with swirling expressive brushstrokes, thick impasto texture, vibrant yellows and deep blues, dynamic movement in sky and landscapes, post-impressionist emotional intensity, visible paint texture",
                compatible_with=["impressionism", "expressionism", "fauvism"],
                tips=["Emphasize brushstroke movement", "Use thick paint texture", "Vibrant color contrasts"]
            ),
            
            "van_gogh": StylePreset(
                id="van_gogh",
                name="Van Gogh Style",
                category=StyleCategory.FAMOUS_ARTISTS,
                description="Post-impressionist with swirling brushstrokes",
                visual_elements=["swirling brushstrokes", "thick impasto", "emotional intensity", "dynamic movement"],
                color_characteristics=["vibrant yellows and blues", "complementary colors", "emotional palette"],
                technique_details=["visible brushwork", "expressive technique", "textured paint"],
                example_prompt="convert to Van Gogh post-impressionist painting with dynamic swirling brushstrokes like Starry Night, vibrant yellows and blues, thick impasto texture",
                compatible_with=["impressionist", "oil_painting", "expressive"],
                tips=["Emphasize movement and emotion", "Use thick paint texture"]
            )
        }