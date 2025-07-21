"""
Cultural styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

CULTURAL_STYLES = {
            "aboriginal": StylePreset(
                id="aboriginal",
                name="Aboriginal Dot Painting",
                category=StyleCategory.CULTURAL,
                description="Australian indigenous art style",
                visual_elements=["dot patterns", "dreamtime symbols", "earth connections", "circular motifs"],
                color_characteristics=["earth tones", "ochre colors", "natural pigments"],
                technique_details=["dot application", "symbolic patterns", "storytelling elements"],
                example_prompt="convert to Aboriginal dot painting with intricate circular patterns, ochre earth tones, and dreamtime symbolism",
                compatible_with=["tribal", "spiritual", "earth"],
                tips=["Use dots for texture", "Include symbolic elements"]
            ),
            
            "ancient_egyptian": StylePreset(
                id="ancient_egyptian",
                name="Ancient Egyptian",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Egyptian palace and temple setting",
                visual_elements=["pyramids", "hieroglyphs", "pharaoh regalia", "columns", "sphinx statues"],
                color_characteristics=["gold and lapis blue", "sandstone", "turquoise", "desert tones"],
                technique_details=["hieroglyphic decorations", "profile art style", "monumental scale", "ritual objects"],
                example_prompt="transform to ancient Egyptian setting, massive stone columns with hieroglyphs, people in linen clothing and golden jewelry, pharaoh headdresses, Eye of Horus symbols, pyramid structures, sphinx statues, papyrus scrolls, golden sarcophagi, desert sand backdrop, monumental architecture scale",
                compatible_with=["ancient", "desert", "monumental"],
                tips=["Add hieroglyphs everywhere", "Gold and blue palette", "Monumental scale"]
            ),
            
            "ancient_slavic": StylePreset(
                id="ancient_slavic",
                name="Ancient Slavic Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Slavic/Russian folklore aesthetic",
                visual_elements=["wooden architecture", "folk patterns", "traditional clothing", "pagan symbols", "birch forests"],
                color_characteristics=["rich reds", "deep blues", "gold accents", "natural wood tones", "white birch"],
                technique_details=["ornate wood carving", "folk art patterns", "traditional embroidery", "onion domes"],
                example_prompt="transform to ancient Slavic cultural setting, wooden log architecture with carved details, people in traditional vyshyvanka embroidered clothing, kokoshnik headdresses, folk patterns and symbols, birch forest surroundings, Orthodox church domes in distance, rich red and gold color scheme, firebird and folkloric elements",
                compatible_with=["medieval", "fantasy", "folklore"],
                tips=["Add folk patterns", "Use red and gold", "Include traditional elements"]
            ),
            
            "arabian_nights": StylePreset(
                id="arabian_nights",
                name="Arabian Nights Fantasy",
                category=StyleCategory.CULTURAL,
                description="Transform to magical Middle Eastern palace setting",
                visual_elements=["minarets", "geometric patterns", "flying carpets", "oil lamps", "desert palace"],
                color_characteristics=["rich jewel tones", "gold trim", "turquoise", "sunset oranges", "deep purples"],
                technique_details=["islamic geometry", "arabesque patterns", "calligraphy", "ornate details"],
                example_prompt="transform to Arabian Nights setting, ornate palace with geometric tile patterns, minarets and onion domes, people in flowing robes and turbans, magic carpets, brass oil lamps, intricate arabesque decorations, fountain courtyards, silk curtains, golden treasures, desert oasis backdrop",
                compatible_with=["fantasy", "desert", "magical"],
                tips=["Geometric patterns", "Rich jewel colors", "Ornate details"]
            ),
            
            "aztec": StylePreset(
                id="aztec",
                name="Aztec Art",
                category=StyleCategory.CULTURAL,
                description="Pre-Columbian Mesoamerican style",
                visual_elements=["geometric patterns", "feathered serpents", "sun symbols", "stepped pyramids"],
                color_characteristics=["turquoise and gold", "jade green", "blood red"],
                technique_details=["stone carving style", "symbolic imagery", "hierarchical scale"],
                example_prompt="convert to Aztec art with stepped geometric patterns, turquoise and gold colors, symbolic imagery, and pre-Columbian stone carving aesthetics",
                compatible_with=["mayan", "ancient", "symbolic"],
                tips=["Use stepped patterns", "Add symbolic animals"]
            ),
            
            "aztec_mayan": StylePreset(
                id="aztec_mayan",
                name="Aztec-Mayan Civilization",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Mesoamerican pyramid city",
                visual_elements=["step pyramids", "jade ornaments", "feathered headdresses", "stone carvings", "jungle setting"],
                color_characteristics=["jade green", "gold", "terracotta", "bright feathers", "jungle greens"],
                technique_details=["geometric patterns", "ritual masks", "calendar stones", "relief carvings"],
                example_prompt="transform to Aztec-Mayan setting, massive step pyramids with steep stairs, people in feathered headdresses and jade jewelry, intricate stone carvings, serpent motifs, ritual masks, jungle vegetation, colorful murals, obsidian and gold ornaments, hieroglyphic calendar stones, ceremonial plaza",
                compatible_with=["ancient", "jungle", "ceremonial"],
                tips=["Geometric patterns", "Feathered decorations", "Jungle integration"]
            ),
            
            "celtic_druid": StylePreset(
                id="celtic_druid",
                name="Celtic Druid Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to ancient Celtic mystical setting",
                visual_elements=["stone circles", "celtic knots", "druid robes", "ancient oaks", "mist"],
                color_characteristics=["forest green", "stone grey", "mystic blue", "earth browns"],
                technique_details=["knotwork patterns", "stone carving", "natural magic", "circular designs"],
                example_prompt="transform to Celtic druid setting, ancient stone circles like Stonehenge, people in hooded druid robes, celtic knotwork patterns, sacred oak groves, mystical mist, carved standing stones with spiral designs, torcs and bronze jewelry, woad face paint, ritual fires, connection to nature magic",
                compatible_with=["fantasy", "mystical", "ancient"],
                tips=["Celtic knot patterns", "Stone circles", "Mystical atmosphere"]
            ),
            
            "chinese_imperial": StylePreset(
                id="chinese_imperial",
                name="Chinese Imperial Dynasty",
                category=StyleCategory.CULTURAL,
                description="Transform to Chinese imperial palace setting",
                visual_elements=["forbidden city", "dragon motifs", "silk robes", "jade ornaments", "pagoda roofs"],
                color_characteristics=["imperial yellow", "jade green", "lacquer red", "gold accents"],
                technique_details=["dragon symbolism", "calligraphy", "porcelain", "silk painting"],
                example_prompt="transform to Chinese imperial setting, Forbidden City architecture with golden roofs, people in silk hanfu robes with dragon embroidery, jade ornaments, red lacquered pillars, imperial throne room, porcelain vases, chinese calligraphy scrolls, guardian lion statues, traditional garden with koi pond",
                compatible_with=["asian", "royal", "traditional"],
                tips=["Dragon motifs", "Imperial colors", "Hierarchical details"]
            ),
            
            "fantasy_elven": StylePreset(
                id="fantasy_elven",
                name="Elven Fantasy Realm",
                category=StyleCategory.CULTURAL,
                description="Transform to elegant elven civilization aesthetic",
                visual_elements=["organic architecture", "living trees", "elegant clothing", "glowing crystals", "nature integration"],
                color_characteristics=["forest greens", "silver and white", "ethereal glows", "natural lights"],
                technique_details=["flowing organic lines", "nature-integrated design", "delicate details", "magical elements"],
                example_prompt="transform to elven fantasy realm, organic architecture grown from living trees, elegant flowing robes with silver embroidery, pointed ears on characters, glowing magical crystals, delicate filigree metalwork, integration with nature, ethereal lighting, mystical forest setting, elvish script decorations",
                compatible_with=["fantasy", "magical", "forest"],
                tips=["Integrate with nature", "Add ethereal glows", "Elegant flowing designs"]
            ),
            
            "hokusai": StylePreset(
                id="hokusai",
                name="Hokusai Ukiyo-e",
                category=StyleCategory.CULTURAL,
                description="Katsushika Hokusai's iconic Japanese woodblock print style",
                visual_elements=["wave patterns", "Mount Fuji", "dynamic composition", "nature forces"],
                color_characteristics=["Prussian blue", "limited palette", "gradient skies", "white foam"],
                technique_details=["woodblock printing", "bold outlines", "flat color areas", "dynamic movement"],
                example_prompt="convert to Hokusai ukiyo-e style, bold woodblock print aesthetic, Great Wave composition, Prussian blue dominant, Mount Fuji in background, dynamic natural forces, traditional Japanese art technique",
                compatible_with=["ukiyo_e", "japanese_art", "woodblock"],
                tips=["Use Prussian blue prominently", "Add dynamic wave patterns", "Include Mount Fuji"]
            ),
            
            "japanese_edo": StylePreset(
                id="japanese_edo",
                name="Japanese Edo Period",
                category=StyleCategory.CULTURAL,
                description="Transform to traditional Japanese Edo period setting",
                visual_elements=["pagodas", "cherry blossoms", "kimono", "paper lanterns", "wooden bridges"],
                color_characteristics=["sakura pink", "indigo blue", "red torii", "natural wood"],
                technique_details=["ukiyo-e influence", "architectural precision", "zen aesthetics", "seasonal elements"],
                example_prompt="transform to Japanese Edo period, traditional wooden architecture with curved roofs, people in elaborate kimono with obi, cherry blossom trees, red torii gates, paper lanterns, koi ponds with wooden bridges, mount fuji in distance, ukiyo-e art influence, tea ceremony elements, zen garden features",
                compatible_with=["asian", "historical", "zen"],
                tips=["Add cherry blossoms", "Traditional architecture", "Seasonal elements"]
            ),
            
            "medieval_european": StylePreset(
                id="medieval_european",
                name="Medieval European",
                category=StyleCategory.CULTURAL,
                description="Transform to medieval European castle and village life",
                visual_elements=["stone castles", "knight armor", "medieval clothing", "heraldry", "market squares"],
                color_characteristics=["stone greys", "royal purples", "heraldic colors", "torch lighting"],
                technique_details=["Gothic architecture", "illuminated manuscripts", "tapestry style", "medieval crafts"],
                example_prompt="transform to medieval European setting, stone castle architecture with Gothic elements, people in period-accurate tunics and dresses, knights in armor, heraldic banners and shields, cobblestone streets, market stalls, thatched roof houses, torch and candlelight illumination, medieval atmosphere",
                compatible_with=["fantasy", "historical", "castle"],
                tips=["Add heraldry", "Stone and wood materials", "Period lighting"]
            ),
            
            "norse_viking": StylePreset(
                id="norse_viking",
                name="Norse Viking Culture",
                category=StyleCategory.CULTURAL,
                description="Transform to Norse Viking settlement and mythology",
                visual_elements=["longhouses", "rune stones", "viking ships", "fur clothing", "norse symbols"],
                color_characteristics=["iron grey", "fur browns", "ice blue", "blood red", "gold metal"],
                technique_details=["runic inscriptions", "wood carving", "metalwork", "woven patterns"],
                example_prompt="transform to Norse Viking setting, wooden longhouses with dragon head decorations, warriors in fur cloaks and iron helmets, rune stone monuments, viking longships, Thor's hammer symbols, mead halls with carved pillars, northern lights sky, fjord landscape, norse mythology elements",
                compatible_with=["medieval", "warrior", "mythology"],
                tips=["Add runes", "Fur and iron materials", "Dragon motifs"]
            ),
            
            "ukiyo_e": StylePreset(
                id="ukiyo_e",
                name="Ukiyo-e",
                category=StyleCategory.CULTURAL,
                description="Japanese woodblock print style",
                visual_elements=["flat colors", "bold outlines", "wave patterns", "nature scenes"],
                color_characteristics=["limited palette", "indigo blues", "natural tones"],
                technique_details=["woodblock texture", "layered printing", "precise lines"],
                example_prompt="convert to Ukiyo-e Japanese woodblock print with flat color areas, bold black outlines, traditional wave patterns, and limited color palette",
                compatible_with=["japanese", "traditional", "nature"],
                tips=["Use limited colors", "Add wave patterns"]
            )
        }