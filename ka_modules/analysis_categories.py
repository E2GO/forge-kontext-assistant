"""
Analysis Categories for Enhanced Image Understanding
Детальные категории для улучшенного анализа изображений в Kontext Assistant
"""

class AnalysisCategories:
    """Comprehensive categories for image analysis"""
    
    # 1. FANTASY & FANTASTICAL ELEMENTS
    # Фэнтези и фантастические элементы
    FANTASY_ELEMENTS = {
        'body_modifications': {
            'horns': [
                'horn', 'horns', 'antler', 'antlers', 'spike on head',
                'demon horn', 'dragon horn', 'ram horn', 'curved horn',
                'bone protrusion', 'head spike'
            ],
            'wings': [
                'wing', 'wings', 'feathered wing', 'bat wing', 'dragon wing',
                'angel wing', 'butterfly wing', 'insect wing', 'membrane wing',
                'energy wing', 'mechanical wing', 'skeletal wing'
            ],
            'tails': [
                'tail', 'demon tail', 'dragon tail', 'cat tail', 'fox tail',
                'scaled tail', 'furry tail', 'prehensile tail', 'spaded tail',
                'scorpion tail', 'lizard tail'
            ],
            'ears': [
                'pointed ear', 'elf ear', 'cat ear', 'fox ear', 'long ear',
                'animal ear', 'bat ear', 'droopy ear', 'fin ear'
            ],
            'eyes': [
                'glowing eye', 'glowing eyes', 'heterochromia', 'third eye',
                'compound eye', 'cat eye', 'slit pupil', 'red eye', 'blue glow',
                'purple eye', 'golden eye', 'empty socket', 'mechanical eye'
            ],
            'skin': [
                'scales', 'fur', 'feathers', 'chitinous', 'crystalline skin',
                'stone skin', 'bark skin', 'metallic skin', 'translucent skin',
                'spotted skin', 'striped skin', 'blue skin', 'green skin',
                'purple skin', 'grey skin', 'pale skin'
            ],
            'limbs': [
                'claw', 'claws', 'talon', 'hoof', 'hooves', 'tentacle',
                'extra arm', 'four arms', 'six arms', 'mechanical limb',
                'prosthetic', 'bone arm', 'energy limb'
            ]
        },
        
        'magical_effects': {
            'auras': [
                'aura', 'glow', 'energy field', 'halo', 'nimbus',
                'holy light', 'dark aura', 'fire aura', 'ice aura',
                'lightning aura', 'divine light', 'ethereal glow'
            ],
            'particles': [
                'sparkles', 'floating particles', 'magical dust', 'embers',
                'snow particles', 'light orbs', 'energy sparks', 'glitter',
                'fireflies', 'magical motes', 'stardust'
            ],
            'energy': [
                'lightning', 'electricity', 'fire magic', 'ice shards',
                'energy beam', 'plasma', 'void energy', 'holy energy',
                'dark magic', 'arcane power', 'ki energy', 'chakra'
            ],
            'runes': [
                'glowing runes', 'magical symbols', 'arcane marks', 'sigils',
                'enchantment', 'spell circle', 'mystic inscription',
                'floating text', 'ancient script', 'power glyph'
            ]
        },
        
        'creature_types': {
            'humanoid': [
                'elf', 'dwarf', 'orc', 'tiefling', 'dragonborn', 'halfling',
                'gnome', 'drow', 'half-elf', 'aasimar', 'genasi', 'goblin',
                'hobgoblin', 'kobold', 'lizardfolk', 'tabaxi'
            ],
            'hybrid': [
                'centaur', 'minotaur', 'harpy', 'mermaid', 'satyr', 'naga',
                'lamia', 'sphinx', 'chimera', 'manticore', 'werewolf',
                'werebear', 'kitsune', 'selkie'
            ],
            'mythical': [
                'dragon', 'phoenix', 'griffin', 'griffon', 'unicorn',
                'pegasus', 'hydra', 'basilisk', 'kraken', 'behemoth',
                'leviathan', 'roc', 'thunderbird'
            ],
            'undead': [
                'skeleton', 'zombie', 'vampire', 'lich', 'wraith', 'ghost',
                'specter', 'banshee', 'mummy', 'revenant', 'death knight',
                'necromancer', 'bone creature'
            ],
            'celestial': [
                'angel', 'archangel', 'seraph', 'cherub', 'valkyrie',
                'divine being', 'celestial', 'solar', 'planetar'
            ],
            'demonic': [
                'demon', 'devil', 'imp', 'succubus', 'incubus', 'balor',
                'pit fiend', 'hellspawn', 'fiend', 'daemon'
            ],
            'elemental': [
                'fire elemental', 'water elemental', 'earth elemental',
                'air elemental', 'ice elemental', 'lightning elemental',
                'nature spirit', 'treant', 'dryad'
            ]
        },
        
        'medieval_fantasy': {
            'armor_types': [
                'plate armor', 'chainmail', 'chain mail', 'leather armor',
                'scale mail', 'brigandine', 'lamellar', 'studded leather',
                'ring mail', 'splint mail', 'half plate', 'full plate'
            ],
            'armor_pieces': [
                'helmet', 'helm', 'pauldron', 'breastplate', 'cuirass',
                'gauntlet', 'greaves', 'sabatons', 'vambrace', 'gorget',
                'tassets', 'couter', 'plackart', 'bevor'
            ],
            'weapons': [
                'sword', 'longsword', 'greatsword', 'rapier', 'katana',
                'staff', 'wand', 'bow', 'crossbow', 'mace', 'hammer',
                'warhammer', 'axe', 'battleaxe', 'spear', 'lance',
                'dagger', 'scimitar', 'falchion', 'halberd', 'glaive'
            ],
            'shields': [
                'shield', 'buckler', 'kite shield', 'tower shield',
                'round shield', 'heater shield', 'pavise'
            ],
            'accessories': [
                'crown', 'circlet', 'tiara', 'amulet', 'pendant', 'tome',
                'scroll', 'potion', 'crystal ball', 'orb', 'scepter',
                'medallion', 'talisman', 'holy symbol'
            ]
        }
    }
    
    # 2. MATERIALS & TEXTURES
    # Материалы и текстуры
    MATERIALS = {
        'metals': {
            'precious': [
                'gold', 'golden', 'silver', 'platinum', 'copper', 'rose gold',
                'white gold', 'electrum', 'palladium'
            ],
            'common': [
                'iron', 'steel', 'bronze', 'brass', 'pewter', 'tin',
                'lead', 'zinc', 'aluminum', 'titanium'
            ],
            'fantasy': [
                'mithril', 'adamantine', 'orichalcum', 'darksteel',
                'starmetal', 'cold iron', 'living steel', 'skyforge steel'
            ],
            'properties': [
                'polished', 'rusted', 'rusty', 'tarnished', 'etched',
                'damascus', 'engraved', 'hammered', 'brushed', 'oxidized',
                'patina', 'weathered metal', 'scratched', 'dented'
            ]
        },
        
        'fabrics': {
            'natural': [
                'cotton', 'wool', 'silk', 'linen', 'leather', 'suede',
                'fur', 'hide', 'felt', 'canvas', 'hemp', 'jute'
            ],
            'luxury': [
                'velvet', 'satin', 'brocade', 'lace', 'damask', 'taffeta',
                'chiffon', 'organza', 'cashmere', 'mohair', 'angora'
            ],
            'rough': [
                'burlap', 'canvas', 'denim', 'tweed', 'corduroy',
                'sackcloth', 'rough linen', 'homespun'
            ],
            'properties': [
                'embroidered', 'torn', 'ripped', 'weathered', 'sheer',
                'flowing', 'draped', 'pleated', 'quilted', 'woven',
                'knitted', 'crocheted', 'tattered', 'frayed'
            ]
        },
        
        'organic': {
            'wood': [
                'wood', 'wooden', 'oak', 'pine', 'ebony', 'mahogany',
                'birch', 'ash', 'maple', 'cherry', 'walnut', 'cedar',
                'bamboo', 'driftwood', 'petrified wood'
            ],
            'stone': [
                'stone', 'marble', 'granite', 'limestone', 'sandstone',
                'obsidian', 'slate', 'quartz', 'basalt', 'jade',
                'onyx', 'alabaster', 'soapstone'
            ],
            'natural': [
                'bone', 'horn', 'ivory', 'pearl', 'coral', 'amber',
                'shell', 'mother of pearl', 'nacre', 'chitin',
                'tooth', 'fang', 'claw material'
            ],
            'plant': [
                'leaves', 'bark', 'vines', 'flowers', 'petals', 'moss',
                'lichen', 'roots', 'branches', 'thorns', 'grass',
                'straw', 'reeds'
            ]
        },
        
        'synthetic': {
            'modern': [
                'plastic', 'rubber', 'glass', 'ceramic', 'porcelain',
                'resin', 'vinyl', 'acrylic', 'polymer', 'fiberglass'
            ],
            'tech': [
                'carbon fiber', 'titanium', 'chrome', 'nanofiber',
                'composite', 'kevlar', 'synthetic diamond', 'graphene'
            ],
            'energy': [
                'plasma', 'holographic', 'ethereal', 'crystalline',
                'energy construct', 'hard light', 'force field',
                'luminescent', 'bioluminescent'
            ]
        }
    }
    
    # 3. FINE DETAILS
    # Мелкие детали
    FINE_DETAILS = {
        'jewelry': {
            'types': [
                'ring', 'rings', 'necklace', 'bracelet', 'earring', 'brooch',
                'anklet', 'armlet', 'torc', 'chain', 'locket', 'choker',
                'diadem', 'bangle', 'cuff', 'nose ring', 'belly chain'
            ],
            'gems': [
                'diamond', 'ruby', 'emerald', 'sapphire', 'amethyst',
                'topaz', 'opal', 'garnet', 'aquamarine', 'citrine',
                'peridot', 'turquoise', 'lapis lazuli', 'moonstone',
                'obsidian', 'crystal', 'gem', 'jewel'
            ],
            'settings': [
                'pendant', 'chain', 'beads', 'filigree', 'setting',
                'clasp', 'prongs', 'bezel', 'pave', 'channel set'
            ]
        },
        
        'body_art': {
            'tattoos': [
                'tattoo', 'tribal tattoo', 'runic tattoo', 'geometric tattoo',
                'sleeve tattoo', 'face tattoo', 'neck tattoo', 'magical tattoo',
                'glowing tattoo', 'moving tattoo', 'ink', 'body art'
            ],
            'scars': [
                'scar', 'battle scar', 'burn mark', 'claw marks', 'slash',
                'old wound', 'healed wound', 'ritual scarring', 'brand',
                'bite mark', 'surgical scar'
            ],
            'paint': [
                'war paint', 'face paint', 'body paint', 'ceremonial paint',
                'tribal markings', 'makeup', 'cosmetics', 'kohl',
                'henna', 'ritual markings'
            ],
            'piercings': [
                'piercing', 'ear piercing', 'nose piercing', 'lip piercing',
                'eyebrow piercing', 'septum', 'gauge', 'stud', 'hoop'
            ]
        },
        
        'decorative': {
            'patterns': [
                'embroidery', 'engravings', 'engraved', 'inlay', 'relief',
                'filigree', 'scrollwork', 'knotwork', 'celtic knots',
                'paisley', 'floral pattern', 'geometric pattern'
            ],
            'symbols': [
                'heraldry', 'coat of arms', 'religious symbol', 'holy symbol',
                'clan mark', 'sigil', 'crest', 'emblem', 'insignia',
                'rune', 'glyph', 'seal'
            ],
            'trim': [
                'gold trim', 'silver trim', 'fur trim', 'lace trim',
                'leather trim', 'fringe', 'tassels', 'braiding',
                'piping', 'ribbon', 'braid'
            ],
            'fastenings': [
                'buckle', 'button', 'laces', 'straps', 'clasps',
                'hooks', 'ties', 'toggles', 'zipper', 'velcro'
            ]
        }
    }
    
    # 4. LIGHTING & ATMOSPHERE
    # Освещение и атмосфера
    LIGHTING_ATMOSPHERE = {
        'time_of_day': {
            'dawn': [
                'dawn', 'sunrise', 'first light', 'daybreak', 'morning twilight',
                'pre-dawn', 'early morning', 'breaking dawn'
            ],
            'morning': [
                'morning', 'morning light', 'morning sun', 'soft daylight',
                'mid-morning', 'late morning', 'morning glow'
            ],
            'noon': [
                'noon', 'midday', 'midday sun', 'high noon', 'harsh sunlight',
                'bright daylight', 'overhead sun'
            ],
            'afternoon': [
                'afternoon', 'afternoon light', 'late afternoon',
                'afternoon sun', 'warm afternoon'
            ],
            'evening': [
                'evening', 'golden hour', 'sunset', 'dusk', 'twilight',
                'magic hour', 'sundown', 'eventide'
            ],
            'night': [
                'night', 'nighttime', 'midnight', 'moonlight', 'starlight',
                'darkness', 'night sky', 'nocturnal'
            ]
        },
        
        'light_quality': {
            'natural': [
                'sunlight', 'daylight', 'moonlight', 'starlight',
                'cloudy', 'overcast', 'diffused light', 'filtered light',
                'dappled light', 'natural lighting'
            ],
            'artificial': [
                'candlelight', 'torchlight', 'lamplight', 'firelight',
                'magical glow', 'neon light', 'fluorescent', 'incandescent',
                'LED light', 'street light', 'lantern light'
            ],
            'dramatic': [
                'rim lighting', 'backlight', 'backlighting', 'silhouette',
                'chiaroscuro', 'high contrast', 'low key', 'high key',
                'split lighting', 'rembrandt lighting', 'contre-jour'
            ],
            'color_temp': [
                'warm light', 'cool light', 'golden light', 'blue light',
                'orange glow', 'purple light', 'green tint', 'red lighting'
            ]
        },
        
        'atmosphere': {
            'weather': [
                'fog', 'mist', 'haze', 'rain', 'drizzle', 'snow', 'snowfall',
                'storm', 'thunderstorm', 'blizzard', 'wind', 'breeze',
                'clear sky', 'cloudy', 'partly cloudy', 'overcast'
            ],
            'effects': [
                'god rays', 'crepuscular rays', 'lens flare', 'bokeh',
                'depth of field', 'motion blur', 'light shafts',
                'volumetric lighting', 'atmospheric perspective',
                'light bloom', 'glare', 'halo effect'
            ],
            'mood': [
                'ethereal', 'mystical', 'ominous', 'serene', 'mysterious',
                'eerie', 'peaceful', 'dramatic', 'romantic', 'melancholic',
                'hopeful', 'apocalyptic', 'dreamlike', 'surreal'
            ],
            'environment': [
                'dusty', 'smoky', 'steamy', 'humid', 'dry', 'crisp air',
                'thick atmosphere', 'thin air', 'underground', 'underwater'
            ]
        }
    }
    
    # 5. ARTISTIC STYLES
    # Стилистические особенности
    ARTISTIC_STYLES = {
        'traditional_art': {
            'painting': [
                'oil painting', 'watercolor', 'watercolour', 'acrylic painting',
                'gouache', 'tempera', 'fresco', 'encaustic', 'mixed media'
            ],
            'drawing': [
                'pencil drawing', 'pencil sketch', 'charcoal drawing',
                'ink drawing', 'pen and ink', 'pastel', 'chalk',
                'graphite', 'colored pencil', 'marker art'
            ],
            'printmaking': [
                'etching', 'engraving', 'woodcut', 'linocut', 'lithograph',
                'screen print', 'monotype', 'relief print'
            ],
            'historical': [
                'renaissance', 'baroque', 'rococo', 'neoclassical',
                'romantic', 'impressionist', 'post-impressionist',
                'art nouveau', 'art deco', 'expressionist',
                'surrealist', 'abstract', 'cubist'
            ]
        },
        
        'digital_art': {
            'rendering': [
                '3d render', '3d rendering', 'cgi', 'photorealistic',
                'hyperrealistic', 'stylized 3d', 'low poly', 'high poly',
                'ray traced', 'path traced', 'pbr render'
            ],
            'painting': [
                'digital painting', 'digital art', 'concept art',
                'matte painting', 'speed painting', 'photobash',
                'digital illustration', 'vector art'
            ],
            'effects': [
                'cel shaded', 'cell shaded', 'toon shading', 'flat shading',
                'gradient shading', 'soft shading', 'hard edges',
                'painterly', 'glossy', 'matte finish'
            ],
            'technical': [
                'wireframe', 'blueprint', 'technical drawing', 'cad',
                'architectural rendering', 'product visualization'
            ]
        },
        
        'cultural_styles': {
            'eastern': [
                'anime', 'manga', 'manhwa', 'ukiyo-e', 'sumi-e',
                'chinese painting', 'chinese watercolor', 'oriental',
                'japanese art', 'korean art', 'thai art'
            ],
            'western': [
                'comic book', 'american comic', 'european comic',
                'cartoon', 'disney style', 'pixar style', 'dreamworks',
                'western animation', 'graphic novel'
            ],
            'gaming': [
                'video game art', 'game concept', 'fantasy art',
                'sci-fi art', 'cyberpunk', 'steampunk', 'dieselpunk',
                'dark fantasy', 'high fantasy', 'jrpg style'
            ],
            'contemporary': [
                'modern art', 'contemporary', 'street art', 'graffiti',
                'pop art', 'minimalist', 'maximalist', 'abstract'
            ]
        },
        
        'technique': {
            'brushwork': [
                'loose brushstrokes', 'tight brushwork', 'detailed',
                'impressionistic', 'expressive', 'controlled',
                'gestural', 'precise', 'rough', 'smooth'
            ],
            'color': [
                'vibrant', 'vivid', 'saturated', 'muted', 'desaturated',
                'monochromatic', 'limited palette', 'full color',
                'pastel colors', 'earth tones', 'jewel tones',
                'complementary colors', 'analogous colors'
            ],
            'composition': [
                'dynamic composition', 'static composition', 'symmetrical',
                'asymmetrical', 'rule of thirds', 'golden ratio',
                'centered', 'off-center', 'diagonal', 'triangular',
                'circular', 'leading lines', 'framing'
            ],
            'texture': [
                'smooth', 'rough', 'textured', 'impasto', 'glazed',
                'layered', 'flat', 'dimensional', 'tactile', 'visual texture'
            ]
        }
    }
    
    @classmethod
    def get_all_keywords(cls, category=None):
        """
        Получить все ключевые слова из категории или всех категорий
        
        Args:
            category: имя категории (FANTASY_ELEMENTS, MATERIALS и т.д.)
                     если None - вернуть все ключевые слова
        
        Returns:
            set: уникальные ключевые слова
        """
        keywords = set()
        
        if category:
            if hasattr(cls, category):
                cls._extract_keywords(getattr(cls, category), keywords)
        else:
            # Извлекаем из всех категорий
            for attr_name in dir(cls):
                if attr_name.isupper() and isinstance(getattr(cls, attr_name), dict):
                    cls._extract_keywords(getattr(cls, attr_name), keywords)
        
        return keywords
    
    @classmethod
    def _extract_keywords(cls, data, keywords):
        """Рекурсивное извлечение ключевых слов из структуры"""
        if isinstance(data, dict):
            for value in data.values():
                cls._extract_keywords(value, keywords)
        elif isinstance(data, list):
            keywords.update(data)
        elif isinstance(data, str):
            keywords.add(data)


# Вспомогательные функции для работы с категориями
def find_category_for_keyword(keyword):
    """
    Найти категорию и подкатегорию для заданного ключевого слова
    
    Args:
        keyword: искомое ключевое слово
    
    Returns:
        tuple: (category, subcategory, subsubcategory) или None
    """
    keyword_lower = keyword.lower()
    
    for category_name in dir(AnalysisCategories):
        if category_name.isupper():
            category = getattr(AnalysisCategories, category_name)
            if isinstance(category, dict):
                result = _search_in_category(category, keyword_lower, category_name)
                if result:
                    return result
    
    return None


def _search_in_category(category_dict, keyword, category_name, path=[]):
    """Рекурсивный поиск в категории"""
    for key, value in category_dict.items():
        current_path = path + [key]
        
        if isinstance(value, dict):
            result = _search_in_category(value, keyword, category_name, current_path)
            if result:
                return result
        elif isinstance(value, list):
            if keyword in [item.lower() for item in value]:
                return (category_name, *current_path)
    
    return None


# Тестовая функция для проверки работы категорий
if __name__ == "__main__":
    # Простой тест
    print("Testing AnalysisCategories...")
    
    # Тест поиска ключевого слова
    test_keywords = ['dragon horn', 'velvet', 'golden hour', 'oil painting']
    
    for kw in test_keywords:
        result = find_category_for_keyword(kw)
        if result:
            print(f"'{kw}' found in: {' > '.join(result)}")
        else:
            print(f"'{kw}' not found")
    
    # Подсчет общего количества ключевых слов
    all_keywords = AnalysisCategories.get_all_keywords()
    print(f"\nTotal keywords: {len(all_keywords)}")
    
    # Подсчет по категориям
    for cat in ['FANTASY_ELEMENTS', 'MATERIALS', 'FINE_DETAILS', 
                'LIGHTING_ATMOSPHERE', 'ARTISTIC_STYLES']:
        cat_keywords = AnalysisCategories.get_all_keywords(cat)
        print(f"{cat}: {len(cat_keywords)} keywords")