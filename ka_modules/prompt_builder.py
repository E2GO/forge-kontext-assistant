"""
Prompt Builder module for Kontext Assistant
Handles prompt generation for various scenarios like object manipulation, style transfer, etc.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from ka_modules.token_utils import validate_prompt as validate_token_count, get_token_display

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompt generation scenarios"""
    OBJECT_ADD = "object_add"
    OBJECT_REMOVE = "object_remove"
    OBJECT_REPLACE = "object_replace"
    STYLE_TRANSFER = "style_transfer"
    POSE_CHANGE = "pose_change"
    EMOTION_CHANGE = "emotion_change"
    DETAIL_ENHANCEMENT = "detail_enhancement"  # Deprecated - use IMAGE_ENHANCEMENT
    COMPOSITION_MERGE = "composition_merge"
    ENVIRONMENT_CHANGE = "environment_change"
    LIGHTING_CHANGE = "lighting_change"
    IMAGE_ENHANCEMENT = "image_enhancement"  # Combines detail enhancement and restoration
    OUTPAINTING = "outpainting"
    CUSTOM_PROMPT = "custom_prompt"  # New custom prompt builder


class PreserveOptions(Enum):
    """What to preserve from original image"""
    SUBJECT_IDENTITY = "subject_identity"
    COMPOSITION = "composition"
    SCENE_CONTENT = "scene_content"
    LIGHTING = "lighting"
    COLOR_PALETTE = "color_palette"
    ARTISTIC_STYLE = "artistic_style"


class RemovalScenario(Enum):
    """Predefined removal scenarios"""
    CLOTHING = "clothing"
    ACCESSORIES = "accessories"
    BACKGROUND_OBJECT = "background_object"
    PERSON = "person"
    TEXT_LOGOS = "text_logos"
    WATERMARK = "watermark"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"
    CUSTOM = "custom"


class EnhancementScenario(Enum):
    """Types of image enhancement and restoration"""
    # Detail Enhancement
    FACIAL_DETAILS = "facial_details"
    EYE_ENHANCEMENT = "eye_enhancement"
    TEXTURE_DETAILS = "texture_details"
    HAIR_DETAILS = "hair_details"
    OVERALL_SHARPNESS = "overall_sharpness"
    # Restoration
    OLD_PHOTO = "old_photo"
    DAMAGED_ART = "damaged_art"
    LOW_RESOLUTION = "low_resolution"
    BLURRY = "blurry"
    EXPOSURE_ISSUES = "exposure_issues"
    NOISY = "noisy"
    COMPRESSED = "compressed"
    FADED_COLORS = "faded_colors"
    # Combined
    GENERAL_ENHANCEMENT = "general_enhancement"
    CUSTOM = "custom"

# Keep for backward compatibility
RestorationScenario = EnhancementScenario


class OutpaintingDirection(Enum):
    """Outpainting direction options"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ALL_SIDES = "all_sides"
    WIDESCREEN = "widescreen"
    SQUARE = "square"
    CUSTOM = "custom"


class LightingScenario(Enum):
    """Types of lighting scenarios"""
    NATURAL = "natural"
    STUDIO = "studio"
    DRAMATIC = "dramatic"
    AMBIENT = "ambient"
    NEON = "neon"
    CANDLELIGHT = "candlelight"
    SUNSET = "sunset"
    NIGHT = "night"
    UNDERWATER = "underwater"
    BACKLIT = "backlit"
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    name: str
    type: PromptType
    template: str
    placeholders: List[str]
    examples: List[Dict[str, str]]
    tips: List[str]


class PromptBuilder:
    """Main class for building Kontext prompts"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.common_objects = self._initialize_common_objects()
        self.position_descriptors = self._initialize_positions()
        
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize prompt templates for each scenario"""
        templates = {
            PromptType.OBJECT_ADD: PromptTemplate(
                name="Add Object",
                type=PromptType.OBJECT_ADD,
                template="Add {object} {position} {style_modifier}",
                placeholders=["object", "position", "style_modifier"],
                examples=[
                    {"object": "a red balloon", "position": "in the character's left hand", "style_modifier": "floating gently"},
                    {"object": "sunglasses", "position": "on the person's face", "style_modifier": "reflecting the surroundings"},
                    {"object": "a golden crown", "position": "on top of the head", "style_modifier": "with jeweled details"}
                ],
                tips=[
                    "Be specific about the object's appearance (color, size, material)",
                    "Use clear positional references (left/right, foreground/background)",
                    "Include style modifiers to match the image aesthetic"
                ]
            ),
            
            PromptType.OBJECT_REMOVE: PromptTemplate(
                name="Remove Object",
                type=PromptType.OBJECT_REMOVE,
                template="Remove {object_description} and {fill_method}",
                placeholders=["object_description", "fill_method"],
                examples=[
                    {"object_description": "the person in the red shirt on the left", "fill_method": "blend seamlessly with the background"},
                    {"object_description": "all text and logos", "fill_method": "maintain the original surface texture"},
                    {"object_description": "the car in the background", "fill_method": "extend the landscape naturally"},
                    {"object_description": "the jacket", "fill_method": "reveal what's underneath naturally"},
                    {"object_description": "the hat from their head", "fill_method": "show their natural hair"},
                    {"object_description": "the glasses", "fill_method": "show their face clearly"}
                ],
                tips=[
                    "Describe the object's location precisely",
                    "For clothing: use 'reveal what's underneath' or specify what should be shown",
                    "For accessories: mention what should appear instead",
                    "Use 'all' for multiple similar objects"
                ]
            ),
            
            PromptType.OBJECT_REPLACE: PromptTemplate(
                name="Replace Object",
                type=PromptType.OBJECT_REPLACE,
                template="Replace {original_object} with {new_object} {maintain_aspects}",
                placeholders=["original_object", "new_object", "maintain_aspects"],
                examples=[
                    {"original_object": "the coffee cup", "new_object": "a tea cup with steam", "maintain_aspects": "keeping the same position and lighting"},
                    {"original_object": "the sedan car", "new_object": "a sports car", "maintain_aspects": "matching the perspective and color scheme"},
                    {"original_object": "the wooden chair", "new_object": "a modern office chair", "maintain_aspects": "preserving the shadow and scale"}
                ],
                tips=[
                    "Clearly identify what to replace",
                    "Describe the replacement in detail",
                    "Specify what aspects to maintain (size, position, lighting)"
                ]
            ),
            
            PromptType.STYLE_TRANSFER: PromptTemplate(
                name="Style Transfer",
                type=PromptType.STYLE_TRANSFER,
                template="Convert to {style_description}, {preserve_elements}",
                placeholders=["style_description", "preserve_elements"],
                examples=[
                    {"style_description": "oil painting style with visible brushstrokes and rich colors", "preserve_elements": "while maintaining the subject's identity and composition"},
                    {"style_description": "cyberpunk aesthetic with neon colors and futuristic elements", "preserve_elements": "while keeping the original scene layout and main subjects"},
                    {"style_description": "watercolor art with soft edges and translucent washes", "preserve_elements": "while preserving the mood and facial expressions"}
                ],
                tips=[
                    "Use 'convert to' or 'make as' for better results",
                    "Describe visual characteristics of the style",
                    "Mention specific techniques or effects",
                    "Always use 'while' before preservation elements"
                ]
            ),
            
            PromptType.POSE_CHANGE: PromptTemplate(
                name="Change Pose",
                type=PromptType.POSE_CHANGE,
                template="Change the {subject} pose to {new_pose} {additional_details}",
                placeholders=["subject", "new_pose", "additional_details"],
                examples=[
                    {"subject": "person", "new_pose": "sitting cross-legged", "additional_details": "with hands resting on knees, maintaining the same outfit"},
                    {"subject": "character", "new_pose": "standing with arms crossed", "additional_details": "confident stance, same facial expression"},
                    {"subject": "figure", "new_pose": "walking motion mid-stride", "additional_details": "natural movement, preserve clothing style"}
                ],
                tips=[
                    "Describe the full body position",
                    "Include hand and arm positions",
                    "Mention what to keep (clothing, expression, etc.)"
                ]
            ),
            
            PromptType.EMOTION_CHANGE: PromptTemplate(
                name="Change Emotion",
                type=PromptType.EMOTION_CHANGE,
                template="Change {subject} expression to {emotion} {intensity}",
                placeholders=["subject", "emotion", "intensity"],
                examples=[
                    {"subject": "the person's", "emotion": "genuine laughter", "intensity": "with eyes crinkled and mouth open"},
                    {"subject": "the character's", "emotion": "thoughtful contemplation", "intensity": "subtle with furrowed brow"},
                    {"subject": "their", "emotion": "surprised excitement", "intensity": "wide eyes and raised eyebrows"}
                ],
                tips=[
                    "Describe facial features involved",
                    "Specify intensity (subtle, moderate, intense)",
                    "Consider full face coherence"
                ]
            ),
            
            # Deprecated - kept for compatibility
            PromptType.DETAIL_ENHANCEMENT: PromptTemplate(
                name="Enhance Details",
                type=PromptType.DETAIL_ENHANCEMENT,
                template="Enhance {area} by {enhancement_type} {specific_changes}",
                placeholders=["area", "enhancement_type", "specific_changes"],
                examples=[
                    {"area": "the eyes", "enhancement_type": "adding more detail", "specific_changes": "with clearer iris patterns and natural highlights"},
                    {"area": "the fabric texture", "enhancement_type": "increasing definition", "specific_changes": "showing individual threads and material properties"},
                    {"area": "the hair", "enhancement_type": "improving realism", "specific_changes": "with individual strands and natural flow"}
                ],
                tips=[
                    "Focus on specific areas",
                    "Describe the type of enhancement",
                    "Be specific about desired changes"
                ]
            ),
            
            PromptType.LIGHTING_CHANGE: PromptTemplate(
                name="Change Lighting",
                type=PromptType.LIGHTING_CHANGE,
                template="Change lighting to {lighting_type} with {specific_effects} {adjustments}",
                placeholders=["lighting_type", "specific_effects", "adjustments"],
                examples=[
                    {"lighting_type": "golden hour sunset", "specific_effects": "warm orange glow and long shadows", "adjustments": "while maintaining subject details"},
                    {"lighting_type": "dramatic studio lighting", "specific_effects": "strong key light from left with rim lighting", "adjustments": "creating depth and contrast"},
                    {"lighting_type": "soft natural window light", "specific_effects": "diffused daylight from the right", "adjustments": "with gentle shadows"},
                    {"lighting_type": "neon cyberpunk lighting", "specific_effects": "pink and blue neon reflections", "adjustments": "on wet surfaces"},
                    {"lighting_type": "candlelit ambiance", "specific_effects": "warm flickering glow with soft shadows", "adjustments": "creating intimate mood"}
                ],
                tips=[
                    "Specify light direction (left, right, above, etc.)",
                    "Mention color temperature (warm, cool, neutral)",
                    "Include shadow characteristics (soft, hard, long)",
                    "Consider reflections and highlights",
                    "Describe the mood you want to create"
                ]
            ),
            
            PromptType.IMAGE_ENHANCEMENT: PromptTemplate(
                name="Image Enhancement & Restoration",
                type=PromptType.IMAGE_ENHANCEMENT,
                template="Enhance {enhancement_target} by {method} to achieve {quality_goal}",
                placeholders=["enhancement_target", "method", "quality_goal"],
                examples=[
                    # Detail enhancement examples
                    {"enhancement_target": "the eyes", "method": "adding more detail with clearer iris patterns", "quality_goal": "photorealistic eye detail"},
                    {"enhancement_target": "facial features", "method": "increasing definition and sharpness", "quality_goal": "professional portrait quality"},
                    {"enhancement_target": "texture details", "method": "revealing fabric weave and material properties", "quality_goal": "tactile realism"},
                    # Restoration examples
                    {"enhancement_target": "old photograph", "method": "removing scratches and fixing faded colors", "quality_goal": "pristine archival quality"},
                    {"enhancement_target": "low resolution image", "method": "AI upscaling with detail synthesis", "quality_goal": "4K sharp output"},
                    {"enhancement_target": "blurry photo", "method": "motion blur correction and sharpening", "quality_goal": "tack-sharp clarity"},
                    # Combined examples
                    {"enhancement_target": "overall image quality", "method": "noise reduction, color correction, and detail enhancement", "quality_goal": "professional grade output"}
                ],
                tips=[
                    "Can enhance specific areas OR restore damaged images",
                    "Describe what needs improvement (detail, damage, quality)",
                    "Specify the enhancement/restoration method",
                    "Define your quality goal clearly",
                    "Use 'Enhance' for improvements, 'Restore' for fixing damage"
                ]
            ),
            
            PromptType.OUTPAINTING: PromptTemplate(
                name="Outpainting / Canvas Extension",
                type=PromptType.OUTPAINTING,
                template="Extend {direction} by {extension_description} maintaining {consistency_elements}",
                placeholders=["direction", "extension_description", "consistency_elements"],
                examples=[
                    {"direction": "the landscape horizontally", "extension_description": "continuing the mountain range and forest", "consistency_elements": "consistent lighting and perspective"},
                    {"direction": "the scene upward", "extension_description": "revealing more of the sky and clouds", "consistency_elements": "matching weather conditions and time of day"},
                    {"direction": "the portrait downward", "extension_description": "showing full body and environment", "consistency_elements": "consistent style and proportions"},
                    {"direction": "all sides equally", "extension_description": "expanding the urban environment naturally", "consistency_elements": "architectural continuity and perspective"},
                    {"direction": "the canvas to widescreen", "extension_description": "adding cinematic side panels", "consistency_elements": "color grading and atmosphere"}
                ],
                tips=[
                    "Specify which direction(s) to extend",
                    "Describe what should appear in extended areas",
                    "Maintain perspective and vanishing points",
                    "Keep lighting and shadows consistent",
                    "Consider the logical continuation of the scene"
                ]
            ),
            
            PromptType.CUSTOM_PROMPT: PromptTemplate(
                name="Custom Prompt Builder",
                type=PromptType.CUSTOM_PROMPT,
                template="{base_action} {elements} {custom_text}",
                placeholders=["base_action", "elements", "custom_text"],
                examples=[
                    {
                        "base_action": "Transform the image into oil painting",
                        "elements": "with photorealistic quality, using ray-traced reflections while maintaining subject identity",
                        "custom_text": ""
                    },
                    {
                        "base_action": "Add magical effects to the scene", 
                        "elements": "with cinematic quality, featuring volumetric lighting, ensuring seamless integration while preserving original composition",
                        "custom_text": "with particle effects and lens flares"
                    },
                    {
                        "base_action": "Enhance the portrait",
                        "elements": "with professional grade output, enhancing fine details, creating dramatic mood while maintaining facial features and expressions",
                        "custom_text": ""
                    }
                ],
                tips=[
                    "Start with a clear base action",
                    "Select multiple elements from different categories",
                    "Preservation options help maintain important aspects",
                    "Add custom text for specific requirements",
                    "Preview helps refine your prompt"
                ]
            )
        }
        return templates
    
    def _initialize_common_objects(self) -> Dict[str, List[str]]:
        """Initialize common objects for quick selection"""
        return {
            "accessories": ["hat", "glasses", "sunglasses", "necklace", "earrings", "watch", "bracelet", "scarf", "bag", "backpack"],
            "clothing": ["shirt", "jacket", "coat", "dress", "pants", "shoes", "boots", "gloves", "tie", "belt"],
            "props": ["book", "phone", "laptop", "coffee cup", "flowers", "umbrella", "camera", "headphones", "keys", "wallet"],
            "environment": ["tree", "building", "car", "bicycle", "bench", "lamp post", "sign", "window", "door", "fence"],
            "effects": ["smoke", "fog", "rain", "snow", "sparkles", "light rays", "shadows", "reflections", "glow", "particles"]
        }
    
    def _initialize_positions(self) -> List[str]:
        """Initialize position descriptors"""
        return [
            # Wearable positions
            "on their head",
            "on the person's head",
            "worn on the head",
            "on their face",
            "covering their eyes",
            "around their neck",
            "on their shoulders",
            "on their wrist",
            "on their finger",
            "on their ears",
            # Held positions
            "in their hand",
            "in their left hand",
            "in their right hand", 
            "held in both hands",
            "carrying it",
            # Spatial positions
            "in the foreground",
            "in the background", 
            "on the left side",
            "on the right side",
            "in the center",
            "behind the subject",
            "in front of the subject",
            "next to the subject",
            "above the subject",
            "below the subject",
            "floating above",
            "on the ground",
            "on the table",
            "on the wall"
        ]
    
    def build_prompt(self, 
                    prompt_type: PromptType,
                    parameters: Dict[str, str],
                    analysis_context: Optional[Dict] = None) -> str:
        """
        Build a prompt based on type and parameters
        
        Args:
            prompt_type: Type of prompt to generate
            parameters: Dictionary of placeholder values
            analysis_context: Optional context from image analysis
            
        Returns:
            Generated prompt string
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        template = self.templates[prompt_type]
        prompt = template.template
        
        # Replace placeholders
        for placeholder in template.placeholders:
            if placeholder in parameters:
                value = parameters[placeholder]
                # Skip empty values to avoid double spaces
                if value:
                    prompt = prompt.replace(f"{{{placeholder}}}", value)
                else:
                    prompt = prompt.replace(f" {{{placeholder}}}", "")  # Remove space before empty placeholder
                    prompt = prompt.replace(f"{{{placeholder}}}", "")
        
        # Clean up any double spaces
        prompt = " ".join(prompt.split())
        
        # Add analysis context if available
        if analysis_context:
            prompt = self._enhance_with_context(prompt, analysis_context)
            
        return prompt
    
    def _enhance_with_context(self, prompt: str, context: Dict) -> str:
        """Enhance prompt with analysis context"""
        # This can be expanded to intelligently incorporate analysis results
        return prompt
    
    def get_template_info(self, prompt_type: PromptType) -> Dict[str, Any]:
        """Get template information for UI display"""
        template = self.templates.get(prompt_type)
        if not template:
            return {}
            
        return {
            "name": template.name,
            "placeholders": template.placeholders,
            "examples": template.examples,
            "tips": template.tips
        }
    
    def get_suggestions(self, prompt_type: PromptType, placeholder: str) -> List[str]:
        """Get suggestions for a specific placeholder"""
        suggestions = []
        
        if placeholder == "object" and prompt_type == PromptType.OBJECT_ADD:
            # Flatten all object categories
            for category, items in self.common_objects.items():
                suggestions.extend(items)
        elif placeholder == "position":
            suggestions = self.position_descriptors
        elif placeholder == "emotion":
            suggestions = [
                # Basic emotions
                "happy", "sad", "angry", "surprised", "disgusted", "fearful",
                # Popular choices  
                "joyful", "excited", "calm", "worried", "confident", "shy",
                "thoughtful", "determined", "curious", "amazed", "peaceful",
                "frustrated", "anxious", "delighted", "contemplative", "focused",
                # Expressive
                "laughing", "crying", "smiling", "frowning", "grinning", "sighing"
            ]
        elif placeholder == "style_modifier":
            suggestions = ["seamlessly integrated", "matching the lighting", "with realistic shadows", "in the same art style", "naturally blended"]
        elif placeholder == "pose" or placeholder == "new_pose":
            suggestions = self.get_pose_suggestions()
        
        return suggestions
    
    def get_pose_suggestions(self) -> List[str]:
        """Get comprehensive list of pose suggestions"""
        return [
            # Standing poses
            "standing straight", "standing with arms crossed", "standing with hands on hips",
            "standing casually", "standing with weight on one leg", "standing in contrapposto",
            "standing with arms behind back", "standing at attention", "standing with arms raised",
            
            # Sitting poses
            "sitting cross-legged", "sitting on chair", "sitting on ground",
            "sitting with legs crossed", "sitting leaning forward", "sitting leaning back",
            "sitting sideways", "sitting meditation pose", "sitting with knees up",
            
            # Dynamic poses
            "walking forward", "running", "jumping", "dancing", "spinning",
            "mid-stride walk", "jogging", "leaping", "skipping", "twirling",
            
            # Action poses
            "punching", "kicking", "fighting stance", "defensive pose", "martial arts stance",
            "throwing", "catching", "reaching out", "pointing", "gesturing",
            
            # Resting poses
            "lying down", "lying on side", "lying on stomach", "reclining",
            "lounging", "sleeping position", "resting on elbow", "relaxed pose",
            
            # Emotional poses
            "victory pose", "thinking pose", "praying", "celebrating", "mourning",
            "contemplative stance", "triumphant pose", "defeated posture", "confident power pose",
            
            # Professional poses
            "presenting", "speaking gesture", "business handshake pose", "lecturing stance",
            "model pose", "fashion pose", "portrait pose", "profile pose",
            
            # Athletic poses
            "yoga tree pose", "yoga warrior pose", "stretching", "weightlifting pose",
            "sports ready stance", "batting stance", "golf swing pose", "tennis serve pose",
            
            # Creative poses
            "dancing ballet", "playing instrument pose", "painting gesture", "photographing pose",
            "writing pose", "reading pose", "sculpting stance", "conducting orchestra",
            
            # Interactive poses
            "hugging pose", "hand holding", "high five gesture", "waving",
            "beckoning gesture", "offering hand", "shaking hands", "embracing"
        ]
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate generated prompt against FLUX.1 Kontext requirements"""
        if not prompt or len(prompt.strip()) < 10:
            return False, "Prompt too short"
            
        if "{" in prompt or "}" in prompt:
            return False, "Unfilled placeholders remaining"
        
        # Validate token count (FLUX.1 Kontext has 512 token limit)
        is_valid, message, token_count = validate_token_count(prompt)
        if not is_valid:
            return False, message
            
        return True, None
    
    def get_removal_suggestions(self, scenario: RemovalScenario) -> Dict[str, List[str]]:
        """Get suggestions for specific removal scenario"""
        if scenario == RemovalScenario.CLOTHING:
            return self._get_categorized_clothing_suggestions()
        else:
            return self._get_standard_removal_suggestions(scenario)
    
    def _get_categorized_clothing_suggestions(self) -> Dict[str, List[str]]:
        """Get clothing suggestions organized by category"""
        return {
            "items": [
                # Most common items first
                "jacket", "shirt", "pants", "dress", "shoes", "hat",
                "coat", "sweater", "jeans", "skirt", "boots", "tie",
                "hoodie", "t-shirt", "blazer", "shorts", "socks", "belt",
                
                # Then categorized items
                "--- OUTERWEAR ---",
                "leather jacket", "denim jacket", "suit jacket", "windbreaker",
                "bomber jacket", "parka", "raincoat", "trench coat", "cape",
                "poncho", "cardigan", "shawl", "cloak", "robe", "kimono",
                
                "--- TOPS ---",
                "blouse", "tank top", "sweatshirt", "pullover", "turtleneck",
                "polo shirt", "crop top", "tube top", "camisole", "vest",
                "waistcoat", "jersey", "tunic", "halter top",
                
                "--- BOTTOMS ---",
                "trousers", "leggings", "tights", "stockings", "cargo pants",
                "sweatpants", "chinos", "capri pants", "culottes", "overalls",
                "jumpsuit", "romper", "mini skirt",
                
                "--- DRESSES ---",
                "gown", "sundress", "cocktail dress", "evening dress", 
                "maxi dress", "mini dress", "wrap dress", "shift dress",
                "bodycon dress", "ball gown",
                
                "--- FOOTWEAR ---",
                "sneakers", "heels", "sandals", "loafers", "flats", "pumps",
                "wedges", "ankle boots", "knee-high boots", "slippers",
                "flip-flops", "oxfords", "stilettos", "platform shoes", "combat boots",
                
                "--- UNDERWEAR ---",
                "bra", "underwear", "panties", "boxers", "briefs", "lingerie",
                "corset", "bodysuit", "slip", "nightgown", "pajamas", "nightwear",
                
                "--- SPORTSWEAR ---",
                "sportswear", "tracksuit", "yoga pants", "sports bra",
                "athletic shorts", "compression shirt", "cycling shorts",
                "swimsuit", "bikini", "wetsuit",
                
                "--- ACCESSORIES ---",
                "bow tie", "suspenders", "apron", "uniform", "costume"
            ],
            "fill_methods": [
                "reveal what's underneath naturally",
                "show the clothing underneath",
                "reveal natural body shape with appropriate clothing",
                "show the shirt/top underneath",
                "show skin naturally",
                "reveal the layer beneath",
                "show what they're wearing underneath",
                "display the inner garment"
            ]
        }
    
    def _get_standard_removal_suggestions(self, scenario: RemovalScenario) -> Dict[str, List[str]]:
        """Get suggestions for specific removal scenario"""
        suggestions = {
            RemovalScenario.ACCESSORIES: {
                "items": ["hat", "cap", "glasses", "sunglasses", "necklace", "earrings", "watch", "bracelet", "belt", "scarf", "mask", "headphones"],
                "fill_methods": [
                    "show their natural appearance",
                    "reveal what's behind/underneath",
                    "show their face clearly",
                    "show their natural hair"
                ]
            },
            RemovalScenario.BACKGROUND_OBJECT: {
                "items": ["car", "building", "tree", "pole", "sign", "trash can", "bench", "fence"],
                "fill_methods": [
                    "extend the background naturally",
                    "blend with surrounding environment",
                    "continue the background pattern",
                    "fill with matching scenery"
                ]
            },
            RemovalScenario.PERSON: {
                "items": ["person in background", "crowd", "photobomber", "bystander"],
                "fill_methods": [
                    "blend seamlessly with the background",
                    "extend the background naturally",
                    "fill with the environment",
                    "continue the scene without them"
                ]
            },
            RemovalScenario.TEXT_LOGOS: {
                "items": ["text", "logo", "brand name", "label", "sign text", "watermark"],
                "fill_methods": [
                    "maintain the original surface texture",
                    "keep the surface clean",
                    "blend with the material",
                    "preserve the underlying pattern"
                ]
            },
            RemovalScenario.WATERMARK: {
                "items": ["watermark", "copyright text", "signature", "stamp"],
                "fill_methods": [
                    "restore the original image content",
                    "blend seamlessly with surroundings",
                    "maintain image continuity",
                    "preserve underlying details"
                ]
            },
            RemovalScenario.VEHICLE: {
                "items": ["car", "truck", "motorcycle", "bicycle", "bus", "van"],
                "fill_methods": [
                    "show the road/street clearly",
                    "extend the background naturally",
                    "reveal what's behind",
                    "continue the landscape"
                ]
            },
            RemovalScenario.FURNITURE: {
                "items": ["chair", "table", "sofa", "desk", "lamp", "shelf", "cabinet"],
                "fill_methods": [
                    "show the floor/wall clearly",
                    "extend the room naturally",
                    "blend with the interior",
                    "maintain the room's appearance"
                ]
            }
        }
        
        return suggestions.get(scenario, {"items": [], "fill_methods": []})
    
    def get_lighting_suggestions(self, scenario: LightingScenario) -> Dict[str, Any]:
        """Get suggestions for lighting scenarios"""
        suggestions = {
            LightingScenario.NATURAL: {
                "presets": [
                    "soft morning light", "bright midday sun", "golden hour", 
                    "blue hour", "overcast daylight", "dappled sunlight through trees",
                    "window light from left", "window light from right", "skylight from above"
                ],
                "effects": [
                    "soft shadows", "natural skin tones", "realistic colors",
                    "gentle gradients", "atmospheric perspective"
                ],
                "adjustments": [
                    "while preserving natural look", "maintaining realistic shadows",
                    "with authentic color temperature", "keeping outdoor atmosphere"
                ]
            },
            LightingScenario.STUDIO: {
                "presets": [
                    "three-point lighting", "butterfly lighting", "Rembrandt lighting",
                    "split lighting", "broad lighting", "short lighting",
                    "high key lighting", "low key lighting", "beauty dish setup"
                ],
                "effects": [
                    "professional look", "controlled shadows", "even illumination",
                    "catchlights in eyes", "rim lighting", "fill light", "background separation"
                ],
                "adjustments": [
                    "creating professional portrait", "emphasizing subject features",
                    "with studio atmosphere", "maintaining skin texture"
                ]
            },
            LightingScenario.DRAMATIC: {
                "presets": [
                    "chiaroscuro lighting", "film noir lighting", "single spotlight",
                    "venetian blind shadows", "harsh side lighting", "bottom lighting",
                    "silhouette backlight", "moody low light", "theatrical spotlight"
                ],
                "effects": [
                    "strong contrast", "deep shadows", "selective illumination",
                    "mystery atmosphere", "emotional impact", "dramatic mood"
                ],
                "adjustments": [
                    "creating tension", "emphasizing drama", "with noir atmosphere",
                    "maintaining subject recognition"
                ]
            },
            LightingScenario.AMBIENT: {
                "presets": [
                    "soft ambient glow", "even room lighting", "bounce light",
                    "diffused overhead", "wraparound light", "shadowless lighting",
                    "museum lighting", "gallery lighting", "soft box lighting"
                ],
                "effects": [
                    "minimal shadows", "even exposure", "soft gradients",
                    "neutral atmosphere", "flattering light"
                ],
                "adjustments": [
                    "reducing harsh shadows", "creating soft atmosphere",
                    "with even illumination", "maintaining detail throughout"
                ]
            },
            LightingScenario.NEON: {
                "presets": [
                    "cyberpunk neon", "pink and blue neon", "green matrix glow",
                    "red neon signs", "multicolor neon", "UV blacklight",
                    "laser lights", "LED strips", "holographic lighting"
                ],
                "effects": [
                    "vivid colors", "glowing edges", "reflective surfaces",
                    "urban night atmosphere", "futuristic mood", "color bleeding"
                ],
                "adjustments": [
                    "on wet streets", "with fog atmosphere", "creating cyberpunk mood",
                    "maintaining neon intensity"
                ]
            },
            LightingScenario.CANDLELIGHT: {
                "presets": [
                    "single candle", "multiple candles", "fireplace glow",
                    "lantern light", "torch lighting", "campfire illumination",
                    "birthday candles", "romantic dinner lighting", "vigil candles"
                ],
                "effects": [
                    "warm orange glow", "flickering shadows", "intimate atmosphere",
                    "soft illumination", "cozy mood", "dancing light"
                ],
                "adjustments": [
                    "creating warmth", "with romantic mood", "maintaining flame colors",
                    "showing light movement"
                ]
            },
            LightingScenario.SUNSET: {
                "presets": [
                    "golden hour glow", "magic hour", "sunset backlight",
                    "orange sky reflection", "purple twilight", "sun flare",
                    "horizon glow", "cloud illumination", "sunset silhouette"
                ],
                "effects": [
                    "warm colors", "long shadows", "golden tones",
                    "sky gradients", "atmospheric haze", "lens flares"
                ],
                "adjustments": [
                    "with golden atmosphere", "creating magical mood",
                    "maintaining warm tones", "showing time of day"
                ]
            },
            LightingScenario.NIGHT: {
                "presets": [
                    "moonlight", "starlight", "street lamps", "city lights",
                    "northern lights", "lightning flash", "car headlights",
                    "shop window glow", "emergency lights"
                ],
                "effects": [
                    "cool blue tones", "deep shadows", "selective lighting",
                    "mysterious atmosphere", "night sky", "artificial light sources"
                ],
                "adjustments": [
                    "maintaining night atmosphere", "with visible light sources",
                    "creating nocturnal mood", "preserving darkness"
                ]
            },
            LightingScenario.UNDERWATER: {
                "presets": [
                    "underwater caustics", "deep ocean blue", "shallow water clarity",
                    "sunbeams through water", "bioluminescence", "coral reef lighting",
                    "murky depths", "crystal clear water", "underwater cave"
                ],
                "effects": [
                    "blue-green tint", "light rays", "floating particles",
                    "distorted light", "depth gradient", "aquatic atmosphere"
                ],
                "adjustments": [
                    "with underwater physics", "creating depth", "maintaining visibility",
                    "showing water effects"
                ]
            },
            LightingScenario.BACKLIT: {
                "presets": [
                    "rim lighting", "halo effect", "silhouette", "translucent glow",
                    "edge lighting", "contre-jour", "backlit hair", "glowing outline",
                    "sun behind subject"
                ],
                "effects": [
                    "glowing edges", "lens flares", "atmospheric haze",
                    "separation from background", "ethereal glow", "dramatic silhouettes"
                ],
                "adjustments": [
                    "creating separation", "with glowing edges", "maintaining subject detail",
                    "adding atmospheric depth"
                ]
            }
        }
        
        return suggestions.get(scenario, {
            "presets": ["custom lighting setup"],
            "effects": ["specific lighting effects"],
            "adjustments": ["maintaining image quality"]
        })
    
    def get_context_aware_lighting(self, context: str) -> Dict[str, List[str]]:
        """Get lighting suggestions based on what was added/changed"""
        context_lighting = {
            "outdoor_object": [
                "match the outdoor lighting conditions",
                "cast appropriate shadows for time of day",
                "integrate with natural light direction",
                "reflect the sky color on the object"
            ],
            "indoor_object": [
                "match the indoor lighting setup",
                "create consistent shadows with room lighting",
                "add subtle reflections from light sources",
                "integrate with ambient room light"
            ],
            "person_added": [
                "ensure face is well-lit",
                "add catchlights to eyes",
                "create natural skin tones for the lighting",
                "match shadows with other people in scene"
            ],
            "background_changed": [
                "adjust overall lighting to match new background",
                "modify color temperature for new environment",
                "update shadows and reflections accordingly",
                "blend foreground and background lighting"
            ],
            "night_scene": [
                "add appropriate artificial light sources",
                "create rim lighting for separation",
                "use cool tones with warm light accents",
                "maintain visibility while preserving night mood"
            ],
            "water_added": [
                "add water reflections of existing lights",
                "create caustic patterns if applicable",
                "adjust underwater portions with blue tint",
                "add specular highlights on water surface"
            ]
        }
        
        return context_lighting.get(context, [
            "adjust lighting to match the changes",
            "ensure consistent light direction",
            "maintain realistic shadows",
            "preserve the original mood"
        ])
    
    def get_enhancement_suggestions(self, scenario: EnhancementScenario) -> Dict[str, List[str]]:
        """Get suggestions for enhancement and restoration scenarios"""
        suggestions = {
            # Detail Enhancement scenarios
            EnhancementScenario.FACIAL_DETAILS: {
                "methods": [
                    "enhancing skin texture and pores",
                    "defining facial features clearly",
                    "adding natural skin detail",
                    "improving facial structure definition",
                    "revealing subtle expressions"
                ],
                "quality_goals": [
                    "photorealistic facial detail",
                    "professional portrait quality",
                    "natural skin appearance",
                    "magazine-quality retouching"
                ]
            },
            EnhancementScenario.EYE_ENHANCEMENT: {
                "methods": [
                    "adding iris detail and patterns",
                    "enhancing natural eye color",
                    "defining eyelashes individually",
                    "adding realistic catchlights",
                    "improving eye clarity and sharpness"
                ],
                "quality_goals": [
                    "striking eye detail",
                    "captivating gaze",
                    "crystal clear eyes",
                    "professional beauty standard"
                ]
            },
            EnhancementScenario.TEXTURE_DETAILS: {
                "methods": [
                    "revealing fabric weave patterns",
                    "enhancing material properties",
                    "adding surface detail",
                    "showing texture depth",
                    "defining material characteristics"
                ],
                "quality_goals": [
                    "tactile realism",
                    "photorealistic textures",
                    "material authenticity",
                    "tangible surface quality"
                ]
            },
            EnhancementScenario.HAIR_DETAILS: {
                "methods": [
                    "defining individual hair strands",
                    "adding natural hair flow",
                    "enhancing hair texture",
                    "revealing hair shine and depth",
                    "improving hairline definition"
                ],
                "quality_goals": [
                    "salon-quality hair",
                    "natural hair appearance",
                    "detailed hair texture",
                    "professional hair photography"
                ]
            },
            EnhancementScenario.OVERALL_SHARPNESS: {
                "methods": [
                    "intelligent sharpening throughout",
                    "edge enhancement and definition",
                    "clarity improvement",
                    "detail extraction",
                    "micro-contrast enhancement"
                ],
                "quality_goals": [
                    "tack-sharp image",
                    "professional clarity",
                    "crisp definition",
                    "optimal sharpness"
                ]
            },
            # Restoration scenarios
            EnhancementScenario.OLD_PHOTO: {
                "methods": [
                    "removing scratches and dust spots",
                    "fixing faded and yellowed colors",
                    "repairing torn edges and creases",
                    "enhancing contrast and clarity",
                    "restoring missing portions intelligently"
                ],
                "quality_goals": [
                    "pristine archival quality",
                    "authentic vintage restoration",
                    "museum-grade preservation",
                    "family heirloom quality"
                ]
            },
            EnhancementScenario.DAMAGED_ART: {
                "methods": [
                    "repairing cracks and paint loss",
                    "removing surface dirt and varnish yellowing",
                    "reconstructing damaged areas",
                    "color matching and blending",
                    "preserving original brushwork"
                ],
                "quality_goals": [
                    "museum-quality restoration",
                    "conservation-grade repair",
                    "gallery-ready presentation",
                    "authentic artistic preservation"
                ]
            },
            EnhancementScenario.LOW_RESOLUTION: {
                "methods": [
                    "AI upscaling with detail enhancement",
                    "intelligent pixel interpolation",
                    "edge sharpening and refinement",
                    "texture synthesis and recovery",
                    "noise reduction while preserving detail"
                ],
                "quality_goals": [
                    "4K ultra-high resolution",
                    "print-ready quality",
                    "sharp detailed output",
                    "professional presentation quality"
                ]
            },
            EnhancementScenario.BLURRY: {
                "methods": [
                    "motion blur correction",
                    "focus enhancement and sharpening",
                    "detail recovery using AI",
                    "edge definition improvement",
                    "selective sharpness adjustment"
                ],
                "quality_goals": [
                    "tack-sharp clarity",
                    "professional focus quality",
                    "crisp detailed image",
                    "publication-ready sharpness"
                ]
            },
            EnhancementScenario.EXPOSURE_ISSUES: {
                "methods": [
                    "exposure correction and tone mapping",
                    "highlight recovery for overexposed areas",
                    "shadow detail recovery for underexposed areas",
                    "dynamic range optimization",
                    "selective brightness adjustment",
                    "color preservation during correction"
                ],
                "quality_goals": [
                    "properly exposed photograph",
                    "balanced tonal range",
                    "natural lighting appearance",
                    "professional exposure quality",
                    "preserved detail in all areas"
                ]
            },
            EnhancementScenario.NOISY: {
                "methods": [
                    "advanced noise reduction",
                    "preserving fine details while denoising",
                    "color noise elimination",
                    "grain structure refinement",
                    "selective noise filtering"
                ],
                "quality_goals": [
                    "clean professional image",
                    "smooth gradients",
                    "preserved detail quality",
                    "noise-free output"
                ]
            },
            EnhancementScenario.COMPRESSED: {
                "methods": [
                    "JPEG artifact removal",
                    "block artifact elimination",
                    "color banding correction",
                    "detail reconstruction",
                    "compression noise reduction"
                ],
                "quality_goals": [
                    "uncompressed quality",
                    "smooth color transitions",
                    "artifact-free image",
                    "original quality restoration"
                ]
            },
            EnhancementScenario.FADED_COLORS: {
                "methods": [
                    "color vibrancy restoration",
                    "selective color enhancement",
                    "white balance correction",
                    "saturation optimization",
                    "color cast removal"
                ],
                "quality_goals": [
                    "vibrant natural colors",
                    "authentic color restoration",
                    "balanced color palette",
                    "true-to-life appearance"
                ]
            },
            EnhancementScenario.GENERAL_ENHANCEMENT: {
                "methods": [
                    "intelligent content-aware fill",
                    "seamless patch reconstruction",
                    "edge blending and matching",
                    "texture continuation",
                    "structural completion",
                    "Make without outer strokes, no silhouette outlines. Remove sticker effect"
                ],
                "quality_goals": [
                    "seamlessly restored image",
                    "invisible repair quality",
                    "complete reconstruction",
                    "authentic completion"
                ]
            }
        }
        
        return suggestions.get(scenario, {
            "methods": ["custom enhancement technique"],
            "quality_goals": ["desired quality outcome"]
        })
    
    # Keep for backward compatibility
    def get_restoration_suggestions(self, scenario: RestorationScenario) -> Dict[str, List[str]]:
        """Backward compatibility wrapper"""
        return self.get_enhancement_suggestions(scenario)
    
    def get_outpainting_suggestions(self, direction: OutpaintingDirection) -> Dict[str, List[str]]:
        """Get suggestions for outpainting scenarios"""
        suggestions = {
            OutpaintingDirection.HORIZONTAL: {
                "extensions": [
                    "continuing the landscape naturally",
                    "extending the horizon line",
                    "adding more environmental context",
                    "revealing hidden scene elements",
                    "expanding the panoramic view"
                ],
                "consistency": [
                    "consistent perspective and vanishing points",
                    "matching lighting and shadows",
                    "continuous textures and patterns",
                    "seamless color transitions"
                ]
            },
            OutpaintingDirection.VERTICAL: {
                "extensions": [
                    "revealing more sky and clouds",
                    "showing ground and foundation",
                    "extending architectural elements",
                    "adding atmospheric depth",
                    "completing the full scene height"
                ],
                "consistency": [
                    "vertical perspective accuracy",
                    "consistent atmospheric perspective",
                    "matching architectural style",
                    "continuous vertical elements"
                ]
            },
            OutpaintingDirection.LEFT: {
                "extensions": [
                    "revealing what's to the left",
                    "continuing the scene leftward",
                    "adding contextual elements",
                    "extending the environment",
                    "showing hidden left portion"
                ],
                "consistency": [
                    "leftward perspective flow",
                    "consistent scene elements",
                    "matching left-side lighting",
                    "seamless left integration"
                ]
            },
            OutpaintingDirection.RIGHT: {
                "extensions": [
                    "revealing what's to the right",
                    "continuing the scene rightward",
                    "adding contextual elements",
                    "extending the environment",
                    "showing hidden right portion"
                ],
                "consistency": [
                    "rightward perspective flow",
                    "consistent scene elements",
                    "matching right-side lighting",
                    "seamless right integration"
                ]
            },
            OutpaintingDirection.TOP: {
                "extensions": [
                    "revealing the sky above",
                    "showing ceiling or canopy",
                    "adding vertical space",
                    "extending upward elements",
                    "completing the upper view"
                ],
                "consistency": [
                    "upward perspective accuracy",
                    "consistent sky conditions",
                    "matching upper lighting",
                    "seamless top integration"
                ]
            },
            OutpaintingDirection.BOTTOM: {
                "extensions": [
                    "showing the ground below",
                    "revealing foundation elements",
                    "adding lower context",
                    "extending downward view",
                    "completing the base"
                ],
                "consistency": [
                    "downward perspective accuracy",
                    "consistent ground plane",
                    "matching lower lighting",
                    "seamless bottom integration"
                ]
            },
            OutpaintingDirection.ALL_SIDES: {
                "extensions": [
                    "expanding the entire scene",
                    "creating a wider context",
                    "revealing surrounding environment",
                    "building complete panorama",
                    "extending in all directions"
                ],
                "consistency": [
                    "radial perspective consistency",
                    "uniform lighting distribution",
                    "seamless edge blending",
                    "coherent scene expansion"
                ]
            },
            OutpaintingDirection.WIDESCREEN: {
                "extensions": [
                    "creating cinematic aspect ratio",
                    "adding side panels for film look",
                    "extending to 16:9 or 21:9",
                    "building theatrical composition",
                    "achieving movie-like framing"
                ],
                "consistency": [
                    "cinematic composition rules",
                    "consistent film aesthetics",
                    "matching color grading",
                    "theatrical lighting continuity"
                ]
            },
            OutpaintingDirection.SQUARE: {
                "extensions": [
                    "converting to square format",
                    "adding content for 1:1 ratio",
                    "centering the composition",
                    "balancing all sides equally",
                    "creating Instagram-ready format"
                ],
                "consistency": [
                    "centered composition balance",
                    "equal side extensions",
                    "symmetric additions",
                    "uniform style matching"
                ]
            }
        }
        
        return suggestions.get(direction, {
            "extensions": ["custom extension description"],
            "consistency": ["maintaining visual coherence"]
        })
    
    def get_custom_prompt_elements(self) -> Dict[str, List[str]]:
        """Get all available elements for custom prompt building"""
        return {
            "preservation": [
                "while maintaining subject identity",
                "while preserving original composition", 
                "while keeping the same position and lighting",
                "while maintaining consistent perspective and lighting",
                "while preserving color palette and mood",
                "while keeping artistic style intact",
                "while maintaining facial features and expressions",
                "while preserving background elements",
                "while keeping texture and material properties",
                "while maintaining scene depth and layers"
            ],
            "quality": [
                "with photorealistic quality",
                "with professional grade output",
                "with high resolution details",
                "with cinematic quality",
                "with studio-quality rendering",
                "with magazine-quality finish",
                "with award-winning composition",
                "with gallery-worthy aesthetics",
                "with production-ready quality",
                "with broadcast-quality standards"
            ],
            "technical": [
                "using ray-traced reflections",
                "with subsurface scattering",
                "featuring ambient occlusion",
                "with realistic depth of field",
                "using global illumination",
                "with volumetric lighting",
                "featuring motion blur effects",
                "with chromatic aberration",
                "using physically-based rendering",
                "with advanced shader effects"
            ],
            "consistency": [
                "ensuring seamless integration",
                "with coherent scene elements",
                "maintaining natural blending",
                "with consistent art direction",
                "ensuring visual harmony",
                "with unified color grading",
                "maintaining stylistic coherence",
                "with matched lighting conditions",
                "ensuring perspective accuracy",
                "with cohesive atmosphere"
            ],
            "enhancement": [
                "enhancing fine details",
                "improving overall sharpness",
                "adding realistic textures",
                "increasing dynamic range",
                "optimizing color vibrancy",
                "refining edge quality",
                "boosting local contrast",
                "clarifying important elements",
                "enriching tonal range",
                "maximizing visual impact"
            ],
            "atmosphere": [
                "creating dramatic mood",
                "establishing ethereal atmosphere",
                "building tension and mystery",
                "evoking nostalgic feeling",
                "generating dreamlike quality",
                "establishing epic scale",
                "creating intimate ambiance",
                "building surreal environment",
                "establishing photojournalistic feel",
                "generating cinematic atmosphere"
            ]
        }
    
    def build_custom_prompt(self, 
                          base_action: str,
                          selected_elements: Dict[str, List[str]],
                          custom_additions: Optional[List[str]] = None) -> str:
        """
        Build a custom prompt from selected elements
        
        Args:
            base_action: The main action/transformation to perform
            selected_elements: Dictionary of category -> list of selected elements
            custom_additions: Additional custom text elements
            
        Returns:
            Complete custom prompt
        """
        prompt_parts = [base_action]
        
        # Add selected elements in a logical order
        order = ["technical", "quality", "enhancement", "atmosphere", "consistency", "preservation"]
        
        for category in order:
            if category in selected_elements and selected_elements[category]:
                # Join elements with appropriate connectors
                if category == "preservation":
                    # Preservation elements typically start with "while"
                    prompt_parts.extend(selected_elements[category])
                else:
                    # Other elements can be joined with commas
                    for element in selected_elements[category]:
                        if element not in prompt_parts:  # Avoid duplicates
                            prompt_parts.append(element)
        
        # Add custom additions if provided
        if custom_additions:
            for addition in custom_additions:
                if addition.strip() and addition not in prompt_parts:
                    prompt_parts.append(addition.strip())
        
        # Join with appropriate punctuation
        final_prompt = prompt_parts[0]
        if len(prompt_parts) > 1:
            # Group preservation elements (those starting with "while")
            preservation_parts = []
            other_parts = []
            
            # Process parts and avoid repetitive patterns
            seen_concepts = set()
            for part in prompt_parts[1:]:
                # Extract key concepts to avoid repetition
                if "while maintaining" in part:
                    concept = part.replace("while maintaining", "").strip()
                elif "while preserving" in part:
                    concept = part.replace("while preserving", "").strip()
                elif "while keeping" in part:
                    concept = part.replace("while keeping", "").strip()
                else:
                    concept = part
                
                # Only add if we haven't seen this concept
                if concept not in seen_concepts:
                    seen_concepts.add(concept)
                    if part.startswith("while"):
                        preservation_parts.append(part)
                    else:
                        # Check for repetitive technical terms
                        if not any(term in part for term in ["using", "with", "featuring", "establishing"] 
                                 if any(term in op for op in other_parts)):
                            other_parts.append(part)
            
            # Build final prompt with better formatting
            if other_parts:
                # Remove duplicate action words
                cleaned_parts = []
                used_verbs = set()
                for part in other_parts:
                    verb = part.split()[0] if part else ""
                    if verb not in used_verbs or verb not in ["using", "with", "featuring", "establishing"]:
                        cleaned_parts.append(part)
                        used_verbs.add(verb)
                
                final_prompt += " " + ", ".join(cleaned_parts)
            
            if preservation_parts:
                # Combine similar preservation concepts
                combined_preservations = []
                preservation_concepts = {}
                
                for part in preservation_parts:
                    if "subject identity" in part or "facial features" in part:
                        if "identity" not in preservation_concepts:
                            preservation_concepts["identity"] = "while maintaining subject identity and facial features"
                    elif "composition" in part or "scene" in part or "layout" in part:
                        if "composition" not in preservation_concepts:
                            preservation_concepts["composition"] = "while preserving original composition"
                    elif "texture" in part or "material" in part:
                        if "texture" not in preservation_concepts:
                            preservation_concepts["texture"] = "while keeping texture and material properties"
                    else:
                        combined_preservations.append(part)
                
                # Add unique preservation concepts
                combined_preservations.extend(preservation_concepts.values())
                
                if combined_preservations:
                    final_prompt += " " + " ".join(combined_preservations[:2])  # Limit to 2 preservation clauses
        
        return final_prompt