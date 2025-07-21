"""
Photography styles for Kontext Assistant
"""
from .base import StylePreset, StyleCategory

PHOTOGRAPHY_STYLES = {
            "cinematic": StylePreset(
                id="cinematic",
                name="Cinematic",
                category=StyleCategory.PHOTOGRAPHY,
                description="Movie-like cinematography with dramatic lighting",
                visual_elements=["wide aspect ratio", "depth of field", "lens flares", "atmospheric haze"],
                color_characteristics=["color grading", "teal and orange", "desaturated tones"],
                technique_details=["anamorphic lens", "dramatic lighting", "film grain"],
                example_prompt="make as cinematic shot with dramatic lighting, shallow depth of field, anamorphic lens characteristics, and film color grading",
                compatible_with=["noir", "documentary", "vintage"],
                tips=["Add specific film references for accuracy", "Mention lighting setup"]
            ),
            
            "cyanotype": StylePreset(
                id="cyanotype",
                name="Cyanotype Photography",
                category=StyleCategory.PHOTOGRAPHY,
                description="Historic photographic printing process with distinctive Prussian blue tones",
                visual_elements=["blueprint aesthetic", "botanical silhouettes", "sun print quality", "negative space"],
                color_characteristics=["Prussian blue monochrome", "white highlights", "cyan gradients", "deep blue shadows"],
                technique_details=["photogram process", "contact printing", "UV exposure", "chemical development"],
                example_prompt="convert to cyanotype photography style, Prussian blue monochromatic print, sun-exposed photogram aesthetic, botanical blueprint quality, white silhouettes on deep cyan background, historic alternative photography process",
                compatible_with=["botanical_art", "vintage_photo", "blueprint"],
                tips=["Use only blue tones", "High contrast works best", "Include botanical elements"]
            ),
            
            "double_exposure": StylePreset(
                id="double_exposure",
                name="Double Exposure",
                category=StyleCategory.PHOTOGRAPHY,
                description="Multiple images blended into one",
                visual_elements=["overlapping images", "transparency effects", "silhouettes", "layered compositions"],
                color_characteristics=["blended tones", "gradient overlays", "ethereal colors"],
                technique_details=["image layering", "masking", "blend modes"],
                example_prompt="create double exposure photography technique with two overlapping transparent images, silhouette masking, and ethereal blended composition",
                compatible_with=["artistic", "portrait", "surreal"],
                tips=["Use contrasting images", "Focus on silhouettes"]
            ),
            
            "long_exposure": StylePreset(
                id="long_exposure",
                name="Long Exposure",
                category=StyleCategory.PHOTOGRAPHY,
                description="Motion blur and light trails",
                visual_elements=["motion blur", "light trails", "smooth water", "star trails"],
                color_characteristics=["ethereal glows", "streaked lights", "smooth gradients"],
                technique_details=["time accumulation", "motion capture", "light painting"],
                example_prompt="create long exposure photography with smooth motion blur, streaming light trails, silky water, and time-lapse effects",
                compatible_with=["night", "landscape", "urban"],
                tips=["Capture movement", "Use for water/clouds"]
            ),
            
            "portrait_studio": StylePreset(
                id="portrait_studio",
                name="Studio Portrait",
                category=StyleCategory.PHOTOGRAPHY,
                description="Professional studio portrait photography",
                visual_elements=["soft lighting", "clean background", "sharp focus", "catchlights"],
                color_characteristics=["natural skin tones", "balanced exposure", "subtle highlights"],
                technique_details=["three-point lighting", "beauty dish", "professional retouching"],
                example_prompt="make as professional studio portrait with three-point lighting, seamless backdrop, natural skin tones, and sharp focus",
                compatible_with=["fashion", "commercial", "headshot"],
                tips=["Specify lighting setup for best results", "Mention background color/type"]
            ),
            
            "tilt_shift": StylePreset(
                id="tilt_shift",
                name="Tilt-Shift Photography",
                category=StyleCategory.PHOTOGRAPHY,
                description="Miniature world effect",
                visual_elements=["selective focus", "miniature appearance", "toy-like quality", "aerial view"],
                color_characteristics=["saturated colors", "high contrast", "vivid tones"],
                technique_details=["depth of field manipulation", "perspective control", "blur gradients"],
                example_prompt="create tilt-shift photography with miniature diorama effect, selective focus blur, high saturation, and toy-like appearance",
                compatible_with=["architectural", "cityscape", "whimsical"],
                tips=["Shoot from above", "Increase color saturation"]
            )
        }