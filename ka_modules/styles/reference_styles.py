"""
Style reference presets for using loaded images as style guides
"""
from .base import StylePreset, StyleCategory

FROM_REFERENCE_STYLES = {
    "reference_with_prompt": StylePreset(
        id="reference_with_prompt",
        name="Using this style create [prompt]",
        category=StyleCategory.FROM_REFERENCE,
        description="Apply the reference image's style to create something new based on your prompt",
        visual_elements=["reference image style", "user-defined subject", "style transfer"],
        color_characteristics=["colors from reference", "matching palette", "tonal consistency"],
        technique_details=["same artistic technique", "matching brushwork", "consistent rendering"],
        example_prompt="Using this style create [prompt]",
        compatible_with=["oil_painting", "watercolor", "digital_art", "anime_modern"],
        tips=[
            "Add your subject after the prompt",
            "Works best with clear style references",
            "The model will match the artistic style",
            "Keep prompts concise for best results"
        ]
    ),
    
    "reference_new_illustration": StylePreset(
        id="reference_new_illustration",
        name="Create new illustration using same style. [prompt]",
        category=StyleCategory.FROM_REFERENCE,
        description="Generate a new illustration maintaining the exact style of the reference",
        visual_elements=["matching illustration style", "consistent art direction", "unified aesthetic"],
        color_characteristics=["identical color treatment", "same color grading", "matching mood"],
        technique_details=["same medium appearance", "matching detail level", "consistent textures"],
        example_prompt="Create new illustration using same style. [prompt]",
        compatible_with=["digital_art", "concept_art", "anime_modern", "comic_book"],
        tips=[
            "Add additional details after the period",
            "Ideal for creating style-consistent sets",
            "Reference should have clear artistic style",
            "Can specify subject or scene after prompt"
        ]
    )
}