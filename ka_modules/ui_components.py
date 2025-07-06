"""
UI components and helpers for Kontext Assistant.
"""

import gradio as gr
from typing import Optional, List, Dict, Any, Tuple
import logging

# Compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

logger = logging.getLogger("KontextAssistant.UIComponents")


class UIHelpers:
    """Helper functions for UI components."""
    
    @staticmethod
    def create_progress_message(step: int, total: int, message: str) -> str:
        """Create a progress message with visual indicator."""
        progress = step / total
        bar_length = 20
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        percentage = int(progress * 100)
        return f"[{bar}] {percentage}% - {message}"
    
    @staticmethod
    def format_analysis_results(analysis_data: Dict[str, Any]) -> str:
        """Format analysis results for display."""
        lines = []
        
        if 'objects' in analysis_data:
            objects = analysis_data['objects']
            if isinstance(objects, dict) and 'main' in objects:
                lines.append(f"**Main Objects**: {', '.join(objects['main'])}")
            elif isinstance(objects, list):
                lines.append(f"**Objects**: {', '.join(objects)}")
        
        if 'style' in analysis_data:
            style = analysis_data['style']
            if isinstance(style, dict):
                if 'artistic' in style:
                    lines.append(f"**Style**: {style['artistic']}")
                if 'mood' in style:
                    lines.append(f"**Mood**: {style['mood']}")
            else:
                lines.append(f"**Style**: {style}")
        
        if 'environment' in analysis_data:
            env = analysis_data['environment']
            if isinstance(env, dict):
                if 'setting' in env:
                    lines.append(f"**Setting**: {env['setting']}")
                if 'time_of_day' in env:
                    lines.append(f"**Time**: {env['time_of_day']}")
        
        return "\n".join(lines) if lines else "No analysis data available"
    
    @staticmethod
    def create_task_info(task_type: str, subtype: str) -> str:
        """Create informative text about selected task."""
        task_descriptions = {
            "object_manipulation": "Modify object properties like color, size, or position",
            "style_transfer": "Apply artistic styles or visual treatments",
            "environment_change": "Change background, weather, or time of day",
            "element_combination": "Merge or blend multiple elements",
            "state_change": "Transform objects between different states",
            "outpainting": "Extend the image beyond current boundaries",
            "lighting_adjustment": "Modify lighting conditions and effects",
            "texture_change": "Change surface materials and textures",
            "perspective_shift": "Change viewpoint or camera angle"
        }
        
        description = task_descriptions.get(task_type, "Custom image modification")
        return f"**Task**: {description}\n**Action**: {subtype.replace('_', ' ').title()}"


class PromptHistoryManager:
    """Manages prompt generation history."""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_entry(self, task_type: str, user_intent: str, generated_prompt: str) -> None:
        """Add a new history entry."""
        entry = {
            'task_type': task_type,
            'user_intent': user_intent,
            'generated_prompt': generated_prompt,
            'timestamp': self._get_timestamp()
        }
        
        self.history.insert(0, entry)  # Add to beginning
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
    
    def get_history_display(self) -> List[List[str]]:
        """Get history formatted for Gradio dataframe."""
        return [
            [
                entry['timestamp'],
                entry['task_type'],
                entry['user_intent'][:50] + '...' if len(entry['user_intent']) > 50 else entry['user_intent'],
                entry['generated_prompt'][:100] + '...' if len(entry['generated_prompt']) > 100 else entry['generated_prompt']
            ]
            for entry in self.history
        ]
    
    def get_entry(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific history entry."""
        if 0 <= index < len(self.history):
            return self.history[index]
        return None
    
    def clear(self) -> None:
        """Clear all history."""
        self.history = []
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")


def create_analysis_display(image_index: int) -> Tuple[gr.Accordion, gr.Markdown]:
    """Create analysis display components."""
    with gr.Accordion(f"Analysis Results {image_index + 1}", open=False, visible=False) as accordion:
        display = gr.Markdown(
            value="",
            elem_id=f"ka_analysis_display_{image_index}"
        )
    return accordion, display


def create_task_selector() -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Create task type and subtype selectors."""
    task_type = gr.Dropdown(
        label="Task Type",
        choices=[
            "object_manipulation",
            "style_transfer",
            "environment_change",
            "element_combination",
            "state_change",
            "outpainting",
            "lighting_adjustment",
            "texture_change",
            "perspective_shift"
        ],
        value="object_manipulation",
        interactive=True,
        elem_id="ka_task_type"
    )
    
    subtype = gr.Dropdown(
        label="Specific Action",
        choices=["color_change", "add_element", "remove_element"],
        value="color_change",
        interactive=True,
        elem_id="ka_subtype"
    )
    
    return task_type, subtype


def create_advanced_options() -> Dict[str, Any]:
    """Create advanced option controls."""
    with gr.Accordion("Advanced Options", open=False) as accordion:
        preserve_strength = gr.Slider(
            label="Preservation Strength",
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            info="How strongly to preserve unchanged elements"
        )
        
        detail_level = gr.Radio(
            label="Detail Level",
            choices=["concise", "balanced", "detailed"],
            value="balanced",
            info="Amount of detail in generated prompt"
        )
        
        include_analysis = gr.Checkbox(
            label="Include image analysis in prompt",
            value=True,
            info="Use Florence-2 analysis to enhance prompt"
        )
        
        # Future options (disabled for now)
        use_phi3 = gr.Checkbox(
            label="Use Phi-3 enhancement (not available)",
            value=False,
            interactive=False,
            info="Advanced prompt enhancement with Phi-3 mini"
        )
    
    return {
        'accordion': accordion,
        'preserve_strength': preserve_strength,
        'detail_level': detail_level,
        'include_analysis': include_analysis,
        'use_phi3': use_phi3
    }