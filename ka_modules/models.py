"""
Data models for Kontext Assistant
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GenerationParams:
    """Parameters for prompt generation"""
    task_type: str
    user_intent: str
    analysis_data: List[Any]