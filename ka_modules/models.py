"""
Data models and parameter classes
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class AnalysisMode(Enum):
    """Analysis mode options"""
    FLORENCE_ONLY = "florence"
    JOYCAPTION_ONLY = "joycaption" 
    COMBINED = "combined"


@dataclass
class AnalysisParams:
    """Parameters for image analysis"""
    # Hardware settings
    force_cpu: bool = False
    
    # Analysis modes
    use_florence: bool = True
    use_joycaption: bool = False  # Default to False due to issues
    
    # Display options
    show_detailed: bool = False
    
    # Derived property
    @property
    def mode(self) -> AnalysisMode:
        """Determine analysis mode from flags"""
        if self.use_florence and self.use_joycaption:
            return AnalysisMode.COMBINED
        elif self.use_florence:
            return AnalysisMode.FLORENCE_ONLY
        else:
            return AnalysisMode.JOYCAPTION_ONLY


@dataclass
class AnalysisResult:
    """Structured analysis result"""
    description: str = ""
    objects: Dict[str, List[str]] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    enhanced_analysis: Optional[Dict[str, Any]] = None
    analysis_time: float = 0.0
    analysis_mode: str = "unknown"
    error: Optional[str] = None
    
    # Metadata
    image_size: Optional[str] = None
    analyzers_used: Dict[str, bool] = field(default_factory=dict)
    
    def has_enhanced_data(self) -> bool:
        """Check if enhanced analysis data is available"""
        return bool(self.enhanced_analysis and 
                   self.enhanced_analysis.get('total_elements_found', 0) > 0)
    
    def get_total_elements(self) -> int:
        """Get total number of detected elements"""
        if self.enhanced_analysis:
            return self.enhanced_analysis.get('total_elements_found', 0)
        return len(self.objects.get('all', []))


@dataclass
class GenerationParams:
    """Parameters for prompt generation"""
    task_type: str
    user_intent: str
    preservation_strength: float = 0.8
    use_analysis: bool = True
    analysis_data: Optional[Dict[str, Any]] = None


@dataclass
class CacheStats:
    """Cache statistics"""
    items: int = 0
    max_items: int = 0
    memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    hit_rate: float = 0.0
    evictions: int = 0