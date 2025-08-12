import gradio
import gradio as gr
import torch
import numpy as np
import logging
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any
from pathlib import Path
from enum import Enum
from PIL import Image
import os
import sys
import gc
import threading

from modules import scripts, shared
from modules.ui_components import InputAccordion
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import IntegratedFluxTransformer2DModel
from einops import rearrange, repeat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForgeKontextUnified")

# Get the extension directory
extension_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add ka_modules to path
modules_path = os.path.join(extension_dir, 'ka_modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Try to import assistant modules
ASSISTANT_AVAILABLE = False
try:
    from ka_modules.shared_state import shared_state
    from ka_modules.smart_analyzer import SmartAnalyzer
    from ka_modules.text_utils import DescriptionCleaner
    from ka_modules.templates import PromptTemplates
    from ka_modules.image_utils import validate_image_list, validate_and_convert_image
    from ka_modules.prompt_builder import PromptBuilder, PromptType, RemovalScenario, LightingScenario, EnhancementScenario, OutpaintingDirection
    from ka_modules.style_library import StyleLibrary, StyleCategory
    from ka_modules.token_utils import get_token_display
    
    ASSISTANT_AVAILABLE = True
    logger.info(f"✅ Kontext Assistant modules loaded successfully from {modules_path}")
except ImportError as e:
    logger.error(f"❌ Failed to import Kontext Assistant modules: {e}")
    SmartAnalyzer = None
    DescriptionCleaner = None
    PromptTemplates = None
    validate_image_list = None
    validate_and_convert_image = None
    PromptBuilder = None
    StyleLibrary = None

# Constants
PATCH_SIZE = 2
VAE_SCALE_FACTOR = 8
MIN_IMAGE_SIZE = 64
MAX_CONTEXT_IMAGES = 3

# Preferred resolutions from BFL
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688),
    (1568, 672),
]

# Load style modifiers configuration
def load_style_modifiers():
    """Load style modifiers from configuration file."""
    try:
        config_path = os.path.join(extension_dir, "configs", "style_modifiers.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load style modifiers: {e}")
        return {"categories": {}, "custom_modifiers": {"name": "Custom", "icon": "✏️"}}

STYLE_MODIFIERS_CONFIG = load_style_modifiers()


class SizingMode(Enum):
    """Sizing modes for context images."""
    NO_CHANGE = "no change"
    TO_OUTPUT = "to output"
    TO_BFL_RECOMMENDED = "to BFL recommended"
    TO_FILL_RESOLUTION = "to fill resolution"


@dataclass
class KontextConfig:
    """Configuration for Kontext processing."""
    max_context_images: int = MAX_CONTEXT_IMAGES
    patch_size: int = PATCH_SIZE
    vae_scale_factor: int = VAE_SCALE_FACTOR
    min_image_size: int = MIN_IMAGE_SIZE
    enable_performance_metrics: bool = True


@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    preprocessing_time: float = 0.0
    vae_encoding_time: float = 0.0
    total_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    context_image_count: int = 0
    
    def log_summary(self):
        """Log performance summary."""
        logger.info(f"Performance Summary:")
        logger.info(f"  Total processing time: {self.total_processing_time:.2f}s")
        logger.info(f"  Preprocessing: {self.preprocessing_time:.2f}s")
        logger.info(f"  VAE encoding: {self.vae_encoding_time:.2f}s")
        logger.info(f"  Memory usage: {self.memory_usage_mb:.1f}MB")
        logger.info(f"  Context images: {self.context_image_count}")


class KontextError(Exception):
    """Custom exception for Kontext-related errors."""
    pass


@dataclass
class KontextState:
    """Encapsulates kontext processing state with metrics."""
    latent: Optional[torch.Tensor] = None
    ids: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    def clear(self) -> None:
        """Clear all tensors and free memory."""
        if self.latent is not None:
            del self.latent
            self.latent = None
        if self.ids is not None:
            del self.ids
            self.ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def is_initialized(self) -> bool:
        """Check if state is properly initialized."""
        return self.latent is not None and self.ids is not None


class ImageProcessor:
    """Handles image preprocessing for kontext."""
    
    @staticmethod
    def validate_image(image, config: KontextConfig) -> None:
        """Validate input image."""
        if image is None:
            raise KontextError("Image cannot be None")
        
        if not hasattr(image, 'size'):
            raise KontextError("Invalid image object - missing size attribute")
        
        width, height = image.size
        if width < config.min_image_size or height < config.min_image_size:
            raise KontextError(
                f"Image too small: {width}x{height}. "
                f"Minimum: {config.min_image_size}x{config.min_image_size}"
            )
        
        # Warn for very large images
        max_dimension = 2048
        if width > max_dimension or height > max_dimension:
            logger.warning(
                f"Large image detected ({width}x{height}). "
                f"Consider resizing for better performance."
            )
    
    @staticmethod
    def calculate_target_size(image, sizing_mode: SizingMode, 
                            output_width: int, output_height: int,
                            config: KontextConfig, is_fill_model: bool = False) -> Tuple[int, int]:
        """Calculate target dimensions based on sizing mode."""
        if sizing_mode == SizingMode.NO_CHANGE:
            return image.size
        elif sizing_mode == SizingMode.TO_OUTPUT:
            return (output_width * config.vae_scale_factor, 
                   output_height * config.vae_scale_factor)
        elif sizing_mode == SizingMode.TO_BFL_RECOMMENDED:
            k_width, k_height = image.size
            aspect_ratio = k_width / k_height
            _, target_width, target_height = min(
                (abs(aspect_ratio - w / h), w, h) 
                for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
            return target_width, target_height
        elif sizing_mode == SizingMode.TO_FILL_RESOLUTION:
            # For Fill models, use specific resolution
            if is_fill_model:
                return (512, 512)  # Fill models typically use 512x512
            else:
                # Fallback to BFL recommended for non-fill models
                return ImageProcessor.calculate_target_size(
                    image, SizingMode.TO_BFL_RECOMMENDED, 
                    output_width, output_height, config
                )
        else:
            raise KontextError(f"Unknown sizing mode: {sizing_mode}")
    
    @staticmethod
    def preprocess_image(image, target_width: int, target_height: int, 
                        reduce: bool = False) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        try:
            # Apply reduction if requested
            if reduce:
                target_width //= 2
                target_height //= 2
                logger.info(f"Reducing image size by half to {target_width}x{target_height}")
            
            # Convert to RGB
            k_image = image.convert('RGB')
            
            # Convert to numpy and normalize
            k_image = np.array(k_image, dtype=np.float32) / 255.0
            k_image = np.transpose(k_image, (2, 0, 1))
            k_image = torch.from_numpy(k_image).unsqueeze(0)
            
            # Resize if needed
            current_height, current_width = k_image.shape[2], k_image.shape[3]
            if current_width != target_width or current_height != target_height:
                logger.info(
                    f"Resizing image from {current_width}x{current_height} "
                    f"to {target_width}x{target_height}"
                )
                k_image = adaptive_resize(
                    k_image, target_width, target_height, "lanczos", "center"
                )
            
            return k_image
            
        except Exception as e:
            raise KontextError(f"Failed to preprocess image: {str(e)}")


class LayoutManager:
    """Manages spatial layout of context images."""
    
    @staticmethod
    def calculate_placement_offset(current_h: int, current_w: int, 
                                 new_h: int, new_w: int) -> Tuple[int, int]:
        """Calculate optimal placement offset to minimize bounding box."""
        # Compare two placement options: below vs right
        height_if_below = current_h + new_h
        width_if_right = current_w + new_w
        
        if height_if_below > width_if_right:
            # Place to the right
            return 0, current_w
        else:
            # Place below
            return current_h, 0
    
    @staticmethod
    def create_position_ids(height: int, width: int, offset_h: int, offset_w: int,
                          device: torch.device, dtype: torch.dtype, 
                          batch_size: int) -> torch.Tensor:
        """Create position ID tensor for image patches."""
        k_id = torch.zeros((height, width, 3), device=device, dtype=dtype)
        k_id[:, :, 0] = 1  # Context image marker
        k_id[:, :, 1] += torch.linspace(
            offset_h, offset_h + height - 1, steps=height, 
            device=device, dtype=dtype
        )[:, None]
        k_id[:, :, 2] += torch.linspace(
            offset_w, offset_w + width - 1, steps=width, 
            device=device, dtype=dtype
        )[None, :]
        
        return repeat(k_id, "h w c -> b (h w) c", b=batch_size)


class AnalysisCache:
    """Simple cache for analysis results"""
    def __init__(self):
        self._cache = {}
        self._max_size = 10  # Keep last 10 analyses
        
    def get(self, key: str) -> Optional[Dict]:
        """Get cached analysis"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Dict):
        """Cache analysis result"""
        # Limit cache size
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def clear(self):
        """Clear all cached results"""
        self._cache.clear()
    
    def invalidate_image(self, image_index: int):
        """Invalidate all cache entries for a specific image index"""
        keys_to_remove = [key for key in self._cache if key.startswith(f"img_{image_index}_")]
        for key in keys_to_remove:
            del self._cache[key]
    
    def get_image_key(self, image_index: int, params: str) -> str:
        """Generate cache key for an image"""
        return f"img_{image_index}_{params}"


def create_patched_forward():
    """Create patched forward function with enhanced error handling."""
    
    def patched_flux_forward(self, x: torch.Tensor, timestep: torch.Tensor,
                           context: torch.Tensor, y: torch.Tensor,
                           guidance: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Patched forward pass that includes kontext images."""
        try:
            bs, c, h, w = x.shape
            input_device = x.device
            input_dtype = x.dtype
            
            # Pad to patch size
            pad_h = (PATCH_SIZE - x.shape[-2] % PATCH_SIZE) % PATCH_SIZE
            pad_w = (PATCH_SIZE - x.shape[-1] % PATCH_SIZE) % PATCH_SIZE
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
            
            # Convert to patches
            img = rearrange(
                x, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=PATCH_SIZE, pw=PATCH_SIZE
            )
            
            # Calculate patch dimensions
            h_len = ((h + (PATCH_SIZE // 2)) // PATCH_SIZE)
            w_len = ((w + (PATCH_SIZE // 2)) // PATCH_SIZE)
            
            # Create position IDs for main image
            img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
            img_ids[..., 1] += torch.linspace(
                0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype
            )[:, None]
            img_ids[..., 2] += torch.linspace(
                0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype
            )[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
            
            img_tokens = img.shape[1]
            
            # Add kontext if available
            current_state = ForgeKontextUnified.get_current_kontext_state()
            if current_state and current_state.is_initialized():
                img = torch.cat([img, current_state.latent], dim=1)
                img_ids = torch.cat([img_ids, current_state.ids], dim=1)
            
            # Create text IDs
            txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
            
            # Forward pass
            out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
            
            # Extract main image tokens and reshape
            out = out[:, :img_tokens]
            out = rearrange(
                out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                h=h_len, w=w_len, ph=PATCH_SIZE, pw=PATCH_SIZE
            )[:, :, :h, :w]
            
            return out
            
        except Exception as e:
            logger.error(f"Error in patched forward: {str(e)}")
            raise
    
    return patched_flux_forward


class SafeModelPatcher:
    """Thread-safe model patcher with reference counting."""
    
    def __init__(self):
        self.original_forward = None
        self.is_patched = False
        self.patch_count = 0
        self._lock = torch.multiprocessing.Lock() if torch.multiprocessing.get_context() else None
    
    @contextmanager
    def _thread_lock(self):
        """Thread-safe context manager."""
        if self._lock:
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock:
                self._lock.release()
    
    def apply_patch(self, model_class) -> bool:
        """Apply patch safely with reference counting."""
        with self._thread_lock():
            if not self.is_patched:
                self.original_forward = model_class.forward
                patched_forward = create_patched_forward()
                model_class.forward = patched_forward
                self.is_patched = True
                logger.info("Model patched successfully")
            
            self.patch_count += 1
            logger.debug(f"Patch reference count: {self.patch_count}")
            return True
    
    def remove_patch(self, model_class) -> bool:
        """Remove patch safely with reference counting."""
        with self._thread_lock():
            if self.patch_count > 0:
                self.patch_count -= 1
                logger.debug(f"Patch reference count: {self.patch_count}")
            
            if self.patch_count == 0 and self.is_patched:
                if self.original_forward is not None:
                    model_class.forward = self.original_forward
                self.is_patched = False
                self.original_forward = None
                logger.info("Model patch removed")
                return True
            return False


class ForgeKontextUnified(scripts.Script):
    """Unified Kontext Pro and Assistant functionality."""
    
    # Class-level components
    _model_patcher = SafeModelPatcher()
    _active_state: Optional[KontextState] = None
    
    # Add image sharing functionality
    _current_kontext_images: List[Optional[Image.Image]] = [None, None, None]
    _images_lock = threading.Lock()
    
    # Assistant components
    _shared_analyzer = None
    _analyzer_lock = threading.Lock()
    _analyzer_settings = None
    _analysis_lock = threading.Lock()
    
    def __init__(self):
        super().__init__()
        self.config = KontextConfig()
        self.kontext_state = KontextState()
        self.image_processor = ImageProcessor()
        self.layout_manager = LayoutManager()
        self.sorting_priority = 0
        
        # Assistant components
        self.analyzer = None
        self.force_cpu = False
        self._analysis_cache = AnalysisCache()
        self._image_hashes = {}
        self._analysis_states = [{}, {}, {}]
        self._analysis_texts = ["", "", ""]
        self._analysis_displays = []
        self._analysis_data_states = []
        self._callback_registered = False
    
    @classmethod
    def get_current_kontext_state(cls) -> Optional[KontextState]:
        """Get the currently active kontext state."""
        return cls._active_state
    
    @classmethod
    def set_current_kontext_state(cls, state: Optional[KontextState]) -> None:
        """Set the currently active kontext state."""
        cls._active_state = state
    
    @classmethod
    def get_kontext_images(cls) -> List[Optional[Image.Image]]:
        """Get currently loaded kontext images for other extensions"""
        return cls._current_kontext_images.copy()
    
    @classmethod  
    def set_kontext_images(cls, images: List[Optional[Image.Image]]) -> None:
        """Store kontext images for access by other extensions"""
        with cls._images_lock:
            cls._current_kontext_images = images.copy() if images else [None, None, None]
            logger.debug(f"Stored {sum(1 for img in images if img is not None)} kontext images for sharing")
    
    def title(self) -> str:
        return "Forge FluxKontext Pro"
    
    def show(self, is_img2img: bool) -> scripts.AlwaysVisible:
        return scripts.AlwaysVisible
    
    def _get_analyzer_settings(self, force_cpu: bool, florence_model_type: str = "base", auto_unload: bool = True, unload_delay: int = 60):
        """Get current analyzer settings"""
        return {
            'force_cpu': force_cpu,
            'florence_model_type': florence_model_type,
            'auto_unload': auto_unload,
            'unload_delay': unload_delay
        }
    
    def _initialize_modules(self, force_cpu=False, florence_model_type="base", auto_unload=True, unload_delay=60):
        """Initialize modules if needed, reusing shared analyzer when possible"""
        if not ASSISTANT_AVAILABLE:
            return False
            
        try:
            # Get device
            device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
            
            # Current settings
            current_settings = self._get_analyzer_settings(force_cpu, florence_model_type, auto_unload, unload_delay)
            
            # Check if we need to reinitialize analyzer
            need_new_analyzer = (
                ForgeKontextUnified._shared_analyzer is None or
                ForgeKontextUnified._analyzer_settings != current_settings
            )
            
            if need_new_analyzer:
                with ForgeKontextUnified._analyzer_lock:
                    # Double-check after acquiring lock
                    if ForgeKontextUnified._shared_analyzer is None or ForgeKontextUnified._analyzer_settings != current_settings:
                        logger.info(f"Creating new SmartAnalyzer with device={device}, florence_model_type={florence_model_type}")
                        
                        # Unload existing analyzer if settings changed
                        if ForgeKontextUnified._shared_analyzer and ForgeKontextUnified._analyzer_settings != current_settings:
                            logger.info("Settings changed, unloading existing analyzer...")
                            ForgeKontextUnified._shared_analyzer.unload_models()
                            ForgeKontextUnified._shared_analyzer = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Create new analyzer with delayed auto_unload for memory efficiency
                        ForgeKontextUnified._shared_analyzer = SmartAnalyzer(
                            device=device,
                            force_cpu=force_cpu,
                            florence_model_type=florence_model_type,
                            auto_unload=auto_unload,
                            unload_delay=unload_delay
                        )
                        ForgeKontextUnified._analyzer_settings = current_settings
            
            # Use the shared analyzer
            self.analyzer = ForgeKontextUnified._shared_analyzer
            
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_model_status(self) -> str:
        """Get current model loading status"""
        if not self.analyzer:
            return "Not initialized"
        
        try:
            status = self.analyzer.get_status()
            loaded_models = []
            
            # Check Florence Base
            if status.get('florence_base', {}).get('loaded', False):
                loaded_models.append("Florence-2 Base")
            
            # Check PromptGen
            if status.get('florence_promptgen', {}).get('loaded', False):
                loaded_models.append("PromptGen v2.0")
            
            if loaded_models:
                return f"Loaded: {', '.join(loaded_models)}"
            else:
                return "No models loaded"
                
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return "Status unavailable"
    
    def _get_performance_info(self) -> str:
        """Get detailed performance and model information"""
        info = []
        
        # Header
        info.append("### 🚀 Model & Performance Information\n")
        
        # Model status
        try:
            if self.analyzer:
                status = self.analyzer.get_status()
                
                # Florence Base info
                base_status = status.get('florence_base', {})
                if base_status:
                    info.append("**Florence-2 Base**")
                    if base_status.get('loaded', False):
                        info.append(f"- Model: {base_status.get('model_id', 'Unknown')}")
                        info.append(f"- Device: {base_status.get('device', 'Unknown')}")
                        info.append(f"- dtype: {base_status.get('dtype', 'Unknown')}")
                        info.append(f"- Compiled: {'Yes' if base_status.get('compiled', False) else 'No'}")
                        if 'gpu_memory' in base_status:
                            info.append(f"- GPU Memory: {base_status['gpu_memory']['allocated_gb']:.2f} GB")
                    else:
                        info.append("- Status: Not loaded (will load on first use)")
                    info.append("")
                
                # PromptGen info
                pg_status = status.get('florence_promptgen', {})
                if pg_status:
                    info.append("**PromptGen v2.0**")
                    if pg_status.get('loaded', False):
                        info.append(f"- Model: {pg_status.get('model_id', 'Unknown')}")
                        info.append(f"- Device: {pg_status.get('device', 'Unknown')}")
                        info.append(f"- dtype: {pg_status.get('dtype', 'Unknown')}")
                        info.append(f"- Compiled: {'Yes' if pg_status.get('compiled', False) else 'No'}")
                        if 'gpu_memory' in pg_status:
                            info.append(f"- GPU Memory: {pg_status['gpu_memory']['allocated_gb']:.2f} GB")
                    else:
                        info.append("- Status: Not loaded (will load on first use)")
                    info.append("")
        except Exception as e:
            logger.error(f"Error getting model status in performance info: {e}")
            pass
        
        # Performance metrics from last analyses
        info.append("### ⏱️ Recent Performance Metrics\n")
        
        perf_data = []
        for i in range(3):
            if self._analysis_states[i] and self._analysis_states[i].get('success', False):
                analysis = self._analysis_states[i]
                img_num = i + 1
                
                # Overall time
                if 'processing_time' in analysis:
                    perf_data.append(f"**Image {img_num}**: {analysis['processing_time']}")
                elif 'total_analysis_time' in analysis:
                    perf_data.append(f"**Image {img_num}**: {analysis['total_analysis_time']:.2f}s")
                
                # Task breakdown
                if 'task_times' in analysis and analysis['task_times']:
                    tasks = analysis['task_times']
                    task_str = ", ".join([f"{k}: {v:.2f}s" for k, v in tasks.items()])
                    perf_data.append(f"  - Tasks: {task_str}")
        
        if perf_data:
            info.extend(perf_data)
        else:
            info.append("*No performance data yet - analyze images to see metrics*")
        
        # GPU memory if available
        if torch.cuda.is_available():
            info.append("\n### 💾 GPU Memory Usage\n")
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info.append(f"- Allocated: {allocated:.2f} GB")
            info.append(f"- Reserved: {reserved:.2f} GB")
            info.append(f"- Total VRAM: {total:.2f} GB")
        
        # Version info
        info.append("\n### ℹ️ Version Info\n")
        info.append("- Kontext Assistant: v1.0.1")
        info.append(f"- PyTorch: {torch.__version__}")
        info.append("- Optimizations: FP16 + torch.compile")
        
        # Get current unload settings from analyzer
        if self.analyzer:
            info.append(f"- Auto-unload: {'Enabled' if self.analyzer.auto_unload else 'Disabled'}")
            if self.analyzer.auto_unload:
                info.append(f"- Unload delay: {self.analyzer.unload_delay} seconds")
        
        return "\n".join(info)
    
    def _has_image_changed(self, image: Image.Image, image_index: int) -> bool:
        """Check if an image has changed since last analysis"""
        if image is None:
            return False
            
        # Calculate image hash
        import hashlib
        import io
        
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        current_hash = hashlib.md5(img_bytes).hexdigest()
        
        # Check if changed
        prev_hash = self._image_hashes.get(image_index)
        if prev_hash != current_hash:
            self._image_hashes[image_index] = current_hash
            if self._analysis_cache:
                self._analysis_cache.invalidate_image(image_index)
            return True
        
        return False
    
    def _get_kontext_images_from_ui(self, *args) -> List[Optional[Image.Image]]:
        """Get kontext images from shared state or UI args"""
        kontext_images = []
        
        # First try to get from shared state
        if ASSISTANT_AVAILABLE and shared_state:
            try:
                kontext_images = shared_state.get_images()
                
                # Validate images to handle any format issues
                if validate_image_list:
                    kontext_images = validate_image_list(kontext_images)
                
                if any(img is not None for img in kontext_images):
                    return kontext_images
            except Exception as e:
                logger.error(f"Error getting images from shared state: {e}")
        
        # Fallback to class-level storage
        return ForgeKontextUnified.get_kontext_images()
    
    def analyze_image(self, image_index: int, force_cpu: bool, 
                     use_florence: bool, use_joycaption: bool, 
                     florence_model: str, show_detailed: bool, 
                     promptgen_instruction: str, auto_unload: bool,
                     unload_delay: int, analysis_mode: str, *args):
        """Analyze a kontext image with timing"""
        logger.info(f"analyze_image called: image_index={image_index}")
        
        # Use lock to prevent concurrent analysis
        with ForgeKontextUnified._analysis_lock:
            try:
                start_time = time.time()
                # Update settings
                self.force_cpu = force_cpu
                
                # Reinitialize if needed
                self._initialize_modules(force_cpu=force_cpu, florence_model_type=florence_model, 
                                       auto_unload=auto_unload, unload_delay=unload_delay)
                
                # Get kontext images from UI args
                kontext_images = self._get_kontext_images_from_ui(*args)
                
                if image_index >= len(kontext_images):
                    return f"❌ Invalid image index {image_index + 1}", {}
            
                image = kontext_images[image_index]
                if image is None:
                    # Clear analysis cache for this slot
                    self._analysis_cache.invalidate_image(image_index)
                    return f"❌ No image in slot {image_index + 1} - please load an image first", {}
                
                # Validate image is accessible
                try:
                    _ = image.size
                except Exception as e:
                    logger.error(f"Invalid image in slot {image_index + 1}: {e}")
                    return f"❌ Invalid or inaccessible image in slot {image_index + 1}", {}
            
                # Check if we should clear due to image change
                if self._has_image_changed(image, image_index):
                    logger.info(f"Image {image_index + 1} has changed, invalidating cache")
                
                # Check cache
                cache_key = f"{use_florence}_{use_joycaption}_{florence_model}_{promptgen_instruction}_{analysis_mode}"
                cached_result = self._analysis_cache.get(self._analysis_cache.get_image_key(image_index, cache_key))
                if cached_result:
                    logger.info(f"Using cached analysis for image {image_index + 1}")
                    return self._format_analysis_output(cached_result, show_detailed), cached_result
                
                # Use SmartAnalyzer with selected models
                if not use_florence and not use_joycaption:
                    return "❌ Please select at least one analysis method", {}
                
                try:
                    # Run analysis with SmartAnalyzer
                    analysis = self.analyzer.analyze(image, use_florence=use_florence, use_joycaption=use_joycaption, 
                                                    promptgen_instruction=promptgen_instruction, analysis_mode=analysis_mode)
                    
                    # Check for errors
                    if not analysis.get('success', True):
                        error_msg = analysis.get('error', 'Unknown error')
                        return f"❌ Analysis failed: {error_msg}", {}
                        
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    return f"❌ Analysis failed: {str(e)}", {}
            
                # Calculate analysis time
                analysis_time = time.time() - start_time
                logger.info(f"Analysis completed in {analysis_time:.2f}s")
                
                # Cache the result
                self._analysis_cache.set(self._analysis_cache.get_image_key(image_index, cache_key), analysis)
                
                # Format output using the method
                return self._format_analysis_output(analysis, show_detailed), analysis
                
            except Exception as e:
                logger.error(f"Error analyzing image {image_index + 1}: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Error analyzing image: {str(e)}", {}
    
    def _format_analysis_output(self, analysis: dict, show_detailed: bool = False) -> str:
        """Format analysis results into readable output for different analysis modes"""
        output = ""
        mode = analysis.get('analysis_mode', 'standard')
        
        # Handle different analysis modes
        if mode == 'fast':
            # Fast mode: Quick description & tags
            if analysis.get('brief_description'):
                output += f"📝 Quick Description:\n{analysis['brief_description']}"
            
            if analysis.get('tags') and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    if output:
                        output += "\n\n"
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
        
        elif mode == 'tags_only':
            # Tags only mode
            if analysis.get('tags') and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
            else:
                output = "❌ No tags generated"
        
        elif mode == 'composition':
            # Composition mode: Spatial analysis
            if analysis.get('composition_analysis'):
                output += f"📸 Composition Analysis:\n{analysis['composition_analysis']}"
            
            if analysis.get('objects'):
                objects_data = analysis['objects']
                if isinstance(objects_data, dict):
                    if output:
                        output += "\n\n"
                    output += "🎯 Detected Objects:\n"
                    
                    # Show objects with positions
                    if objects_data.get('with_positions'):
                        for obj in objects_data['with_positions']:
                            bbox = obj.get('bbox', [])
                            if bbox and len(bbox) == 4:
                                output += f"- {obj['label']}: ({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})\n"
                            else:
                                output += f"- {obj['label']}\n"
                    elif objects_data.get('all'):
                        output += ', '.join(objects_data['all'][:10])
            
            if analysis.get('dense_regions'):
                regions = analysis['dense_regions']
                if output:
                    output += "\n\n"
                output += "📍 Region Descriptions:\n"
                # Format dense regions data
                if isinstance(regions, dict) and 'labels' in regions:
                    for i, label in enumerate(regions.get('labels', [])):
                        output += f"- {label}\n"
        
        elif mode == 'detailed':
            # Detailed mode: Everything
            # Brief description from Florence-2 Base
            if analysis.get('brief_description'):
                output += f"📝 Brief Description:\n{analysis['brief_description']}"
            
            # Detailed description from PromptGen v2.0
            if analysis.get('description'):
                if output:
                    output += "\n\n"
                desc = DescriptionCleaner.clean(analysis['description']) if DescriptionCleaner else analysis['description']
                output += f"📝 Detailed Description:\n{desc}"
            
            # Tags
            if analysis.get('tags') and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    if output:
                        output += "\n\n"
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
            
            # Mixed caption PLUS
            if analysis.get('mixed_caption'):
                if output:
                    output += "\n\n"
                output += f"📝 Mixed Caption (FULL):\n{analysis['mixed_caption']}"
            
            # Composition
            if analysis.get('composition_analysis'):
                if output:
                    output += "\n\n"
                output += f"📸 Composition Analysis:\n{analysis['composition_analysis']}"
            
            # Objects (if detailed mode enabled)
            if show_detailed and analysis.get('objects'):
                objects_data = analysis['objects']
                if isinstance(objects_data, dict) and 'all' in objects_data:
                    object_list = objects_data['all']
                    if object_list:
                        if output:
                            output += "\n\n"
                        output += f"🎯 Detected Objects:\n{', '.join(object_list[:10])}"
            
            # Dense regions
            if show_detailed and analysis.get('dense_regions'):
                regions = analysis['dense_regions']
                if output:
                    output += "\n\n"
                output += "📍 Region Descriptions:\n"
                if isinstance(regions, dict) and 'labels' in regions:
                    for label in regions.get('labels', [])[:5]:
                        output += f"- {label}\n"
        
        # Standard mode or dual_model_automatic (backward compatibility)
        elif mode == 'standard' or mode == 'dual_model_automatic':
            # Brief description from Florence-2 Base
            if analysis.get('brief_description'):
                output += f"📝 Brief Description:\n{analysis['brief_description']}"
            
            # Detailed description from PromptGen v2.0
            if analysis.get('description'):
                if output:
                    output += "\n\n"
                desc = DescriptionCleaner.clean(analysis['description'])
                output += f"📝 Detailed Description:\n{desc}"
            
            # Tags from PromptGen v2.0
            if analysis.get('tags') and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    if output:
                        output += "\n\n"
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
            
            # Mixed caption from PromptGen
            if analysis.get('mixed_caption'):
                # First check if there's any meaningful content
                caption_parts = analysis['mixed_caption'].split('\n\n')
                has_content = False
                content_parts = []
                
                for part in caption_parts:
                    part_cleaned = part.strip()
                    # Skip empty parts and pure tag lists
                    if not part_cleaned or (', ' in part_cleaned and any(tag_word in part_cleaned.lower() for tag_word in ['solo', 'background', 'looking', 'standing', 'sitting', 'full body', '1girl', '1boy'])):
                        continue
                    has_content = True
                    content_parts.append(part_cleaned)
                
                # Only show Mixed Caption section if there's actual content
                if has_content and content_parts:
                    if output:
                        output += "\n\n"
                    output += "📝 Mixed Caption (FULL):\n"
                    
                    description_shown = False
                    for i, part_cleaned in enumerate(content_parts):
                        # Replace NA with circle icon
                        part_cleaned = part_cleaned.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                        
                        # Check if this is a structured data part (contains colons and semicolons)
                        if ':' in part_cleaned and ';' in part_cleaned:
                            if not description_shown:
                                output += "🎨 Structured prompt elements:\n"
                                description_shown = True
                            else:
                                output += "\n\n🎨 Additional elements:\n"
                            output += part_cleaned
                        else:
                            # This is a natural language description
                            if description_shown:
                                continue  # Skip duplicate descriptions
                            output += f"🎨 {part_cleaned}"
                            description_shown = True
            
            # Composition analysis from PromptGen
            if analysis.get('composition_analysis'):
                if output:
                    output += "\n\n"
                output += "📸 Composition Analysis:\n"
                # Replace NA with circle icon
                comp_analysis = analysis['composition_analysis']
                comp_analysis = comp_analysis.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                output += comp_analysis
            
            # Object detection from Florence-2 Base (if detailed mode)
            if show_detailed and analysis.get('objects'):
                objects_data = analysis['objects']
                if isinstance(objects_data, dict) and 'all' in objects_data:
                    object_list = objects_data['all']
                    if object_list:
                        if output:
                            output += "\n\n"
                        output += f"🎯 Detected Objects:\n{', '.join(object_list[:10])}"
            
            # Mood if available
            if 'mood' in analysis and analysis['mood']:
                if output:
                    output += "\n\n"
                output += f"💭 Mood: {analysis['mood']}"
        
        return output or "❌ No analysis data available"
    
    def ui(self, is_img2img: bool):
        """Create unified user interface."""
        
        # Debug logging
        logger.info(f"Creating UI - ASSISTANT_AVAILABLE: {ASSISTANT_AVAILABLE}")
        logger.info(f"PromptBuilder available: {PromptBuilder is not None}")
        logger.info(f"StyleLibrary available: {StyleLibrary is not None}")
        
        # Initialize modules on first UI creation with default settings
        if ASSISTANT_AVAILABLE:
            self._initialize_modules(force_cpu=False)
        
        # JavaScript for dimension setting
        set_dimensions_js = """
        function kontext_set_dimensions(tab_id, dims) {
            const [width, height] = dims.split(',').map(Number);
            if (width === 0 || height === 0) return;
            
            // Находим поля ввода размеров
            const widthInput = gradioApp().querySelector(`#${tab_id}_width input[type="number"]`);
            const heightInput = gradioApp().querySelector(`#${tab_id}_height input[type="number"]`);
            
            if (widthInput && heightInput) {
                // Устанавливаем новые значения
                widthInput.value = width;
                heightInput.value = height;
                
                // Вызываем события для обновления Gradio
                widthInput.dispatchEvent(new Event('input', { bubbles: true }));
                heightInput.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Дополнительно вызываем change event
                widthInput.dispatchEvent(new Event('change', { bubbles: true }));
                heightInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        
        // Убедимся, что функция доступна глобально
        window.kontext_set_dimensions = kontext_set_dimensions;
        """
        
        with InputAccordion(False, label=self.title()) as enabled:
            # Add custom CSS
            custom_css = """
            <style>
            .refresh-button, .clear-button {
                min-width: 40px !important;
                max-width: 40px !important;
                width: 40px !important;
                height: 40px !important;
                padding: 0 !important;
                font-size: 1.2em !important;
            }
            .refresh-button {
                color: #2196F3 !important;
                transition: transform 0.3s ease;
            }
            .refresh-button:hover {
                color: #1976D2 !important;
                transform: rotate(180deg);
            }
            .clear-button {
                color: #f44336 !important;
            }
            .clear-button:hover {
                color: #d32f2f !important;
            }
            </style>
            """
            gradio.HTML(custom_css)
            
            # Info text with lightbulb
            gradio.Markdown("💡 Select a FluxKontext model in the Checkpoint menu. Add reference image(s) here.")
            
            # Context images with dimension info
            kontext_images = []
            image_dimensions = []
            set_dims_buttons = []
            
            with gradio.Row():
                for i in range(3):
                    with gradio.Column():
                        img = gradio.Image(
                            label="",
                            type="pil",
                            height=200,
                            sources=["upload", "clipboard"],
                            show_label=False,
                            elem_id=f"kontext_image_{i}"
                        )
                        kontext_images.append(img)
                        
                        # Dimension info and button row
                        with gradio.Row():
                            dim_text = gradio.Textbox(
                                value="",
                                label=None,
                                interactive=False,
                                scale=3,
                                container=False,
                                elem_id=f"kontext_dim_info_{i}"
                            )
                            image_dimensions.append(dim_text)
                            
                            set_btn = gradio.Button(
                                "📐",
                                scale=1,
                                elem_id=f"kontext_set_dims_btn_{i}",
                                size="sm"
                            )
                            set_dims_buttons.append(set_btn)
            
            # Initialize shared images storage
            ForgeKontextUnified.set_kontext_images(kontext_images)
            
            # Single update function that avoids loops
            def store_kontext_images(img0, img1, img2):
                """Store images when they change"""
                images = [img0, img1, img2]
                
                # Validate images before storing
                if ASSISTANT_AVAILABLE and validate_image_list:
                    validated_images = validate_image_list(images)
                    ForgeKontextUnified.set_kontext_images(validated_images)
                    # Also update shared state for kontext assistant
                    if shared_state:
                        shared_state.set_images(validated_images)
                else:
                    ForgeKontextUnified.set_kontext_images(images)
                    if ASSISTANT_AVAILABLE and shared_state:
                        shared_state.set_images(images)
                
            # Connect each image individually to avoid circular updates
            kontext_images[0].change(
                fn=store_kontext_images,
                inputs=kontext_images,
                outputs=[]
            )
            kontext_images[1].change(
                fn=store_kontext_images,
                inputs=kontext_images,
                outputs=[]
            )
            kontext_images[2].change(
                fn=store_kontext_images,
                inputs=kontext_images,
                outputs=[]
            )
            
            # Kontext image size/crop section
            gradio.Markdown("**Kontext image size/crop**")
            
            with gradio.Row():
                with gradio.Column(scale=1):
                    sizing = gradio.Dropdown(
                        label=None,
                        choices=[mode.value for mode in SizingMode],
                        value=SizingMode.TO_BFL_RECOMMENDED.value,
                        elem_id="kontext_sizing_dropdown",
                        container=False
                    )
                    
                    # Dynamic help text based on selected mode
                    mode_help = gradio.Markdown(
                        value="**to BFL recommended** - Resize to the nearest recommended resolution for optimal quality",
                        elem_id="kontext_mode_help"
                    )
                
                with gradio.Column(scale=1):
                    # Empty column for spacing
                    pass
            
            # Hidden component for dimension calculation
            dims_info = gradio.Textbox(
                value="0",
                visible=False,
                elem_id="kontext_dims_info"
            )
            
            # Function to update image dimensions display
            def update_image_dimensions(img, reduce_checked=False):
                """Update dimension display for an image."""
                if img is None:
                    return ""
                w, h = img.size
                if reduce_checked:
                    return f"{w} × {h} ({w//2} × {h//2})"
                else:
                    return f"{w} × {h}"
            
            # Function to get dimensions from specific image
            def get_image_dimensions(img):
                """Get dimensions string from image."""
                if img is None:
                    return "0"
                return f"{img.size[0]},{img.size[1]}"
            
            # Function to update mode help text
            def update_mode_help(mode):
                """Update help text based on selected sizing mode."""
                help_texts = {
                    "no change": "**no change** - Use context images at their original resolution",
                    "to output": "**to output** - Scale context images to match the output resolution",
                    "to BFL recommended": "**to BFL recommended** - Resize to the nearest recommended resolution for optimal quality",
                    "to fill resolution": "**to fill resolution** - For inpainting models, uses 512×512 resolution"
                }
                return help_texts.get(mode, "")
            
            # Connect sizing mode change to help text update
            sizing.change(
                fn=update_mode_help,
                inputs=[sizing],
                outputs=[mode_help]
            )
            
            # Connect image changes to dimension updates
            for i, (img, dim_text) in enumerate(zip(kontext_images, image_dimensions)):
                img.change(
                    fn=update_image_dimensions,
                    inputs=[img],
                    outputs=[dim_text]
                )
            
            # Connect set dimension buttons
            for i, (img, btn) in enumerate(zip(kontext_images, set_dims_buttons)):
                btn.click(
                    fn=get_image_dimensions,
                    inputs=[img],
                    outputs=[dims_info]
                ).then(
                    fn=None,
                    _js=f"(dims) => kontext_set_dimensions('{'img2img' if is_img2img else 'txt2img'}', dims)",
                    inputs=[dims_info],
                    outputs=[]
                )
            
            # Add JavaScript to page
            gradio.HTML(f"<script>{set_dimensions_js}</script>")
            
            # Assistant functionality only if available
            if ASSISTANT_AVAILABLE:
                # Hidden values for compatibility
                use_florence = gradio.State(value=True)
                use_joycaption = gradio.State(value=False)
                florence_model = gradio.State(value="base")
                promptgen_instruction = gradio.State(value="<MORE_DETAILED_CAPTION>")
                
                # Image Analysis accordion
                with gradio.Accordion("📸 Image Analysis", open=False, elem_classes=["kontext-image-analysis"]):
                    # Analysis mode selection
                    analysis_mode = gradio.Radio(
                        choices=[
                            ("⚡ Quick Description (fast) - Florence only", "fast"),
                            ("⚖️ Standard descriptions & tags (balanced)", "standard"),
                            ("🔍 All-in-one analysis (slowest)", "detailed"),
                            ("📸 Spatial analysis (slow)", "composition"),
                            ("🏷️ Tags Only (fast)", "tags")
                        ],
                        value="standard",
                        label="Analysis Mode",
                        info="Choose based on your needs and time constraints"
                    )
                    
                    # Analyze and refresh/clear buttons
                    with gradio.Row():
                        analyze_all_btn = gradio.Button(
                            "🔍 Analyze",
                            variant="primary",
                            size="sm"
                        )
                        refresh_btn = gradio.Button(
                            "🔄",
                            size="sm",
                            elem_classes="refresh-button",
                            min_width=40
                        )
                        clear_btn = gradio.Button(
                            "🗑️",
                            size="sm",
                            elem_classes="clear-button",
                            min_width=40
                        )
                        analyze_status = gradio.Markdown("", elem_id="analyze_status")
                    
                    analysis_displays = []
                    analysis_data = []
                    
                    for i in range(3):
                        analysis_text = gradio.Textbox(
                            value="",
                            label=f"Analysis {i+1}",
                            interactive=False,
                            lines=4,
                            placeholder=f"Click 🔍 Analyze to analyze kontext image {i+1}...",
                            elem_id=f"analysis_text_{i}"
                        )
                        
                        analysis_displays.append(analysis_text)
                        # Hidden state to store analysis data
                        analysis_state = gradio.State(value={})
                        analysis_data.append(analysis_state)
                
            
            # Combined Advanced Settings
            with gradio.Accordion("⚙️ Advanced Settings", open=False, elem_classes=["kontext-advanced-settings"]):
                # Kontext processing settings
                gradio.Markdown("### Image Processing")
                gradio.Markdown("This reduction is independent of the size/crop setting.")
                
                reduce_checkbox = gradio.Checkbox(
                    False,
                    label="reduce to half width and height",
                    elem_id="kontext_reduce_checkbox"
                )
                
                enable_metrics = gradio.Checkbox(
                    True,
                    label="Enable performance metrics"
                )
                
                if ASSISTANT_AVAILABLE:
                    # Assistant settings
                    gradio.Markdown("### Assistant Settings")
                    
                    with gradio.Row():
                        force_cpu = gradio.Checkbox(
                            label="Force CPU Mode",
                            value=False,
                            info="Use CPU instead of GPU (slower but more compatible)"
                        )
                        
                        show_detailed = gradio.Checkbox(
                            label="Show Detailed Output",
                            value=False,
                            info="Display additional technical information"
                        )
                    
                    with gradio.Row():
                        auto_unload = gradio.Checkbox(
                            label="Auto-unload Models",
                            value=True,
                            info="Automatically unload models after delay to save VRAM"
                        )
                        
                        unload_delay = gradio.Slider(
                            label="Unload Delay (seconds)",
                            minimum=0,
                            maximum=300,
                            value=60,
                            step=15,
                            info="Time to wait before unloading models (0 = immediate)",
                            visible=True
                        )
                    
                    # Model status display
                    model_status = gradio.Markdown(
                        value=self._get_model_status(),
                        elem_id="model_status"
                    )
                
                # Update dimensions when reduce checkbox changes
                def update_all_dimensions(reduce_checked, *images):
                    """Update all dimension displays when reduce changes."""
                    return [update_image_dimensions(img, reduce_checked) for img in images]
                
                reduce_checkbox.change(
                    fn=update_all_dimensions,
                    inputs=[reduce_checkbox] + kontext_images,
                    outputs=image_dimensions
                )
            
            # Performance metrics accordion (if assistant available)
            if ASSISTANT_AVAILABLE:
                with gradio.Accordion("📊 Performance & Model Info", open=False, elem_classes=["kontext-performance"]):
                    performance_info = gradio.Markdown(
                        value=self._get_performance_info(),
                        elem_id="performance_info"
                    )
            
            # Prompt Builder accordion
            if ASSISTANT_AVAILABLE and PromptBuilder and StyleLibrary:
                with gradio.Accordion("🎨 Prompt Builder", open=False, elem_classes=["kontext-prompt-builder"]):
                    # Initialize builders
                    prompt_builder = PromptBuilder()
                    style_library = StyleLibrary()
                    
                    # Debug log style library initialization
                    logger.info(f"StyleLibrary initialized with {len(style_library.styles)} styles")
                    artistic_styles = style_library.get_styles_by_category(StyleCategory.TRADITIONAL_ART)
                    logger.info(f"Found {len(artistic_styles)} artistic styles")
                    
                    # Scenario selection
                    prompt_scenario = gradio.Radio(
                        choices=[
                            ("🎨 Style Transfer", PromptType.STYLE_TRANSFER.value),
                            ("✏️ Add/Remove Object", "object_manipulation"),  # Combined scenario
                            ("🔄 Replace Object", PromptType.OBJECT_REPLACE.value),
                            ("🚶 Change Pose", PromptType.POSE_CHANGE.value),
                            ("😊 Change Emotion", PromptType.EMOTION_CHANGE.value),
                            ("💡 Change Lighting", PromptType.LIGHTING_CHANGE.value),
                            ("🔧 Enhance/Restore Image", PromptType.IMAGE_ENHANCEMENT.value),
                            ("🖼️ Extend Canvas", PromptType.OUTPAINTING.value),
                            ("🔀 Dual-Image Mode", "dual_image"),  # NEW dual-image mode
                            ("📁 User Prompts", "user_prompts")
                        ],
                        value=PromptType.STYLE_TRANSFER.value,
                        label="What do you want to do?",
                        interactive=True
                    )
                    
                    # Dynamic input fields container
                    with gradio.Column() as prompt_inputs_container:
                        # Object manipulation inputs (visible by default)
                        with gradio.Group(visible=False) as object_inputs:
                            # Add/Remove mode selection
                            object_mode = gradio.Radio(
                                label="Mode",
                                choices=[("Add Object", "add"), ("Remove Object", "remove")],
                                value="add",
                                interactive=True
                            )
                            
                            object_field = gradio.Textbox(
                                label="Object",
                                placeholder="e.g., red balloon, sunglasses, golden crown",
                                interactive=True
                            )
                            
                            # Fields for adding objects
                            with gradio.Group(visible=True) as add_fields:
                                position_field = gradio.Dropdown(
                                    label="Position",
                                    choices=prompt_builder.position_descriptors,
                                    value="in the center",
                                    interactive=True,
                                    allow_custom_value=True
                                )
                                # Style modifier with categories
                                with gradio.Row():
                                    style_modifier_category = gradio.Dropdown(
                                        label="Style Category",
                                        choices=[
                                            ("💡 Lighting & Environment", "lighting_integration"),
                                            ("📍 Positioning & Composition", "positioning"),
                                            ("🎨 Style Integration", "style_matching"),
                                            ("✨ Effects & Interactions", "effects"),
                                            ("📏 Scale & Proportions", "scale"),
                                            ("✏️ Custom", "custom")
                                        ],
                                        value="lighting_integration",
                                        interactive=True
                                    )
                                
                                # Create choices for the initial category
                                initial_category = "lighting_integration"
                                initial_modifiers = []
                                if initial_category in STYLE_MODIFIERS_CONFIG.get("categories", {}):
                                    for mod in STYLE_MODIFIERS_CONFIG["categories"][initial_category]["modifiers"]:
                                        initial_modifiers.append((mod["name"], mod["value"]))
                                
                                style_modifier_preset = gradio.Dropdown(
                                    label="Style Modifier",
                                    choices=initial_modifiers,
                                    value=initial_modifiers[0][1] if initial_modifiers else "",
                                    interactive=True,
                                    allow_custom_value=False
                                )
                                
                                style_modifier_field = gradio.Textbox(
                                    label="Custom Style Modifier",
                                    placeholder="e.g., floating gently, reflecting surroundings",
                                    interactive=True,
                                    visible=False
                                )
                            
                            # Fields for removing objects
                            with gradio.Group(visible=False) as remove_fields:
                                removal_scenario = gradio.Radio(
                                    label="What type of object?",
                                    choices=[
                                        ("👔 Clothing", RemovalScenario.CLOTHING.value),
                                        ("👓 Accessories", RemovalScenario.ACCESSORIES.value),
                                        ("🏠 Background Object", RemovalScenario.BACKGROUND_OBJECT.value),
                                        ("👤 Person", RemovalScenario.PERSON.value),
                                        ("📝 Text/Logos", RemovalScenario.TEXT_LOGOS.value),
                                        ("©️ Watermark", RemovalScenario.WATERMARK.value),
                                        ("🚗 Vehicle", RemovalScenario.VEHICLE.value),
                                        ("🪑 Furniture", RemovalScenario.FURNITURE.value),
                                        ("✏️ Custom", RemovalScenario.CUSTOM.value)
                                    ],
                                    value=RemovalScenario.CUSTOM.value,
                                    interactive=True
                                )
                                
                                # Quick select dropdown for items
                                with gradio.Row() as removal_quick_select:
                                    removal_item_dropdown = gradio.Dropdown(
                                        label="Quick select item:",
                                        choices=[],
                                        interactive=True,
                                        visible=False,
                                        allow_custom_value=False,
                                        multiselect=False
                                    )
                                
                                fill_method = gradio.Textbox(
                                    label="How to Fill Empty Space",
                                    placeholder="e.g., blend with background, extend landscape naturally",
                                    value="blend seamlessly with the background",
                                    interactive=True
                                )
                                
                                # Quick fill method dropdown
                                with gradio.Row() as fill_quick_select:
                                    fill_method_dropdown = gradio.Dropdown(
                                        label="Quick fill method:",
                                        choices=[],
                                        interactive=True,
                                        visible=False,
                                        allow_custom_value=False,
                                        multiselect=False
                                    )
                        
                        # Style transfer inputs (visible by default since style transfer is default scenario)
                        with gradio.Group(visible=True) as style_inputs:
                            # Style selection
                            style_category = gradio.Dropdown(
                                label="Style Category",
                                choices=[(cat.value.replace('_', ' ').title(), cat.value) for cat in StyleCategory if cat != StyleCategory.CUSTOM],
                                value=StyleCategory.TRADITIONAL_ART.value,
                                interactive=True
                            )
                            
                            # Initialize with artistic styles
                            initial_styles = style_library.get_styles_by_category(StyleCategory.TRADITIONAL_ART)
                            style_choices = [(style.name, style.id) for style in initial_styles]
                            
                            style_preset = gradio.Dropdown(
                                label="Style Preset",
                                choices=style_choices,
                                value=style_choices[0][1] if style_choices else None,
                                interactive=True
                            )
                            
                            # Style details display
                            style_details = gradio.Markdown(
                                value="",
                                visible=False
                            )
                            
                            # Prompt field for reference styles
                            style_reference_prompt = gradio.Textbox(
                                label="Complete the prompt",
                                placeholder="e.g., a cat portrait, futuristic cityscape, fantasy warrior",
                                visible=False,
                                interactive=True,
                                info="This will replace [prompt] in the style template"
                            )
                            
                            # Preserve options
                            preserve_elements = gradio.CheckboxGroup(
                                label="Preserve from original",
                                choices=[
                                    "subject identity",
                                    "composition",
                                    "scene content",
                                    "lighting",
                                    "color palette"
                                ],
                                value=["subject identity", "composition"],
                                interactive=True
                            )
                        
                        # Replace object inputs (hidden by default)
                        with gradio.Group(visible=False) as replace_inputs:
                            original_object = gradio.Textbox(
                                label="Original Object",
                                placeholder="e.g., the coffee cup, the sedan car",
                                interactive=True
                            )
                            new_object = gradio.Textbox(
                                label="New Object",
                                placeholder="e.g., tea cup with steam, sports car",
                                interactive=True
                            )
                            maintain_aspects = gradio.Textbox(
                                label="What to Maintain",
                                placeholder="e.g., same position and lighting, perspective and color",
                                value="keeping the same position and lighting",
                                interactive=True
                            )
                        
                        # Pose change inputs (hidden by default)
                        with gradio.Group(visible=False) as pose_inputs:
                            subject_field = gradio.Textbox(
                                label="Subject",
                                value="person",
                                interactive=True
                            )
                            # Get pose suggestions if available
                            pose_suggestions = prompt_builder.get_pose_suggestions() if prompt_builder else [
                                "standing straight", "sitting cross-legged", "walking forward", "running"
                            ]
                            
                            new_pose = gradio.Dropdown(
                                label="New Pose",
                                choices=pose_suggestions,
                                value=pose_suggestions[0] if pose_suggestions else None,
                                interactive=True,
                                allow_custom_value=True
                            )
                            pose_details = gradio.Textbox(
                                label="Additional Details",
                                placeholder="e.g., confident stance, same facial expression",
                                value="maintaining the same outfit",
                                interactive=True
                            )
                        
                        # Emotion change inputs (hidden by default)
                        with gradio.Group(visible=False) as emotion_inputs:
                            emotion_subject = gradio.Textbox(
                                label="Subject",
                                value="the person's",
                                interactive=True
                            )
                            emotion_type = gradio.Dropdown(
                                label="Emotion",
                                choices=[
                                    "happy", "sad", "angry", "surprised", 
                                    "thoughtful", "confident", "shy", 
                                    "excited", "calm", "worried",
                                    "laughing", "contemplative", "determined"
                                ],
                                value="happy",
                                interactive=True,
                                allow_custom_value=True
                            )
                            emotion_intensity = gradio.Textbox(
                                label="Intensity/Details",
                                placeholder="e.g., with eyes crinkled, subtle expression",
                                value="natural and genuine",
                                interactive=True
                            )
                        
                        # Detail enhancement inputs (hidden by default)
                        with gradio.Group(visible=False) as detail_inputs:
                            detail_area = gradio.Dropdown(
                                label="Area to Enhance",
                                choices=[
                                    "the eyes", "the face", "the hair", 
                                    "the hands", "the clothing", "the background",
                                    "the skin", "the fabric texture"
                                ],
                                interactive=True,
                                allow_custom_value=True
                            )
                            enhancement_type = gradio.Dropdown(
                                label="Enhancement Type",
                                choices=[
                                    "adding more detail",
                                    "increasing definition",
                                    "improving realism",
                                    "fixing artifacts",
                                    "enhancing texture",
                                    "adding highlights"
                                ],
                                value="adding more detail",
                                interactive=True
                            )
                            specific_changes = gradio.Textbox(
                                label="Specific Changes",
                                placeholder="e.g., clearer iris patterns, individual hair strands",
                                interactive=True
                            )
                        
                        # Lighting change inputs (hidden by default)
                        with gradio.Group(visible=False) as lighting_inputs:
                            lighting_scenario = gradio.Radio(
                                label="Lighting Type",
                                choices=[
                                    ("🌞 Natural", LightingScenario.NATURAL.value),
                                    ("📸 Studio", LightingScenario.STUDIO.value),
                                    ("🎭 Dramatic", LightingScenario.DRAMATIC.value),
                                    ("💫 Ambient", LightingScenario.AMBIENT.value),
                                    ("🌃 Neon", LightingScenario.NEON.value),
                                    ("🕯️ Candlelight", LightingScenario.CANDLELIGHT.value),
                                    ("🌅 Sunset", LightingScenario.SUNSET.value),
                                    ("🌙 Night", LightingScenario.NIGHT.value),
                                    ("🌊 Underwater", LightingScenario.UNDERWATER.value),
                                    ("✨ Backlit", LightingScenario.BACKLIT.value),
                                    ("🎨 Custom", LightingScenario.CUSTOM.value)
                                ],
                                value=LightingScenario.NATURAL.value,
                                interactive=True
                            )
                            
                            # Initialize with Natural lighting suggestions
                            natural_suggestions = prompt_builder.get_lighting_suggestions(LightingScenario.NATURAL) if prompt_builder else {}
                            
                            lighting_preset = gradio.Dropdown(
                                label="Lighting Preset",
                                choices=natural_suggestions.get("presets", ["soft morning light"]),
                                value=natural_suggestions.get("presets", ["soft morning light"])[0] if natural_suggestions.get("presets") else None,
                                interactive=True,
                                allow_custom_value=True
                            )
                            
                            lighting_effects = gradio.Dropdown(
                                label="Effects",
                                choices=natural_suggestions.get("effects", []),
                                interactive=True,
                                multiselect=True
                            )
                            
                            lighting_adjustments = gradio.Dropdown(
                                label="Adjustments",
                                choices=natural_suggestions.get("adjustments", ["while maintaining subject details"]),
                                value="while maintaining subject details",
                                interactive=True
                            )
                            
                            # Context-aware lighting suggestions
                            lighting_context = gradio.Radio(
                                label="What did you change? (for lighting recommendations)",
                                choices=[
                                    ("Added outdoor object", "outdoor_object"),
                                    ("Added indoor object", "indoor_object"),
                                    ("Added person", "person_added"),
                                    ("Changed background", "background_changed"),
                                    ("Made it night", "night_scene"),
                                    ("Added water", "water_added"),
                                    ("No specific change", "general")
                                ],
                                value="general",
                                interactive=True,
                                visible=False
                            )
                            
                            lighting_direction = gradio.Dropdown(
                                label="Light Direction (optional)",
                                choices=[
                                    "from above", "from below",
                                    "from left", "from right",
                                    "from front", "from behind",
                                    "all around", "multiple sources"
                                ],
                                interactive=True,
                                allow_custom_value=True
                            )
                        
                        # Image restoration inputs (hidden by default)
                        with gradio.Group(visible=False) as restoration_inputs:
                            restoration_type = gradio.Radio(
                                label="Enhancement/Restoration Type",
                                choices=[
                                    # Detail Enhancement options
                                    ("👁️ Facial Details", EnhancementScenario.FACIAL_DETAILS.value),
                                    ("👀 Eye Enhancement", EnhancementScenario.EYE_ENHANCEMENT.value),
                                    ("🧺 Texture Details", EnhancementScenario.TEXTURE_DETAILS.value),
                                    ("💇 Hair Details", EnhancementScenario.HAIR_DETAILS.value),
                                    ("🔍 Overall Sharpness", EnhancementScenario.OVERALL_SHARPNESS.value),
                                    # Restoration options
                                    ("🖼️ Old Photo", EnhancementScenario.OLD_PHOTO.value),
                                    ("🎨 Damaged Art", EnhancementScenario.DAMAGED_ART.value),
                                    ("🔍 Low Resolution", EnhancementScenario.LOW_RESOLUTION.value),
                                    ("😵 Blurry", EnhancementScenario.BLURRY.value),
                                    ("📦 Exposure Issues", EnhancementScenario.EXPOSURE_ISSUES.value),
                                    ("🔊 Noisy", EnhancementScenario.NOISY.value),
                                    ("🗜️ Compressed", EnhancementScenario.COMPRESSED.value),
                                    ("🎨 Faded Colors", EnhancementScenario.FADED_COLORS.value),
                                    ("✨ General Enhancement", EnhancementScenario.GENERAL_ENHANCEMENT.value),
                                    ("🎯 Custom", EnhancementScenario.CUSTOM.value)
                                ],
                                value=EnhancementScenario.FACIAL_DETAILS.value,
                                interactive=True
                            )
                            
                            # Initialize with Facial Details enhancement suggestions
                            initial_suggestions = prompt_builder.get_enhancement_suggestions(EnhancementScenario.FACIAL_DETAILS) if prompt_builder else {}
                            
                            restoration_method = gradio.Dropdown(
                                label="Enhancement/Restoration Method",
                                choices=initial_suggestions.get("methods", ["custom enhancement technique"]),
                                value=initial_suggestions.get("methods", ["custom enhancement technique"])[0] if initial_suggestions.get("methods") else None,
                                interactive=True,
                                allow_custom_value=True
                            )
                            
                            quality_goal = gradio.Dropdown(
                                label="Quality Goal",
                                choices=initial_suggestions.get("quality_goals", ["professional quality"]),
                                value=initial_suggestions.get("quality_goals", ["professional quality"])[0] if initial_suggestions.get("quality_goals") else "professional quality",
                                interactive=True,
                                allow_custom_value=True
                            )
                        
                        # Outpainting inputs (hidden by default)
                        with gradio.Group(visible=False) as outpainting_inputs:
                            outpaint_direction = gradio.Radio(
                                label="Extension Direction",
                                choices=[
                                    ("↔️ Horizontal", OutpaintingDirection.HORIZONTAL.value),
                                    ("↕️ Vertical", OutpaintingDirection.VERTICAL.value),
                                    ("←️ Left", OutpaintingDirection.LEFT.value),
                                    ("→️ Right", OutpaintingDirection.RIGHT.value),
                                    ("↑️ Top", OutpaintingDirection.TOP.value),
                                    ("↓️ Bottom", OutpaintingDirection.BOTTOM.value),
                                    ("🔄 All Sides", OutpaintingDirection.ALL_SIDES.value),
                                    ("🎥 Widescreen", OutpaintingDirection.WIDESCREEN.value),
                                    ("◻️ Square", OutpaintingDirection.SQUARE.value),
                                    ("🎯 Custom", OutpaintingDirection.CUSTOM.value)
                                ],
                                value=OutpaintingDirection.HORIZONTAL.value,
                                interactive=True
                            )
                            
                            # Initialize with Horizontal outpainting suggestions
                            horizontal_suggestions = prompt_builder.get_outpainting_suggestions(OutpaintingDirection.HORIZONTAL) if prompt_builder else {}
                            
                            extension_description = gradio.Dropdown(
                                label="What to Add",
                                choices=horizontal_suggestions.get("extensions", ["custom extension"]),
                                value=horizontal_suggestions.get("extensions", ["custom extension"])[0] if horizontal_suggestions.get("extensions") else None,
                                interactive=True,
                                allow_custom_value=True
                            )
                            
                            consistency_elements = gradio.Dropdown(
                                label="Maintain Consistency",
                                choices=horizontal_suggestions.get("consistency", ["consistent perspective and lighting"]),
                                value="consistent perspective and lighting",
                                interactive=True,
                                allow_custom_value=True
                            )
                        
                        # User prompts section (hidden by default)
                        with gradio.Group(visible=False) as user_prompts_inputs:
                            # User saved custom styles selector
                            user_custom_styles = style_library.get_styles_by_category(StyleCategory.CUSTOM) if style_library else []
                            user_style_choices = [(style.name, style.id) for style in user_custom_styles]
                            
                            user_prompt_selector = gradio.Dropdown(
                                label="Select Saved User Prompt",
                                choices=user_style_choices,
                                interactive=True,
                                allow_custom_value=False
                            )
                            
                            user_prompt_display = gradio.Textbox(
                                label="Selected Prompt",
                                lines=3,
                                interactive=False,
                                elem_classes=["user-prompt-display"]
                            )
                            
                            with gradio.Row():
                                use_user_prompt_btn = gradio.Button(
                                    "✅ Use This Prompt",
                                    variant="primary"
                                )
                                delete_user_prompt_btn = gradio.Button(
                                    "🗑️ Delete",
                                    variant="stop"
                                )
                            
                            user_prompt_status = gradio.Markdown(visible=False)
                        
                        # Dual-image inputs section (hidden by default)
                        with gradio.Group(visible=False) as dual_image_inputs:
                            # Load dual-image configurations
                            dual_image_config_path = os.path.join(extension_dir, "configs", "dual_image_configs.json")
                            try:
                                with open(dual_image_config_path, 'r', encoding='utf-8') as f:
                                    dual_image_config = json.load(f)
                            except:
                                dual_image_config = {"scenarios": {}}
                            
                            # Dual-image scenario selection
                            dual_scenario_choices = []
                            for key, scenario in dual_image_config.get("scenarios", {}).items():
                                dual_scenario_choices.append((f"{scenario['icon']} {scenario['name']}", key))
                            
                            dual_image_scenario = gradio.Radio(
                                label="Select Dual-Image Mode",
                                choices=dual_scenario_choices,
                                value="style_content" if dual_scenario_choices else None,
                                interactive=True
                            )
                            
                            # Image indicators
                            gradio.Markdown("ℹ️ **Note**: Describe elements visually. FLUX.1 Kontext will automatically identify them from your loaded images.")
                            
                            # Dynamic parameters based on scenario
                            dual_param1 = gradio.Textbox(
                                label="Primary Element",
                                placeholder="Describe what you want to use",
                                interactive=True
                            )
                            
                            dual_param2 = gradio.Textbox(
                                label="Secondary Element",
                                placeholder="Describe additional elements",
                                interactive=True
                            )
                            
                            # Additional field for some scenarios
                            dual_param3 = gradio.Textbox(
                                label="Additional Parameter",
                                placeholder="Extra details",
                                interactive=True,
                                visible=False
                            )
                            
                            # Character arrangement presets (hidden by default)
                            with gradio.Group(visible=False) as character_merge_options:
                                # Load unified arrangements config
                                unified_path = os.path.join(extension_dir, "configs", "unified_arrangements.json")
                                try:
                                    with open(unified_path, 'r', encoding='utf-8') as f:
                                        unified_config = json.load(f)
                                except:
                                    unified_config = {"arrangements": {}, "interaction_styles": {}, "integration_methods": {}}
                                
                                # Unified arrangement dropdown (combines preset + spatial)
                                arrangement_choices = []
                                for arr_key, arr_data in unified_config.get("arrangements", {}).items():
                                    if arr_key != "custom":
                                        category = arr_data.get("category", "")
                                        icon = arr_data.get("icon", "")
                                        name = arr_data.get("name", "")
                                        # Add category header
                                        arrangement_choices.append((f"━━━ {icon} {name} ━━━", f"__category_{arr_key}"))
                                        # Add options
                                        for option in arr_data.get("options", []):
                                            arrangement_choices.append((f"    {option['name']}", f"{arr_key}:{option['name']}"))
                                
                                # Add custom option at the end
                                arrangement_choices.append(("━━━ ✏️ Custom ━━━", "__category_custom"))
                                arrangement_choices.append(("    Custom Arrangement", "custom:custom"))
                                
                                unified_arrangement = gradio.Dropdown(
                                    label="Arrangement & Position",
                                    choices=arrangement_choices,
                                    value=None,
                                    interactive=True
                                )
                                
                                # Interaction style dropdown (optional)
                                interaction_choices = []
                                for key, style in unified_config.get("interaction_styles", {}).items():
                                    interaction_choices.append((style['name'], key))
                                
                                interaction_style = gradio.Dropdown(
                                    label="Additional Interaction Style (Optional)",
                                    choices=interaction_choices,
                                    value="none",
                                    interactive=True
                                )
                                
                                # Custom arrangement field (visible when custom is selected)
                                custom_arrangement_field = gradio.Textbox(
                                    label="Custom Arrangement Description",
                                    placeholder="Describe how the characters should be positioned",
                                    interactive=True,
                                    visible=False
                                )
                                
                                # Integration method info
                                integration_info = gradio.Markdown(
                                    "**Integration Method** affects how the two images are visually combined:",
                                    visible=True
                                )
                            
                            # Integration method with descriptions
                            integration_choices = []
                            for key, method in unified_config.get("integration_methods", {}).items():
                                if key != "note":
                                    choice_label = f"{method['name']} - {method['best_for']}"
                                    integration_choices.append((choice_label, key))
                            
                            integration_method = gradio.Radio(
                                label="Integration Method",
                                choices=integration_choices if integration_choices else [
                                    ("Natural - Photorealistic results", "natural"),
                                    ("Artistic - Stylized compositions", "artistic"),
                                    ("Seamless - Professional blending", "seamless")
                                ],
                                value="natural",
                                interactive=True
                            )
                            
                            # Additional options
                            preservation_options = gradio.CheckboxGroup(
                                label="What to Preserve",
                                choices=[
                                    "Original lighting",
                                    "Color palette",
                                    "Artistic style",
                                    "Composition structure",
                                    "Texture details",
                                    "Atmospheric mood"
                                ],
                                value=[],
                                interactive=True
                            )
                    
                    # Tips display
                    prompt_tips = gradio.Markdown(
                        value="💡 **Tips:** Be specific about object appearance, position, and style",
                        elem_id="prompt_tips"
                    )
                    
                    # Generated prompt output
                    with gradio.Row():
                        build_prompt_btn = gradio.Button(
                            "🔨 Build Prompt",
                            variant="primary",
                            size="lg"
                        )
                        copy_to_prompt_btn = gradio.Button(
                            "📋 Copy to Clipboard",
                            variant="secondary",
                            size="lg",
                            elem_id="ka_copy_to_clipboard_btn"
                        )
                        generate_btn = gradio.Button(
                            "🚀 Generate",
                            variant="primary",
                            size="lg",
                            elem_id="ka_generate_btn"
                        )
                    
                    generated_prompt_output = gradio.Textbox(
                        label="Generated Prompt",
                        lines=3,
                        interactive=True,
                        elem_id="generated_prompt_output",
                        placeholder="Your generated prompt will appear here. You can edit it before copying or generating."
                    )
                    
                    # Token count display
                    token_count_display = gradio.Markdown(
                        value="",
                        elem_id="token_count_display"
                    )
                    
                    # Custom style save UI
                    with gradio.Accordion("💾 Save as Custom Style", open=False):
                        with gradio.Row():
                            custom_style_name = gradio.Textbox(
                                label="Style Name",
                                placeholder="Enter a name for your custom style",
                                interactive=True
                            )
                            custom_style_description = gradio.Textbox(
                                label="Description (optional)",
                                placeholder="Brief description of this style",
                                interactive=True
                            )
                        save_custom_style_btn = gradio.Button(
                            "💾 Save Custom Style",
                            variant="secondary"
                        )
                        custom_style_status = gradio.Markdown(visible=False)
                    
                    # Examples gallery
                    with gradio.Accordion("📚 Examples", open=False):
                        examples_display = gradio.Markdown(
                            value="",
                            elem_id="prompt_examples"
                        )
            
            # Event handlers for assistant functionality
            if ASSISTANT_AVAILABLE:
                # Store references to analysis displays and data states
                self._analysis_displays = analysis_displays
                self._analysis_data_states = analysis_data
                
                
                # Analyze all images function
                def analyze_all_images(*args):
                    """Analyze all available images"""
                    results = []
                    
                    # Extract the inputs we need
                    force_cpu_val = args[0]
                    use_florence_val = args[1]
                    use_joycaption_val = args[2]
                    florence_model_val = args[3]
                    show_detailed_val = args[4]
                    promptgen_instruction_val = args[5]
                    auto_unload_val = args[6]
                    unload_delay_val = args[7]
                    analysis_mode_val = args[8]
                    
                    # The rest are extra inputs for images
                    extra_args = args[9:]
                    
                    # Get current images
                    kontext_images = self._get_kontext_images_from_ui(*extra_args)
                    
                    for i in range(3):
                        current_image = kontext_images[i] if i < len(kontext_images) else None
                        
                        # Check if image changed or was removed
                        if current_image is None:
                            if i in self._image_hashes:
                                # Image was removed
                                logger.info(f"Image {i+1} was removed")
                                del self._image_hashes[i]
                                self._analysis_states[i] = {}
                                self._analysis_texts[i] = ""
                                if self._analysis_cache:
                                    self._analysis_cache.invalidate_image(i)
                            results.extend(["", {}])
                        else:
                            # Check if image changed
                            if self._has_image_changed(current_image, i):
                                logger.info(f"Image {i+1} has changed - will re-analyze")
                                self._analysis_states[i] = {}
                                self._analysis_texts[i] = ""
                                if self._analysis_cache:
                                    self._analysis_cache.invalidate_image(i)
                            
                            # Analyze the image
                            text_result, data_result = self.analyze_image(
                                i, force_cpu_val, use_florence_val, use_joycaption_val,
                                florence_model_val, show_detailed_val, promptgen_instruction_val,
                                auto_unload_val, unload_delay_val, analysis_mode_val, *extra_args
                            )
                            results.extend([text_result, data_result])
                            self._analysis_states[i] = data_result
                            self._analysis_texts[i] = text_result
                    
                    return results
                
                # Refresh function
                def refresh_missing_analyses(*args):
                    """Clear analysis for missing images and generated prompt"""
                    # Get extra inputs for images
                    extra_args = args[0:]
                    
                    # Get current images
                    kontext_images = self._get_kontext_images_from_ui(*extra_args)
                    
                    results = []
                    cleared_count = 0
                    has_empty_slots = False
                    
                    for i in range(3):
                        current_image = kontext_images[i] if i < len(kontext_images) else None
                        
                        if current_image is None:
                            # No image in this slot - clear analysis
                            has_empty_slots = True
                            if self._analysis_states[i]:
                                logger.info(f"Clearing analysis for empty slot {i+1}")
                                cleared_count += 1
                            self._analysis_states[i] = {}
                            self._analysis_texts[i] = ""
                            if i in self._image_hashes:
                                del self._image_hashes[i]
                            if self._analysis_cache:
                                self._analysis_cache.invalidate_image(i)
                            results.extend(["", {}])
                        else:
                            # Image exists - keep current analysis
                            if self._analysis_states[i] and self._analysis_states[i].get('success', False):
                                results.extend([self._analysis_texts[i], self._analysis_states[i]])
                            else:
                                results.extend(["", self._analysis_states[i]])
                    
                    # Clear generated prompt if there are empty slots
                    prompt_text = "" if has_empty_slots else generated_prompt.value
                    
                    logger.info(f"Refresh completed: cleared {cleared_count} empty slots" + (" and prompt" if has_empty_slots else ""))
                    return results + [prompt_text]
                
                # Clear all analyses function
                def clear_all_analyses():
                    """Clear all analyses and cache"""
                    self._analysis_cache.clear()
                    self._image_hashes.clear()
                    self._analysis_states = [{}, {}, {}]
                    self._analysis_texts = ["", "", ""]
                    
                    results = []
                    for i in range(3):
                        results.extend(["", {}])
                    
                    # Also clear generated prompt
                    results.append("")
                    
                    logger.info("Cleared all analyses and cache")
                    return results
                
                # Extra inputs for image data
                extra_inputs = []
                
                # Set up event handlers
                analyze_all_btn.click(
                    fn=analyze_all_images,
                    inputs=[
                        force_cpu, use_florence, use_joycaption, florence_model, show_detailed, promptgen_instruction,
                        auto_unload, unload_delay, analysis_mode
                    ] + extra_inputs,
                    outputs=[item for pair in zip(analysis_displays, analysis_data) for item in pair]
                ).then(
                    fn=lambda: self._get_performance_info(),
                    inputs=[],
                    outputs=performance_info
                )
                
                refresh_btn.click(
                    fn=refresh_missing_analyses,
                    inputs=extra_inputs,
                    outputs=[item for pair in zip(analysis_displays, analysis_data) for item in pair]
                ).then(
                    fn=lambda: self._get_performance_info(),
                    inputs=[],
                    outputs=performance_info,
                    show_progress=False
                )
                
                clear_btn.click(
                    fn=clear_all_analyses,
                    outputs=[item for pair in zip(analysis_displays, analysis_data) for item in pair]
                ).then(
                    fn=lambda: self._get_performance_info(),
                    inputs=[],
                    outputs=performance_info,
                    show_progress=False
                )
                
                
                # Update status when force_cpu changes
                force_cpu.change(
                    fn=lambda: self._get_model_status(),
                    outputs=model_status
                )
                
                # Event handlers for prompt builder
                if PromptBuilder and StyleLibrary:
                    # Update input visibility based on scenario
                    def update_prompt_inputs(scenario):
                        """Show/hide appropriate input fields based on scenario"""
                        # Update visibility
                        object_vis = scenario == "object_manipulation"  # Combined add/remove
                        style_vis = scenario == PromptType.STYLE_TRANSFER.value
                        replace_vis = scenario == PromptType.OBJECT_REPLACE.value
                        pose_vis = scenario == PromptType.POSE_CHANGE.value
                        emotion_vis = scenario == PromptType.EMOTION_CHANGE.value
                        # Handle both old DETAIL_ENHANCEMENT and new IMAGE_ENHANCEMENT
                        enhancement_vis = scenario == PromptType.IMAGE_ENHANCEMENT.value or scenario == PromptType.DETAIL_ENHANCEMENT.value
                        lighting_vis = scenario == PromptType.LIGHTING_CHANGE.value
                        outpainting_vis = scenario == PromptType.OUTPAINTING.value
                        user_prompts_vis = scenario == "user_prompts"
                        dual_image_vis = scenario == "dual_image"
                        
                        # Get template info for tips
                        tips_text = "💡 **Tips:**\n"
                        if scenario == "object_manipulation":
                            tips_text += "• Use Add mode to insert new objects into the scene\n"
                            tips_text += "• Use Remove mode to cleanly remove unwanted elements\n"
                            tips_text += "• Be specific about object descriptions and positions"
                        elif scenario == "user_prompts":
                            tips_text += "• Select from your saved custom prompts\n"
                            tips_text += "• Click 'Use This Prompt' to apply it\n"
                            tips_text += "• Delete prompts you no longer need"
                        elif scenario == "dual_image":
                            tips_text += "• Make sure you have at least 2 context images loaded\n"
                            tips_text += "• Describe elements visually - FLUX.1 Kontext will identify them\n"
                            tips_text += "• For characters: describe clothing, appearance, or unique features\n"
                            tips_text += "• Choose integration method based on your desired outcome"
                        else:
                            try:
                                template_info = prompt_builder.get_template_info(PromptType(scenario))
                                if template_info and template_info.get('tips'):
                                    tips_text += "\n".join([f"• {tip}" for tip in template_info['tips']])
                            except ValueError:
                                # Handle invalid enum values
                                pass
                        
                        # If switching to user prompts, update style selector
                        user_prompt_choices = []
                        if user_prompts_vis:
                            try:
                                custom_styles = style_library.get_styles_by_category(StyleCategory.CUSTOM)
                                user_prompt_choices = [(style.name, style.id) for style in custom_styles]
                            except:
                                pass
                        
                        return (
                            gradio.update(visible=object_vis),  # object_inputs (combined add/remove)
                            gradio.update(visible=style_vis),   # style_inputs
                            gradio.update(visible=replace_vis),  # replace_inputs
                            gradio.update(visible=pose_vis),     # pose_inputs
                            gradio.update(visible=emotion_vis),  # emotion_inputs
                            gradio.update(visible=False),        # detail_inputs (deprecated)
                            gradio.update(visible=lighting_vis), # lighting_inputs
                            gradio.update(visible=enhancement_vis),  # restoration_inputs
                            gradio.update(visible=outpainting_vis), # outpainting_inputs
                            gradio.update(visible=user_prompts_vis),  # user_prompts_inputs
                            gradio.update(visible=dual_image_vis),  # dual_image_inputs
                            tips_text,
                            gradio.update(choices=user_prompt_choices)  # user_prompt_selector
                        )
                    
                    # Update style presets based on category
                    def update_style_presets(category):
                        """Update style preset dropdown based on category and manage reference prompt field"""
                        try:
                            styles = style_library.get_styles_by_category(StyleCategory(category))
                            choices = [(style.name, style.id) for style in styles]
                            logger.info(f"Updated style presets for {category}: {len(choices)} styles")
                            
                            # Check if it's FROM_REFERENCE category
                            is_reference = (category == StyleCategory.FROM_REFERENCE.value)
                            
                            # If FROM_REFERENCE, show prompt field and clear preserve checkboxes
                            # Otherwise hide prompt field and use default preserve values
                            return (
                                gradio.update(choices=choices, value=choices[0][1] if choices else None),  # style_preset
                                gradio.update(visible=is_reference),  # style_reference_prompt
                                gradio.update(value=[] if is_reference else ["subject identity", "composition"])  # preserve_elements
                            )
                        except Exception as e:
                            logger.error(f"Error updating style presets: {e}")
                            return (
                                gradio.update(choices=[], value=None),  # style_preset
                                gradio.update(visible=False),  # style_reference_prompt
                                gradio.update(value=["subject identity", "composition"])  # preserve_elements
                            )
                    
                    # Update style modifier presets based on category
                    def update_style_modifier_presets(category):
                        """Update style modifier dropdown based on selected category"""
                        try:
                            if category == "custom":
                                # Show custom text field, hide preset dropdown
                                return (
                                    gradio.update(visible=False),  # style_modifier_preset
                                    gradio.update(visible=True)    # style_modifier_field
                                )
                            else:
                                # Update preset choices
                                modifiers = []
                                if category in STYLE_MODIFIERS_CONFIG.get("categories", {}):
                                    for mod in STYLE_MODIFIERS_CONFIG["categories"][category]["modifiers"]:
                                        modifiers.append((mod["name"], mod["value"]))
                                
                                return (
                                    gradio.update(
                                        choices=modifiers,
                                        value=modifiers[0][1] if modifiers else "",
                                        visible=True
                                    ),  # style_modifier_preset
                                    gradio.update(visible=False)  # style_modifier_field
                                )
                        except Exception as e:
                            logger.error(f"Error updating style modifier presets: {e}")
                            return (
                                gradio.update(choices=[], value="", visible=True),
                                gradio.update(visible=False)
                            )
                    
                    # Show style details
                    def show_style_details(style_id):
                        """Display detailed information about selected style"""
                        if not style_id:
                            return gradio.update(value="", visible=False)
                            
                        style = style_library.get_style(style_id)
                        if not style:
                            return gradio.update(value="", visible=False)
                        
                        details = f"### {style.name}\n\n"
                        details += f"**Description:** {style.description}\n\n"
                        details += f"**Visual Elements:** {', '.join(style.visual_elements)}\n\n"
                        details += f"**Color Characteristics:** {', '.join(style.color_characteristics)}\n\n"
                        details += f"**Technique:** {', '.join(style.technique_details)}\n\n"
                        details += f"**Example:** {style.example_prompt}"
                        
                        return gradio.update(value=details, visible=True)
                    
                    # Build prompt function
                    def build_prompt_from_inputs(scenario, *inputs):
                        """Build prompt based on scenario and inputs"""
                        try:
                            # Index mapping for all_prompt_inputs:
                            # 0: object_field, 1: object_mode, 2: position_field, 3: style_modifier_field,
                            # 4: style_modifier_category, 5: style_modifier_preset,
                            # 6: removal_scenario, 7: fill_method,
                            # 8: style_category, 9: style_preset, 10: style_details, 11: preserve_elements, 12: style_reference_prompt,
                            # 13: original_object, 14: new_object, 15: maintain_aspects,
                            # 16: subject_field, 17: new_pose, 18: pose_details,
                            # 19: emotion_subject, 20: emotion_type, 21: emotion_intensity,
                            # 22: detail_area, 23: enhancement_type, 24: specific_changes,
                            # 25: lighting_scenario, 26: lighting_preset, 27: lighting_effects, 28: lighting_adjustments,
                            # 29: restoration_type, 30: restoration_method, 31: quality_goal,
                            # 32: outpaint_direction, 33: extension_description, 34: consistency_elements,
                            # 35-40: dual_image fields, 41-43: character merge fields
                            
                            # Handle combined object manipulation scenario
                            if scenario == "object_manipulation":
                                # Check the object mode (add or remove)
                                mode = inputs[1]  # object_mode value
                                if mode == "add":
                                    # Determine which style modifier to use
                                    style_modifier_category = inputs[4]
                                    if style_modifier_category == "custom":
                                        style_modifier = inputs[3]  # Use custom text field
                                    else:
                                        style_modifier = inputs[5]  # Use preset value
                                    
                                    parameters = {
                                        "object": inputs[0],      # object_field
                                        "position": inputs[2],    # position_field
                                        "style_modifier": style_modifier
                                        }
                                    return prompt_builder.build_prompt(PromptType.OBJECT_ADD, parameters)
                                else:  # remove mode
                                    parameters = {
                                        "object_description": inputs[0],  # object_field
                                        "fill_method": inputs[7]  # fill_method (updated index)
                                        }
                                    return prompt_builder.build_prompt(PromptType.OBJECT_REMOVE, parameters)
                            
                            # Skip prompt_type assignment for non-standard scenarios
                            if scenario not in ["dual_image", "user_prompts"]:
                                prompt_type = PromptType(scenario)
                                parameters = {}
                            
                            if scenario not in ["dual_image", "user_prompts"]:
                                if prompt_type == PromptType.OBJECT_ADD:
                                    parameters = {
                                        "object": inputs[0],
                                        "position": inputs[1],
                                        "style_modifier": inputs[2]
                                        }
                                elif prompt_type == PromptType.STYLE_TRANSFER:
                                    # Indices: style_category=8, style_preset=9, style_details=10, preserve_elements=11, style_reference_prompt=12
                                    style_ids = [inputs[9]] if inputs[9] else []
                                    preserve = inputs[11] if inputs[11] else []
                                    reference_prompt = inputs[12] if inputs[12] else None
                                    return style_library.build_style_prompt(style_ids, preserve_elements=preserve, reference_prompt=reference_prompt)
                                elif prompt_type == PromptType.OBJECT_REMOVE:
                                    # This should never be reached directly, handled in object_manipulation
                                    # But if it is: object_field=0, fill_method=7
                                    parameters = {
                                        "object_description": inputs[0],
                                        "fill_method": inputs[7]
                                        }
                                elif prompt_type == PromptType.OBJECT_REPLACE:
                                    parameters = {
                                        "original_object": inputs[13],
                                        "new_object": inputs[14],
                                        "maintain_aspects": inputs[15]
                                        }
                                elif prompt_type == PromptType.POSE_CHANGE:
                                    parameters = {
                                        "subject": inputs[16],
                                        "new_pose": inputs[17],
                                        "additional_details": inputs[18]
                                        }
                                elif prompt_type == PromptType.EMOTION_CHANGE:
                                    parameters = {
                                        "subject": inputs[19],
                                        "emotion": inputs[20],
                                        "intensity": inputs[21]
                                        }
                                elif prompt_type == PromptType.DETAIL_ENHANCEMENT:
                                    parameters = {
                                        "area": inputs[22],
                                        "enhancement_type": inputs[23],
                                        "specific_changes": inputs[24]
                                        }
                                elif prompt_type == PromptType.LIGHTING_CHANGE:
                                    # Inputs[25] = lighting_scenario, [26] = preset, [27] = effects, [28] = adjustments
                                    effects_list = inputs[27] if inputs[27] else []
                                    
                                    # Build parameters differently based on whether effects are provided
                                    if effects_list:
                                        effects_str = ", ".join(effects_list)
                                        parameters = {
                                            "lighting_type": inputs[26] if inputs[26] else inputs[25],
                                            "specific_effects": effects_str,
                                            "adjustments": inputs[28] if inputs[28] else "while maintaining subject details"
                                            }
                                    else:
                                        # If no effects, combine lighting type and adjustments directly
                                        parameters = {
                                            "lighting_type": inputs[26] if inputs[26] else inputs[25],
                                            "specific_effects": inputs[28] if inputs[28] else "while maintaining subject details",
                                            "adjustments": ""  # Empty to avoid double spacing
                                            }
                                elif prompt_type == PromptType.IMAGE_ENHANCEMENT:
                                    # Inputs[29] = enhancement_type, [30] = method, [31] = quality_goal
                                    enhancement_map = {
                                            EnhancementScenario.FACIAL_DETAILS.value: "facial features",
                                        EnhancementScenario.EYE_ENHANCEMENT.value: "the eyes",
                                        EnhancementScenario.TEXTURE_DETAILS.value: "texture details",
                                        EnhancementScenario.HAIR_DETAILS.value: "hair details",
                                        EnhancementScenario.OVERALL_SHARPNESS.value: "overall sharpness",
                                        EnhancementScenario.OLD_PHOTO.value: "old photograph",
                                        EnhancementScenario.DAMAGED_ART.value: "damaged artwork",
                                        EnhancementScenario.LOW_RESOLUTION.value: "low resolution image",
                                        EnhancementScenario.BLURRY.value: "blurry photo",
                                        EnhancementScenario.EXPOSURE_ISSUES.value: "exposure issues",
                                        EnhancementScenario.NOISY.value: "noisy image",
                                        EnhancementScenario.COMPRESSED.value: "compressed image",
                                        EnhancementScenario.FADED_COLORS.value: "faded colors",
                                        EnhancementScenario.GENERAL_ENHANCEMENT.value: "overall image quality"
                                    }
                                    enhancement_text = enhancement_map.get(inputs[29], inputs[29])
                                    parameters = {
                                        "enhancement_target": enhancement_text,
                                        "method": inputs[30],
                                        "quality_goal": inputs[31]
                                        }
                                elif prompt_type == PromptType.OUTPAINTING:
                                    # Inputs[32] = direction, [33] = extension_description, [34] = consistency
                                    direction_map = {
                                        OutpaintingDirection.HORIZONTAL.value: "the landscape horizontally",
                                        OutpaintingDirection.VERTICAL.value: "the scene vertically",
                                        OutpaintingDirection.LEFT.value: "leftward",
                                        OutpaintingDirection.RIGHT.value: "rightward",
                                        OutpaintingDirection.TOP.value: "upward",
                                        OutpaintingDirection.BOTTOM.value: "downward",
                                        OutpaintingDirection.ALL_SIDES.value: "all sides equally",
                                        OutpaintingDirection.WIDESCREEN.value: "to widescreen format",
                                        OutpaintingDirection.SQUARE.value: "to square format"
                                        }
                                    direction_text = direction_map.get(inputs[32], inputs[32])
                                    parameters = {
                                        "direction": direction_text,
                                        "extension_description": inputs[33],
                                        "consistency_elements": inputs[34]
                                        }
                            
                            # Handle user prompts scenario
                            if scenario == "user_prompts":
                                # This should be handled by the user prompt selector
                                return "❌ Please select a saved prompt from the dropdown"
                            
                            # Handle dual-image scenario
                            if scenario == "dual_image":
                                # Inputs[35]=scenario, [36]=param1, [37]=param2, [38]=param3, [39]=method, [40]=preservation
                                # Additional inputs for character merge: [41]=arrangement_preset, [42]=interaction_type
                                dual_scenario = inputs[35]
                                
                                if dual_scenario not in dual_image_config.get("scenarios", {}):
                                    return "❌ Invalid dual-image scenario selected"
                                
                                scenario_data = dual_image_config["scenarios"][dual_scenario]
                                template = scenario_data.get("template", "")
                                fields = scenario_data.get("fields", {})
                                field_keys = list(fields.keys())
                                
                                # Get integration method modifiers
                                integration = inputs[39]
                                integration_modifiers = ""
                                if integration in dual_image_config.get("integration_methods", {}):
                                    method_data = dual_image_config["integration_methods"][integration]
                                    modifiers = method_data.get("modifiers", [])
                                    if modifiers:
                                        integration_modifiers = modifiers[0]  # Use first modifier as default
                                
                                # Build the prompt from template
                                prompt = template
                                
                                # Replace placeholders based on field keys
                                for i, field_key in enumerate(field_keys):
                                    if i < 3:  # We have up to 3 input fields
                                        value = inputs[36 + i] if inputs[36 + i] else scenario_data.get("defaults", {}).get(field_key, "")
                                        # Escape any braces in user input to avoid template conflicts
                                        value = value.replace("{", "{{").replace("}", "}}")
                                        prompt = prompt.replace(f"{{{field_key}}}", value)
                                
                                # Special handling for character_merge to avoid empty "and"
                                if dual_scenario == "character_merge":
                                    # Check if both characters are provided
                                    char1 = inputs[36] if inputs[36] else ""
                                    char2 = inputs[37] if inputs[37] else ""
                                    if not char2.strip():
                                        # If no second character, adjust the template
                                        prompt = prompt.replace(" and  together", " alone")
                                        prompt = prompt.replace("both ", "")
                                
                                # Handle character merge scenario with interaction modifiers
                                if dual_scenario == "character_merge" and len(inputs) > 42:
                                    # Get interaction style from unified config
                                    interaction_key = inputs[42] if inputs[42] else "none"
                                    interaction_text = ""
                                    
                                    # First check if the arrangement already includes interaction
                                    arrangement_value = inputs[41] if len(inputs) > 41 else ""
                                    has_builtin_interaction = False
                                    
                                    if arrangement_value and ":" in arrangement_value:
                                        arr_type, option_name = arrangement_value.split(":", 1)
                                        if arr_type in unified_config.get("arrangements", {}):
                                            options = unified_config["arrangements"][arr_type].get("options", [])
                                            for opt in options:
                                                if opt["name"] == option_name and opt.get("interaction"):
                                                    has_builtin_interaction = True
                                                    interaction_text = opt["interaction"]
                                                    break
                                    
                                    # If no built-in interaction and user selected one, use it
                                    if not has_builtin_interaction and interaction_key != "none":
                                        if interaction_key in unified_config.get("interaction_styles", {}):
                                            interaction_text = unified_config["interaction_styles"][interaction_key]["value"]
                                    
                                    # Replace interaction placeholder
                                    if interaction_text:
                                        prompt = prompt.replace("{interaction}", interaction_text)
                                    else:
                                        prompt = prompt.replace(", {interaction}", "")  # Remove if empty
                                
                                # Replace integration-related placeholders
                                for placeholder in ["{preservation}", "{integration}", "{balance}", "{method}", "{style}"]:
                                    if placeholder in prompt:
                                        prompt = prompt.replace(placeholder, integration_modifiers)
                                
                                # Add preservation options if selected
                                preservation = inputs[39]
                                if preservation:
                                    preservation_text = ", preserving " + ", ".join(preservation).lower()
                                    prompt += preservation_text
                                
                                # Quick token check for dual-image prompts
                                if len(prompt) > 400:  # Rough estimate: 400 chars ≈ 450+ tokens
                                    logger.warning(f"Long dual-image prompt: {len(prompt)} chars")
                                
                                return prompt
                            
                            # Build the prompt for standard scenarios
                            if scenario != "dual_image" and scenario != "user_prompts":
                                if prompt_type in [PromptType.OBJECT_ADD, PromptType.OBJECT_REMOVE, PromptType.OBJECT_REPLACE,
                                                 PromptType.POSE_CHANGE, PromptType.EMOTION_CHANGE, PromptType.DETAIL_ENHANCEMENT,
                                                 PromptType.LIGHTING_CHANGE, PromptType.IMAGE_ENHANCEMENT, PromptType.OUTPAINTING]:
                                    prompt = prompt_builder.build_prompt(prompt_type, parameters)
                                
                                # Validate
                                is_valid, error = prompt_builder.validate_prompt(prompt)
                                if not is_valid:
                                    return f"❌ {error}"
                                    
                                return prompt
                            
                            # For dual_image, we already returned the prompt above
                            # For user_prompts, it should be handled separately
                            return "❌ Invalid scenario type"
                            
                        except Exception as e:
                            logger.error(f"Failed to build prompt: {e}")
                            return f"❌ Failed to build prompt: {str(e)}"
                    
                    
                    # Update examples based on scenario
                    def update_examples(scenario):
                        """Show relevant examples for selected scenario"""
                        if scenario == "object_manipulation":
                            examples_text = "### Examples:\n\n"
                            examples_text += "**Add Object Example:**\n"
                            examples_text += "- Object: red balloon\n"
                            examples_text += "- Position: floating in the top right corner\n"
                            examples_text += "- Style: semi-transparent and glowing\n"
                            examples_text += "- **Result:** add a red balloon floating in the top right corner, semi-transparent and glowing\n\n"
                            examples_text += "**Remove Object Example:**\n"
                            examples_text += "- Object: the jacket\n"
                            examples_text += "- Fill Method: show their natural clothing underneath\n"
                            examples_text += "- **Result:** remove the jacket and show their natural clothing underneath\n"
                            return examples_text
                        elif scenario == "dual_image":
                            examples_text = "### Dual-Image Mode Examples:\n\n"
                            
                            # Get current dual scenario if available
                            if dual_image_config and "scenarios" in dual_image_config:
                                for scenario_key, scenario_data in dual_image_config["scenarios"].items():
                                    if "examples" in scenario_data:
                                        examples_text += f"**{scenario_data['name']}:**\n"
                                        for example in scenario_data["examples"][:2]:  # Show first 2 examples
                                            examples_text += f"• {example}\n"
                                        examples_text += "\n"
                            
                            return examples_text
                        
                        try:
                            template_info = prompt_builder.get_template_info(PromptType(scenario))
                            if not template_info or not template_info.get('examples'):
                                return ""
                            
                            examples_text = "### Examples:\n\n"
                            for i, example in enumerate(template_info['examples'], 1):
                                example_prompt = prompt_builder.build_prompt(PromptType(scenario), example)
                                examples_text += f"**Example {i}:**\n"
                                for key, value in example.items():
                                    examples_text += f"- {key.replace('_', ' ').title()}: {value}\n"
                                examples_text += f"- **Result:** {example_prompt}\n\n"
                            
                            return examples_text
                        except ValueError:
                            return "No examples available for this scenario."
                    
                    # Connect event handlers
                    prompt_scenario.change(
                        fn=update_prompt_inputs,
                        inputs=prompt_scenario,
                        outputs=[
                            object_inputs, style_inputs, replace_inputs,
                            pose_inputs, emotion_inputs, detail_inputs, lighting_inputs,
                            restoration_inputs, outpainting_inputs, user_prompts_inputs, 
                            dual_image_inputs, prompt_tips,
                            user_prompt_selector
                        ]
                    ).then(
                        fn=update_examples,
                        inputs=prompt_scenario,
                        outputs=examples_display
                    )
                    
                    style_category.change(
                        fn=update_style_presets,
                        inputs=style_category,
                        outputs=[style_preset, style_reference_prompt, preserve_elements]
                    )
                    
                    style_preset.change(
                        fn=show_style_details,
                        inputs=style_preset,
                        outputs=style_details
                    )
                    
                    # Object mode change handler
                    def update_object_fields(mode):
                        """Toggle between add and remove fields"""
                        is_add = mode == "add"
                        return (
                            gradio.update(visible=is_add),    # add_fields
                            gradio.update(visible=not is_add) # remove_fields
                        )
                    
                    object_mode.change(
                        fn=update_object_fields,
                        inputs=object_mode,
                        outputs=[add_fields, remove_fields]
                    )
                    
                    # Style modifier category change handler
                    style_modifier_category.change(
                        fn=update_style_modifier_presets,
                        inputs=style_modifier_category,
                        outputs=[style_modifier_preset, style_modifier_field]
                    )
                    
                    # Dual-image scenario change handler
                    def update_dual_image_fields(scenario):
                        """Update dual-image fields based on selected scenario"""
                        if scenario in dual_image_config.get("scenarios", {}):
                            scenario_data = dual_image_config["scenarios"][scenario]
                            fields = scenario_data.get("fields", {})
                            
                            # Get field configurations
                            field1_config = fields.get(list(fields.keys())[0], {}) if fields else {}
                            field2_config = fields.get(list(fields.keys())[1], {}) if len(fields) > 1 else {}
                            field3_config = fields.get(list(fields.keys())[2], {}) if len(fields) > 2 else {}
                            
                            # Check if scenario needs third field
                            needs_third_field = len(fields) > 2
                            
                            # Show character merge options for character_merge scenario
                            show_character_options = scenario == "character_merge"
                            
                            return (
                                gradio.update(
                                    label=field1_config.get("label", "Primary Element"),
                                    placeholder=field1_config.get("placeholder", "")
                                ),
                                gradio.update(
                                    label=field2_config.get("label", "Secondary Element"),
                                    placeholder=field2_config.get("placeholder", "")
                                ),
                                gradio.update(
                                    label=field3_config.get("label", "Additional Details"),
                                    placeholder=field3_config.get("placeholder", ""),
                                    visible=needs_third_field
                                ),
                                gradio.update(visible=show_character_options)  # character_merge_options
                            )
                        
                        return (
                            gradio.update(label="Primary Element"),
                            gradio.update(label="Secondary Element"),
                            gradio.update(visible=False),
                            gradio.update(visible=False)  # character_merge_options
                        )
                    
                    dual_image_scenario.change(
                        fn=update_dual_image_fields,
                        inputs=dual_image_scenario,
                        outputs=[dual_param1, dual_param2, dual_param3, character_merge_options]
                    )
                    
                    # Unified arrangement handler
                    def update_from_unified_arrangement(arrangement_value):
                        """Update arrangement field and show/hide custom input"""
                        if not arrangement_value or arrangement_value.startswith("__category_"):
                            # Category separator - don't update
                            return gradio.update(), gradio.update()
                        
                        if arrangement_value == "custom:custom":
                            # Show custom field, clear arrangement
                            return "", gradio.update(visible=True)
                        
                        # Parse the arrangement value
                        if ":" in arrangement_value:
                            arr_type, option_name = arrangement_value.split(":", 1)
                            # Find the actual value from config
                            if arr_type in unified_config.get("arrangements", {}):
                                options = unified_config["arrangements"][arr_type].get("options", [])
                                for opt in options:
                                    if opt["name"] == option_name:
                                        return opt["value"], gradio.update(visible=False)
                        
                        return arrangement_value, gradio.update(visible=False)
                    
                    unified_arrangement.change(
                        fn=update_from_unified_arrangement,
                        inputs=unified_arrangement,
                        outputs=[dual_param3, custom_arrangement_field]
                    )
                    
                    # Update dual_param3 when custom field changes
                    custom_arrangement_field.change(
                        fn=lambda x: x,
                        inputs=custom_arrangement_field,
                        outputs=dual_param3
                    )
                    
                    # Update removal suggestions based on scenario
                    def update_removal_suggestions(scenario):
                        """Update quick select options for removal scenario"""
                        # Check if this is a valid removal scenario
                        try:
                            removal_scenario_enum = RemovalScenario(scenario)
                        except ValueError:
                            # Not a removal scenario, hide the quick select options
                            return (
                                gradio.update(visible=False, choices=[]),
                                gradio.update(visible=False, choices=[])
                            )
                        
                        if removal_scenario_enum == RemovalScenario.CUSTOM:
                            return (
                                gradio.update(visible=False, choices=[]),
                                gradio.update(visible=False, choices=[])
                            )
                        
                        suggestions = prompt_builder.get_removal_suggestions(removal_scenario_enum)
                        items = suggestions.get("items", [])
                        methods = suggestions.get("fill_methods", [])
                        
                        return (
                            gradio.update(visible=True, choices=items, value=None),
                            gradio.update(visible=True, choices=methods, value=None)
                        )
                    
                    # Update text fields from quick select
                    def update_object_description(selected_item):
                        """Update object description from quick select"""
                        if selected_item and not selected_item.startswith("---"):
                            return f"the {selected_item}"
                        return ""
                    
                    def update_fill_method_text(selected_method):
                        """Update fill method from quick select"""
                        if selected_method:
                            return selected_method
                        return "blend seamlessly with the background"
                    
                    removal_scenario.change(
                        fn=update_removal_suggestions,
                        inputs=removal_scenario,
                        outputs=[removal_item_dropdown, fill_method_dropdown]
                    )
                    
                    removal_item_dropdown.change(
                        fn=update_object_description,
                        inputs=removal_item_dropdown,
                        outputs=object_field
                    )
                    
                    fill_method_dropdown.change(
                        fn=update_fill_method_text,
                        inputs=fill_method_dropdown,
                        outputs=fill_method
                    )
                    
                    # Update lighting dropdowns based on scenario
                    def update_lighting_dropdowns(scenario):
                        """Update lighting preset, effects and adjustments based on scenario"""
                        try:
                            lighting_scenario_enum = LightingScenario(scenario)
                        except ValueError:
                            # Use CUSTOM as fallback
                            lighting_scenario_enum = LightingScenario.CUSTOM
                        
                        suggestions = prompt_builder.get_lighting_suggestions(lighting_scenario_enum)
                        
                        return (
                            gradio.update(choices=suggestions.get("presets", []), value=suggestions["presets"][0] if suggestions.get("presets") else None),
                            gradio.update(choices=suggestions.get("effects", []), value=[]),
                            gradio.update(choices=suggestions.get("adjustments", []), value=suggestions["adjustments"][0] if suggestions.get("adjustments") else None)
                        )
                    
                    lighting_scenario.change(
                        fn=update_lighting_dropdowns,
                        inputs=lighting_scenario,
                        outputs=[lighting_preset, lighting_effects, lighting_adjustments]
                    )
                    
                    # Update enhancement/restoration dropdowns based on scenario
                    def update_restoration_dropdowns(scenario):
                        """Update enhancement method and quality goal based on scenario"""
                        try:
                            enhancement_scenario_enum = EnhancementScenario(scenario)
                        except ValueError:
                            enhancement_scenario_enum = EnhancementScenario.CUSTOM
                        
                        suggestions = prompt_builder.get_enhancement_suggestions(enhancement_scenario_enum)
                        
                        return (
                            gradio.update(choices=suggestions.get("methods", []), value=suggestions["methods"][0] if suggestions.get("methods") else None),
                            gradio.update(choices=suggestions.get("quality_goals", []), value=suggestions["quality_goals"][0] if suggestions.get("quality_goals") else None)
                        )
                    
                    restoration_type.change(
                        fn=update_restoration_dropdowns,
                        inputs=restoration_type,
                        outputs=[restoration_method, quality_goal]
                    )
                    
                    # Update outpainting dropdowns based on direction
                    def update_outpainting_dropdowns(direction):
                        """Update extension description and consistency based on direction"""
                        try:
                            outpainting_direction_enum = OutpaintingDirection(direction)
                        except ValueError:
                            outpainting_direction_enum = OutpaintingDirection.CUSTOM
                        
                        suggestions = prompt_builder.get_outpainting_suggestions(outpainting_direction_enum)
                        
                        return (
                            gradio.update(choices=suggestions.get("extensions", []), value=suggestions["extensions"][0] if suggestions.get("extensions") else None),
                            gradio.update(choices=suggestions.get("consistency", []), value=suggestions["consistency"][0] if suggestions.get("consistency") else None)
                        )
                    
                    outpaint_direction.change(
                        fn=update_outpainting_dropdowns,
                        inputs=outpaint_direction,
                        outputs=[extension_description, consistency_elements]
                    )
                    
                    
                    # Collect all input fields
                    # IMPORTANT: The order here must match the index comments in build_prompt_from_inputs
                    all_prompt_inputs = [
                        object_field, object_mode, position_field, style_modifier_field,  # 0-3
                        style_modifier_category, style_modifier_preset,  # 4-5 (NEW)
                        removal_scenario, fill_method,  # 6-7 (was 4-5)
                        style_category, style_preset, style_details, preserve_elements, style_reference_prompt,  # 8-12
                        original_object, new_object, maintain_aspects,  # 13-15
                        subject_field, new_pose, pose_details,  # 16-18
                        emotion_subject, emotion_type, emotion_intensity,  # 19-21
                        detail_area, enhancement_type, specific_changes,  # 22-24
                        lighting_scenario, lighting_preset, lighting_effects, lighting_adjustments,  # 25-28
                        restoration_type, restoration_method, quality_goal,  # 29-31
                        outpaint_direction, extension_description, consistency_elements,  # 32-34
                        dual_image_scenario, dual_param1, dual_param2, dual_param3, integration_method, preservation_options,  # 35-40
                        unified_arrangement, interaction_style, custom_arrangement_field  # 41-43
                    ]
                    
                    # Wrapper function to handle both prompt and token display
                    def build_prompt_with_tokens(scenario, *inputs):
                        """Build prompt and return both prompt and token info"""
                        prompt = build_prompt_from_inputs(scenario, *inputs)
                        
                        # Get token count display
                        if prompt and not prompt.startswith("❌"):
                            token_info = get_token_display(prompt)
                        else:
                            token_info = ""
                        
                        return prompt, token_info
                    
                    build_prompt_btn.click(
                        fn=build_prompt_with_tokens,
                        inputs=[prompt_scenario] + all_prompt_inputs,
                        outputs=[generated_prompt_output, token_count_display]
                    )
                    
                    # Simple copy to clipboard function
                    # Update token count when prompt is edited
                    def update_token_count(prompt):
                        """Update token count display when prompt changes"""
                        if prompt:
                            return get_token_display(prompt)
                        return ""
                    
                    generated_prompt_output.change(
                        fn=update_token_count,
                        inputs=generated_prompt_output,
                        outputs=token_count_display
                    )
                    
                    copy_to_prompt_btn.click(
                        fn=None,
                        inputs=generated_prompt_output,
                        outputs=None,
                        _js="""
                        function(prompt_text) {
                            if (!prompt_text || prompt_text.startsWith('❌')) {
                                return;
                            }
                            
                            // Copy to clipboard
                            navigator.clipboard.writeText(prompt_text).then(function() {
                                // Show success message
                                const btns = document.querySelectorAll('button');
                                btns.forEach(btn => {
                                    if (btn.textContent.includes('Copy to Clipboard')) {
                                        const originalText = btn.textContent;
                                        btn.textContent = '✅ Copied!';
                                        btn.style.backgroundColor = '#4CAF50';
                                        setTimeout(function() {
                                            btn.textContent = originalText;
                                            btn.style.backgroundColor = '';
                                            }, 2000);
                                        }
                                    });
                            }).catch(function(err) {
                                // Fallback method
                                const textArea = document.createElement("textarea");
                                textArea.value = prompt_text;
                                textArea.style.position = "fixed";
                                textArea.style.left = "-999999px";
                                document.body.appendChild(textArea);
                                textArea.focus();
                                textArea.select();
                                try {
                                    document.execCommand('copy');
                                    // Show success
                                    const btns = document.querySelectorAll('button');
                                    btns.forEach(btn => {
                                        if (btn.textContent.includes('Copy to Clipboard')) {
                                            const originalText = btn.textContent;
                                            btn.textContent = '✅ Copied!';
                                            btn.style.backgroundColor = '#4CAF50';
                                            setTimeout(function() {
                                                btn.textContent = originalText;
                                                btn.style.backgroundColor = '';
                                                }, 2000);
                                            }
                                        });
                                    } catch (err) {
                                    console.error('Copy failed:', err);
                                    alert('Failed to copy to clipboard. Please copy manually.');
                                    } finally {
                                    document.body.removeChild(textArea);
                                    }
                            });
                        }
                        """
                    )
                    
                    # Generate button handler
                    generate_btn.click(
                        fn=None,
                        inputs=generated_prompt_output,
                        outputs=None,
                        _js="""
                        function(prompt_text) {
                            if (!prompt_text || prompt_text.startsWith('❌')) {
                                alert('Please build a prompt first');
                                return;
                            }
                            
                            // Get the Generate button in Prompt Builder
                            const kaGenerateBtn = document.querySelector('#ka_generate_btn');
                            
                            // Check if already generating
                            if (kaGenerateBtn && kaGenerateBtn.disabled) {
                                console.log('Generation already in progress');
                                return;
                            }
                            
                            try {
                                // Disable the button immediately
                                if (kaGenerateBtn) {
                                    kaGenerateBtn.disabled = true;
                                    kaGenerateBtn.style.opacity = '0.5';
                                    kaGenerateBtn.style.cursor = 'not-allowed';
                                    }
                                
                                // Check which tab is active
                                const txt2imgTab = gradioApp().querySelector('#tab_txt2img');
                                const img2imgTab = gradioApp().querySelector('#tab_img2img');
                                
                                let promptField = null;
                                let generateBtn = null;
                                let interruptBtn = null;
                                
                                // Determine active tab and get appropriate elements
                                if (txt2imgTab && txt2imgTab.style.display !== 'none') {
                                    // txt2img tab is active
                                    promptField = gradioApp().querySelector('#txt2img_prompt textarea');
                                    generateBtn = gradioApp().querySelector('#txt2img_generate');
                                    interruptBtn = gradioApp().querySelector('#txt2img_interrupt');
                                    } else if (img2imgTab && img2imgTab.style.display !== 'none') {
                                    // img2img tab is active
                                    promptField = gradioApp().querySelector('#img2img_prompt textarea');
                                    generateBtn = gradioApp().querySelector('#img2img_generate');
                                    interruptBtn = gradioApp().querySelector('#img2img_interrupt');
                                    }
                                
                                if (!promptField || !generateBtn) {
                                    console.warn('Could not find prompt field or generate button');
                                    alert('Please make sure you are in txt2img or img2img tab');
                                    // Re-enable button on error
                                    if (kaGenerateBtn) {
                                        kaGenerateBtn.disabled = false;
                                        kaGenerateBtn.style.opacity = '1';
                                        kaGenerateBtn.style.cursor = 'pointer';
                                        }
                                    return;
                                    }
                                
                                // Insert prompt text
                                promptField.value = prompt_text;
                                promptField.dispatchEvent(new Event('input', { bubbles: true }));
                                
                                // Update UI to reflect the change
                                const updateEvent = new Event('change', { bubbles: true });
                                promptField.dispatchEvent(updateEvent);
                                
                                // Small delay to ensure UI updates
                                setTimeout(() => {
                                    // Click generate button
                                    generateBtn.click();
                                    
                                    // Visual feedback
                                    if (kaGenerateBtn) {
                                        const originalText = kaGenerateBtn.textContent;
                                        kaGenerateBtn.textContent = '⏳ Generating...';
                                        kaGenerateBtn.style.backgroundColor = '#ff9800';
                                        
                                        // Monitor generation status
                                        const checkInterval = setInterval(() => {
                                            // More precise check - look for the progress bar and generation state
                                            const progressBar = gradioApp().querySelector('.progressDiv');
                                            const isProgressVisible = progressBar && progressBar.style.display !== 'none';
                                            const generateBtnHidden = generateBtn && generateBtn.style.display === 'none';
                                            const interruptBtnVisible = interruptBtn && interruptBtn.style.display !== 'none';
                                            
                                            // Check if truly still generating
                                            const isGenerating = isProgressVisible || generateBtnHidden || interruptBtnVisible;
                                            
                                            if (!isGenerating) {
                                                // Add small delay to ensure UI has fully updated
                                                setTimeout(() => {
                                                    // Double-check generation is really complete
                                                    const stillGenerating = (generateBtn && generateBtn.style.display === 'none') || 
                                                                          (interruptBtn && interruptBtn.style.display !== 'none');
                                                    
                                                    if (!stillGenerating) {
                                                        // Generation complete - re-enable button
                                                        clearInterval(checkInterval);
                                                        if (kaGenerateBtn) {
                                                            kaGenerateBtn.textContent = originalText;
                                                            kaGenerateBtn.style.backgroundColor = '';
                                                            kaGenerateBtn.disabled = false;
                                                            kaGenerateBtn.style.opacity = '1';
                                                            kaGenerateBtn.style.cursor = 'pointer';
                                                            }
                                                        }
                                                    }, 1000); // Wait 1 second after progress disappears
                                                }
                                            }, 500); // Check every 500ms
                                        
                                        // Fallback timeout after 5 minutes
                                        setTimeout(() => {
                                            clearInterval(checkInterval);
                                            if (kaGenerateBtn) {
                                                kaGenerateBtn.textContent = originalText;
                                                kaGenerateBtn.style.backgroundColor = '';
                                                kaGenerateBtn.disabled = false;
                                                kaGenerateBtn.style.opacity = '1';
                                                kaGenerateBtn.style.cursor = 'pointer';
                                                }
                                            }, 300000); // 5 minutes
                                        }
                                    }, 200);
                                
                            } catch (error) {
                                console.error('Generate failed:', error);
                                alert('Failed to start generation. Please try copying the prompt manually.');
                                // Re-enable button on error
                                if (kaGenerateBtn) {
                                    kaGenerateBtn.disabled = false;
                                    kaGenerateBtn.style.opacity = '1';
                                    kaGenerateBtn.style.cursor = 'pointer';
                                    }
                            }
                        }
                        """
                    )
                    
                    # Save custom style handler
                    def save_custom_style(prompt_text, style_name, style_description):
                        """Save generated prompt as custom style"""
                        if not prompt_text or prompt_text.startswith('❌'):
                            return gr.update(value="❌ No prompt to save", visible=True)
                        
                        if not style_name:
                            return gr.update(value="❌ Please enter a style name", visible=True)
                        
                        try:
                            # Save the custom style
                            style_id = style_library.save_custom_style(
                                name=style_name,
                                prompt=prompt_text,
                                description=style_description
                            )
                            
                            # Success message
                            status_msg = f"✅ Saved custom style: {style_name}\n\nTo use it, select 'Custom' from the Style Category dropdown"
                            return gr.update(value=status_msg, visible=True)
                        except Exception as e:
                            logger.error(f"Failed to save custom style: {e}")
                            return gr.update(
                                value=f"❌ Failed to save style: {str(e)}",
                                visible=True
                            )
                    
                    save_custom_style_btn.click(
                        fn=save_custom_style,
                        inputs=[generated_prompt_output, custom_style_name, custom_style_description],
                        outputs=custom_style_status
                    )
                    
                    
                    # User prompts handlers
                    def show_user_prompt(style_id):
                        """Display selected user prompt"""
                        if not style_id:
                            return gr.update(value="")
                        
                        prompt = style_library.get_custom_style_prompt(style_id)
                        return gr.update(value=prompt or "")
                    
                    def use_user_prompt(prompt_text):
                        """Copy user prompt to generated prompt output"""
                        if not prompt_text:
                            return gr.update(value="❌ No prompt selected")
                        return gr.update(value=prompt_text)
                    
                    def delete_user_prompt(style_id):
                        """Delete selected user prompt"""
                        if not style_id:
                            return gr.update(value="❌ No prompt selected", visible=True), gr.update()
                        
                        success = style_library.delete_custom_style(style_id)
                        if success:
                            # Update dropdown choices
                            custom_styles = style_library.get_styles_by_category(StyleCategory.CUSTOM)
                            choices = [(style.name, style.id) for style in custom_styles]
                            return (
                                gr.update(value="✅ Prompt deleted successfully", visible=True),
                                gr.update(choices=choices, value=None)
                            )
                        else:
                            return gr.update(value="❌ Failed to delete prompt", visible=True), gr.update()
                    
                    user_prompt_selector.change(
                        fn=show_user_prompt,
                        inputs=user_prompt_selector,
                        outputs=user_prompt_display
                    )
                    
                    use_user_prompt_btn.click(
                        fn=use_user_prompt,
                        inputs=user_prompt_display,
                        outputs=generated_prompt_output
                    )
                    
                    delete_user_prompt_btn.click(
                        fn=delete_user_prompt,
                        inputs=user_prompt_selector,
                        outputs=[user_prompt_status, user_prompt_selector]
                    )
                    
        
        
        # Add CSS to reorder accordions
        css = """
        <style>
        /* Reorder Kontext accordions using flexbox order */
        .kontext-prompt-builder { order: 1 !important; }
        .kontext-prompt-generation { order: 2 !important; }
        .kontext-image-analysis { order: 3 !important; }
        .kontext-advanced-settings { order: 4 !important; }
        .kontext-performance { order: 5 !important; }
        
        /* Ensure parent container uses flexbox */
        .kontext-prompt-builder:parent,
        .kontext-prompt-generation:parent,
        .kontext-image-analysis:parent,
        .kontext-advanced-settings:parent,
        .kontext-performance:parent {
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Square button styling */
        .square-button {
            width: 40px !important;
            height: 40px !important;
            min-width: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 18px !important;
        }
        </style>
        """
        gradio.HTML(css)
        
        # Return UI elements
        ui_elements = [enabled] + kontext_images + [sizing, reduce_checkbox, enable_metrics, dims_info]
        if ASSISTANT_AVAILABLE:
            ui_elements.extend([force_cpu, show_detailed, auto_unload, unload_delay])
        
        return ui_elements
    
    def _encode_image_to_latent(self, image_tensor: torch.Tensor, sd_model) -> torch.Tensor:
        """Encode image tensor to latent space with timing."""
        try:
            start_time = time.time()
            
            approximation_method = shared.opts.sd_vae_encode_method
            approximation_index = approximation_indexes.get(approximation_method)
            result = images_tensor_to_samples(image_tensor, approximation_index, sd_model)
            
            self.kontext_state.metrics.vae_encoding_time += time.time() - start_time
            return result
            
        except Exception as e:
            raise KontextError(f"Failed to encode image to latent: {str(e)}")
    
    def _process_single_image(self, image, sizing_mode: str, output_width: int,
                            output_height: int, reduce: bool, sd_model,
                            device: torch.device, dtype: torch.dtype,
                            is_fill_model: bool = False) -> Tuple[torch.Tensor, int, int]:
        """Process a single kontext image with validation."""
        
        # Validate
        self.image_processor.validate_image(image, self.config)
        
        # Convert sizing mode
        sizing_enum = SizingMode(sizing_mode)
        
        # Calculate target size
        target_width, target_height = self.image_processor.calculate_target_size(
            image, sizing_enum, output_width, output_height, self.config, is_fill_model
        )
        
        # Preprocess
        start_time = time.time()
        image_tensor = self.image_processor.preprocess_image(
            image, target_width, target_height, reduce
        )
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        self.kontext_state.metrics.preprocessing_time += time.time() - start_time
        
        # Encode to latent
        latent = self._encode_image_to_latent(image_tensor, sd_model)
        
        # Pad to patch size
        pad_h = latent.shape[2] % self.config.patch_size
        pad_w = latent.shape[3] % self.config.patch_size
        if pad_h > 0 or pad_w > 0:
            latent = torch.nn.functional.pad(
                latent, (0, pad_w, 0, pad_h), mode="circular"
            )
        
        # Convert to patches
        patch_latent = rearrange(
            latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.config.patch_size, pw=self.config.patch_size
        )
        
        latent_h, latent_w = latent.shape[2], latent.shape[3]
        patch_h = ((latent_h + (self.config.patch_size // 2)) // self.config.patch_size)
        patch_w = ((latent_w + (self.config.patch_size // 2)) // self.config.patch_size)
        
        return patch_latent, patch_h, patch_w
    
    def process_before_every_sampling(self, params, *script_args, **kwargs) -> None:
        """Main processing function."""
        # Parse arguments
        enabled = script_args[0]
        kontext_images = script_args[1:4]  # 3 images
        sizing_mode = script_args[4]
        reduce = script_args[5]
        enable_metrics = script_args[6] if len(script_args) > 6 else True
        dims_info = script_args[7] if len(script_args) > 7 else "0"
        
        # Update shared state with current images
        if ASSISTANT_AVAILABLE and shared_state:
            # Validate images before storing
            if validate_image_list:
                validated_images = validate_image_list(kontext_images)
                shared_state.set_images(validated_images)
            else:
                shared_state.set_images(kontext_images)
        
        # Update config
        self.config.enable_performance_metrics = enable_metrics
        
        # Check if any images provided
        has_images = any(img is not None for img in kontext_images)
        
        # Skip if disabled or no images
        if not enabled or not has_images:
            return
        
        # Only process on first iteration
        if params.iteration > 0:
            return
        
        # Check model compatibility
        if params.sd_model.is_webui_legacy_model():
            logger.warning("Legacy model detected, skipping kontext processing")
            return
        
        # Check if this is a Fill model
        is_fill_model = False
        model_name = getattr(params.sd_model, 'model_name', '').lower()
        if 'fill' in model_name or hasattr(params.sd_model, 'is_fill_model'):
            is_fill_model = True
            logger.info("Fill model detected, adjusting processing")
        
        try:
            total_start_time = time.time()
            
            # Clear previous state
            self.kontext_state.clear()
            
            # Get main latent info
            x = kwargs.get('x')
            if x is None:
                raise KontextError("No main latent tensor found")
            
            batch_size, channels, height, width = x.shape
            device, dtype = x.device, x.dtype
            
            # Store device info
            self.kontext_state.device = device
            self.kontext_state.dtype = dtype
            
            # Process images
            processed_latents = []
            processed_ids = []
            accum_h, accum_w = 0, 0
            valid_image_count = 0
            
            for idx, image in enumerate(kontext_images):
                if image is None:
                    continue
                
                if valid_image_count >= self.config.max_context_images:
                    logger.warning(
                        f"Maximum context images ({self.config.max_context_images}) exceeded. "
                        f"Ignoring additional images."
                    )
                    break
                
                logger.info(f"Processing context image {idx + 1}")
                
                # Process single image
                patch_latent, patch_h, patch_w = self._process_single_image(
                    image, sizing_mode, width, height, reduce,
                    params.sd_model, device, dtype, is_fill_model
                )
                
                # Calculate placement
                offset_h, offset_w = self.layout_manager.calculate_placement_offset(
                    accum_h, accum_w, patch_h, patch_w
                )
                
                # Create position IDs
                position_ids = self.layout_manager.create_position_ids(
                    patch_h, patch_w, offset_h, offset_w, device, dtype, batch_size
                )
                
                # Store results
                processed_latents.append(patch_latent)
                processed_ids.append(position_ids)
                
                # Update accumulator
                accum_h = max(accum_h, patch_h + offset_h)
                accum_w = max(accum_w, patch_w + offset_w)
                valid_image_count += 1
            
            # Combine all processed data
            if processed_latents:
                self.kontext_state.latent = torch.cat(processed_latents, dim=1)
                self.kontext_state.ids = torch.cat(processed_ids, dim=1)
                self.kontext_state.metrics.context_image_count = valid_image_count
                
                # Calculate memory usage
                if self.config.enable_performance_metrics:
                    memory_bytes = (
                        self.kontext_state.latent.element_size() * 
                        self.kontext_state.latent.nelement()
                    )
                    self.kontext_state.metrics.memory_usage_mb = memory_bytes / (1024 * 1024)
                
                # Set as active state
                ForgeKontextUnified.set_current_kontext_state(self.kontext_state)
                
                # Apply model patch
                ForgeKontextUnified._model_patcher.apply_patch(IntegratedFluxTransformer2DModel)
                
                # Log summary
                logger.info(
                    f"Successfully processed {valid_image_count} context images "
                    f"(layout: {accum_h}x{accum_w} patches)"
                )
                
                # Record total time
                self.kontext_state.metrics.total_processing_time = time.time() - total_start_time
                
                # Log performance metrics
                if self.config.enable_performance_metrics:
                    self.kontext_state.metrics.log_summary()
            
        except Exception as e:
            logger.error(f"Error during kontext processing: {str(e)}")
            self.kontext_state.clear()
            # Don't re-raise to avoid breaking generation
    
    def postprocess(self, params, processed, *args) -> None:
        """Clean up after processing."""
        enabled = args[0] if args else False
        
        if enabled:
            logger.info("Cleaning up kontext resources")
            
            # Remove model patch
            ForgeKontextUnified._model_patcher.remove_patch(IntegratedFluxTransformer2DModel)
            
            # Clear active state
            ForgeKontextUnified.set_current_kontext_state(None)
            
            # Clear local state
            self.kontext_state.clear()
            
            logger.info("Kontext cleanup complete")


# Backwards compatibility
forgeKontextUnified = ForgeKontextUnified()