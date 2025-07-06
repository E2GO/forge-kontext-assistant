import gradio
import torch
import numpy as np
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any
from pathlib import Path
from enum import Enum
from PIL import Image

from modules import scripts, shared
from modules.ui_components import InputAccordion
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import IntegratedFluxTransformer2DModel
from einops import rearrange, repeat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForgeKontext")

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
            current_state = ForgeKontext.get_current_kontext_state()
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


class ForgeKontext(scripts.Script):
    """Main script class for Flux Kontext functionality."""
    
    # Class-level components
    _model_patcher = SafeModelPatcher()
    _active_state: Optional[KontextState] = None
    
    def __init__(self):
        super().__init__()
        self.config = KontextConfig()
        self.kontext_state = KontextState()
        self.image_processor = ImageProcessor()
        self.layout_manager = LayoutManager()
        self.sorting_priority = 0
    
    @classmethod
    def get_current_kontext_state(cls) -> Optional[KontextState]:
        """Get the currently active kontext state."""
        return cls._active_state
    
    @classmethod
    def set_current_kontext_state(cls, state: Optional[KontextState]) -> None:
        """Set the currently active kontext state."""
        cls._active_state = state
    
    def title(self) -> str:
        return "Forge FluxKontext Pro"
    
    def show(self, is_img2img: bool) -> scripts.AlwaysVisible:
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img: bool):
        """Create user interface."""
        
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
                            show_label=False
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
                                "📐",  # Using icon for compact design
                                scale=1,
                                elem_id=f"kontext_set_dims_btn_{i}",
                                size="sm"
                            )
                            set_dims_buttons.append(set_btn)
            
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
                    
                    # Dynamic help text based on selected mode - directly under dropdown
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
            
            # Advanced settings
            with gradio.Accordion("Advanced Settings", open=False):
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
                
                # Update dimensions when reduce checkbox changes
                def update_all_dimensions(reduce_checked, *images):
                    """Update all dimension displays when reduce changes."""
                    return [update_image_dimensions(img, reduce_checked) for img in images]
                
                reduce_checkbox.change(
                    fn=update_all_dimensions,
                    inputs=[reduce_checkbox] + kontext_images,
                    outputs=image_dimensions
                )
        
        return (enabled, *kontext_images, sizing, reduce_checkbox, enable_metrics, dims_info)
    
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
                ForgeKontext.set_current_kontext_state(self.kontext_state)
                
                # Apply model patch
                ForgeKontext._model_patcher.apply_patch(IntegratedFluxTransformer2DModel)
                
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
            ForgeKontext._model_patcher.remove_patch(IntegratedFluxTransformer2DModel)
            
            # Clear active state
            ForgeKontext.set_current_kontext_state(None)
            
            # Clear local state
            self.kontext_state.clear()
            
            logger.info("Kontext cleanup complete")


# Backwards compatibility
forgeKontext = ForgeKontext()