"""
Integration module for communication between Kontext and Kontext Assistant.
Handles shared state and image discovery.
"""

import gradio as gr
from typing import List, Optional, Any, Dict, Tuple
import logging
from PIL import Image
import weakref

# Compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

logger = logging.getLogger("KontextAssistant.Integration")


class ForgeIntegration:
    """Manages integration between Kontext scripts in Forge WebUI."""
    
    # Class-level storage for cross-script communication
    _kontext_images: List[Optional[Image.Image]] = [None, None, None]
    _kontext_enabled: bool = False
    _image_observers: List[weakref.ref] = []
    _shared_state: Dict[str, Any] = {}
    
    @classmethod
    def register_kontext_images(cls, images: List[Optional[Image.Image]]) -> None:
        """Register images from the main Kontext script."""
        cls._kontext_images = images[:3]  # Max 3 images
        cls._notify_observers('images_updated', images)
        logger.info(f"Registered {sum(1 for img in images if img is not None)} kontext images")
    
    @classmethod
    def get_kontext_images(cls) -> List[Optional[Image.Image]]:
        """Get currently loaded kontext images."""
        return cls._kontext_images
    
    @classmethod
    def register_observer(cls, callback) -> None:
        """Register a callback to be notified of changes."""
        cls._image_observers.append(weakref.ref(callback))
        # Clean up dead references
        cls._image_observers = [ref for ref in cls._image_observers if ref() is not None]
    
    @classmethod
    def _notify_observers(cls, event: str, data: Any) -> None:
        """Notify all registered observers of an event."""
        for ref in cls._image_observers:
            callback = ref()
            if callback:
                try:
                    callback(event, data)
                except Exception as e:
                    logger.error(f"Observer notification failed: {e}")
    
    @classmethod
    def set_kontext_enabled(cls, enabled: bool) -> None:
        """Set whether Kontext is enabled."""
        cls._kontext_enabled = enabled
        cls._notify_observers('enabled_changed', enabled)
    
    @classmethod
    def is_kontext_enabled(cls) -> bool:
        """Check if Kontext is enabled."""
        return cls._kontext_enabled
    
    @classmethod
    def set_shared_value(cls, key: str, value: Any) -> None:
        """Set a shared value accessible across scripts."""
        cls._shared_state[key] = value
        cls._notify_observers('state_changed', {key: value})
    
    @classmethod
    def get_shared_value(cls, key: str, default: Any = None) -> Any:
        """Get a shared value."""
        return cls._shared_state.get(key, default)
    
    @classmethod
    def clear_shared_state(cls) -> None:
        """Clear all shared state."""
        cls._shared_state.clear()
        cls._kontext_images = [None, None, None]
        cls._kontext_enabled = False
        cls._notify_observers('state_cleared', None)


class KontextImageFinder:
    """Finds and monitors Kontext images in the Gradio interface."""
    
    def __init__(self):
        self.last_images = [None, None, None]
        ForgeIntegration.register_observer(self._on_integration_event)
    
    def _on_integration_event(self, event: str, data: Any) -> None:
        """Handle integration events."""
        if event == 'images_updated':
            logger.info("Kontext images were updated")
    
    def find_kontext_images(self) -> List[Optional[Image.Image]]:
        """
        Find Kontext images from various sources.
        
        Returns:
            List of up to 3 PIL Images or None
        """
        # First try integrated approach
        integrated_images = ForgeIntegration.get_kontext_images()
        if any(img is not None for img in integrated_images):
            return integrated_images
        
        # Try to find through Gradio state (if needed)
        try:
            images = self._find_through_gradio()
            if images:
                return images
        except Exception as e:
            logger.debug(f"Gradio search failed: {e}")
        
        # Return last known images
        return self.last_images
    
    def _find_through_gradio(self) -> Optional[List[Optional[Image.Image]]]:
        """Attempt to find images through Gradio components."""
        # This would need to be implemented based on specific Gradio setup
        # For now, return None
        return None
    
    def update_last_images(self, images: List[Optional[Image.Image]]) -> None:
        """Update the last known images."""
        self.last_images = images[:3]


def patch_kontext_script():
    """
    Patch the Kontext script to share images with the assistant.
    This should be called early in the kontext_assistant initialization.
    """
    try:
        # Try to import the kontext module
        import sys
        for module_name in ['kontext', 'scripts.kontext']:
            if module_name in sys.modules:
                kontext_module = sys.modules[module_name]
                
                # Find the ForgeKontext class
                if hasattr(kontext_module, 'ForgeKontext'):
                    ForgeKontext = kontext_module.ForgeKontext
                    
                    # Patch the UI method to capture image references
                    original_ui = ForgeKontext.ui
                    
                    def patched_ui(self, is_img2img):
                        result = original_ui(self, is_img2img)
                        
                        # Extract image components (assuming they're in positions 1-3)
                        if len(result) >= 4:
                            # Register any gradio Image components found
                            image_components = []
                            for item in result[1:4]:
                                if hasattr(item, 'value') and hasattr(item, 'height'):
                                    # Likely a gr.Image component
                                    image_components.append(item)
                            
                            if image_components:
                                logger.info(f"Found {len(image_components)} image components to monitor")
                                
                                # Set up change handlers
                                for i, comp in enumerate(image_components):
                                    def make_handler(idx):
                                        def handler(img):
                                            current = ForgeIntegration.get_kontext_images()
                                            current[idx] = img
                                            ForgeIntegration.register_kontext_images(current)
                                        return handler
                                    
                                    comp.change(make_handler(i), inputs=[comp], outputs=[])
                        
                        return result
                    
                    ForgeKontext.ui = patched_ui
                    logger.info("Successfully patched Kontext script for integration")
                    return True
                    
    except Exception as e:
        logger.warning(f"Could not patch Kontext script: {e}")
    
    return False


def create_linked_analyze_button(image_index: int) -> gr.Button:
    """
    Create an analyze button that automatically uses the corresponding Kontext image.
    
    Args:
        image_index: Index of the Kontext image (0-2)
        
    Returns:
        Gradio Button component
    """
    def analyze_handler():
        images = ForgeIntegration.get_kontext_images()
        if image_index < len(images) and images[image_index] is not None:
            return f"analyze_image_{image_index}"
        else:
            return f"no_image_{image_index}"
    
    button = gr.Button(
        f"Analyze Kontext Image {image_index + 1}",
        variant="secondary"
    )
    
    return button


class SmartImageSync:
    """Provides smart synchronization between Kontext images and analysis."""
    
    def __init__(self):
        self.auto_analyze = False
        self.analyzed_hashes = set()
    
    def enable_auto_analysis(self) -> None:
        """Enable automatic analysis when new images are loaded."""
        self.auto_analyze = True
        ForgeIntegration.register_observer(self._on_images_changed)
    
    def _on_images_changed(self, event: str, data: Any) -> None:
        """Handle image change events."""
        if event == 'images_updated' and self.auto_analyze:
            images = data
            for i, img in enumerate(images):
                if img is not None:
                    img_hash = hash(img.tobytes())
                    if img_hash not in self.analyzed_hashes:
                        # Trigger analysis for new image
                        logger.info(f"Auto-analyzing new image {i+1}")
                        self.analyzed_hashes.add(img_hash)
                        # This would trigger the analysis through Gradio events
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        images = ForgeIntegration.get_kontext_images()
        return {
            'kontext_enabled': ForgeIntegration.is_kontext_enabled(),
            'images_loaded': [img is not None for img in images],
            'auto_analyze': self.auto_analyze,
            'analyzed_count': len(self.analyzed_hashes)
        }