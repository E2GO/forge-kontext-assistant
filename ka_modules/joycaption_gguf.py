"""
JoyCaption GGUF implementation with automatic model downloading
Uses llama-cpp-python for quantized model inference
"""

import os
import sys
import logging
import hashlib
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch
import base64
import io
import json
import time

logger = logging.getLogger(__name__)

# Model configuration
GGUF_MODELS = {
    'Q6_K': {
        'url': 'https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-hf-llava.Q6_K.gguf',
        'size': '6.7GB',
        'quality': 0.97,
        'filename': 'llama-joycaption-beta-one-hf-llava.Q6_K.gguf'
    },
    'mmproj': {
        'url': 'https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-mmproj-f16.gguf',
        'size': '596MB',
        'filename': 'llama-joycaption-mmproj-f16.gguf'
    }
}

class JoyCaptionGGUF:
    """JoyCaption analyzer using GGUF quantized models"""
    
    def __init__(self, model_dir: Optional[str] = None, quantization: str = 'Q6_K', 
                 device: str = 'cuda', force_cpu: bool = False):
        self.quantization = quantization
        self.device = 'cpu' if force_cpu else device
        self.model_dir = Path(model_dir) if model_dir else self._get_default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.llama_cpp_available = False
        self._check_llama_cpp()
        
        # Prompt templates (same as original)
        self.prompts = {
            'descriptive_casual_medium': 'Describe this image in detail, including any text present. Use natural language with short, descriptive sentences.',
            'booru_tags_medium': 'Please caption this image in danbooru style. Output the caption as text separated by commas.',
            'descriptive': 'Write a descriptive caption for this image in a formal tone.',
            'training_prompt': 'Write a stable diffusion prompt for this image.'
        }
        
    def _get_default_model_dir(self) -> Path:
        """Get default model directory"""
        # Try to use Forge's model directory
        forge_root = Path(__file__).parent.parent.parent.parent
        models_dir = forge_root / "models" / "JoyCaption"
        
        if not models_dir.exists():
            # Fallback to extension directory
            models_dir = Path(__file__).parent.parent / "models" / "JoyCaption"
        
        return models_dir
    
    def _check_llama_cpp(self):
        """Check if llama-cpp-python is available"""
        try:
            import llama_cpp
            self.llama_cpp_available = True
            logger.info("llama-cpp-python is available")
        except ImportError:
            logger.warning("llama-cpp-python not installed, trying to install...")
            self._install_llama_cpp()
    
    def _install_llama_cpp(self):
        """Try to install llama-cpp-python"""
        try:
            import subprocess
            # Try with CUDA support first
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--extra-index-url", 
                "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                # Fallback to CPU version
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "llama-cpp-python"
                ])
            
            import llama_cpp
            self.llama_cpp_available = True
            logger.info("Successfully installed llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to install llama-cpp-python: {e}")
            self.llama_cpp_available = False
    
    def _download_file(self, url: str, filepath: Path, desc: str) -> bool:
        """Download a file with progress bar"""
        try:
            logger.info(f"Downloading {desc} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Downloading {desc}: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded {desc}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {desc}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def _ensure_models_downloaded(self) -> bool:
        """Ensure all required models are downloaded"""
        # Check main model
        model_info = GGUF_MODELS[self.quantization]
        model_path = self.model_dir / model_info['filename']
        
        if not model_path.exists():
            logger.info(f"JoyCaption {self.quantization} model not found, downloading...")
            if not self._download_file(model_info['url'], model_path, f"JoyCaption {self.quantization}"):
                return False
        
        # Check mmproj file
        mmproj_info = GGUF_MODELS['mmproj']
        mmproj_path = self.model_dir / mmproj_info['filename']
        
        if not mmproj_path.exists():
            logger.info("JoyCaption mmproj file not found, downloading...")
            if not self._download_file(mmproj_info['url'], mmproj_path, "mmproj"):
                return False
        
        return True
    
    def _load_model(self):
        """Load the GGUF model"""
        if not self.llama_cpp_available:
            raise RuntimeError("llama-cpp-python is not available")
        
        if not self._ensure_models_downloaded():
            raise RuntimeError("Failed to download required models")
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            model_path = self.model_dir / GGUF_MODELS[self.quantization]['filename']
            mmproj_path = self.model_dir / GGUF_MODELS['mmproj']['filename']
            
            logger.info(f"Loading JoyCaption {self.quantization} from {model_path}")
            
            # Create chat handler for multimodal support
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=str(mmproj_path),
                verbose=False
            )
            
            # Load model
            n_gpu_layers = -1 if self.device == 'cuda' else 0
            
            self.model = Llama(
                model_path=str(model_path),
                chat_handler=self.chat_handler,
                n_ctx=2048,
                n_gpu_layers=n_gpu_layers,
                seed=42,
                verbose=False
            )
            
            logger.info(f"Successfully loaded JoyCaption {self.quantization} GGUF model")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def analyze(self, image: Image.Image, mode: str = 'booru_tags_medium') -> Dict[str, Any]:
        """Analyze image using GGUF model"""
        if self.model is None:
            self._load_model()
        
        start_time = time.time()
        
        try:
            # Convert PIL image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare prompt
            prompt = self.prompts.get(mode, self.prompts['descriptive_casual_medium'])
            
            # Create message with image
            messages = [
                {
                    "role": "system",
                    "content": "You are an image captioning assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/png;base64,{img_str}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Generate response
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.6,
                top_p=0.9,
                max_tokens=300
            )
            
            content = response['choices'][0]['message']['content']
            
            # Parse output based on mode
            result = {
                'success': True,
                'analysis_time': time.time() - start_time,
                'model': f'JoyCaption {self.quantization} GGUF'
            }
            
            if 'booru' in mode or 'tags' in mode:
                result['booru_tags_medium'] = content
                result['danbooru_tags'] = content
                result['categories'] = self._parse_tags(content)
            else:
                result['descriptive_casual_medium'] = content
                result['descriptive'] = content
            
            return result
            
        except Exception as e:
            logger.error(f"Error during GGUF analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_time': time.time() - start_time
            }
    
    def _parse_tags(self, tags_str: str) -> Dict[str, list]:
        """Parse tags into categories"""
        categories = {
            'characters': [],
            'objects': [],
            'environment': [],
            'style': [],
            'colors': [],
            'lighting': []
        }
        
        if not tags_str:
            return categories
        
        # Split tags
        tags = [tag.strip() for tag in tags_str.replace('\n', ',').split(',') if tag.strip()]
        
        # Simple categorization
        for tag in tags:
            tag_lower = tag.lower()
            
            # Characters
            if any(kw in tag_lower for kw in ['girl', 'boy', 'woman', 'man', 'person', 'people']):
                categories['characters'].append(tag)
            # Colors
            elif any(kw in tag_lower for kw in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'pink']):
                categories['colors'].append(tag)
            # Lighting
            elif any(kw in tag_lower for kw in ['light', 'shadow', 'dark', 'bright', 'sunset', 'sunrise']):
                categories['lighting'].append(tag)
            # Environment
            elif any(kw in tag_lower for kw in ['indoor', 'outdoor', 'room', 'street', 'nature', 'city']):
                categories['environment'].append(tag)
            # Style
            elif any(kw in tag_lower for kw in ['anime', 'realistic', 'cartoon', 'drawing', 'painting']):
                categories['style'].append(tag)
            else:
                categories['objects'].append(tag)
        
        return categories
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if hasattr(self, 'chat_handler'):
                del self.chat_handler
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("JoyCaption GGUF model unloaded")