# 📚 Technical Documentation - Kontext Assistant

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Components and Modules](#components-and-modules)
3. [Algorithms and Workflows](#algorithms-and-workflows)
4. [APIs and Interfaces](#apis-and-interfaces)
5. [Configuration](#configuration)
6. [AI Models](#ai-models)
7. [Security and Permissions](#security-and-permissions)
8. [Performance Optimization](#performance-optimization)

## 🏗️ System Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    WebUI Interface                       │
│                 (Gradio Components)                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  ForgeKontextUnified                     │
│              (scripts/kontext.py)                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Image     │  │   Prompt     │  │   Advanced     │ │
│  │  Analysis   │  │   Builder    │  │   Settings     │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘ │
└─────────┼────────────────┼───────────────────┼─────────┘
          │                │                   │
┌─────────┴────────┐ ┌─────┴─────┐ ┌─────────┴──────────┐
│  SmartAnalyzer   │ │  Prompt   │ │  Configuration     │
│  (Dual-Model)    │ │  Builder  │ │  Management        │
├──────────────────┤ ├───────────┤ ├────────────────────┤
│ • Florence-2     │ │ • Task    │ │ • Settings.json    │
│ • PromptGen v2   │ │   Types   │ │ • Style Modifiers  │
│ • Auto-unload    │ │ • Token   │ │ • Dual-Image       │
│ • Thread-safe    │ │   Count   │ │ • Arrangements     │
└──────────────────┘ └───────────┘ └────────────────────┘
```

### Data Flow

1. **Image Loading** → KontextPro Extension
2. **Image Retrieval** → Kontext Assistant via shared state
3. **Analysis** → SmartAnalyzer (Florence-2 + PromptGen)
4. **Processing** → PromptBuilder with task and settings
5. **Generation** → FLUX.1 Kontext via WebUI

## 📦 Components and Modules

### 1. scripts/kontext.py

**Main extension module**

```python
class ForgeKontextUnified:
    def __init__(self):
        # Initialize UI components
        # Load configurations
        # Setup event handlers
```

**Key methods:**
- `create_ui()` - Create Gradio interface
- `analyze_images()` - Start image analysis
- `generate_prompt()` - Generate prompt
- `_get_kontext_images()` - Get images from KontextPro

### 2. ka_modules/smart_analyzer.py

**Dual-model analysis system**

```python
class SmartAnalyzer:
    def __init__(self, auto_unload=True, unload_delay=60):
        self.florence_analyzer = None
        self.promptgen_analyzer = None
        self.florence_lock = threading.Lock()
        self.promptgen_lock = threading.Lock()
```

**Main features:**
- Automatic memory management
- Thread-safe model initialization
- Model selection based on task
- Result caching

### 3. ka_modules/image_analyzer.py

**Florence-2 model handling**

```python
class ImageAnalyzer:
    def __init__(self):
        self.model_name = "microsoft/Florence-2-base"
        self.model = None
        self.processor = None
        self.device = self._get_device()
```

**Features:**
- Auto GPU detection (RTX 4090/5090)
- FP16 support for compatible GPUs
- Protection against processor = None errors
- `_run_florence_task` method for task execution

### 4. ka_modules/prompt_builder.py

**Prompt generation and processing**

**Functionality:**
- 9 task types with different templates
- Style and modifier integration
- Dual-image scenario handling
- Character arrangement management

### 5. ka_modules/token_utils.py

**Token counting for FLUX.1**

```python
def count_flux_tokens(text: str) -> int:
    """Count tokens considering FLUX.1 specifics"""
    # Uses cl100k_base encoder
    # Accounts for special tokens
```

**Important:** 512 tokens != 512 characters

## 🔄 Algorithms and Workflows

### Image Analysis Algorithm

```python
def analyze_image(image, mode="standard"):
    # 1. Select tasks based on mode
    if mode == "fast":
        tasks = ["<CAPTION>"]
    elif mode == "detailed":
        tasks = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    
    # 2. Choose model
    if need_promptgen(tasks):
        analyzer = get_promptgen_analyzer()
    else:
        analyzer = get_florence_analyzer()
    
    # 3. Execute analysis
    results = analyzer.analyze(image, tasks)
    
    # 4. Format results
    return format_results(results)
```

### Prompt Generation Algorithm

```python
def generate_prompt(analysis_results, task_type, description):
    # 1. Load template for task
    template = load_task_template(task_type)
    
    # 2. Extract key elements
    elements = extract_key_elements(analysis_results)
    
    # 3. Apply modifiers
    if style_modifiers:
        elements = apply_style_modifiers(elements, modifiers)
    
    # 4. Build prompt
    prompt = build_prompt(template, elements, description)
    
    # 5. Validate tokens
    if count_tokens(prompt) > 512:
        prompt = truncate_prompt(prompt)
    
    return prompt
```

## 🔌 APIs and Interfaces

### SmartAnalyzer Interface

```python
# Initialize
analyzer = SmartAnalyzer(auto_unload=True, unload_delay=60)

# Analyze image
results = analyzer.analyze_image(
    image=PIL.Image,
    mode="standard",  # fast/standard/detailed/tags_only/composition
    force_model=None  # "florence"/"promptgen" or None for auto
)

# Force unload models
analyzer.unload_all()
```

### PromptBuilder Interface

```python
# Create prompt
prompt = build_prompt(
    task_type="enhance_details",
    image_analyses=["analysis1", "analysis2"],
    num_images=2,
    description="make it more vibrant",
    style_modifier="vibrant_colors",
    dual_image_scenario="character_interaction",
    arrangement_preset="side_by_side",
    integration_method="natural"
)
```

## ⚙️ Configuration

### settings.json

```json
{
    "default_analysis_mode": "standard",
    "auto_unload_enabled": true,
    "auto_unload_delay": 60,
    "max_custom_styles": 50,
    "debug_mode": false,
    "performance": {
        "batch_size": 1,
        "use_fp16": "auto",
        "cache_analyses": true
    }
}
```

### style_modifiers.json

```json
{
    "categories": {
        "colors": {
            "name": "Color Schemes",
            "modifiers": {
                "vibrant_colors": {
                    "name": "Vibrant Colors",
                    "description": "Bright and saturated colors",
                    "prompt": "vibrant colors, high saturation..."
                }
            }
        }
    }
}
```

### dual_image_configs.json

```json
{
    "scenarios": {
        "character_interaction": {
            "name": "Character Interaction",
            "description": "Two characters interacting",
            "template": "Place both {char1} and {char2} together..."
        }
    }
}
```

## 🤖 AI Models

### Florence-2 Base

- **Source**: Microsoft
- **Size**: ~450MB
- **License**: GNU AGPL v3
- **Capabilities**: 
  - Object detection
  - Image captioning
  - Composition analysis
  - Text OCR

### PromptGen v2.0

- **Source**: Mitsua Diffusion
- **Size**: ~550MB
- **License**: Apache 2.0
- **Capabilities**:
  - Booru-style tag generation
  - Detailed descriptions
  - Style and mood analysis

### Automatic Download

Models download on first use to:
```
~/.cache/kontext_assistant/
├── florence-2-base/
└── promptgen-v2/
```

## 🔒 Security and Permissions

### Permissions Used

1. **File System**:
   - Read: Configuration files
   - Write: Model cache, custom styles
   - Path: `~/.cache/kontext_assistant/`

2. **Network**:
   - Model download from Hugging Face
   - HTTPS only
   - Checksum verification

3. **GPU/CUDA**:
   - GPU access for inference
   - VRAM memory management

### Data Security

- Images processed locally
- No data sent to external servers
- Temporary files deleted automatically
- Custom styles stored locally

## 🚀 Performance Optimization

### Memory Management

```python
# Automatic model unloading
def auto_unload_models():
    if time_since_last_use > unload_delay:
        unload_model("florence")
        unload_model("promptgen")
        torch.cuda.empty_cache()
```

### GPU Optimizations

1. **FP16 for RTX 4090/5090**:
   ```python
   if is_rtx_4090_or_5090():
       model = model.half()
   ```

2. **Batch processing**:
   - Process multiple tasks in one pass
   - Minimize context switching

3. **Caching**:
   - Image hashing for change detection
   - Analysis result saving

### Performance Recommendations

1. **For speed**:
   - Use "Fast" mode
   - Enable auto-unload
   - Reduce unload_delay to 30 seconds

2. **For quality**:
   - "Detailed" mode
   - Disable auto-unload for batch processing
   - Use PromptGen for tags

3. **With limited VRAM**:
   - Enable auto-unload
   - Use FP16 (automatic for RTX 4090/5090)
   - Process one image at a time

## 📊 Monitoring and Debugging

### Performance Info Panel

Displays:
- Current VRAM usage
- Model status (loaded/unloaded)
- Last analysis time
- Number of processed images

### Debug Mode

Enable in settings.json:
```json
{
    "debug_mode": true
}
```

Additional information:
- Detailed operation logs
- Execution time for each stage
- Memory usage by module
- Error tracing

---

**Created with Claude AI** | [Back to README](README.md)