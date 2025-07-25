# 🤖 Kontext Assistant for Stable Diffusion WebUI Forge

[![Version](https://img.shields.io/badge/version-1.0.1-blue.svg)](https://github.com/yourusername/forge-kontext-assistant)
[![License](https://img.shields.io/badge/license-AGPL%20v3-green.svg)](LICENSE)
[![Forge](https://img.shields.io/badge/Forge%20WebUI-Compatible-orange.svg)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![Created with](https://img.shields.io/badge/Created%20with-Claude%20AI-purple.svg)](https://claude.ai)

An intelligent assistant for FLUX.1 Kontext models in Stable Diffusion WebUI Forge. Analyzes context images and generates optimized prompts using dual AI models.

## ⚠️ Important Notice: Multiple Images Warning

> **FLUX.1 Kontext Image Limitations:**
> - **1 image**: ✅ Works perfectly - recommended for best results
> - **2 images**: ⚠️ May experience issues - use with caution
> - **3 images**: 🚨 **DANGER ZONE** - use at your own risk!
> 
> **Memory Usage**: Each additional image significantly increases VRAM consumption. Monitor your GPU memory when using multiple images.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Credits](#credits)
- [License](#license)

## 🌟 Overview

Kontext Assistant is an extension for Stable Diffusion WebUI Forge that automates prompt creation for FLUX.1 Kontext models. The extension uses advanced computer vision models to analyze images and generate detailed descriptions, which are then transformed into optimized prompts.

### What is FLUX.1 Kontext?

FLUX.1 Kontext is a specialized version of the FLUX.1 model capable of working with context images. Unlike standard models, it can use up to 3 reference images for better understanding of the desired output.

### Why use this assistant?

- **Automation**: Eliminates the need to manually describe images
- **Accuracy**: Uses AI for detailed analysis of composition, objects, and style
- **Optimization**: Generates prompts considering FLUX.1 Kontext limitations (512 tokens)
- **Flexibility**: Provides numerous settings and operation modes

## ✨ Key Features

### 🔍 Dual-Model Analysis System
- **Florence-2 Base**: Fast analysis and object detection
- **PromptGen v2.0**: Detailed descriptions and tag generation
- **Automatic model switching** based on task requirements

### 🎯 Analysis Modes
1. **Fast**: Basic description in 2-3 seconds
2. **Standard**: Balanced analysis
3. **Detailed**: Complete analysis of all aspects
4. **Tags Only**: Booru-style tag generation
5. **Composition**: Focus on element arrangement

### 🎨 Prompt Builder
- **Hundreds of styles** across 14 comprehensive categories
- **Material & Environment transforms** for creative effects
- **10 scenarios** including dual-image workflows
- **Multiple arrangement options** for dual-image mode
- **User prompt management** system

### 🚀 Performance Optimization
- RTX 4090/5090 support with automatic FP16
- Automatic model unloading from memory
- Multi-threaded processing with conflict protection
- Analysis result caching

### 💾 Custom Styles System
- Save favorite prompts
- Quick access via dropdown
- Import/export styles

## 🛠️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **VRAM**: 4GB
- **RAM**: 8GB
- **Disk Space**: 2GB for models

### Recommended Requirements
- **VRAM**: 6GB or more
- **RAM**: 16GB
- **GPU**: NVIDIA RTX 3060 or higher
- **Disk Space**: 5GB (including cache)

## 📦 Installation

### Method 1: Install from WebUI (Easiest)

1. Open Stable Diffusion WebUI Forge in your browser
2. Go to **Extensions** tab
3. Click **Install from URL** sub-tab
4. Paste this URL: `https://github.com/yourusername/forge-kontext-assistant`
5. Click **Install**
6. Go to **Installed** tab
7. Click **Apply and restart UI**

### Method 2: Command Line Installation

1. Clone the repository to extensions folder:
```bash
cd stable-diffusion-webui-forge/extensions
git clone https://github.com/E2GO/forge-kontext-assistant
```

2. Restart WebUI - dependencies will install automatically

### Method 3: Manual Installation

1. Download the release archive
2. Extract to `stable-diffusion-webui-forge/extensions/`
3. Install dependencies:
```bash
cd forge-kontext-assistant
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Workflow: Change Image Style

1. **Load your image** in Forge FluxKontext Pro
2. **Open** "🎨 Prompt Builder" section
3. **Select** "Style Transfer" from Scenario dropdown
4. **Choose styles** from extensive categories or type custom style
5. **Click** "Build Prompt" to generate optimized prompt
6. **Click** "Generate" button to create your styled image

### When to Use Image Analysis?

Not sure what you're seeing in the image? Don't know what something is called or what color it is? Want to recreate an image and need a prompt for it?

**Use the Analyze feature:**
1. Click **"Analyze All Images"** button
2. Choose analysis mode (Fast for quick overview, Detailed for full description)
3. Use the analysis results to understand your image better

## 📚 Features

### Main Components

#### 1. Image Analysis
- **Analysis Mode**: Choose analysis mode
- **Analyze All Images**: Analyze all loaded images
- **Analysis Results**: Results with detailed description

#### 2. Prompt Builder
- **Scenario Selection**: 10 scenarios for different tasks
- **Style Library**: 14 categories with hundreds of styles
- **Dual-Image Mode**: Combine two images with arrangement options
- **Material/Environment Transforms**: Creative transformation effects
- **Token Counter**: Real-time token tracking
- **Direct Generation**: Generate from within Prompt Builder

#### 3. Advanced Settings
- **Prompt Templates**: Prompt templates
- **Custom Styles**: User styles management
- **Auto-unload**: Model auto-unload settings
- **Performance Info**: Performance information

### Scenarios

1. **Style Transfer**: Apply artistic styles to images
2. **Add/Remove Object**: Add new elements or remove unwanted ones
3. **Replace Object**: Transform one object into another
4. **Change Pose**: Modify character poses
5. **Change Emotion**: Alter facial expressions
6. **Change Lighting**: Adjust lighting and mood
7. **Enhance/Restore Image**: Improve quality and details
8. **Extend Canvas**: Expand image boundaries (Outpainting)
9. **Dual-Image Mode**: Combine two images creatively
10. **User Prompts**: Use saved custom prompts

### Token Limitations

- **Maximum**: 512 tokens (not characters!)
- **Warning**: At 450+ tokens
- **Counter**: Real-time display
- **Validation**: Automatic check before generation

## 📖 Documentation

- [Detailed Documentation](DOCUMENTATION.md) - Technical information
- [Credits and Sources](CREDITS.md) - Resources used

## 🔧 Troubleshooting

### Images not loading?
- Check that images are loaded in KontextPro
- Click refresh button (🔄)
- Ensure image format is supported (PNG, JPG, WebP)

### Errors on first run?
- Models download on first use (~1GB)
- Process may take 1-5 minutes depending on internet speed
- Check free disk space

### Cache and h11 errors?

The extension includes a **universal cache clearing script** that works on all platforms (Windows, Linux, macOS).

**When to use it:**
- Getting `h11` or `httpcore` errors
- WebUI crashes or behaves strangely after updates
- Extension stops working after system changes
- UI elements not updating properly

**How to use:**
1. Navigate to the extension folder
2. Run the cache cleaner:
   ```bash
   python clear_cache.py
   ```

**What it does:**
- Clears Python `__pycache__` directories (compiled bytecode)
- Clears pip cache (downloaded packages)
- Clears Gradio temporary files (UI components)
- Optionally clears Kontext Assistant model cache
- Reinstalls the h11 module (common source of errors)
- Provides browser cache clearing instructions

**Note:** This is a safe operation that only removes temporary files. Your settings and custom styles are preserved.

### Low performance?
- Enable Auto-unload in Advanced Settings
- Use Fast mode for quick tasks
- Check VRAM usage in Performance Info panel

### CUDA/GPU errors?
- Ensure latest NVIDIA drivers are installed
- Check PyTorch version compatibility with your GPU
- Try reducing batch size in WebUI settings

## 👨‍💻 Development

### Created with Claude AI

This extension was fully developed using [Claude AI](https://claude.ai) by Anthropic. Claude helped with:
- System architecture and design
- Writing all code
- AI model integration
- Performance optimization
- Documentation creation

### Based on

This project is based on and inspired by:
- **[forge2_flux_kontext](https://github.com/DenOfEquity/forge2_flux_kontext)** by DenOfEquity - Base script code and resolution transfer from script to main interface
- **[4o-ghibli-at-home](https://github.com/TheAhmadOsman/4o-ghibli-at-home)** by TheAhmadOsman - Many styles were used or inspired by this project

### Project Structure

```
forge-kontext-assistant/
├── scripts/
│   └── kontext.py          # Main UI and logic
├── ka_modules/
│   ├── smart_analyzer.py   # Dual-model system
│   ├── image_analyzer.py   # Florence-2 handling
│   ├── prompt_builder.py   # Prompt generation
│   ├── token_utils.py      # Token counting
│   ├── styles/             # Style library modules
│   │   ├── anime_manga_styles.py
│   │   ├── art_movements_styles.py
│   │   ├── cartoon_styles.py
│   │   ├── cultural_styles.py
│   │   ├── digital_art_styles.py
│   │   ├── environment_transform_styles.py
│   │   ├── famous_artists_styles.py
│   │   ├── material_transform_styles.py
│   │   ├── photography_styles.py
│   │   └── traditional_art_styles.py
│   └── ...                 # Other modules
├── configs/
│   ├── settings.json       # Main settings
│   ├── style_modifiers.json # Style modifiers
│   └── ...                 # Other configs
└── javascript/
    └── kontext_set_dimensions.js # UI scripts
```

### Contributing

Pull requests are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Credits

- **DenOfEquity** - For the base [forge2_flux_kontext](https://github.com/DenOfEquity/forge2_flux_kontext) code
- **TheAhmadOsman** - For style inspiration from [4o-ghibli-at-home](https://github.com/TheAhmadOsman/4o-ghibli-at-home)
- **Anthropic Claude AI** - For development assistance
- **Microsoft** - For Florence-2 model
- **Mitsua Diffusion** - For PromptGen v2.0 model
- **AUTOMATIC1111 & lllyasviel** - For Stable Diffusion WebUI Forge
- **Community** - For testing and feedback

## 📄 License

Distributed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

---

**Version**: 1.0.1 | **Status**: Production Ready | **Created with**: Claude AI
