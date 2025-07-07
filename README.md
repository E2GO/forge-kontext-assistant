# 🤖 Kontext Smart Assistant for Forge WebUI

[![Version](https://img.shields.io/badge/version-2.75-blue.svg)](https://github.com/yourusername/forge-kontext-assistant)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Forge](https://img.shields.io/badge/Forge%20WebUI-Compatible-orange.svg)](https://github.com/lllyasviel/stable-diffusion-webui-forge)

AI-powered assistant for FLUX.1 Kontext that analyzes context images and generates optimal instructional prompts through an intuitive interface.

## 🌟 Features

### ✨ Current Features (v2.75)
- **🔍 Smart Image Analysis** - Powered by Florence-2 Large for detailed scene understanding
- **🎯 Automatic Prompt Generation** - Converts simple intentions into FLUX.1 Kontext instructions
- **🖼️ Multi-Image Support** - Analyze up to 3 context images simultaneously
- **⚡ Performance Modes** - GPU, CPU, and Mock modes for different hardware configurations
- **🎨 Enhanced Object Detection** - Identifies clothing, accessories, props, and architectural elements
- **💾 Smart Memory Management** - Automatic model loading/unloading to prevent OOM errors
- **🆕 RTX 5090 Support** - Full compatibility with latest NVIDIA GPUs

### 📸 Analysis Capabilities
- Main objects and characters
- Clothing and accessories
- Props and held items
- Background elements
- Architectural details
- Lighting and atmosphere
- Art style detection

## 📋 Requirements

- **Forge WebUI** (latest version)
- **Python** 3.10+
- **VRAM**: 
  - Minimum: 8GB (CPU mode)
  - Recommended: 12GB+ (GPU mode)
- **Compatible GPUs**: All NVIDIA GPUs (including RTX 4090/5090)

### Python Dependencies
```bash
transformers>=4.36.0
torch>=2.0.0
Pillow>=9.0.0
gradio>=3.50.0
einops
timm
```

## 🚀 Installation

1. **Clone the repository** into your Forge extensions folder:
```bash
cd extensions
git clone https://github.com/yourusername/forge-kontext-assistant.git
```

2. **Install dependencies**:
```bash
cd forge-kontext-assistant
pip install -r requirements.txt
```

3. **Restart Forge WebUI**

## 📖 Usage

### Basic Workflow

1. **Load a FLUX.1 Kontext model** in the Checkpoint menu
2. **Add context images** to the Forge FluxKontext Pro section
3. **Open Kontext Smart Assistant** accordion
4. **Analyze images** by clicking "Analyze Image" buttons
5. **Select task type** from the dropdown
6. **Describe your intent** in natural language
7. **Generate prompt** and copy to main prompt field

### Performance Settings

- **🚀 GPU Mode** (Default): Fast analysis using CUDA
- **🖥️ CPU Mode**: Compatible mode for problematic GPUs
- **⚡ Mock Mode**: Instant results for testing (less accurate)

### Example Prompts

**Input**: "make the car blue"  
**Output**: "Change the red car color to blue while maintaining the exact same model, shadows, reflections, and position. Keep all background elements, lighting conditions, and street environment unchanged."

**Input**: "sunset lighting"  
**Output**: "Transform the lighting to golden hour sunset conditions with warm orange-yellow tones casting long shadows. Preserve all objects, positions, and scene composition while adjusting only the lighting atmosphere."

## 🔧 Troubleshooting

### Common Issues

**Florence-2 won't load on GPU**
- Enable "Force CPU mode" in Performance Settings
- This is common with RTX 4090/5090 GPUs

**Analysis takes too long**
- Use "Mock analysis" for quick testing
- Consider upgrading to GPU with more VRAM

**"Ready to download" in descriptions**
- This artifact has been fixed in v2.75
- Update if you're on an older version

### Error Messages

| Error | Solution |
|-------|----------|
| `IndentationError` | Run `fix_indentation.py` script |
| `CUDA out of memory` | Enable CPU mode or reduce image size |
| `Module not found` | Reinstall requirements.txt |

## 🛠️ Advanced Configuration

### Adjusting Analysis Detail
```python
# In ka_modules/image_analyzer.py
# Modify Florence-2 generation parameters
max_new_tokens=512  # Increase for more detail
temperature=0.7     # Adjust creativity (0.1-1.0)
```

### Custom Task Types
Add new task types in `configs/prompts/templates.json`

## 📊 Performance Benchmarks

| Hardware | Mode | Analysis Time |
|----------|------|---------------|
| RTX 5090 | GPU* | 2-5 seconds |
| RTX 5090 | CPU | 8-15 seconds |
| RTX 3080 | GPU | 3-7 seconds |
| CPU Only | CPU | 15-30 seconds |

*GPU mode may require CPU fallback on some RTX 4090/5090 cards

## 🗺️ Roadmap

### v3.0 (Planned)
- [ ] Phi-3 integration for complex prompts
- [ ] Batch image processing
- [ ] Prompt history with search
- [ ] Custom prompt templates
- [ ] API endpoint for external tools

### v2.8 (Next)
- [ ] Enhanced fantasy/creature detection
- [ ] Material and texture recognition
- [ ] Multi-language support
- [ ] Performance optimizations

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** for Florence-2 Large model
- **Black Forest Labs** for FLUX.1 Kontext
- **Forge WebUI** community for the amazing platform
- All contributors and testers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/forge-kontext-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/forge-kontext-assistant/discussions)
- **Wiki**: [Documentation](https://github.com/yourusername/forge-kontext-assistant/wiki)

---

**Note**: This extension enhances but does not replace the original Forge FluxKontext functionality. Both work together seamlessly.

*Last updated: January 2025*