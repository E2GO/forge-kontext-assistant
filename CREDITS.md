# 🙏 Credits and Acknowledgments

This document contains complete information about the resources, models, libraries, and sources of inspiration used in the Kontext Assistant project.

## 🤖 Project Creation

### Claude AI by Anthropic

This extension was **fully created with Claude AI** - an advanced language AI by Anthropic.

**Claude AI's contributions:**
- ✅ Complete system architecture and design
- ✅ Writing 100% of the project's code
- ✅ AI model integration and optimization
- ✅ Dual-model analysis system development
- ✅ User interface development
- ✅ Performance optimization and memory management
- ✅ All documentation writing
- ✅ Technical problem solving and debugging

**Link**: [claude.ai](https://claude.ai)

## 🎯 Base Code and Inspiration

### forge2_flux_kontext by DenOfEquity

This project is based on the excellent work by **DenOfEquity**.

**Contributions:**
- Base script code structure
- FLUX.1 Kontext integration approach
- Resolution transfer from script to main interface
- Core workflow implementation

**Repository**: [github.com/DenOfEquity/forge2_flux_kontext](https://github.com/DenOfEquity/forge2_flux_kontext)

### 4o-ghibli-at-home by TheAhmadOsman

Many styles and creative approaches were inspired by **TheAhmadOsman**'s work.

**Contributions:**
- Style modifier concepts
- Creative prompt engineering approaches
- Many specific styles used or adapted
- UI/UX inspiration for style selection

**Repository**: [github.com/TheAhmadOsman/4o-ghibli-at-home](https://github.com/TheAhmadOsman/4o-ghibli-at-home)

## 🧠 AI Models Used

### Florence-2 Base

**Developer**: Microsoft  
**License**: GNU AGPL v3  
**Source**: [Hugging Face - microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)  
**Version**: Latest (auto-downloaded)

**Usage in project:**
- Fast image analysis
- Object detection
- Basic caption generation
- Composition analysis
- OCR (optical character recognition)

**Technical details:**
- Model size: ~450MB
- Architecture: Vision Transformer
- FP16 support for RTX 4090/5090

### PromptGen v2.0

**Developer**: Mitsua Diffusion  
**License**: Apache License 2.0  
**Source**: [Hugging Face - MitsuaDiffusion/PromptGenV2](https://huggingface.co/MitsuaDiffusion/PromptGenV2)  
**Version**: v2.0

**Usage in project:**
- Booru-style tag generation
- Detailed description creation
- Image style and mood analysis
- "Mixed captions" generation

**Technical details:**
- Model size: ~550MB
- Specialization: Prompt engineering for Stable Diffusion
- Trained on millions of images dataset

## 📚 Core Libraries

### Python Dependencies

```
transformers==4.36.0    # For AI model operations
torch>=2.0.0           # PyTorch for GPU computing
Pillow>=9.0.0         # Image processing
numpy>=1.24.0         # Numerical computations
tiktoken>=0.5.0       # Token counting
gradio                # UI components (via WebUI)
```

### WebUI Forge Built-in Components

- **Gradio**: User interface creation
- **Modules system**: WebUI integration
- **Shared state**: Data exchange between extensions

## 🎨 Design and UI

### CSS Styling

Custom styles for buttons and interface elements were developed specifically for the project:
- 40x40px square buttons for refresh/clear
- Styling to match WebUI Forge's overall design
- Responsive layout for different screen sizes

### Icons and Emoji

Standard Unicode emoji are used for function indicators:
- 🤖 - Main assistant logo
- 🔄 - Refresh
- 🗑️ - Clear
- ⚠️ - Warnings
- ✅ - Successful operations

## 🔧 Development Tools

### Development Environment
- **Claude AI**: Primary development tool
- **Git**: Version control
- **Python 3.10+**: Programming language

### Testing
- Manual testing in Stable Diffusion WebUI Forge
- Testing on various GPUs (RTX 3060, 4090, 5090)
- Testing with different FLUX.1 Kontext versions

## 📖 Knowledge Sources and Documentation

### FLUX.1 Kontext
- Official model documentation
- Community usage examples
- Token and prompt guidelines

### Stable Diffusion WebUI Forge
- [Official repository](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- Extension development documentation
- Integration API

### Hugging Face Transformers
- [Documentation](https://huggingface.co/docs/transformers)
- Florence-2 usage examples
- Performance optimization guides

## 🌟 Ideas and Inspiration

### Stable Diffusion Community
- Dual-model analysis concept
- Prompt generation approaches
- UI/UX solutions

### Computer Vision Research
- Vision Transformers papers
- Image composition analysis methods
- Visual content description techniques

## 🤝 Special Thanks

1. **Anthropic and the Claude AI team** - For creating an amazing AI assistant that could develop this extension

2. **DenOfEquity** - For the foundational forge2_flux_kontext code that this project builds upon

3. **TheAhmadOsman** - For creative style inspiration and prompt engineering approaches

4. **Microsoft Research** - For the open-source Florence-2 model and documentation

5. **Mitsua Diffusion** - For PromptGen v2.0 and community contributions

6. **AUTOMATIC1111 & lllyasviel** - For Stable Diffusion WebUI and Forge

7. **Hugging Face** - For the platform and model hosting

8. **The Community** - For testing, feedback, and suggestions

## 📊 Project Statistics

- **Lines of code**: ~3,500+
- **Development time**: Created with Claude AI
- **Supported languages**: English (documentation and UI)
- **Compatibility**: Windows, Linux, macOS

## ⚖️ Third-Party Licenses

All components used have open licenses:
- Florence-2: GNU AGPL v3
- PromptGen v2.0: Apache 2.0
- Transformers: Apache 2.0
- PyTorch: BSD
- Other dependencies: See requirements.txt

## 📞 Documentation

- **Documentation**: See README.md and DOCUMENTATION.md

---

**Kontext Assistant v1.0.1** | **Created with Claude AI** | [Back to README](README.md)