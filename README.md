# FluxKontext Smart Assistant V2

Advanced AI-powered prompt generation for FLUX.1 Kontext in Forge WebUI.

## 🚀 What's New in V2

- **Florence-2 Integration**: Real image analysis using Microsoft's Florence-2 Large model
- **Extended Task Types**: Added lighting, texture, perspective, and seasonal changes
- **Smart Integration**: Automatic detection and analysis of Kontext images
- **Performance**: Optimized memory usage with lazy loading and caching
- **Python 3.10.17 Compatibility**: Full support with collections patches

## 📋 Requirements

- Forge WebUI (latest version)
- Python 3.10+
- PyTorch with CUDA support (recommended)
- ~1.5GB VRAM for Florence-2
- ~2GB disk space for models

## 🔧 Installation

1. Clone into your Forge extensions folder:
```bash
cd extensions
git clone https://github.com/E2GO/forge-kontext-assistant.git
cd forge-kontext-assistant
git checkout V2
```

2. Install dependencies (if not already installed):
```bash
pip install transformers torch torchvision
```

## 🎯 Quick Start

1. **Test the installation**:
```bash
# Quick check
python tests/quick_check.py

# Test Florence-2 (downloads model on first run)
python tests/test_florence2_simple.py
```

2. **In Forge WebUI**:
   - Load FLUX.1 Kontext model
   - Switch to img2img tab
   - Enable "Forge FluxKontext Pro"
   - Enable "Kontext Smart Assistant"
   - Load your context images
   - Click "Analyze Image" for each
   - Select task type and describe your intent
   - Generate optimized prompt!

## 🎨 Supported Task Types

### Object Manipulation
- Color changes
- Adding/removing elements
- Modifying attributes
- Resizing and repositioning

### Style Transfer
- Artistic styles
- Time period conversion
- Medium changes (oil painting, sketch, etc.)
- Mood adjustments

### Environment Changes
- Location transport
- Weather modification
- Time of day shifts
- Seasonal transformations

### Advanced Tasks
- Element combination
- State changes (aging, repair)
- Outpainting
- Lighting adjustments
- Texture modifications
- Perspective shifts

## 🧠 How It Works

1. **Image Analysis**: Florence-2 analyzes your context images to understand:
   - Objects and their positions
   - Artistic style and mood
   - Environment and setting
   - Lighting conditions
   - Composition

2. **Smart Prompt Generation**: Based on analysis and your intent:
   - Selects appropriate template
   - Fills in context-aware details
   - Adds preservation rules
   - Optimizes for FLUX.1 Kontext

3. **Integration**: Seamlessly works with Forge FluxKontext Pro:
   - Auto-detects loaded images
   - Shares state between scripts
   - Manages memory efficiently

## 💡 Tips for Best Results

1. **Be Specific**: Instead of "make it better", say "add dramatic sunset lighting"
2. **Use Task Types**: Select the most appropriate task type for better templates
3. **Analyze First**: Always analyze images before generating prompts
4. **Preserve Details**: Use the preservation strength slider to control changes

## 🐛 Troubleshooting

### "ImportError: cannot import name 'Mapping'"
This is fixed in V2. If you still see it, ensure you're using the V2 branch.

### Florence-2 won't load
- Check internet connection (first download ~1.5GB)
- Ensure sufficient disk space
- Try: `pip install --upgrade transformers`

### Out of Memory
- Florence-2 only needs ~1.5GB VRAM
- Use the "Unload Model" option when done
- Restart Forge if needed

## 🛠️ Development

### Running Tests
```bash
# All tests
python tests/test_basic.py

# Florence-2 specific
python tests/test_florence2_simple.py

# Quick diagnostic
python tests/quick_check.py
```

### Project Structure
```
forge-kontext-assistant/
├── scripts/
│   ├── kontext.py              # Main Kontext script
│   └── kontext_assistant.py    # Smart Assistant
├── ka_modules/                 # Assistant modules
│   ├── image_analyzer.py       # Florence-2 integration
│   ├── prompt_generator.py     # Prompt generation
│   ├── templates.py           # Template library
│   └── forge_integration.py    # Script integration
├── configs/
│   └── task_configs.json      # Task configurations
└── tests/                     # Test suite
```

## 📊 Performance

- Image analysis: 2-5 seconds (first run), <0.1s (cached)
- Prompt generation: <0.5 seconds
- Memory usage: ~1.5GB (Florence-2) + base Forge usage
- Supports RTX 30/40/50 series, AMD RX 6000+

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Credits

- Microsoft for Florence-2
- Black Forest Labs for FLUX.1 Kontext
- Forge WebUI team
- All contributors

---

**Version**: 2.0.0  
**Status**: Active Development  
**Support**: [Issues](https://github.com/E2GO/forge-kontext-assistant/issues)