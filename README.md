# 🤖 Kontext Assistant for Stable Diffusion WebUI Forge

[![Version](https://img.shields.io/badge/version-2.77-blue.svg)](https://github.com/yourusername/forge-kontext-assistant)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Forge](https://img.shields.io/badge/Forge%20WebUI-Compatible-orange.svg)](https://github.com/lllyasviel/stable-diffusion-webui-forge)

Smart context-aware assistant for FLUX.1 Kontext models. Analyzes your context images and generates optimized prompts.

## ✨ Features

- **🔍 Dual-Model Analysis**: Florence-2 Base + PromptGen v2.0
- **🎯 Smart Prompt Generation**: 7 task types for FLUX.1 Kontext
- **⚡ Performance Optimized**: FP16, auto-unload, RTX 5090 support
- **🖼️ Multi-Image Support**: Up to 3 context images
- **💾 Memory Management**: Automatic model loading/unloading
- **🔄 Auto-refresh**: Detects image changes

## 📦 Installation

1. Clone to extensions folder:
```bash
cd extensions
git clone https://github.com/yourusername/forge-kontext-assistant
```

2. Restart WebUI - dependencies install automatically

## 🚀 Quick Start

1. Load images in **Forge FluxKontext Pro**
2. Open **🤖 Kontext Assistant**
3. Click **Analyze All Images**
4. Select task type and describe changes
5. Click **Generate FLUX.1 Kontext Prompt**

## 📖 Documentation

- [Usage Guide](docs/USAGE.md)
- [Changelog](CHANGELOG.md)

## 🛠️ Troubleshooting

**Images not loading?**
- Check images are in KontextPro
- Click 🔄 refresh button

**h11 errors?**
- Run `clear_all_caches.bat`

**Low performance?**
- Adjust Auto-unload settings in Advanced Options
- Check Performance Info panel

## 📊 System Requirements

- Stable Diffusion WebUI Forge
- FLUX.1 Kontext model
- 6GB+ VRAM recommended
- Python 3.10+

## 🤝 Contributing

Pull requests welcome! Please test changes thoroughly.

## 📄 License

MIT License - see LICENSE file

---
**Version**: 2.77 | **Status**: Production Ready