# FluxKontext Smart Assistant

An intelligent prompt generation assistant for FLUX.1 Kontext in Forge WebUI. Automatically analyzes context images and generates proper instructional prompts for the FLUX.1 Kontext model.

## Features

- 🖼️ **Smart Image Analysis** - Comprehensive analysis using Florence-2 Large
- 🤖 **Intelligent Prompt Generation** - Converts natural language to FLUX.1 Kontext instructions
- 🎯 **Task-Specific Templates** - Optimized prompts for different editing tasks
- 🧠 **Optional LLM Enhancement** - Phi-3-mini for complex creative requests
- 🚀 **Seamless Integration** - Works alongside existing Kontext functionality

## Requirements

- Forge WebUI (latest version)
- Python 3.10+
- CUDA-capable GPU with:
  - Minimum: 6GB VRAM (without Phi-3)
  - Recommended: 12GB+ VRAM (with Phi-3)
- FluxKontext model loaded

## Installation

1. Navigate to your Forge WebUI extensions directory:
```bash
cd extensions
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/forge-kontext-assistant.git
```

3. Install dependencies:
```bash
cd forge-kontext-assistant
pip install -r requirements.txt
```

4. Restart Forge WebUI

## Usage

1. **Load a FluxKontext model** in the checkpoint dropdown
2. **Add context images** using the existing Kontext interface (up to 3)
3. **Click "Analyze Image"** to analyze each context image
4. **Select task type** from the dropdown (Object Color, Style Transfer, etc.)
5. **Enter your intent** in natural language (e.g., "make the car blue")
6. **Generate prompt** and review/edit the result
7. **Use the generated prompt** for image generation

### Example Workflow

```
1. Upload image: Red car on street
2. Analyze → Detects: car (red), street, buildings
3. Task: "Object Color Change"
4. Intent: "blue car"
5. Generated: "Change the red car color to blue while maintaining 
   the exact same model, shadows, reflections, and position..."
```

## Project Structure

```
forge-kontext-assistant/
├── __init__.py
├── kontext_assistant.py      # Main Script class
├── modules/
│   ├── image_analyzer.py     # Florence-2 integration
│   ├── prompt_generator.py   # Template-based generation
│   ├── llm_enhancer.py      # Phi-3 enhancement (optional)
│   ├── templates.py         # Prompt templates
│   └── ui_components.py     # Gradio UI components
├── configs/
│   └── task_configs.json    # Task configurations
└── utils/
    ├── cache.py            # Result caching
    └── validators.py       # Input validation
```

## Supported Task Types

1. **Object Manipulation**
   - Color changes
   - State modifications
   - Attribute editing

2. **Style Transfer**
   - Artistic styles (impressionist, anime, etc.)
   - Time periods (vintage, futuristic)
   - Cultural aesthetics

3. **Environment Changes**
   - Location/background replacement
   - Weather conditions
   - Time of day

4. **Element Combination**
   - Merging multiple elements
   - Scene composition

5. **State Changes**
   - Aging/weathering
   - Physical transformations

## Configuration

Edit `configs/task_configs.json` to customize:
- Task types and subtypes
- Default preservation rules
- Complexity thresholds
- Template patterns

## Performance Tips

- **First run** will download models (~3GB total)
- **Enable caching** for repeated analyses
- **Disable Phi-3** if VRAM limited (<8GB)
- **Use batch analysis** for multiple images

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Disable Phi-3 enhancement
   - Reduce batch size
   - Use CPU offloading

2. **Slow Analysis**
   - Check CUDA installation
   - Enable model caching
   - Use smaller context images

3. **Poor Prompts**
   - Ensure clear intent description
   - Try different task types
   - Enable Phi-3 for complex requests

## Development

### Adding New Task Types

1. Add configuration to `configs/task_configs.json`
2. Create template in `modules/templates.py`
3. Add UI option in task dropdown

### Running Tests

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FLUX.1 Kontext by Black Forest Labs
- Florence-2 by Microsoft
- Phi-3 by Microsoft
- Forge WebUI community

## Links

- [FLUX.1 Kontext Documentation](https://blackforestlabs.ai/flux-1-kontext/)
- [Forge WebUI](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- [Report Issues](https://github.com/yourusername/forge-kontext-assistant/issues)