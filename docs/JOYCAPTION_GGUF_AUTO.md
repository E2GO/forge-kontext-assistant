# JoyCaption GGUF Automatic Support

## Overview
Starting from version 2.8, Kontext Assistant automatically uses quantized GGUF models for JoyCaption, reducing memory usage from 16GB to 6.7GB while maintaining 97% quality.

## Features
- **Automatic download**: Models download on first use
- **Reduced memory**: 6.7GB instead of 16GB
- **Faster inference**: ~15% speed improvement
- **No manual setup**: Works out of the box

## How It Works

1. **First Run**: When you enable JoyCaption for the first time, the extension will:
   - Check for existing GGUF models
   - Download Q6_K model (6.7GB) if not found
   - Download mmproj file (596MB) if not found
   - Save to `models/JoyCaption/` directory

2. **Automatic Selection**: The extension automatically uses GGUF when available

3. **Fallback**: If GGUF fails, it falls back to HuggingFace model

## Model Details

### Q6_K Quantization
- **Size**: 6.7GB (vs 16GB original)
- **Quality**: 97% of original
- **Speed**: 15% faster
- **Memory**: ~8GB GPU required

### Model Files
```
models/JoyCaption/
├── llama-joycaption-beta-one-hf-llava.Q6_K.gguf (6.7GB)
└── llama-joycaption-mmproj-f16.gguf (596MB)
```

## Requirements

### Automatic Installation
The extension will try to install llama-cpp-python automatically with CUDA support.

### Manual Installation (if auto-install fails)
```bash
# For CUDA GPUs
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For CPU only
pip install llama-cpp-python
```

## Usage
No changes needed! Just use JoyCaption as before:

1. Enable "Generate advanced description and tags. Slow mode"
2. Analyze your images
3. Models download automatically on first use

## Troubleshooting

### Download Issues
If download fails:
1. Check internet connection
2. Check disk space (need ~8GB free)
3. Try manual download:
```bash
cd models/JoyCaption
wget https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-hf-llava.Q6_K.gguf
wget https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-mmproj-f16.gguf
```

### CUDA Not Detected
```bash
# Reinstall with CUDA support
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### Out of Memory
- Close other applications
- Try CPU inference (slower but works)
- Consider using Florence-2 only

## Performance Comparison

| Model Type | Size | Memory | Speed | Quality |
|------------|------|--------|-------|---------|
| Original HF | 16GB | ~18GB | 1.0x | 100% |
| GGUF Q6_K | 6.7GB | ~8GB | 1.15x | 97% |

## Future Enhancements

### Planned Features
- UI option to choose quantization level
- Support for other quantizations (Q4_K, Q8_0)
- Batch processing optimization
- Progress bar for downloads

### Coming Soon
- Settings to disable auto-download
- Custom model directory configuration
- Multi-GPU support