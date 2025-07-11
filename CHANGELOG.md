# Changelog

## [2.77] - 2025-07-11
### Added
- Performance Info display panel with model details and metrics
- Configurable auto-unload settings (0-300s delay)
- PyTorch version display in diagnostics
- Better torch.compile error handling
- Image validation utilities for Gradio compatibility

### Fixed
- h11 Content-Length errors via cache cleanup script
- Image loading issues with validation layer
- Performance data capture for dual-model system

### Changed
- Default unload delay: 15s → 60s for better performance
- Memory threshold: 90% → 75% for stability
- Performance Info accordion collapsed by default

## [2.76] - 2025-01-10
### Added
- Refresh button (🔄) for clearing empty image slots
- Automatic image change detection
- Clear analysis when images are removed
- Memory optimization controls

### Fixed
- GPU memory warnings during model switching
- Analysis cache invalidation

## [2.75] - 2025-01-09
### Major Update
- Implemented dual-model system: Florence-2 Base + PromptGen v2.0
- Full RTX 5090 GPU support
- FP16 model prioritization for memory efficiency
- Automatic dependency installation

### Changed
- PromptGen v2.0 now primary model for tags and captions
- Simplified UI - removed model selection options
- Improved thread safety with locks

### Fixed
- PromptGen v2.0 output generation
- Model loading on RTX 5090
- Memory leaks during model switching

### Removed
- JoyCaption support (replaced by PromptGen v2.0)
- GGUF model support
- Manual model selection UI