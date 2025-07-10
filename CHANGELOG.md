# Changelog

All notable changes to the Kontext Assistant extension will be documented in this file.

## [2.75] - 2025-07-10

### Added
- Full support for PromptGen v2.0 model with all 7 task prompts
- RTX 5090 GPU compatibility 
- Comprehensive test suite with 21 new tests
- Model reset functionality for error recovery
- Enhanced logging throughout the pipeline
- Organized documentation structure

### Changed
- PromptGen v2.0 now uses standard post-processing for proper output
- RTX 4090/5090 now use GPU by default (removed forced CPU mode)
- Improved model switching with proper cache management
- Enhanced thread safety with locks
- Better error handling with fallback mechanisms

### Fixed
- PromptGen v2.0 "minimal output" issue - now generates full captions and tags
- UI error "8 values expected, 7 received" when no images loaded
- Model loading failures on RTX 5090
- Memory leaks during model switching
- Warning about Florence2LanguageForConditionalGeneration

### Removed
- 19 temporary debug and test files
- Forced CPU mode for high-end GPUs
- Redundant manual output parsing

### Performance
- Faster model switching with improved memory management
- RTX 5090 now runs at full GPU speed
- Reduced memory usage with proper cleanup

## [2.5] - Previous Version
- Initial PromptGen v2.0 support (had issues)
- Basic RTX 4090 compatibility
- JoyCaption integration