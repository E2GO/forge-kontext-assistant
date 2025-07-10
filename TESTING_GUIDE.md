# PromptGen v2.0 Testing Guide

This guide explains the comprehensive testing suite for PromptGen v2.0 integration.

## Quick Start

### Run All Tests
```bash
python run_all_tests.py
```

### Quick Functionality Test
```bash
python run_all_tests.py --quick
```

### Debug Issues
```bash
python debug_promptgen.py
```

## Test Suite Overview

### 1. Core Functionality Tests (`tests/test_promptgen_v2.py`)

Comprehensive tests for PromptGen v2.0 model:

- **Model Initialization**: Verifies correct model loading and configuration
- **All Task Prompts**: Tests all 7 PromptGen task types
- **Output Format Validation**: Ensures outputs match documentation
- **Model Comparison**: Compares base Florence-2 vs PromptGen outputs
- **Real-World Images**: Tests with different image types
- **Generation Parameters**: Validates correct parameters are used
- **Error Handling**: Tests edge cases and error conditions
- **Full Pipeline**: Tests complete analysis workflow
- **Performance Metrics**: Measures processing times
- **Output Consistency**: Ensures deterministic outputs

### 2. UI Integration Tests (`tests/test_ui_integration.py`)

Tests integration with Forge WebUI:

- **Model Switching**: Tests switching between base and PromptGen
- **Output Format**: Validates UI-compatible output format
- **Performance**: Ensures acceptable response times
- **Error Handling**: Tests UI error scenarios
- **Template Integration**: Tests prompt template generation
- **Batch Processing**: Tests multiple image processing
- **Model Unloading**: Tests memory management

### 3. Output Validation (`tests/validate_promptgen_output.py`)

Validates output formats against documentation:

- **Tags Format**: Validates Danbooru-style tags
- **Caption Format**: Checks caption structure and content
- **Mixed Format**: Validates combined caption+tags format
- **Analysis Format**: Checks composition analysis output

## Expected PromptGen v2.0 Outputs

### Task: `<GENERATE_TAGS>`
- **Format**: Comma-separated Danbooru-style tags
- **Example**: `1girl, solo, long hair, blue eyes, dress, standing, outdoors, scenery`
- **Validation**: 5-50 tags, lowercase with underscores

### Task: `<MORE_DETAILED_CAPTION>`
- **Format**: Detailed paragraph description
- **Length**: 100-1000 characters
- **Content**: Comprehensive scene description

### Task: `<MIXED_CAPTION_PLUS>`
- **Format**: Combined caption, tags, and analysis
- **Length**: 200+ characters
- **Use**: Optimal for Flux model prompts

## Debugging Guide

### If PromptGen outputs are minimal:

1. **Run Debug Script**:
   ```bash
   python debug_promptgen.py
   ```

2. **Check Debug Output**:
   - Model loading status
   - Processor configuration
   - Generation process details
   - Task-by-task results

3. **Review Logs**:
   - Look for tokenization issues
   - Check post-processing errors
   - Verify output extraction

### Common Issues and Solutions

1. **Empty/Short Output**
   - Check if post-processing is working
   - Verify correct task prompts are used
   - Ensure model weights loaded correctly

2. **Model Not Switching**
   - Clear analysis cache
   - Force model reload
   - Check florence_model_type parameter

3. **Slow Performance**
   - Use CUDA if available
   - Check batch size (use 1 for stability)
   - Monitor memory usage

## Test Reports

After running tests, check these files:

- `tests/test_report_promptgen_v2.json` - Core test results
- `tests/promptgen_validation_report.json` - Output validation details
- `promptgen_debug_report.json` - Debugging information

## Integration Checklist

Before deployment, ensure:

- [ ] All core tests pass
- [ ] UI integration tests pass
- [ ] Output validation passes
- [ ] Performance is acceptable (<30s per image)
- [ ] Error handling works correctly
- [ ] Memory management (unloading) works
- [ ] Model switching works reliably

## Adding New Tests

To add new tests:

1. Add test method to appropriate test class
2. Follow naming convention: `test_description_of_test`
3. Include assertions and logging
4. Update this guide if needed

## Continuous Testing

For development:

```bash
# Watch mode (requires pytest-watch)
ptw tests/

# Run specific test
pytest tests/test_promptgen_v2.py::TestPromptGenV2::test_all_task_prompts -v
```

## Performance Benchmarks

Expected performance on CUDA:

- Model loading: ~30-60s (first time)
- Caption generation: 2-5s
- Tag generation: 1-3s
- Full analysis: 5-15s

## Support

If tests fail:

1. Check console logs for detailed errors
2. Run debug_promptgen.py for diagnosis
3. Review generated reports
4. Check GPU compatibility
5. Verify dependencies are installed