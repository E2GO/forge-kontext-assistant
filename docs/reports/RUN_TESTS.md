# How to Run PromptGen v2.0 Tests

## Simple Python Scripts (No WebUI Required)

All these scripts can be run directly with Python from any directory.

### 1. Minimal Test (Fastest)
```bash
cd /path/to/forge-kontext-assistant
python test_minimal.py
```
- Tests basic functionality
- Shows if PromptGen produces any output
- Takes ~3-4 minutes

### 2. Quick Debug
```bash
python quick_debug.py
```
- Quick functionality check
- Shows output preview
- Basic error detection

### 3. Check All Tasks
```bash
python check_promptgen.py
```
- Tests all 7 PromptGen tasks
- Shows which tasks work/fail
- Includes direct generation test
- Most comprehensive

### 4. Simple Test
```bash
python simple_test.py
```
- Step-by-step testing
- Detailed logging
- Full analysis test

## What You Should See

### ✅ Working Output Example:
```
Testing...
✓ SUCCESS! Got 245 characters:
A vibrant red square fills the entire frame, creating a bold and striking visual composition. The uniform crimson hue dominates the image with its intense saturation, presenting a minimalist aesthetic that emphasizes color as the primary element...
```

### ❌ Problem Output:
```
Testing...
✗ FAILED: No result
```
or
```
✓ SUCCESS! Got 15 characters:
Mood: neutral
```

## If Tests Fail

1. **Check Error Messages**
   - Import errors → Install missing packages
   - CUDA errors → Try CPU mode
   - Model loading errors → Check disk space

2. **Install Dependencies**
   ```bash
   pip install torch torchvision transformers pillow
   ```

3. **Run Most Detailed Test**
   ```bash
   python check_promptgen.py
   ```
   This will show exactly which part is failing.

## Expected Results

When PromptGen v2.0 works correctly:
- Captions: 100-1000 characters
- Tags: Comma-separated list
- All tasks produce different outputs
- Model loads in 3-4 minutes

## Quick Diagnosis

Run this command to see what's happening:
```bash
python -c "import sys; sys.path.insert(0, '.'); from ka_modules.image_analyzer import ImageAnalyzer; from PIL import Image; a = ImageAnalyzer(model_type='promptgen_v2'); a._ensure_initialized(); r = a._run_florence_task(Image.new('RGB', (256, 256), 'blue'), '<MORE_DETAILED_CAPTION>'); print(f'Result: {r}')"
```

This one-liner will show the raw output from PromptGen.