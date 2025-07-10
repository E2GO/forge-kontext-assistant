# PromptGen Instruction Processing Fix Summary

## Issue Found

The PromptGen instruction parameter was not being passed from the UI through to the actual model, causing all instructions to produce the same output (using the default `<MORE_DETAILED_CAPTION>`).

## Root Causes

1. **Missing Parameter Pass-Through**: In `scripts/kontext_assistant.py`, the `promptgen_instruction` parameter was not being passed to the `analyzer.analyze()` method call.

2. **Missing Method**: The `_extract_key_elements()` method was being called in `image_analyzer.py` but didn't exist, which would cause an error when using `<GENERATE_TAGS>` instruction.

## Fixes Applied

### 1. Fixed Parameter Pass-Through
In `scripts/kontext_assistant.py` line 331:
```python
# Before:
analysis = self.analyzer.analyze(image, use_florence=use_florence, use_joycaption=use_joycaption)

# After:
analysis = self.analyzer.analyze(image, use_florence=use_florence, use_joycaption=use_joycaption, promptgen_instruction=promptgen_instruction)
```

### 2. Fixed Missing Method Reference
In `ka_modules/image_analyzer.py` line 286:
```python
# Before:
'general': self._extract_key_elements(raw_description)

# After:
'general': raw_description  # Use raw description directly for now
```

## How the Flow Works Now

1. User selects a PromptGen instruction in the UI dropdown
2. The instruction is passed to `analyze_image()` in `kontext_assistant.py`
3. It's then passed to `analyzer.analyze()` in `smart_analyzer.py`
4. SmartAnalyzer passes it to `florence.analyze()` in `image_analyzer.py`
5. ImageAnalyzer uses it in `_run_florence_task()` to generate the appropriate output
6. The output is processed differently based on the instruction type:
   - `<GENERATE_TAGS>`: Stored in `analysis['tags']`
   - `<ANALYZE>`: Stored in `analysis['composition_analysis']`
   - All caption types: Stored in `analysis['description']`

## Expected Behavior

Now each PromptGen instruction should produce different outputs:
- `<CAPTION>`: Brief caption
- `<DETAILED_CAPTION>`: More detailed caption
- `<MORE_DETAILED_CAPTION>`: Very detailed caption (default)
- `<ANALYZE>`: Composition and artistic analysis
- `<GENERATE_TAGS>`: Danbooru-style tags

## Testing Recommendation

To verify the fix works:
1. Load an image in Forge FluxKontext Pro
2. Select "Florence-2 PromptGen v2.0" as the model
3. Try each instruction type and click "Analyze Image"
4. You should see different outputs for each instruction type