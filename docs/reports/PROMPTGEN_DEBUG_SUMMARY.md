# PromptGen v2.0 Debug Summary

## Issue
PromptGen v2.0 model loads successfully (~3 minutes for 3.29GB) but produces minimal output (only "💭 Mood: neutral ⏱️ 187.56s | 🤖 Florence-2").

## Debugging Steps Taken

### 1. Enhanced Logging
- Added extensive logging throughout the pipeline:
  - Model initialization logging
  - Input tensor shapes and types
  - Raw generated text (with and without special tokens)
  - Post-processing results
  - Output parsing at each step

### 2. Fixed Output Processing
- Changed from manual parsing to using `processor.post_process_generation()`
- This is the standard method used in the official HuggingFace examples
- Added fallback to clean text if post-processing fails

### 3. Improved Result Extraction
- Added multiple methods to extract results from dict/string responses
- Checks for task keys with and without angle brackets
- Falls back to first value if only one key exists

### 4. Generation Parameters
- Confirmed we're using the recommended parameters:
  - `max_new_tokens=1024`
  - `num_beams=3`
  - `do_sample=False`

### 5. Task Prompts
- Verified we're using the correct PromptGen v2.0 task prompts:
  - `<MORE_DETAILED_CAPTION>` for detailed descriptions
  - `<GENERATE_TAGS>` for Danbooru-style tags
  - `<MIXED_CAPTION_PLUS>` for combined captions
  - `<ANALYZE>` for composition analysis

## Next Steps

After these changes, run another analysis with PromptGen v2.0 and check the console logs. You should see:

1. **Processor and model type information**
2. **Input tensor details** - verify correct shapes
3. **Raw generated text** - both with and without special tokens
4. **Parsed answer structure** - see what keys are returned
5. **Final extracted text** - the actual captions and tags

The enhanced logging will help identify exactly where the output is being lost or truncated.

## Expected Output Format

Based on the HuggingFace examples, the output should be a dict with the task name as key:
```python
{
    '<MORE_DETAILED_CAPTION>': 'Detailed description of the image...',
    '<GENERATE_TAGS>': 'tag1, tag2, tag3, ...',
    '<MIXED_CAPTION_PLUS>': 'Combined caption with tags...',
    '<ANALYZE>': 'Composition analysis...'
}
```

## Common Issues
1. **Empty/Short Output**: Usually means the post-processing failed
2. **Wrong Format**: The model might return different keys than expected
3. **Special Tokens**: Some models include special tokens that need proper handling

The current implementation should handle all these cases with the fallback mechanisms in place.