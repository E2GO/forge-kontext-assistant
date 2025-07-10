# PromptGen v2.0 FLUX Integration Analysis

## The Problem

PromptGen v2.0 was specifically designed for FLUX models with dual text encoders:
- **T5XXL**: For detailed natural language descriptions
- **CLIP_L**: For shorter, tag-based descriptions

Our current implementation treats PromptGen as a standard Florence-2 model, which causes:
- `<GENERATE_TAGS>` returns empty results (designed for CLIP_L)
- `<ANALYZE>` returns empty results
- Missing support for FLUX-specific features

## How It Should Work

### FLUX Dual Encoder System:
```
Image → PromptGen v2.0 → 
    ├── T5XXL encoder: Detailed descriptions
    └── CLIP_L encoder: Tags and short captions
```

### PromptGen Instructions by Encoder:

**For T5XXL (long descriptions):**
- `<MORE_DETAILED_CAPTION>` ✅ Works
- `<DETAILED_CAPTION>` ✅ Works
- `<CAPTION>` ⚠️ Works but too long

**For CLIP_L (tags/short):**
- `<GENERATE_TAGS>` ❌ Broken - expects CLIP_L context
- `<ANALYZE>` ❌ Broken - expects special parsing

**For Both (FLUX optimized):**
- `<MIXED_CAPTION>` ✅ Works - combines both
- `<MIXED_CAPTION_PLUS>` ✅ Works - adds analysis

## Why Some Instructions Fail

1. **GENERATE_TAGS** expects to output in a format suitable for CLIP_L encoder
2. **ANALYZE** expects different post-processing than standard captions
3. The Florence-2 post_process_generation() doesn't understand PromptGen v2.0's special formats

## Solution Approach

### Option 1: Use Working Instructions Only
For FLUX workflows, use:
- `<MIXED_CAPTION>` - Best for FLUX, provides both T5XXL and CLIP_L content
- `<MORE_DETAILED_CAPTION>` - For T5XXL only

### Option 2: Implement Custom Parsing
1. Detect FLUX model usage
2. Implement custom parsing for GENERATE_TAGS and ANALYZE
3. Split MIXED_CAPTION output for dual encoders

### Option 3: ComfyUI Integration
The model was designed for ComfyUI-Miaoshouai-Tagger which has:
- "Flux CLIP Text Encode" node
- Proper dual encoder support
- Custom parsing for all instructions

## Current Workaround

Since we're in WebUI (not ComfyUI), the best approach is:
1. Use `<MIXED_CAPTION>` for FLUX workflows
2. Extract tags from the mixed output if needed
3. Avoid broken instructions until proper FLUX integration is added

## Example MIXED_CAPTION Output

For FLUX, a MIXED_CAPTION should output something like:
```
"A cheerful anthropomorphic rat wearing a blue hooded robe, gold medallion. 
1girl, solo, furry, rat, anthro, hood, blue_robe, medallion, smile"
```

Where:
- First part → T5XXL encoder (detailed description)
- Second part → CLIP_L encoder (tags)