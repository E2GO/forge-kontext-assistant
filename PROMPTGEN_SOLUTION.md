# PromptGen v2.0 Solution Summary

## Root Cause Analysis

Based on research, PromptGen v2.0 has these characteristics:

1. **Designed for FLUX Dual Encoders**
   - T5XXL encoder: For natural language descriptions
   - CLIP_L encoder: For tags and keywords
   - Special instructions optimize for each encoder

2. **Post-Processing Issues**
   - The standard Florence-2 `post_process_generation()` doesn't handle PromptGen v2.0's special formats
   - GENERATE_TAGS and ANALYZE produce output that needs custom parsing
   - The model generates content, but it's not extracted properly

3. **ComfyUI vs WebUI Differences**
   - PromptGen v2.0 was designed for ComfyUI with custom nodes
   - ComfyUI has special "Flux CLIP Text Encode" node for proper handling
   - WebUI Forge loads CLIP_L and T5XXL but doesn't provide this context to PromptGen

## Working Instructions

### ✅ Fully Working:
- `<MORE_DETAILED_CAPTION>` - Detailed descriptions
- `<DETAILED_CAPTION>` - Structured descriptions
- `<MIXED_CAPTION>` - Combined format (best for FLUX)
- `<MIXED_CAPTION_PLUS>` - Everything combined

### ❌ Not Working Properly:
- `<GENERATE_TAGS>` - Returns empty (parsing issue)
- `<ANALYZE>` - Returns empty (parsing issue)
- `<CAPTION>` - Returns full description instead of one line

## Recommended Workflow for FLUX in WebUI

Since you have CLIP_L and T5XXL encoders loaded:

1. **Use MIXED_CAPTION mode** - This is specifically designed for FLUX
   ```
   Output format: "Detailed description. tag1, tag2, tag3, tag4..."
   ```

2. **Parse the output** manually if needed:
   - First part (before tags) → T5XXL encoder
   - Tag portion → CLIP_L encoder

3. **Avoid broken modes** until custom parsing is implemented

## Implementation Plan

To fully fix this, we would need to:

1. **Detect FLUX context** when dual encoders are loaded
2. **Implement custom parsing** for GENERATE_TAGS and ANALYZE
3. **Split MIXED_CAPTION output** automatically for dual encoders
4. **Add FLUX-aware UI options**

## Current Status

- Added fallback messages explaining the issue
- Marked broken modes in UI
- Recommended using MIXED_CAPTION for FLUX workflows
- Created debug scripts for testing

The core issue is that PromptGen v2.0 expects a FLUX-aware environment that understands its special output formats, which standard Florence-2 processing doesn't provide.