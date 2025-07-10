# PromptGen v2.0 Known Issues

## Current Status (as of testing with Rat_Thief.png)

### ✅ Working Instructions:
- **Very Detailed Description** (`<MORE_DETAILED_CAPTION>`) - Works perfectly
- **Detailed Caption with Positions** (`<DETAILED_CAPTION>`) - Works well
- **Mixed Caption + Tags (FLUX)** (`<MIXED_CAPTION>`) - Works, returns description + tags
- **Mixed Caption + Analysis** (`<MIXED_CAPTION_PLUS>`) - Works, returns very detailed combined output

### ❌ Broken Instructions:
- **Tags Only (Danbooru style)** (`<GENERATE_TAGS>`) - Returns empty/only "Mood: neutral"
- **Composition Analysis** (`<ANALYZE>`) - Returns empty/only "Mood: neutral"

### ⚠️ Incorrect Behavior:
- **One Line Caption** (`<CAPTION>`) - Returns full description instead of short caption

## Root Cause Analysis

The issues appear to be with the PromptGen v2.0 model itself, not our implementation:

1. The model seems to be primarily trained for description generation
2. Tag generation and composition analysis features may not be fully implemented
3. The model doesn't differentiate between caption lengths properly

## Workarounds

### For Tags:
Use `<MIXED_CAPTION>` and extract the tags portion from the output

### For Analysis:
Use `<MIXED_CAPTION_PLUS>` which includes some analytical information

### For Short Captions:
Use `<MORE_DETAILED_CAPTION>` and truncate to first sentence

## Testing Commands

To test with your own images:

```bash
# First activate the WebUI environment
cd "J:\Stability Matrix\Packages\Stable Diffusion WebUI Forge"
venv\Scripts\activate

# Then run the test
python extensions\forge-kontext-assistant\debug_promptgen.py your_image.png
```

## Temporary Solution

We've added fallback mechanisms in the code:
- Empty results will trigger alternate processing
- Warning messages in logs when results are too short
- Fallback text when generation fails

## Future Improvements

Consider:
1. Contacting MiaoshouAI about these issues
2. Using a different model for tag generation
3. Implementing our own tag extraction from descriptions
4. Waiting for PromptGen v3.0 or updates