# Florence-2 Tag Generation

## Overview
Florence-2 can now generate tags similar to JoyCaption, making it a viable standalone option for image analysis when JoyCaption is unavailable or to reduce memory usage.

## Tag Categories Generated

### 1. Character Tags
- `1girl`, `1boy`, `solo`
- `multiple_girls`, `multiple_boys`
- Based on detected persons in the image

### 2. Clothing Tags
- Extracted from detailed object analysis
- Includes items like: dress, shirt, suit, coat, jacket, cloak, armor
- Color-specific clothing (e.g., "red dress", "blue shirt")

### 3. Style Tags
- `realistic`, `anime`, `digital_art`
- `watercolor`, `oil_painting`, `3d`
- Detected from image description

### 4. Environment Tags
- Setting: `indoor`, `outdoor`, `urban`
- Time: `morning`, `evening`, `night`, `day`
- Weather conditions when applicable

### 5. Object Tags
- Top 5 most prominent objects from detection
- Based on Florence-2's object detection task

### 6. Color Tags
- Primary colors found in the description
- Includes: red, blue, green, yellow, purple, orange, black, white, etc.

### 7. Composition Tags
- `close-up`, `full_body`, `upper_body`
- `portrait`, `from_above`, `from_below`
- Based on camera angle and framing

## Comparison with JoyCaption

### Florence-2 Tags
**Advantages:**
- Faster generation (no additional model needed)
- Less memory usage
- Better object detection accuracy
- Consistent technical analysis

**Limitations:**
- Less nuanced artistic tags
- No Danbooru-specific training
- Simpler tag vocabulary

### JoyCaption Tags
**Advantages:**
- Trained on Danbooru dataset
- More artistic and stylistic tags
- Better anime/manga specific tags
- Richer tag vocabulary

**Limitations:**
- Requires 8GB additional model
- Slower processing
- Higher memory usage

## Usage

### Florence-2 Only Mode
When JoyCaption is disabled, Florence-2 will automatically generate tags:
```python
# Tags will be available in analysis['tags']
{
    'general': ['1girl', 'solo', 'outdoor', 'day', 'red', 'dress'],
    'character': ['1girl', 'solo'],
    'clothing': ['red dress'],
    'style': ['realistic'],
    'environment': ['outdoor', 'day'],
    'objects': ['tree', 'flower'],
    'colors': ['red', 'green'],
    'composition': ['full_body']
}
```

### Combined Mode
When both analyzers are enabled:
- JoyCaption tags take precedence
- Florence-2 provides technical data
- Objects and composition from both are merged

## Recommendations

1. **For Fast Analysis**: Use Florence-2 only
2. **For Artistic Work**: Use JoyCaption
3. **For Technical Accuracy**: Use Florence-2
4. **For Best Results**: Use both (if memory allows)

## Performance Impact
- Florence-2 tag generation adds ~0.1-0.2 seconds to analysis time
- No additional memory required (uses existing model)
- Tags are generated from the detailed description already created