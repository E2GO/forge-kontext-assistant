# Kontext Assistant - Testing Checklist

## 🧪 Manual Testing Guide for UI

### Pre-test Setup
- [ ] Restart Forge WebUI to load latest changes
- [ ] Load FLUX.1 Kontext model
- [ ] Have test images ready (various types)

### 1. Basic UI Elements
- [ ] Kontext Smart Assistant accordion appears in UI
- [ ] Two checkboxes visible:
  - [ ] "Use Florence-2 (objects, positions, OCR)"
  - [ ] "Use JoyCaption (descriptions, tags, style)"
- [ ] "Analyze All Images" button present
- [ ] Individual analyze buttons (🔍) for each slot

### 2. Florence-2 Testing
- [ ] Load image in Kontext slot 1
- [ ] Enable only Florence-2 checkbox
- [ ] Click analyze button for slot 1
- [ ] Check output contains:
  - [ ] 📝 Description
  - [ ] 🎯 Objects detected
  - [ ] 🔬 Technical data (size, mode)
  - [ ] ⏱️ Analysis time

### 3. JoyCaption Testing
- [ ] Load image in Kontext slot 2
- [ ] Enable only JoyCaption checkbox
- [ ] Click analyze button for slot 2
- [ ] Expected behavior:
  - [ ] First run: May take time to download model (~8GB)
  - [ ] Should show descriptive caption
  - [ ] May show tags if successful

### 4. Combined Analysis
- [ ] Load image in Kontext slot 3
- [ ] Enable both checkboxes
- [ ] Click analyze button
- [ ] Should show combined results from both analyzers

### 5. Error Handling
- [ ] Disable both checkboxes
- [ ] Click analyze - should show error message
- [ ] Load non-image file (if possible) - should handle gracefully

### 6. Analyze All Button
- [ ] Load images in all 3 slots
- [ ] Enable Florence-2
- [ ] Click "Analyze All Images"
- [ ] All 3 slots should show analysis

### 7. Caching Test
- [ ] Analyze an image
- [ ] Click analyze again on same image
- [ ] Should be faster (using cache)

### 8. Image Removal Test
- [ ] Analyze an image
- [ ] Remove image from Kontext slot
- [ ] Analysis should auto-clear (or on next analyze)

### 9. Prompt Generation
- [ ] Analyze an image
- [ ] Select task type (e.g., "Apply Artistic Style")
- [ ] Enter intent (e.g., "make it oil painting")
- [ ] Click "Generate Kontext Prompt"
- [ ] Should generate appropriate prompt

### 10. Advanced Options
- [ ] Check "Show Detailed Results"
- [ ] Re-analyze - should show more details
- [ ] Test "Force CPU mode" if needed

## 🐛 Known Issues to Watch For

1. **JoyCaption First Run**
   - May fail on first attempt while downloading
   - Normal behavior - retry after download

2. **Memory Usage**
   - Monitor VRAM usage with both models loaded
   - Florence-2: ~2-3GB
   - JoyCaption: ~8-10GB

3. **Performance**
   - Florence-2: Should be fast (<5s)
   - JoyCaption: Slower, especially first run

## ✅ Expected Results

### Florence-2 Output Example:
```
📝 Description: A gradient pattern with colors...
🎯 Main objects: gradient, pattern, colors
🔬 Size: 512x512 | Mode: RGB
⏱️ Analysis time: 2.3s | Using: Florence-2
```

### JoyCaption Output Example:
```
📝 Description: A vibrant abstract composition...
🏷️ Danbooru tags: abstract, colorful, gradient
🎨 Style: abstract art
⏱️ Analysis time: 5.1s | Using: JoyCaption
```

## 📊 Success Criteria

- [ ] No Python errors in console
- [ ] All basic functions work
- [ ] Reasonable analysis times
- [ ] Meaningful analysis results
- [ ] Prompt generation produces valid prompts

## 🚀 Ready for Production?

If all tests pass:
1. System is ready for use
2. Consider enabling JoyCaption by default if stable
3. Update documentation with final status