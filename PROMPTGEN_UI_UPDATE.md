# PromptGen UI Update Plan

## 1. Add PromptGen Instruction Selection

### Current Implementation:
- Uses fixed instructions: `<MORE_DETAILED_CAPTION>`, `<GENERATE_TAGS>`, `<MIXED_CAPTION_PLUS>`, `<ANALYZE>`

### New Feature:
Add dropdown to select PromptGen instruction mode:
- `<GENERATE_TAGS>` - Danbooru style tags only
- `<CAPTION>` - One line caption
- `<DETAILED_CAPTION>` - Structured caption with positions
- `<MORE_DETAILED_CAPTION>` - Very detailed description (current default)
- `<ANALYZE>` - Composition analysis
- `<MIXED_CAPTION>` - Mixed caption and tags (for FLUX)
- `<MIXED_CAPTION_PLUS>` - Combined mixed caption with analysis

## 2. Fix Model Status Display

### Issues:
1. Status doesn't update after unloading models
2. Shows "✅ Loaded" even after unload
3. Success message appears at top, status remains unchanged at bottom

### Solution:
1. Update `_get_model_status()` to check actual model state
2. Return model status update from `unload_models()`
3. Connect status update to UI element

## Implementation Code Changes

### 1. Add PromptGen instruction dropdown in UI:
```python
# After florence_model dropdown
promptgen_instruction = gradio.Dropdown(
    label="PromptGen Instruction",
    choices=[
        ("Tags Only (Danbooru style)", "<GENERATE_TAGS>"),
        ("One Line Caption", "<CAPTION>"),
        ("Detailed Caption with Positions", "<DETAILED_CAPTION>"),
        ("Very Detailed Description", "<MORE_DETAILED_CAPTION>"),
        ("Composition Analysis", "<ANALYZE>"),
        ("Mixed Caption + Tags (FLUX)", "<MIXED_CAPTION>"),
        ("Mixed Caption + Analysis", "<MIXED_CAPTION_PLUS>")
    ],
    value="<MORE_DETAILED_CAPTION>",
    info="Select the type of output from PromptGen v2.0",
    visible=False  # Initially hidden
)

# Show/hide based on florence_model
florence_model.change(
    fn=lambda x: gradio.update(visible=(x == "promptgen_v2")),
    inputs=[florence_model],
    outputs=[promptgen_instruction]
)
```

### 2. Update image_analyzer.py to use selected instruction:
```python
def _real_analyze(self, image: Image.Image, detailed: bool = True, 
                  promptgen_instruction: str = "<MORE_DETAILED_CAPTION>") -> Dict[str, Any]:
    # Use the selected instruction instead of hardcoded
    if self.model_type == "promptgen_v2":
        caption_result = self._run_florence_task(image, promptgen_instruction)
```

### 3. Fix model status update:
```python
def unload_models():
    """Manually unload all models to free memory"""
    if hasattr(KontextAssistant, '_shared_analyzer') and KontextAssistant._shared_analyzer:
        try:
            logger.info("Manually unloading analyzer models...")
            KontextAssistant._shared_analyzer.unload_models()
            KontextAssistant._shared_analyzer = None
            KontextAssistant._analyzer_settings = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return both message and updated status
            return "✅ Models unloaded successfully!", self._get_model_status()
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
            return f"❌ Error unloading models: {str(e)}", self._get_model_status()
    return "ℹ️ No models to unload", self._get_model_status()

# Update button connection
unload_models_btn.click(
    fn=unload_models,
    outputs=[analyze_status, model_status]  # Update both status areas
)
```

### 4. Improve _get_model_status to check actual state:
```python
def _get_model_status(self) -> str:
    """Get current model status information"""
    status_lines = ["### 🤖 Model Status\n"]
    
    try:
        # Check if analyzer exists
        if not hasattr(self, 'analyzer') or not self.analyzer:
            status_lines.append("**Models**: Not loaded")
            return "\n".join(status_lines)
            
        # Check JoyCaption status
        if hasattr(self.analyzer, 'joycaption') and self.analyzer.joycaption:
            if self.analyzer.joycaption._initialized and self.analyzer.joycaption.model is not None:
                status_lines.append("**JoyCaption**: ✅ Loaded (16GB HuggingFace model)")
            else:
                status_lines.append("**JoyCaption**: Not loaded")
        else:
            status_lines.append("**JoyCaption**: Not available")
            
        # Check Florence status
        if hasattr(self.analyzer, 'florence') and self.analyzer.florence:
            if self.analyzer.florence._initialized and self.analyzer.florence.model is not None:
                model_name = self.analyzer.florence.model_name
                model_size = "~3GB" if "promptgen" in model_name.lower() else "~1GB"
                status_lines.append(f"**Florence-2 {model_name}**: ✅ Loaded ({model_size})")
            else:
                status_lines.append("**Florence-2**: Not loaded")
        else:
            status_lines.append("**Florence-2**: Not available")
```