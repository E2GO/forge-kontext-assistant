#!/usr/bin/env python3
"""
Quick script to check current Florence-2 loading code and test the fix.
"""

import sys
import os

# Fix for Python 3.10
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check current code
print("Checking current image_analyzer.py code...")
analyzer_path = project_root / "ka_modules" / "image_analyzer.py"

if analyzer_path.exists():
    with open(analyzer_path, 'r') as f:
        content = f.read()
    
    # Check if device_map is still there
    if 'device_map="auto"' in content or "device_map='auto'" in content:
        print("❌ Found device_map='auto' in the code!")
        print("   This needs to be removed.")
        
        # Show the problematic lines
        for i, line in enumerate(content.split('\n')):
            if 'device_map' in line:
                print(f"   Line {i+1}: {line.strip()}")
    else:
        print("✅ No device_map='auto' found in the code.")
        print("   The file should be correct.")

print("\n" + "="*60)
print("Testing direct Florence-2 loading...")
print("="*60)

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    if device == "cuda":
        model = model.to(device)
    
    print("✅ Model loaded successfully!")
    
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    print("✅ Processor loaded successfully!")
    
    print("\nTesting with simple image...")
    from PIL import Image
    test_image = Image.new('RGB', (256, 256), color='red')
    
    # Test processing
    inputs = processor(text="<CAPTION>", images=test_image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print("✅ Processing test successful!")
    
    # Clean up
    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Florence-2 is working correctly!")
    print("The issue is in the image_analyzer.py file.")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()