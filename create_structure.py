#!/usr/bin/env python3
"""Create project directory structure"""

import os
from pathlib import Path

def create_project_structure():
    """Create all necessary directories and empty __init__.py files"""
    
    # Define directory structure
    directories = [
        "modules",
        "configs",
        "configs/prompts",
        "utils",
        "tests",
        "tests/test_modules",
        "docs",
        "examples"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "modules/__init__.py",
        "utils/__init__.py",
        "tests/__init__.py",
        "tests/test_modules/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"Created file: {init_file}")
    
    # Create placeholder files
    placeholder_files = {
        "modules/image_analyzer.py": '"""Florence-2 image analysis module"""\n\n# TODO: Implement ImageAnalyzer class',
        "modules/prompt_generator.py": '"""Template-based prompt generation"""\n\n# TODO: Implement PromptGenerator class',
        "modules/llm_enhancer.py": '"""Phi-3 prompt enhancement module"""\n\n# TODO: Implement Phi3Enhancer class',
        "modules/templates.py": '"""Prompt templates for different task types"""\n\n# TODO: Define template structures',
        "modules/ui_components.py": '"""Gradio UI components"""\n\n# TODO: Implement UI building functions',
        "utils/cache.py": '"""Caching utilities for analysis results"""\n\n# TODO: Implement caching logic',
        "utils/validators.py": '"""Input validation utilities"""\n\n# TODO: Implement validators',
        "configs/prompts/system_prompts.json": '{\n  "phi3_system": "You are an expert at converting user intentions into precise FLUX.1 Kontext editing instructions."\n}',
        "configs/prompts/templates.json": '{\n  "object_color_change": "Change the {object} from {current_color} to {new_color}"\n}'
    }
    
    for file_path, content in placeholder_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {file_path}")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Initialize git repository: git init")
    print("2. Add remote: git remote add origin https://github.com/yourusername/forge-kontext-assistant.git")
    print("3. Create initial commit: git add . && git commit -m 'Initial project structure'")
    print("4. Push to GitHub: git push -u origin main")

if __name__ == "__main__":
    create_project_structure()