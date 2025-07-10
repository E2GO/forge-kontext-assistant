#!/usr/bin/env python3
"""
WebUI-compatible model loading test
Run from WebUI directory: python extensions/forge-kontext-assistant/webui_model_test.py
"""

import os
import sys
import time

# Set up paths for WebUI environment
webui_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, webui_dir)
ext_dir = os.path.join(webui_dir, "extensions", "forge-kontext-assistant")
sys.path.insert(0, ext_dir)

print("="*60)
print("WEBUI MODEL LOADING TEST")
print("="*60)
print(f"WebUI dir: {webui_dir}")
print(f"Extension dir: {ext_dir}")
print()

def run_test():
    """Run the loading test"""
    try:
        # Import after paths are set
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        from ka_modules.smart_analyzer import SmartAnalyzer
        print("✓ Modules imported successfully\n")
        
        models = ["base", "promptgen_v2"]
        cycles = 3
        
        for model_type in models:
            print(f"\n{'='*40}")
            print(f"Testing: {model_type}")
            print(f"{'='*40}")
            
            for cycle in range(cycles):
                print(f"\nCycle {cycle + 1}/{cycles}:")
                
                # Memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    vram_before = torch.cuda.memory_allocated() / 1024**3
                    print(f"  VRAM before: {vram_before:.2f} GB")
                
                # Create and load
                print("  Creating analyzer...", end="", flush=True)
                start = time.time()
                
                try:
                    analyzer = SmartAnalyzer(florence_model_type=model_type)
                    print(" ✓")
                    
                    print("  Loading model...", end="", flush=True)
                    analyzer._ensure_florence()
                    load_time = time.time() - start
                    print(f" ✓ ({load_time:.1f}s)")
                    
                    # Check loaded
                    if hasattr(analyzer.florence, 'model') and analyzer.florence.model is not None:
                        print("  Model loaded: ✓")
                        print(f"  Model class: {type(analyzer.florence.model).__name__}")
                    else:
                        print("  Model loaded: ✗")
                        if hasattr(analyzer.florence, '_initialized'):
                            print(f"  Initialized flag: {analyzer.florence._initialized}")
                        if hasattr(analyzer.florence, '_init_error'):
                            print(f"  Init error: {analyzer.florence._init_error}")
                    
                    # Memory after load
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        vram_after = torch.cuda.memory_allocated() / 1024**3
                        print(f"  VRAM after load: {vram_after:.2f} GB (+{vram_after - vram_before:.2f} GB)")
                    
                    # Unload
                    print("  Unloading...", end="", flush=True)
                    analyzer.unload_models()
                    print(" ✓")
                    
                    # Cleanup
                    del analyzer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        vram_final = torch.cuda.memory_allocated() / 1024**3
                        print(f"  VRAM after unload: {vram_final:.2f} GB")
                    
                except Exception as e:
                    print(f" ✗\n  Error: {str(e)[:100]}...")
                    if 'analyzer' in locals():
                        try:
                            analyzer.unload_models()
                            del analyzer
                        except:
                            pass
                
                # Wait between cycles
                if cycle < cycles - 1:
                    print("  Waiting 3 seconds...")
                    time.sleep(3)
        
        print("\n" + "="*60)
        print("✅ TEST COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()