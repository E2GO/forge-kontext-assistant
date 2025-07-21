"""
Forge Kontext Assistant - AI-powered prompt generation for FLUX.1 Kontext
Version 1.0.1
"""

__version__ = "1.0.1"
__author__ = "E2GO"

# Auto-install optional dependencies on first import
try:
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(current_dir, 'install.py')
    
    # Check if install script exists and run it
    if os.path.exists(install_script):
        # Only run installation once per session
        if not hasattr(sys, '_kontext_assistant_installed'):
            sys._kontext_assistant_installed = True
            import subprocess
            subprocess.run([sys.executable, install_script], capture_output=True, text=True)
except Exception:
    # Don't break the extension if installation fails
    pass