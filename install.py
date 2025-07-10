"""
Automatic dependency installation script for Forge Kontext Assistant
This script ensures all required dependencies are installed when the extension loads
"""

import subprocess
import sys
import importlib.util
import importlib.metadata
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[Kontext Assistant] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check and install missing dependencies from requirements.txt"""
    extension_dir = Path(__file__).parent
    requirements_file = extension_dir / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    # Read requirements
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Separate required and optional packages
    required_packages = []
    optional_packages = []
    
    for req in requirements:
        if '# Optional' in req:
            optional_packages.append(req.split('#')[0].strip())
        elif req and not req.startswith('pytest'):  # Skip test dependencies
            required_packages.append(req)
    
    # Check and install required packages
    missing_required = []
    for package_spec in required_packages:
        package_name = package_spec.split('>=')[0].split('==')[0].strip()
        if not is_package_installed(package_name):
            missing_required.append(package_spec)
    
    if missing_required:
        logger.info(f"Installing missing required packages: {', '.join(missing_required)}")
        for package in missing_required:
            if not install_package(package):
                logger.error(f"Failed to install required package: {package}")
                return False
    
    # Check and install optional packages
    for package_spec in optional_packages:
        package_name = package_spec.split('>=')[0].split('==')[0].strip()
        import_name = get_import_name(package_name)
        
        if not is_package_installed(import_name):
            logger.info(f"Optional package {package_name} not found, attempting to install...")
            if install_package(package_spec):
                logger.info(f"{package_name} installed successfully!")
            else:
                logger.info(f"{package_name} installation failed - will run without this optimization")
    
    return True

def is_package_installed(package_name):
    """Check if a package is installed"""
    try:
        # First try to import it
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            return True
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Also check using metadata (for packages with different import names)
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        pass
    
    return False

def get_import_name(package_name):
    """Get the import name for a package (may differ from package name)"""
    # Known mappings
    mappings = {
        'liger-kernel': 'liger_kernel',
        'pillow': 'PIL',
    }
    return mappings.get(package_name.lower(), package_name.replace('-', '_'))

def install_package(package_spec):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_spec}...")
        
        # Use subprocess to install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"{package_spec} installed successfully")
            return True
        else:
            logger.error(f"Failed to install {package_spec}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Installation of {package_spec} timed out")
        return False
    except Exception as e:
        logger.error(f"Error installing {package_spec}: {e}")
        return False

def verify_core_dependencies():
    """Verify that core dependencies are available"""
    core_deps = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'gradio': 'Gradio'
    }
    
    all_available = True
    for import_name, display_name in core_deps.items():
        if not is_package_installed(import_name):
            logger.error(f"{display_name} is not installed! This is a core dependency.")
            all_available = False
    
    return all_available

def main():
    """Main installation function"""
    logger.info("Checking Kontext Assistant dependencies...")
    
    # First verify core dependencies
    if not verify_core_dependencies():
        logger.error("Core dependencies are missing! Please install them manually or use the WebUI's extension installer.")
        return False
    
    # Then check and install any missing dependencies
    if check_and_install_dependencies():
        logger.info("All dependencies are satisfied!")
        return True
    else:
        logger.error("Some dependencies could not be installed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)