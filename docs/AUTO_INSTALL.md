# Automatic Dependencies Installation

## How it works

The extension now automatically attempts to install optional dependencies that can improve performance:

### 1. **On Extension Load**
When the extension is first loaded by Forge, it runs `install.py` which:
- Checks if `liger-kernel` is installed
- If not, attempts to install it automatically
- Installation is done quietly to not interrupt Forge startup

### 2. **On JoyCaption First Use**
If `liger-kernel` wasn't installed during startup, JoyCaption will:
- Try to install it when first initialized
- This happens only once per session
- Installation timeout is 30 seconds to prevent hanging

### 3. **Manual Installation**
You can also manually install dependencies:
```bash
# From extension directory
python install.py

# Or directly with pip
pip install liger-kernel
```

## Benefits

- **No manual setup required** - Dependencies install automatically
- **Faster JoyCaption** - Liger Kernel can speed up inference by 10-30%
- **Graceful fallback** - If installation fails, JoyCaption still works normally

## Troubleshooting

If automatic installation fails:
1. Check internet connection
2. Check pip permissions
3. Install manually: `pip install liger-kernel`

The extension will work without Liger Kernel, just slightly slower.