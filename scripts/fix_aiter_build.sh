#!/bin/bash
# Fix aiter JIT module build directory
# Run this inside the Docker container as root

echo "=== Fixing aiter JIT build directory ==="

# Try to find aiter via Python import first
AITER_PATH=$(python3 -c "import aiter; import os; print(os.path.dirname(aiter.__file__))" 2>/dev/null)

# If that fails, check for editable install (egg-link)
if [ -z "$AITER_PATH" ] || [ ! -d "$AITER_PATH" ]; then
    echo "Checking for editable install (egg-link)..."
    EGG_LINK=$(find /usr/local/lib/python3.*/dist-packages -name "aiter.egg-link" 2>/dev/null | head -1)
    if [ -n "$EGG_LINK" ] && [ -f "$EGG_LINK" ]; then
        AITER_SOURCE=$(cat "$EGG_LINK" | head -1)
        echo "Found egg-link pointing to: $AITER_SOURCE"
        if [ -d "$AITER_SOURCE" ]; then
            AITER_PATH="$AITER_SOURCE/aiter"
            echo "Using source path: $AITER_PATH"
        fi
    fi
fi

# If still not found, check common locations
if [ -z "$AITER_PATH" ] || [ ! -d "$AITER_PATH" ]; then
    echo "Checking common source locations..."
    for CHECK_PATH in "/aiter/aiter" "/home/runner/aiter/aiter" "/root/aiter/aiter"; do
        if [ -d "$CHECK_PATH" ]; then
            AITER_PATH="$CHECK_PATH"
            echo "Found aiter at: $AITER_PATH"
            break
        fi
    done
fi

if [ -z "$AITER_PATH" ] || [ ! -d "$AITER_PATH" ]; then
    echo "ERROR: Could not find aiter installation"
    echo "Searched in: /aiter, /home/runner/aiter, /root/aiter"
    exit 1
fi

echo "Found aiter at: $AITER_PATH"

# Create build directory in the aiter package
BUILD_DIR="$AITER_PATH/jit/build"
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
chmod 777 "$BUILD_DIR"  # Make it writable by anyone

# Verify jit directory exists
if [ ! -d "$AITER_PATH/jit" ]; then
    echo "WARNING: jit directory not found at $AITER_PATH/jit"
    echo "Contents of $AITER_PATH:"
    ls -la "$AITER_PATH" || true
fi

# Also create build directories in other potential locations
for ALT_PATH in "/aiter/aiter" "/home/runner/aiter/aiter" "/root/aiter/aiter"; do
    if [ -d "$ALT_PATH" ] && [ "$ALT_PATH" != "$AITER_PATH" ]; then
        echo "Also fixing $ALT_PATH build directory"
        mkdir -p "$ALT_PATH/jit/build"
        chmod 777 "$ALT_PATH/jit/build"
    fi
done

# Set environment variable to help aiter find the build directory
export AITER_BUILD_DIR="$BUILD_DIR"

# Patch aiter for PyTorch pybind11 compatibility
echo ""
echo "Applying PyTorch pybind11 compatibility patch..."
if [ -f "/workspace/scripts/patch_aiter_pybind11_simple.py" ]; then
    python3 /workspace/scripts/patch_aiter_pybind11_simple.py
    if [ $? -eq 0 ]; then
        echo "✓ Patch applied successfully"
    else
        echo "⚠ Patch failed, but continuing..."
    fi
else
    echo "⚠ Patch script not found, skipping..."
fi

# Try to trigger a build by importing aiter and its JIT modules
echo ""
echo "Testing aiter import and JIT module build..."
python3 << 'PYTHON_EOF'
import os
import sys

try:
    import aiter
    print(f"✓ aiter imported successfully from: {aiter.__file__}")
    
    # Check if jit directory exists
    aiter_dir = os.path.dirname(aiter.__file__)
    jit_dir = os.path.join(aiter_dir, "jit")
    build_dir = os.path.join(jit_dir, "build")
    
    print(f"  aiter directory: {aiter_dir}")
    print(f"  jit directory: {jit_dir} (exists: {os.path.exists(jit_dir)})")
    print(f"  build directory: {build_dir} (exists: {os.path.exists(build_dir)})")
    
    if os.path.exists(build_dir):
        contents = os.listdir(build_dir)
        print(f"  build directory contents: {contents if contents else '(empty)'}")
    else:
        print(f"  WARNING: Build directory does not exist! Creating it...")
        os.makedirs(build_dir, exist_ok=True)
        os.chmod(build_dir, 0o777)
    
    # Try to import JIT core
    print("\n  Attempting to import aiter.jit.core...")
    try:
        from aiter.jit import core
        print("✓ aiter.jit.core imported successfully")
    except Exception as e:
        print(f"✗ Could not import aiter.jit.core: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Try to trigger module_aiter_enum build by calling get_module
    print("\n  Attempting to build module_aiter_enum...")
    try:
        from aiter.jit.core import get_module
        # This should trigger the build of module_aiter_enum
        module = get_module("module_aiter_enum")
        print("✓ module_aiter_enum built and imported successfully")
    except Exception as e:
        print(f"✗ Could not build/import module_aiter_enum: {e}")
        print("  This is the module that was failing. Error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"✗ ERROR importing aiter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All aiter JIT modules are working!")
PYTHON_EOF

echo "=== Done ==="
