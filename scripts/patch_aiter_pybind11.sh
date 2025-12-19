#!/bin/bash
# Patch aiter to handle missing PyTorch pybind11 attributes
# Run this inside the Docker container as root

echo "=== Patching aiter for PyTorch pybind11 compatibility ==="

# Find aiter installation
AITER_SOURCE="/home/runner/aiter"
CPP_EXT_FILE="$AITER_SOURCE/aiter/jit/utils/cpp_extension.py"

if [ ! -f "$CPP_EXT_FILE" ]; then
    echo "ERROR: Could not find $CPP_EXT_FILE"
    exit 1
fi

echo "Found aiter cpp_extension.py at: $CPP_EXT_FILE"

# Backup the original file
cp "$CPP_EXT_FILE" "${CPP_EXT_FILE}.bak"
echo "Backed up original file to ${CPP_EXT_FILE}.bak"

# Check if already patched
if grep -q "# PATCHED: Handle missing PyTorch pybind11 attributes" "$CPP_EXT_FILE"; then
    echo "File already patched, skipping..."
    exit 0
fi

# Find the _get_pybind11_abi_build_flags function and patch it
python3 << 'PYTHON_PATCH'
import re
import sys

file_path = "/home/runner/aiter/aiter/jit/utils/cpp_extension.py"

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the _get_pybind11_abi_build_flags function
    # We need to patch the part that accesses torch._C attributes
    pattern = r'(def _get_pybind11_abi_build_flags\(\):.*?)(\s+pval = getattr\(torch\._C, f"_PYBIND11_\{pname\}"\))'
    
    replacement = r'''\1
        # PATCHED: Handle missing PyTorch pybind11 attributes
        try:
            pval = getattr(torch._C, f"_PYBIND11_{pname}")
        except AttributeError:
            # PyTorch ROCm builds may not have these attributes
            # Use default values compatible with pybind11
            if pname == "COMPILER_TYPE":
                pval = "gcc"  # Default compiler type
            elif pname == "STDLIB":
                pval = "libstdc++"  # Default stdlib
            elif pname == "BUILD_ABI":
                pval = "cxx11abi"  # Default ABI
            else:
                pval = ""  # Empty for unknown attributes
\2'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        # Try a different approach - find the exact line and replace it
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for the problematic line
            if 'pval = getattr(torch._C, f"_PYBIND11_{pname}")' in line:
                # Add try-except around it
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                new_lines.append(f'{indent_str}# PATCHED: Handle missing PyTorch pybind11 attributes')
                new_lines.append(f'{indent_str}try:')
                new_lines.append(f'{indent_str}    pval = getattr(torch._C, f"_PYBIND11_{{pname}}")')
                new_lines.append(f'{indent_str}except AttributeError:')
                new_lines.append(f'{indent_str}    # PyTorch ROCm builds may not have these attributes')
                new_lines.append(f'{indent_str}    # Use default values compatible with pybind11')
                new_lines.append(f'{indent_str}    if pname == "COMPILER_TYPE":')
                new_lines.append(f'{indent_str}        pval = "gcc"')
                new_lines.append(f'{indent_str}    elif pname == "STDLIB":')
                new_lines.append(f'{indent_str}        pval = "libstdc++"')
                new_lines.append(f'{indent_str}    elif pname == "BUILD_ABI":')
                new_lines.append(f'{indent_str}        pval = "cxx11abi"')
                new_lines.append(f'{indent_str}    else:')
                new_lines.append(f'{indent_str}        pval = ""')
            else:
                new_lines.append(line)
            i += 1
        
        new_content = '\n'.join(new_lines)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("✓ Successfully patched cpp_extension.py")
        sys.exit(0)
    else:
        print("✗ Could not find the code to patch")
        print("Looking for function _get_pybind11_abi_build_flags...")
        if '_get_pybind11_abi_build_flags' in content:
            print("Function found, but pattern matching failed")
            # Show the function
            import re
            match = re.search(r'def _get_pybind11_abi_build_flags\(\):.*?(?=\n\ndef|\nclass|\Z)', content, re.DOTALL)
            if match:
                print("Function content:")
                print(match.group(0)[:500])
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error patching file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_PATCH

if [ $? -eq 0 ]; then
    echo "=== Patch applied successfully ==="
else
    echo "=== Patch failed, trying manual patch ==="
    # Manual patch using sed as fallback
    sed -i.bak2 '/pval = getattr(torch\._C, f"_PYBIND11_{pname}")/i\
    # PATCHED: Handle missing PyTorch pybind11 attributes\
    try:\
        pval = getattr(torch._C, f"_PYBIND11_{pname}")\
    except AttributeError:\
        # PyTorch ROCm builds may not have these attributes\
        if pname == "COMPILER_TYPE":\
            pval = "gcc"\
        elif pname == "STDLIB":\
            pval = "libstdc++"\
        elif pname == "BUILD_ABI":\
            pval = "cxx11abi"\
        else:\
            pval = ""\
' "$CPP_EXT_FILE" 2>/dev/null || echo "Manual patch also failed - may need to edit file manually"
fi

echo "=== Done ==="
