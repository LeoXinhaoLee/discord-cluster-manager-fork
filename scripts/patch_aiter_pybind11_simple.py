#!/usr/bin/env python3
"""
Patch aiter to handle missing PyTorch pybind11 attributes.
This fixes the AttributeError when PyTorch doesn't have _PYBIND11_COMPILER_TYPE etc.
"""

import sys
import re

AITER_SOURCE = "/home/runner/aiter"
CPP_EXT_FILE = f"{AITER_SOURCE}/aiter/jit/utils/cpp_extension.py"

def patch_file():
    try:
        with open(CPP_EXT_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: Could not find {CPP_EXT_FILE}")
        return False
    
    # Check if already patched
    content = ''.join(lines)
    if "# PATCHED: Handle missing PyTorch pybind11 attributes" in content:
        print("File already patched, skipping...")
        return True
    
    # Find the line with the problematic getattr call
    new_lines = []
    patched = False
    
    for i, line in enumerate(lines):
        # Look for the problematic line
        if 'pval = getattr(torch._C, f"_PYBIND11_{pname}")' in line:
            # Get the indentation
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Insert try-except block
            new_lines.append(f'{indent_str}# PATCHED: Handle missing PyTorch pybind11 attributes\n')
            new_lines.append(f'{indent_str}try:\n')
            new_lines.append(f'{indent_str}    pval = getattr(torch._C, f"_PYBIND11_{{pname}}")\n')
            new_lines.append(f'{indent_str}except AttributeError:\n')
            new_lines.append(f'{indent_str}    # PyTorch ROCm builds may not have these attributes\n')
            new_lines.append(f'{indent_str}    # Use default values compatible with pybind11\n')
            new_lines.append(f'{indent_str}    if pname == "COMPILER_TYPE":\n')
            new_lines.append(f'{indent_str}        pval = "gcc"\n')
            new_lines.append(f'{indent_str}    elif pname == "STDLIB":\n')
            new_lines.append(f'{indent_str}        pval = "libstdc++"\n')
            new_lines.append(f'{indent_str}    elif pname == "BUILD_ABI":\n')
            new_lines.append(f'{indent_str}        pval = "cxx11abi"\n')
            new_lines.append(f'{indent_str}    else:\n')
            new_lines.append(f'{indent_str}        pval = ""\n')
            patched = True
        else:
            new_lines.append(line)
    
    if not patched:
        print("ERROR: Could not find the line to patch")
        print("Searching for _get_pybind11_abi_build_flags function...")
        # Try to find the function
        in_function = False
        for i, line in enumerate(lines):
            if 'def _get_pybind11_abi_build_flags' in line:
                in_function = True
                print(f"Found function at line {i+1}")
            if in_function and 'pval = getattr' in line:
                print(f"Found getattr at line {i+1}: {line.strip()}")
        return False
    
    # Backup and write
    import shutil
    shutil.copy(CPP_EXT_FILE, f"{CPP_EXT_FILE}.bak")
    
    with open(CPP_EXT_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Successfully patched {CPP_EXT_FILE}")
    print(f"  Backup saved to {CPP_EXT_FILE}.bak")
    return True

if __name__ == "__main__":
    if patch_file():
        sys.exit(0)
    else:
        sys.exit(1)
