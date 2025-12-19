#!/bin/bash
# Test script to verify ROCm GPU access inside container
# Run this inside the Docker container

echo "=== Checking GPU Devices ==="
ls -la /dev/dri/
echo ""
ls -la /dev/kfd 2>/dev/null || echo "/dev/kfd not accessible"
echo ""

echo "=== Checking ROCm Installation ==="
if [ -d "/opt/rocm" ]; then
    echo "ROCM_PATH: $ROCM_PATH"
    echo "ROCm binaries:"
    ls -la /opt/rocm/bin/ | head -5
else
    echo "ERROR: /opt/rocm not found"
fi
echo ""

echo "=== Testing rocm-smi ==="
if command -v rocm-smi &> /dev/null; then
    rocm-smi || echo "rocm-smi failed"
else
    /opt/rocm/bin/rocm-smi || echo "rocm-smi not found"
fi
echo ""

echo "=== Testing PyTorch GPU Detection ==="
python3 << 'PYTHON_EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ERROR: No CUDA/HIP devices detected")
    print("This is the issue you're experiencing")
PYTHON_EOF
