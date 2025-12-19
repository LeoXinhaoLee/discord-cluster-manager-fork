#!/bin/bash

# Mounts your current directory to /workspace
# Runs as root for full access without sudo
# Includes GPU device access for ROCm
#
# If GPU still doesn't work, try privileged mode by replacing the docker run command with:
# docker run -it --rm --privileged --name amd-runner-test ...

# Check if devices exist
if [ ! -d "/dev/dri" ]; then
    echo "Warning: /dev/dri not found on host"
fi

if [ ! -e "/dev/kfd" ]; then
    echo "Warning: /dev/kfd not found on host"
fi

docker run -it --rm \
    --name amd-runner-test \
    --device=/dev/dri \
    --device=/dev/kfd \
    --group-add video \
    --group-add render \
    --ipc=host \
    --security-opt seccomp=unconfined \
    -v $(pwd):/workspace \
    -e ROCM_PATH=/opt/rocm \
    -e PATH="/opt/openmpi/bin:${PATH}" \
    -e LD_LIBRARY_PATH="/opt/openmpi/lib:/opt/ucx/lib:/opt/rocm/lib:/opt/rocshmem/lib" \
    -u root \
    amd-runner:latest \
    /bin/bash -c "cd /workspace && exec /bin/bash"

