#!/bin/bash
# Setup script for RunPod AMD MI300 instance to match repository environment
# This script replicates the exact environment from docker/amd-docker.Dockerfile

set -e  # Exit on any error

echo "=========================================="
echo "Setting up RunPod instance for GitHub Actions"
echo "Matching repository environment exactly"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

# ==========================================
# Step 1: System Dependencies
# ==========================================
print_step "Installing system dependencies..."

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:git-core/ppa
apt-get update -y

apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    jq \
    sudo \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    python3.10-venv \
    autoconf \
    automake \
    libtool \
    pkg-config \
    build-essential \
    gfortran \
    flex \
    bison \
    libomp-dev \
    libhwloc-dev \
    libnuma-dev \
    libicu-dev

apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python-is-python3 \
    python3-setuptools \
    python3-wheel \
    libpython3.10

# Install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

print_step "System dependencies installed"

# ==========================================
# Step 2: Setup user groups (for GPU access)
# ==========================================
print_step "Setting up user groups for GPU access..."

groupadd -g 109 render || true
# Note: We'll add the runner user to these groups after creating it

# ==========================================
# Step 3: Install AMD GPU Drivers and ROCm
# ==========================================
print_step "Installing AMD GPU drivers and ROCm..."

# Install kernel headers (needed for GPU drivers)
KERNEL_VERSION=$(uname -r)
apt-get update -y
apt-get install -y "linux-headers-${KERNEL_VERSION}" "linux-modules-extra-${KERNEL_VERSION}"

# Download and install AMD GPU installer
cd /tmp
wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/jammy/amdgpu-install_6.3.60301-1_all.deb
apt-get install -y ./amdgpu-install_6.3.60301-1_all.deb
apt-get update -y
apt-get install -y rocm

# Verify ROCm installation
if command -v rocm-smi &> /dev/null; then
    print_step "ROCm installed successfully"
    rocm-smi || print_warning "rocm-smi command available but may need GPU access"
else
    print_error "ROCm installation may have failed"
    exit 1
fi

# ==========================================
# Step 4: Set Environment Variables
# ==========================================
print_step "Setting up environment variables..."

export CXX=clang++
export UCX_CXX=g++
export UCX_CC=gcc
export UCX_INSTALL_DIR=/opt/ucx
export OMPI_INSTALL_DIR=/opt/openmpi
export ROCSHMEM_INSTALL_DIR=/opt/rocshmem
export ROCM_PATH=/opt/rocm

# Add to profile for persistence
cat >> /etc/environment <<EOF
CXX=clang++
UCX_CXX=g++
UCX_CC=gcc
UCX_INSTALL_DIR=/opt/ucx
OMPI_INSTALL_DIR=/opt/openmpi
ROCSHMEM_INSTALL_DIR=/opt/rocshmem
ROCM_PATH=/opt/rocm
EOF

# ==========================================
# Step 5: Install PyTorch with ROCm
# ==========================================
print_step "Installing PyTorch with ROCm (exact version from Dockerfile)..."

pip install --upgrade pip
pip install --no-cache-dir \
    torch==2.10.0.dev20250916+rocm6.3 \
    pytorch-triton-rocm \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# Install additional Python packages
pip install \
    ninja \
    numpy \
    packaging \
    wheel \
    tinygrad

print_step "PyTorch installed"

# ==========================================
# Step 6: Install aiter
# ==========================================
print_step "Installing aiter..."

cd /tmp
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git checkout 1d88633958236e942cba3c283864282f7af3ebc5
pip install -r requirements.txt
python3 setup.py develop

# Create build directory with proper permissions
mkdir -p /root/aiter/aiter/jit/build
chown -R root:root /root/aiter/aiter/jit/build

cd /
rm -rf /tmp/aiter

print_step "aiter installed"

# ==========================================
# Step 7: Install iris
# ==========================================
print_step "Installing iris..."

pip install git+https://github.com/ROCm/iris.git

print_step "iris installed"

# ==========================================
# Step 8: Build UCX (Unified Communication X)
# ==========================================
print_step "Building UCX (this may take a while)..."

cd /tmp
git clone https://github.com/openucx/ucx.git -b v1.17.x
cd ucx
./autogen.sh
CC=gcc CXX=g++ ./configure \
    --prefix=${UCX_INSTALL_DIR} \
    --with-rocm=${ROCM_PATH} \
    --enable-mt \
    --disable-optimizations

make -j$(nproc)
make install

cd /
rm -rf /tmp/ucx

print_step "UCX built and installed"

# ==========================================
# Step 9: Build OpenMPI
# ==========================================
print_step "Building OpenMPI (this may take a while)..."

cd /tmp
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi
./autogen.pl
./configure \
    --prefix=${OMPI_INSTALL_DIR} \
    --with-rocm=${ROCM_PATH} \
    --with-ucx=${UCX_INSTALL_DIR}

make -j$(nproc)
make install

cd /
rm -rf /tmp/ompi

print_step "OpenMPI built and installed"

# ==========================================
# Step 10: Build rocSHMEM
# ==========================================
print_step "Building rocSHMEM (this may take a while)..."

cd /tmp
git clone https://github.com/ROCm/rocSHMEM.git
cd rocSHMEM
mkdir build
cd build

MPI_ROOT=${OMPI_INSTALL_DIR} \
UCX_ROOT=${UCX_INSTALL_DIR} \
CMAKE_PREFIX_PATH="${ROCM_PATH}:${CMAKE_PREFIX_PATH}" \
    ../scripts/build_configs/ipc_single -DCMAKE_INSTALL_PREFIX=${ROCSHMEM_INSTALL_DIR}

cd /
rm -rf /tmp/rocSHMEM

print_step "rocSHMEM built and installed"

# ==========================================
# Step 11: Update PATH and LD_LIBRARY_PATH
# ==========================================
print_step "Updating PATH and LD_LIBRARY_PATH..."

export PATH="${OMPI_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib:${ROCSHMEM_INSTALL_DIR}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# Add to profile
cat >> /etc/environment <<EOF
PATH="${OMPI_INSTALL_DIR}/bin:\${PATH}"
LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib:${ROCSHMEM_INSTALL_DIR}/lib:${ROCM_PATH}/lib:\${LD_LIBRARY_PATH}"
EOF

# Also add to bashrc for interactive sessions
cat >> /root/.bashrc <<EOF
export PATH="${OMPI_INSTALL_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib:${ROCSHMEM_INSTALL_DIR}/lib:${ROCM_PATH}/lib:\${LD_LIBRARY_PATH}"
EOF

print_step "Environment variables updated"

# ==========================================
# Step 12: Install GitHub Actions Runner
# ==========================================
print_step "Installing GitHub Actions Runner..."

# Create runner user (if it doesn't exist)
if ! id -u runner &>/dev/null; then
    useradd -m -s /bin/bash runner
    usermod -a -G render,video runner
    print_step "Created 'runner' user"
else
    usermod -a -G render,video runner
    print_step "Updated 'runner' user groups"
fi

# Install runner in runner's home directory
RUNNER_DIR=/home/runner/actions-runner
mkdir -p ${RUNNER_DIR}
cd ${RUNNER_DIR}

# Download runner
RUNNER_VERSION="2.311.0"
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L \
    https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Set ownership
chown -R runner:runner ${RUNNER_DIR}

print_step "GitHub Actions Runner downloaded"

# ==========================================
# Step 13: Verification
# ==========================================
print_step "Verifying installation..."

echo ""
echo "=== ROCm Version ==="
rocm-smi --version || rocminfo --version || echo "ROCm version check (may need GPU)"

echo ""
echo "=== Python Version ==="
python3 --version

echo ""
echo "=== PyTorch Installation ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || print_warning "PyTorch import failed (may need GPU)"

echo ""
echo "=== Environment Variables ==="
echo "ROCM_PATH: ${ROCM_PATH}"
echo "OMPI_INSTALL_DIR: ${OMPI_INSTALL_DIR}"
echo "UCX_INSTALL_DIR: ${UCX_INSTALL_DIR}"

# ==========================================
# Step 14: Final Instructions
# ==========================================
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Switch to the runner user:"
echo "   sudo su - runner"
echo ""
echo "2. Navigate to runner directory:"
echo "   cd ~/actions-runner"
echo ""
echo "3. Configure the runner with your GitHub token:"
echo "   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \\"
echo "               --token YOUR_REGISTRATION_TOKEN \\"
echo "               --name runpod-mi300-x86-64 \\"
echo "               --labels runpod,amd,mi300,self-hosted \\"
echo "               --work _work"
echo ""
echo "4. Install as a service (optional, to keep it running):"
echo "   sudo ./svc.sh install"
echo "   sudo ./svc.sh start"
echo ""
echo "5. Verify the runner appears in GitHub:"
echo "   Repository → Settings → Actions → Runners"
echo ""
echo "=========================================="
echo ""
print_warning "Note: GPU access will be available once the runner is configured"
print_warning "The runner user has been added to 'render' and 'video' groups for GPU access"
echo ""
