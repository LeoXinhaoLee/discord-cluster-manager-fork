#!/bin/bash
# Setup script for RunPod using the existing Dockerfile
# This uses docker/amd-docker.Dockerfile from the repository

set -e

echo "=========================================="
echo "Setting up RunPod with existing Dockerfile"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# ==========================================
# Step 1: Install Docker (if not already installed)
# ==========================================
print_step "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    print_step "Installing Docker..."
    apt-get update -y
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    print_step "Docker installed"
else
    print_step "Docker already installed"
fi

# Ensure Docker daemon is running
print_step "Starting Docker daemon..."

# Stop any existing Docker daemon
pkill dockerd 2>/dev/null || true
sleep 1

# Backup existing daemon.json if it exists (to avoid conflicts with --iptables flag)
if [ -f /etc/docker/daemon.json ]; then
    print_step "Backing up existing daemon.json to avoid conflicts"
    cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    rm /etc/docker/daemon.json
fi

# Start dockerd directly with iptables disabled (fixes nf_tables permission issues)
if command -v dockerd &> /dev/null; then
    dockerd --iptables=false > /tmp/dockerd.log 2>&1 &
    sleep 3
else
    echo "Error: dockerd not found"
    exit 1
fi

# Wait for Docker to be ready with retries
print_step "Waiting for Docker daemon to be ready..."
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker info > /dev/null 2>&1; then
        print_step "Docker daemon is running"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        sleep 1
    fi
done

# Final check
if ! docker info > /dev/null 2>&1; then
    echo "Error: Cannot connect to Docker daemon."
    echo ""
    echo "Docker daemon log (last 30 lines):"
    tail -30 /tmp/dockerd.log 2>&1 || echo "Could not read log"
    echo ""
    echo "Docker process: $(pgrep -a dockerd || echo 'not running')"
    exit 1
fi

# ==========================================
# Step 2: Install NVIDIA Container Toolkit (for GPU access in containers)
# Actually, for AMD, we need to check if RunPod supports GPU passthrough
# ==========================================
print_step "Checking GPU access setup..."

# For AMD GPUs, we need to ensure devices are accessible
# RunPod should already have this configured, but we'll verify
if [ -d "/dev/dri" ]; then
    print_step "GPU devices found at /dev/dri"
    ls -la /dev/dri/
else
    echo "Warning: /dev/dri not found. GPU access may not work."
fi

# ==========================================
# Step 3: Verify Dockerfile exists
# ==========================================
print_step "Checking for Dockerfile..."

if [ ! -f "docker/amd-docker.Dockerfile" ]; then
    echo "Error: docker/amd-docker.Dockerfile not found in current directory"
    echo "Please run this script from the repository root directory"
    exit 1
fi

print_step "Dockerfile found"

# ==========================================
# Step 4: Build the Docker image
# ==========================================
print_step "Building Docker image from docker/amd-docker.Dockerfile..."

# Build the image
docker build \
    -f docker/amd-docker.Dockerfile \
    -t amd-runner:latest \
    .

print_step "Docker image built successfully"

# ==========================================
# Step 5: Create a script to run the container with runner
# ==========================================
print_step "Creating runner container script..."

cat > /usr/local/bin/start-github-runner.sh <<'EOF'
#!/bin/bash
# Script to start GitHub Actions runner in Docker container

CONTAINER_NAME="github-runner-amd"
IMAGE_NAME="amd-runner:latest"

# Check if container already exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME already exists"
    echo "Stopping and removing..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# Get registration token from user if not provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Please provide GitHub registration token:"
    echo "Get it from: Repository → Settings → Actions → Runners → New self-hosted runner"
    read -p "Token: " GITHUB_TOKEN
fi

if [ -z "$GITHUB_REPO" ]; then
    echo "Please provide GitHub repository URL (e.g., https://github.com/org/repo):"
    read -p "Repo URL: " GITHUB_REPO
fi

if [ -z "$RUNNER_NAME" ]; then
    RUNNER_NAME="runpod-mi300-x86-64"
    echo "Using default runner name: $RUNNER_NAME"
    echo "To customize, set RUNNER_NAME environment variable"
fi

# Create directory for runner work
RUNNER_WORK_DIR="/opt/runner-work"
mkdir -p $RUNNER_WORK_DIR

# Start container with GPU access
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    -v $RUNNER_WORK_DIR:/home/runner/_work \
    -e RUNNER_TOKEN="$GITHUB_TOKEN" \
    -e RUNNER_REPO="$GITHUB_REPO" \
    -e RUNNER_NAME="$RUNNER_NAME" \
    $IMAGE_NAME \
    /bin/bash -c "cd /home/runner && ./config.sh --url $GITHUB_REPO --token $GITHUB_TOKEN --name $RUNNER_NAME --work _work && ./run.sh"

echo "Runner container started!"
echo "Check status with: docker logs $CONTAINER_NAME"
echo "Stop with: docker stop $CONTAINER_NAME"
EOF

chmod +x /usr/local/bin/start-github-runner.sh

print_step "Runner script created at /usr/local/bin/start-github-runner.sh"

# ==========================================
# Step 6: Alternative: Manual container run instructions
# ==========================================
# echo ""
# echo "=========================================="
# echo -e "${GREEN}Setup Complete!${NC}"
# echo "=========================================="
# echo ""
# echo "You have two options to run the runner:"
# echo ""
# echo "Option 1: Use the convenience script"
# echo "  /usr/local/bin/start-github-runner.sh"
# echo ""
# echo "Option 2: Run manually"
# echo "  docker run -d \\"
# echo "    --name github-runner-amd \\"
# echo "    --restart unless-stopped \\"
# echo "    --device=/dev/dri \\"
# echo "    --group-add video \\"
# echo "    --group-add render \\"
# echo "    -v /opt/runner-work:/home/runner/_work \\"
# echo "    amd-runner:latest \\"
# echo "    /bin/bash -c 'cd /home/runner && ./config.sh --url YOUR_REPO_URL --token YOUR_TOKEN --name runpod-mi300-x86-64 --work _work && ./run.sh'"
# echo ""
# echo "To get your GitHub token:"
# echo "  Repository → Settings → Actions → Runners → New self-hosted runner"
# echo ""
# echo "To check runner logs:"
# echo "  docker logs -f github-runner-amd"
# echo ""
# echo "To stop the runner:"
# echo "  docker stop github-runner-amd"
# echo ""
