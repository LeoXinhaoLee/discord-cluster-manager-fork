# RunPod Setup Using Existing Dockerfile

**Much simpler approach!** Use the existing `docker/amd-docker.Dockerfile` from this repository.

## Why Use the Dockerfile?

✅ **Already configured** - The Dockerfile has everything set up  
✅ **Consistent** - Matches the exact environment used in the repo  
✅ **Maintainable** - Updates to Dockerfile automatically apply  
✅ **Isolated** - Runner runs in a container, easier to manage  

## Quick Start

### 1. Create RunPod Instance

1. Go to [RunPod](https://www.runpod.io)
2. Select **AMD MI300X** GPU instance
3. Choose **On-Demand** or **Spot**
4. Select **Ubuntu 22.04** base image
5. Create the pod

### 2. Connect and Setup

```bash
# SSH into your RunPod instance
ssh root@<runpod-ip>

# Clone this repository (or upload the files)
git clone https://github.com/YOUR_ORG/YOUR_REPO.git /opt/discord-cluster-manager

# Run the setup script
cd /opt/discord-cluster-manager
sudo bash scripts/setup-runpod-docker.sh
```

### 3. Build the Docker Image

The setup script will build the image, or do it manually:

```bash
cd /opt/discord-cluster-manager
docker build -f docker/amd-docker.Dockerfile -t amd-runner:latest .
```

**Note:** This build will take 30-60 minutes (compiling UCX, OpenMPI, rocSHMEM).

### 4. Run the Runner Container

#### Option A: Use the convenience script

```bash
# Get your GitHub registration token from:
# Repository → Settings → Actions → Runners → New self-hosted runner

# Run the script
GITHUB_TOKEN="your_token_here" \
GITHUB_REPO="https://github.com/YOUR_ORG/YOUR_REPO" \
RUNNER_NAME="runpod-mi300-x86-64" \
/usr/local/bin/start-github-runner.sh
```

#### Option B: Run manually

```bash
# Create work directory
mkdir -p /opt/runner-work

# Run container
docker run -d \
    --name github-runner-amd \
    --restart unless-stopped \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    -v /opt/runner-work:/home/runner/_work \
    amd-runner:latest \
    /bin/bash -c 'cd /home/runner && \
        ./config.sh \
            --url https://github.com/YOUR_ORG/YOUR_REPO \
            --token YOUR_TOKEN \
            --name runpod-mi300-x86-64 \
            --labels runpod,amd,mi300,self-hosted \
            --work _work && \
        ./run.sh'
```

### 5. Verify

```bash
# Check container is running
docker ps

# Check runner logs
docker logs -f github-runner-amd

# Verify in GitHub
# Repository → Settings → Actions → Runners
# You should see "runpod-mi300-x86-64" as online
```

## What's Included in the Docker Image?

The Dockerfile (`docker/amd-docker.Dockerfile`) includes:

- ✅ GitHub Actions runner (from `ghcr.io/actions/actions-runner:latest`)
- ✅ AMD GPU drivers and ROCm 6.3
- ✅ PyTorch 2.10.0.dev20250916+rocm6.3 (nightly build)
- ✅ aiter (ROCm AI iteration library)
- ✅ iris (ROCm IRIS library)
- ✅ UCX (Unified Communication X)
- ✅ OpenMPI (Message Passing Interface)
- ✅ rocSHMEM (ROCm SHMEM library)
- ✅ All Python dependencies (numpy, ninja, packaging, wheel, tinygrad)

## Managing the Runner

### View Logs

```bash
docker logs -f github-runner-amd
```

### Stop Runner

```bash
docker stop github-runner-amd
```

### Start Runner (if stopped)

```bash
docker start github-runner-amd
```

### Remove Runner

```bash
docker stop github-runner-amd
docker rm github-runner-amd
```

### Update Runner (rebuild image)

```bash
cd /opt/discord-cluster-manager
git pull  # Get latest Dockerfile changes
docker build -f docker/amd-docker.Dockerfile -t amd-runner:latest .
docker stop github-runner-amd
docker rm github-runner-amd
# Then restart using the commands above
```

## GPU Access in Container

The container needs access to GPU devices. The setup uses:

- `--device=/dev/dri` - Direct access to GPU devices
- `--group-add video` - Video group for GPU access
- `--group-add render` - Render group for GPU access

If GPU access doesn't work, verify:

```bash
# Check devices exist
ls -la /dev/dri/

# Test in container
docker exec github-runner-amd rocm-smi
```

## Multiple Runners

To run multiple concurrent jobs, start multiple containers:

```bash
# Runner 1
docker run -d \
    --name github-runner-amd-1 \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    -v /opt/runner-work-1:/home/runner/_work \
    amd-runner:latest \
    /bin/bash -c 'cd /home/runner && ./config.sh --url YOUR_REPO --token TOKEN --name runpod-mi300-1 && ./run.sh'

# Runner 2
docker run -d \
    --name github-runner-amd-2 \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    -v /opt/runner-work-2:/home/runner/_work \
    amd-runner:latest \
    /bin/bash -c 'cd /home/runner && ./config.sh --url YOUR_REPO --token TOKEN --name runpod-mi300-2 && ./run.sh'
```

Then use labels in workflows:

```yaml
runs-on: [self-hosted, mi300]
```

## Persistence

RunPod instances are ephemeral. To persist:

1. **Save the built image**: Push to a registry
   ```bash
   docker tag amd-runner:latest your-registry/amd-runner:latest
   docker push your-registry/amd-runner:latest
   ```

2. **Use RunPod Network Storage**: Mount persistent storage
   ```bash
   docker run ... -v /mnt/persistent/runner:/home/runner/_work ...
   ```

3. **Create RunPod Template**: Save the instance as a template for quick recreation

## Troubleshooting

### Container won't start

```bash
# Check Docker logs
docker logs github-runner-amd

# Check if image exists
docker images | grep amd-runner

# Rebuild if needed
docker build -f docker/amd-docker.Dockerfile -t amd-runner:latest .
```

### GPU not accessible

```bash
# Test GPU access on host
rocm-smi

# Test in container
docker exec github-runner-amd rocm-smi

# If fails, check device permissions
ls -la /dev/dri/
```

### Runner not connecting to GitHub

```bash
# Check network connectivity
docker exec github-runner-amd curl -I https://github.com

# Verify token is correct
# Get new token from GitHub if needed
```

### Build takes too long

The first build compiles UCX, OpenMPI, and rocSHMEM which takes 30-60 minutes. This is normal. Subsequent builds are faster if you use Docker layer caching.

## Advantages Over Manual Setup

| Aspect | Docker Approach | Manual Setup |
|--------|----------------|--------------|
| Setup Time | ~1 hour (mostly build) | ~1 hour (manual steps) |
| Consistency | ✅ Exact match to repo | ⚠️ Manual errors possible |
| Updates | ✅ Rebuild image | ⚠️ Manual updates needed |
| Isolation | ✅ Container isolation | ❌ Direct on host |
| Cleanup | ✅ Remove container | ⚠️ Manual cleanup |
| Portability | ✅ Works anywhere | ❌ Host-specific |

## Next Steps

1. Build the image (one-time, takes ~1 hour)
2. Start the runner container
3. Verify it appears in GitHub
4. Test with a workflow
5. Set up auto-start if using on-demand instances

## Cost Optimization

- **Build once, reuse**: The Docker image can be saved and reused
- **Use spot instances**: For non-critical workloads
- **Auto-start/stop**: Only run when jobs are queued
- **Push image to registry**: Avoid rebuilding on new instances
