# RunPod AMD MI300 GitHub Actions Runner Setup

This guide helps you set up a RunPod AMD MI300 instance as a GitHub Actions self-hosted runner with the **exact environment** matching this repository's Dockerfile.

## Quick Start

### 1. Create RunPod Instance

1. Go to [RunPod](https://www.runpod.io)
2. Select **AMD MI300X** GPU instance
3. Choose **On-Demand** (for persistent runner) or **Spot** (for cost savings)
4. Select **Ubuntu 22.04** as base image
5. Configure storage (at least 50GB recommended)
6. Create the pod

### 2. Connect to Instance

```bash
# SSH into your RunPod instance
ssh root@<runpod-ip-address>
```

### 3. Run Setup Script

```bash
# Download or copy the setup script to your instance
# Then run it:
sudo bash setup-runpod-runner.sh
```

**Note:** The script will take 30-60 minutes to complete (compiling UCX, OpenMPI, and rocSHMEM takes time).

### 4. Configure GitHub Actions Runner

After the setup script completes:

```bash
# Switch to runner user
sudo su - runner

# Navigate to runner directory
cd ~/actions-runner

# Get registration token from GitHub:
# Repository → Settings → Actions → Runners → New self-hosted runner

# Configure the runner
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \
            --token YOUR_REGISTRATION_TOKEN \
            --name runpod-mi300-x86-64 \
            --labels runpod,amd,mi300,self-hosted \
            --work _work
```

### 5. Install as Service (Optional)

To keep the runner running automatically:

```bash
# Still as runner user
sudo ./svc.sh install
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

### 6. Update Workflow

Update your workflow to use the new runner name:

```yaml
# .github/workflows/amd_workflow.yml
jobs:
  run:
    runs-on: runpod-mi300-x86-64  # Or use labels: [self-hosted, mi300]
```

Or update the runner name in your codebase:

```python
# src/libkernelbot/launchers/github.py
runner_name = {
    "MI300": "runpod-mi300-x86-64",  # Update this
    # ...
}
```

## Environment Details

The setup script installs **exactly** what's in `docker/amd-docker.Dockerfile`:

### System Packages
- Python 3.10
- Build tools (cmake, ninja, clang, gcc, etc.)
- Git, jq, curl, wget
- Development libraries

### AMD GPU Stack
- AMD GPU drivers (6.3.1)
- ROCm 6.3
- PyTorch 2.10.0.dev20250916+rocm6.3 (nightly build)
- pytorch-triton-rocm

### Additional Libraries
- **aiter**: ROCm AI iteration library
- **iris**: ROCm IRIS library
- **UCX**: Unified Communication X (v1.17.x)
- **OpenMPI**: Message Passing Interface (v5.0.x)
- **rocSHMEM**: ROCm SHMEM library

### Python Packages
- torch (ROCm version)
- numpy
- ninja
- packaging
- wheel
- tinygrad

## Verification

After setup, verify the environment:

```bash
# Check ROCm
rocm-smi

# Check PyTorch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check aiter
python3 -c "import aiter; print('aiter installed')"

# Check OpenMPI
mpirun --version
```

## Cost Considerations

- **On-Demand**: ~$2.99/hour = ~$71.76/day = ~$2,153/month
- **Spot**: Lower cost but interruptible
- **Savings Plans**: 3-6 month commitments for discounts

### Cost Optimization Tips

1. **Auto-start/stop**: Only run the instance when jobs are queued
2. **Use spot instances**: For non-critical workloads
3. **Monitor usage**: Track actual hours used
4. **Consider savings plans**: If running 24/7

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU access
rocm-smi

# Verify user is in correct groups
groups  # Should include 'render' and 'video'

# Check permissions
ls -la /dev/dri/
```

### PyTorch Can't Find GPU

```bash
# Verify PyTorch ROCm build
python3 -c "import torch; print(torch.version.hip)"

# Check ROCm paths
echo $ROCM_PATH
ls -la /opt/rocm
```

### Runner Not Connecting

```bash
# Check runner status
sudo systemctl status actions.runner.*.service

# Check runner logs
journalctl -u actions.runner.*.service -f

# Verify network connectivity
curl -I https://github.com
```

### Build Failures

If UCX, OpenMPI, or rocSHMEM fail to build:

1. Check available disk space: `df -h`
2. Ensure all dependencies installed: `apt-get install -f`
3. Try building with fewer parallel jobs: `make -j2` instead of `make -j$(nproc)`

## Persistence

RunPod instances are ephemeral. To persist runner configuration:

1. **Use RunPod Network Storage**: Mount persistent storage for runner state
2. **Save configuration**: Export runner config and restore on new instances
3. **Use RunPod Templates**: Create a template with the setup script for quick recreation

## Multiple Runners

To run multiple concurrent jobs:

1. Create multiple runner instances with different names:
   - `runpod-mi300-x86-64-1`
   - `runpod-mi300-x86-64-2`
   - `runpod-mi300-x86-64-3`

2. Use labels in workflows:
   ```yaml
   runs-on: [self-hosted, mi300]
   ```

3. Ensure each instance has sufficient resources

## Security Notes

- Use a GitHub Personal Access Token (PAT) with minimal permissions
- Store tokens securely (don't commit to git)
- Consider using runner groups for access control
- Regularly update the runner software
- Monitor runner logs for suspicious activity

## Support

If you encounter issues:

1. Check the setup script output for errors
2. Verify all steps completed successfully
3. Check GitHub Actions runner logs
4. Review RunPod instance logs
5. Ensure network connectivity to GitHub

## Next Steps

After setup:

1. Test with a simple workflow
2. Monitor costs and usage
3. Set up auto-scaling if needed
4. Configure monitoring/alerting
5. Document your specific configuration
