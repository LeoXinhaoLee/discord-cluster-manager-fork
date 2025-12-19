#!/bin/bash
# Build script for AMD Docker container

set -e  # Exit on error

# Default image name/tag
IMAGE_NAME="${IMAGE_NAME:-amd-runner}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building AMD Docker container..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile: ${PROJECT_ROOT}/docker/amd-docker.Dockerfile"
echo ""

# Build the Docker image
docker build \
    -f "${PROJECT_ROOT}/docker/amd-docker.Dockerfile" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${PROJECT_ROOT}"

echo ""
echo "Build complete! Image: ${IMAGE_NAME}:${IMAGE_TAG}"
