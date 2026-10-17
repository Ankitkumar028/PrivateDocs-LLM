#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Checking ROCm Devices ==="
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd not found. ROCm kernel driver is missing or not loaded!"
    exit 1
fi

if [ ! -d /dev/dri ]; then
    echo "ERROR: /dev/dri directory missing."
    exit 1
fi
echo "ROCm devices /dev/kfd and /dev/dri verified."

echo "=== 2. Checking rocm-smi Status ==="
if command -v rocm-smi &> /dev/null; then
    rocm-smi
else
    echo "WARNING: rocm-smi command not in PATH on host (checking inside container later)."
fi

echo "=== 3. Checking Disk Space ==="
df -h / /mnt/scratch || df -h .

echo "=== 4. Validating Docker GPU Access ==="
VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm6.2_mi300_ubuntu22.04_py3.10_vllm_0.6.3}"
echo "Running quick ROCm container check with image: $VLLM_IMAGE"
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    "$VLLM_IMAGE" rocm-smi || {
        echo "ERROR: Docker container could not access ROCm devices."
        exit 1
    }

echo "All GPU and system checks passed!"
