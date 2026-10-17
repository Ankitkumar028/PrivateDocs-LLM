#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${2:-bfloat16}"
VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm6.2_mi300_ubuntu22.04_py3.10_vllm_0.6.3}"
SCRATCH_DIR="/mnt/scratch/hf_cache"

echo "Launching vLLM on MI300X..."
echo "Model: $MODEL | Precision: $DTYPE"

# Stop existing container if running
docker stop vllm-rocm 2>/dev/null || true
docker rm vllm-rocm 2>/dev/null || true

# Run vLLM with ROCm device bindings and shared memory size
docker run -d \
    --name vllm-rocm \
    --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=32g \
    -v "$SCRATCH_DIR":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --dtype "$DTYPE" \
        --port 8000 \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90

echo "Waiting for vLLM to initialize on port 8000..."
READY=0
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models | grep -q "$MODEL"; then
        READY=1
        break
    fi
    sleep 5
    echo -n "."
done

if [ "$READY" -ne 1 ]; then
    echo ""
    echo "ERROR: vLLM did not become ready within 5 minutes. Check docker logs vllm-rocm."
    exit 1
fi

echo ""
echo "vLLM is READY! Running one-prompt smoke check..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Respond with OK if you are working."}],
        "max_tokens": 10
    }' | jq .

echo "vLLM test passed."
