#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${2:-bfloat16}"
# VERIFY: Ensure container image tag matches host ROCm driver version
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:v0.27.1}"
SCRATCH_DIR="/mnt/scratch/hf_cache"

# Configurable options
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
QUANTIZATION="${QUANTIZATION:-}"

EXTRA_ARGS=()

# Opt-in remote code execution per model
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    EXTRA_ARGS+=(--trust-remote-code)
fi

# FP8 path: check if DTYPE is fp8 or QUANTIZATION is explicitly requested
if [ "$DTYPE" = "fp8" ] || [ -n "$QUANTIZATION" ]; then
    QUANT_VAL="${QUANTIZATION:-fp8}"
    # VERIFY against vLLM's ROCm docs: check whether host ROCm vLLM image supports --quantization fp8 with CDNA3 native FP8 matrix cores
    EXTRA_ARGS+=(--quantization "$QUANT_VAL")
    # For FP8 checkpoints, dtype is typically float16/bfloat16 for unquantized layers
    if [ "$DTYPE" = "fp8" ]; then
        DTYPE="bfloat16"
    fi
fi

echo "Launching vLLM on MI300X..."
echo "Model: $MODEL | Precision: $DTYPE | MaxLen: $MAX_MODEL_LEN | TrustRemoteCode: $TRUST_REMOTE_CODE"
if [ -n "$QUANTIZATION" ] || [ "${2:-}" = "fp8" ]; then
    echo "Quantization: ${QUANT_VAL:-fp8} (# VERIFY against vLLM's ROCm docs)"
fi

# Stop existing container if running
docker stop vllm-rocm 2>/dev/null || true
docker rm vllm-rocm 2>/dev/null || true

# Run vLLM with ROCm device bindings and shared memory size
mkdir -p "$SCRATCH_DIR"
docker run -d \
    --name vllm-rocm \
    -p 127.0.0.1:8000:8000 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=32g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$SCRATCH_DIR":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$VLLM_IMAGE" \
    "$MODEL" \
    --generation-config vllm \
    --dtype "$DTYPE" \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    "${EXTRA_ARGS[@]}"

echo "Waiting for vLLM to initialize on port 8000..."
READY=0
for _ in {1..60}; do
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

