#!/usr/bin/env bash
set -euo pipefail

echo "Running stubbed docker test for GPU automation scripts..."

TEMP_BIN=$(mktemp -d 2>/dev/null || mktemp -d -t 'stubbin')
trap 'rm -rf "$TEMP_BIN"' EXIT

# Create stubbed docker binary
cat << 'EOF' > "$TEMP_BIN/docker"
#!/usr/bin/env bash
echo "[STUB-DOCKER] Called: docker $*"
if [ "${1:-}" = "run" ]; then
    echo "stub_container_id_12345"
fi
exit 0
EOF
chmod +x "$TEMP_BIN/docker"

# Create stubbed curl binary that simulates healthy vLLM responses
cat << 'EOF' > "$TEMP_BIN/curl"
#!/usr/bin/env bash
echo "[STUB-CURL] Called: curl $*"
if echo "$*" | grep -q "models"; then
    echo '{"data":[{"id":"meta-llama/Llama-3.1-8B-Instruct"}]}'
elif echo "$*" | grep -q "chat/completions"; then
    echo '{"choices":[{"message":{"content":"OK"}}]}'
else
    echo '{"status":"ok"}'
fi
exit 0
EOF
chmod +x "$TEMP_BIN/curl"

# Create stubbed jq binary
cat << 'EOF' > "$TEMP_BIN/jq"
#!/usr/bin/env bash
cat
exit 0
EOF
chmod +x "$TEMP_BIN/jq"

# Run 02_start_vllm.sh in dry-run with stubs on PATH
export PATH="$TEMP_BIN:$PATH"
echo "Testing 02_start_vllm.sh with stubbed docker..."
bash scripts/gpu/02_start_vllm.sh meta-llama/Llama-3.1-8B-Instruct bfloat16

echo "Testing 99_teardown.sh with stubbed docker..."
bash scripts/gpu/99_teardown.sh

echo "Stubbed Docker validation completed successfully."
