#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "PrivateDocs-LLM: Droplet Teardown Routine"
echo "=========================================="

# 1. Stop all Docker containers immediately
echo "Stopping all active Docker containers..."
docker stop $(docker ps -q) 2>/dev/null || true
echo "Containers stopped."

# 2. Stop GPU budget tracker and report elapsed hours and dollars
echo ""
echo "=== GPU Session Cost & Duration Summary ==="
if [ -f .gpu_session_state.json ]; then
    python3 scripts/gpu_budget.py stop || python scripts/gpu_budget.py stop
else
    echo "Notice: No active .gpu_session_state.json found."
    echo "Check docs/gpu-session-log.md for historical recorded sessions."
fi

# 3. Synchronize experiment artifacts
echo ""
echo "Checking git status for unsaved results..."
git status --short results/ docs/gpu-session-log.md

echo ""
echo "========================================================"
echo "CRITICAL: DESTROY DROPLET TO PREVENT IDLE CHARGES (\$1.99/HR)"
echo "========================================================"
echo "1. Log into AMD Developer Cloud Console."
echo "2. Select this MI300X instance."
echo "3. Click DESTROY / TERMINATE."
echo "4. Refresh console to verify instance state is TERMINATED."
echo "========================================================"
