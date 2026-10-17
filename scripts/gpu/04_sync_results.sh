#!/usr/bin/env bash
set -euo pipefail

echo "Syncing experiment results off droplet..."

# Check git status
git status --short results/ docs/gpu-session-log.md

# Add and commit results
git add results/ docs/gpu-session-log.md docs/rocm-notes.md || true
COMMIT_MSG="sync(results): automated experiment results [$(date -u +'%Y-%m-%d %H:%M UTC')]"
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

# Push if GITHUB_TOKEN or remote push is configured
if [ -n "${GITHUB_TOKEN:-}" ]; then
    REPO_URL=$(git config --get remote.origin.url | sed -e 's|^https://||')
    git push "https://${GITHUB_TOKEN}@${REPO_URL}" HEAD:main
else
    git push origin HEAD || {
        echo "WARNING: Git push failed. Ensure SSH key or GITHUB_TOKEN is configured."
    }
fi

echo "Results synchronized successfully."
