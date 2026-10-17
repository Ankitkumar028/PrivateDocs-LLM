#!/usr/bin/env bash
set -euo pipefail

echo "Setting up local development environment for PrivateDocs-LLM..."

if command -v uv >/dev/null 2>&1; then
    echo "Using uv for fast virtualenv creation and package installation..."
    uv venv .venv
    uv pip install -e ".[app,eval,dev]"
else
    echo "Using standard python venv and pip..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -e ".[app,eval,dev]"
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Generated .env from .env.example"
fi

mkdir -p data results
echo "Local development environment setup complete."
