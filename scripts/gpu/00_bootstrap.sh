#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "PrivateDocs-LLM: Droplet Bootstrap Setup"
echo "=========================================="

# Update and install basic dependencies
sudo apt-get update
sudo apt-get install -y git curl jq build-essential

# Ensure Docker is installed and service is running
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
fi

# Ensure user is in video and render groups for ROCm access
sudo usermod -aG video,render "$USER" || true

# Setup scratch disk for Hugging Face cache
SCRATCH_DIR="/mnt/scratch/hf_cache"
sudo mkdir -p "$SCRATCH_DIR"
sudo chown -R "$USER:$USER" /mnt/scratch || true
export HF_HOME="$SCRATCH_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Copy environment file if not present
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example. Please review settings."
fi

echo "Droplet bootstrap completed successfully."
