# Operational Runbook: Local Dev & AMD MI300X Droplet

## 1. Local Development Quickstart

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional for local Qdrant/Ollama, or in-memory mode)

```bash
# Clone and setup
git clone https://github.com/example/privatedocs-llm.git
cd privatedocs-llm
cp .env.example .env

# Install dependencies using uv or venv
uv venv && uv pip install -e ".[app,eval,dev]"

# Run tests
pytest -v tests/

# Run local smoke test
python scripts/local/smoke_test.sh
```

## 2. AMD MI300X Droplet Operations

### Step 1: Provision Droplet
1. In the AMD Developer Cloud console, spin up a single MI300X instance ($1.99/hr).
2. Attach your SSH public key.
3. Note the public IP address.

### Step 2: Bootstrap Node
```bash
ssh root@<DROPLET_IP>
git clone https://github.com/<YOUR_REPO>/privatedocs-llm.git
cd privatedocs-llm
bash scripts/gpu/00_bootstrap.sh
bash scripts/gpu/01_check_gpu.sh
```

### Step 3: Run vLLM and Experiments
```bash
# Start GPU budget tracker
python scripts/gpu_budget.py start "Experiment E0 - 8B BF16"

# Launch vLLM with target model and dtype
bash scripts/gpu/02_start_vllm.sh meta-llama/Llama-3.1-8B-Instruct bfloat16

# Run automated experiment
bash scripts/gpu/03_run_experiment.sh E0

# Sync results immediately
bash scripts/gpu/04_sync_results.sh

# Stop GPU budget tracker
python scripts/gpu_budget.py stop
```

### Step 4: Access Web UIs via SSH Port Forwarding
Never expose Grafana, Streamlit, or vLLM directly on public ports. Use SSH local tunneling:
```bash
ssh -L 8000:localhost:8000 -L 8501:localhost:8501 -L 3000:localhost:3000 root@<DROPLET_IP>
```
- Streamlit UI: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin / admin)

### Step 5: Teardown
Follow `scripts/gpu/99_teardown_checklist.md` before terminating the instance.
