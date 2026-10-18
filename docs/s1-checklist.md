# Session 1 (S1) GPU Execution Checklist & Abort Runbook

This document defines the strict, step-by-step protocol for provisioning and operating an **AMD Instinct™ MI300X** (or equivalent ROCm GPU instance) at **$1.99/hour**, with mandatory cost-capping and time-boxed abort triggers.

---

## 0. Hard Financial & Operational Constraints

- **Hourly Rate:** $1.99/hour ($0.033/minute).
- **Session Budget Limit:** 2.0 hours maximum ($3.98 limit for S1; overall project cap: 17.5 hours / $35.00).
- **Golden Rule:** Never troubleshoot cloud environment issues while the GPU droplet is ticking. If a blocker occurs, capture logs, immediately destroy the droplet, and debug offline locally.

---

## 1. Hard Abort Rules (Non-Negotiable)

| Milestone | Max Elapsed Time from Droplet Launch | Failure Condition | Mandatory Action |
| :--- | :--- | :--- | :--- |
| **GPU Detection** | **20 Minutes** | `/dev/kfd`, `/dev/dri`, or `rocm-smi` fails or returns no AMD GPU devices. | **DESTROY DROPLET IMMEDIATELY**; debug driver/kernel issues offline. |
| **vLLM Serving** | **30 Minutes** | `vllm-rocm` container fails to return HTTP 200 on `/v1/models` within 30 min. | **DESTROY DROPLET IMMEDIATELY**; inspect container logs offline. |
| **Total Session** | **120 Minutes (2h)** | Session exceeds 2 hours regardless of experiment completion. | **DESTROY DROPLET IMMEDIATELY**; reconcile results. |

---

## 2. Pre-Session Checklist (Local Machine - Prior to Cloud Boot)

- [ ] Local unit tests pass cleanly: `pytest tests/` (21/21 passing).
- [ ] Code formatting & lint passes: `ruff check .` (0 errors).
- [ ] Static type check passes: `mypy src/` (24/24 files error-free).
- [ ] Dry-run passes on all shell scripts: `make gpu-dryrun` (or `./make.ps1 gpu-dryrun`).
- [ ] Git working tree is clean and pushed to `origin/master`.
- [ ] SSH public key is added to DigitalOcean / Cloud provider dashboard.
- [ ] Hugging Face access token with Llama-3.1 license access is verified (`HF_TOKEN`).

---

## 3. Step-by-Step Droplet Execution Protocol

### Step 1: Provision Droplet & Initial SSH Access (Minutes 0–5)
1. In Cloud Provider console, launch **1x AMD Instinct MI300X (192GB VRAM)** on Ubuntu 22.04 with ROCm support.
2. Note IP address (`DROPLET_IP`).
3. Connect via SSH:
   ```bash
   ssh -i ~/.ssh/id_rsa root@<DROPLET_IP>
   ```

### Step 2: Clone Repo & Bootstrap (Minutes 5–10)
```bash
git clone https://github.com/Ankitkumar028/PrivateDocs-LLM.git
cd PrivateDocs-LLM
bash scripts/gpu/00_bootstrap.sh
```

### Step 3: Verify GPU & ROCm Drivers (Minutes 10–15)
Run hardware diagnostics:
```bash
bash scripts/gpu/01_check_gpu.sh
```
- Verify `rocm-smi` displays the MI300X device.
- Verify scratch disk has $\ge 80$ GB free: `df -h /mnt/scratch`.
- **CRITICAL CHECKPOINT (Minute 20):** If `/dev/kfd` is missing or `rocm-smi` fails, execute **ABORT RULE #1**: destroy the droplet now.

### Step 4: Launch vLLM Inference Server (Minutes 15–25)
Start the vLLM ROCm container:
```bash
export HF_TOKEN="hf_your_token_here"
bash scripts/gpu/02_start_vllm.sh meta-llama/Llama-3.1-8B-Instruct bfloat16
```
- Wait for the smoke-test prompt to return `"OK"`.
- **CRITICAL CHECKPOINT (Minute 30):** If the server is not responding to `/v1/models` by Minute 30, execute **ABORT RULE #2**: destroy the droplet now.

### Step 5: Execute Evaluation & Concurrency Benchmarks (Minutes 25–45)
Run the automated experiment suite:
```bash
bash scripts/gpu/03_run_experiment.sh E1_mi300x_bfloat16
```
This automatically runs:
- `eval/run_answer_eval.py` (answers generated on GPU, saved to `results/E1_mi300x_bfloat16/raw_answers.jsonl`).
- `benchmarks/load_test.py` across concurrency levels $c \in \{1, 8, 32, 64\}$ (minimum 50 measured requests per tier).
- GPU budget tracking in `results/E1_mi300x_bfloat16/gpu_budget.json`.

### Step 6: Synchronize Results to Git (Minutes 45–50)
Commit and push benchmark artifacts:
```bash
bash scripts/gpu/04_sync_results.sh
```
Confirm that `results/E1_mi300x_bfloat16/` is committed and pushed to GitHub.

### Step 7: Teardown & Cloud Instance Destruction (Minutes 50–55)
Execute the teardown script to stop containers and calculate exact session expenditure:
```bash
bash scripts/gpu/99_teardown.sh
```
Output will report:
```
Total Session Time: 0.85 hours
Hourly Rate:        $1.99/hr
Total Session Cost: $1.69
```
**MANDATORY FINAL STEP:** Open the Cloud Console and click **Destroy Droplet** (confirming disk and IP are released). Do NOT leave the droplet shut down without destroying it, as compute is billed while droplets exist.

---

## 4. Post-Session Verification (Local Machine)

1. Verify git commit landed on `master` with GPU results.
2. Run `eval/report.py` to aggregate GPU latency, throughput, and answer quality into `docs/evaluation-report.md`.
3. Feed measured aggregate decode tokens/sec into `benchmarks/cost_model.py`.
