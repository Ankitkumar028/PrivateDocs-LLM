# Session 1 (S1) GPU Execution Checklist & Abort Runbook

This document defines the strict, copy-pasteable protocol for provisioning and operating an **AMD Instinct™ MI300X** (or equivalent ROCm GPU instance) at **$1.99/hour**, with mandatory cost-capping, time-boxed abort triggers, and the **two-pass evaluation flow**.

---

## 0. Hard Financial & Operational Constraints

- **Hourly Rate:** $1.99/hour ($0.033/minute).
- **Session Budget Limit:** 2.0 hours maximum ($3.98 limit for S1; overall project cap: 17.5 hours / $35.00).
- **Golden Rule:** Never troubleshoot cloud environment issues while the GPU droplet is ticking. If a blocker occurs, capture logs, immediately destroy the droplet, and debug offline locally.

---

## 1. Hard Abort Rules (Non-Negotiable)

| Milestone | Max Elapsed Time from Boot | Failure Condition | Mandatory Action & Terminal Command |
| :--- | :--- | :--- | :--- |
| **GPU Detection** | **20 Minutes** | `/dev/kfd`, `/dev/dri`, or `rocm-smi` fails or detects 0 GPUs. | `bash scripts/gpu/99_teardown.sh` -> **Destroy droplet in Cloud Console** |
| **vLLM Serving** | **30 Minutes** | `vllm-rocm` container fails HTTP 200 on `/v1/models`. | `docker logs --tail 100 vllm-rocm` -> `bash scripts/gpu/99_teardown.sh` -> **Destroy droplet** |
| **Total Session** | **120 Minutes (2h)** | Session exceeds 2 hours regardless of completion. | `bash scripts/gpu/99_teardown.sh` -> **Destroy droplet immediately** |

---

## 2. Pre-Session Checklist (Run Locally on Development Machine)

Run the local validation target to ensure zero broken dependencies before cloud provisioning:
```powershell
# Windows
.\make.ps1 test
.\make.ps1 gpu-dryrun

# Linux / MacOS
make test
make gpu-dryrun
```

Verify the following gates:
- [ ] Local unit tests pass cleanly (21/21 passing): `pytest -v tests/`.
- [ ] All shell scripts pass syntax check: `bash -n scripts/gpu/*.sh`.
- [ ] Stubbed Docker workflow passes: `bash scripts/gpu/test_stubbed_docker.sh`.
- [ ] Git working tree is clean and pushed: `git push origin master`.
- [ ] Hugging Face token with Llama-3.1 license access is ready: `HF_TOKEN`.

---

## 3. Step-by-Step Droplet Execution Protocol (Copy-Pasteable Commands)

### Step 1: Provision Droplet & Initial SSH Access (Minutes 0–5)
1. In Cloud Provider console, launch **1x AMD Instinct MI300X (192GB VRAM)** on Ubuntu 22.04 with ROCm 6.2+.
2. Note IP address (`DROPLET_IP`).
3. Connect via SSH:
```bash
ssh -i ~/.ssh/id_rsa root@<DROPLET_IP>
```

### Step 2: Clone Repo & Bootstrap (Minutes 5–10)
Paste into the droplet terminal:
```bash
git clone https://github.com/Ankitkumar028/PrivateDocs-LLM.git
cd PrivateDocs-LLM
bash scripts/gpu/00_bootstrap.sh
```

### Step 3: Verify GPU Hardware & ROCm Drivers (Minutes 10–15)
Run hardware diagnostics:
```bash
bash scripts/gpu/01_check_gpu.sh
```
Checklist:
- [ ] `rocm-smi` displays the MI300X device.
- [ ] `/dev/kfd` and `/dev/dri` exist.
- [ ] Scratch disk has $\ge 80$ GB free: `df -h /mnt/scratch`.
- **CRITICAL GATE (Minute 20):** If `rocm-smi` fails or `/dev/kfd` is absent, run `bash scripts/gpu/99_teardown.sh` and destroy droplet immediately.

### Step 4: Start Ingestion & Qdrant Vector DB (Minutes 15–20)
Start local Qdrant and verify vector collections:
```bash
docker compose -f docker/compose.local.yml up -d qdrant
python -m privatedocs.ingestion.indexer
```

### Step 5: Two-Pass Evaluation Architecture
To ensure fair comparisons across models without benchmark interference, evaluation follows a **two-pass flow**:
- **Pass 1 (During Benchmarking)**: For each model and precision variant, generate and save raw answers to `results/$EXP_ID/raw_answers.jsonl` (along with latency, TTFT, and tokens). No LLM judge is invoked during latency runs.
- **Pass 2 (Post-Experiment Judging)**: At the end of the session, load the largest model (or dedicated judge model), export `JUDGE_MODEL`, and evaluate all saved `raw_answers.jsonl` files in a single batch pass.

---

### Step 6: Pass 1 — Model Execution & Benchmarks (Minutes 20–70)

#### Experiment E0: Small Model (~8B), BF16 Baseline
```bash
export HF_TOKEN="hf_your_token_here"
bash scripts/gpu/02_start_vllm.sh meta-llama/Llama-3.1-8B-Instruct bfloat16
bash scripts/gpu/03_run_experiment.sh E0_8b_bf16
```
*Saves raw answers to `results/E0_8b_bf16/raw_answers.jsonl` and load test metrics across concurrency levels $c \in \{1, 8, 32, 64\}$.*

#### Experiment E1: Small Model (~8B), FP8 Precision
```bash
bash scripts/gpu/02_start_vllm.sh meta-llama/Llama-3.1-8B-Instruct fp8
bash scripts/gpu/03_run_experiment.sh E1_8b_fp8
```
*Saves raw answers to `results/E1_8b_fp8/raw_answers.jsonl`.*

---

### Step 7: Pass 2 — Batch LLM-as-a-Judge Evaluation (Minutes 70–85)
Ensure the primary judge model is running on vLLM (e.g. `meta-llama/Llama-3.1-8B-Instruct` or `meta-llama/Llama-3.1-70B-Instruct`):
```bash
export JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export JUDGE_BASE_URL="http://localhost:8000/v1"

# Judge Experiment E0 raw answers
python eval/judge.py \
    --input results/E0_8b_bf16/raw_answers.jsonl \
    --out results/E0_8b_bf16/judged_answers.jsonl \
    --sample-out results/E0_8b_bf16/manual_review_sample_20pct.jsonl

# Judge Experiment E1 raw answers
python eval/judge.py \
    --input results/E1_8b_fp8/raw_answers.jsonl \
    --out results/E1_8b_fp8/judged_answers.jsonl \
    --sample-out results/E1_8b_fp8/manual_review_sample_20pct.jsonl
```

---

### Step 8: Synchronize Results to Git (Minutes 85–90)
Commit and push all benchmark data to GitHub:
```bash
bash scripts/gpu/04_sync_results.sh
```

---

### Step 9: Teardown & Mandatory Droplet Destruction (Minutes 90–95)
Stop all containers and record final dollar expenditures:
```bash
bash scripts/gpu/99_teardown.sh
```
Example terminal output:
```
Total Session Time: 1.42 hours
Hourly Rate:        $1.99/hr
Total Session Cost: $2.83
```

**MANDATORY CLOUD STEP:**
1. Open Cloud Console.
2. Click **Destroy / Terminate Droplet**.
3. Confirm deletion to halt hourly billing immediately.

---

## 4. Post-Session Offline Analysis (Local Machine)

Once the droplet is destroyed, generate the final reports locally:
```powershell
# 1. Pull results from git
git pull origin master

# 2. Run report compiler
python eval/report.py --results-dir results/E0_8b_bf16 --out docs/evaluation-report.md

# 3. Update cost model with empirical GPU throughput numbers
python benchmarks/cost_model.py \
    --device gpu \
    --decode-tok-sec <MEASURED_DECODE_TOK_SEC> \
    --prefill-tok-sec <MEASURED_PREFILL_TOK_SEC> \
    --concurrency 32 \
    --out results/cost_model_summary.json
```
