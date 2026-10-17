# Droplet Teardown Checklist

Run this checklist at the conclusion of **every single working block** to avoid burning idle GPU credits ($1.99/hr).

- [ ] 1. Stop active vLLM and docker containers:
  ```bash
  docker stop $(docker ps -q) 2>/dev/null || true
  ```
- [ ] 2. Confirm GPU budget session has stopped:
  ```bash
  python scripts/gpu_budget.py status
  # If still running:
  python scripts/gpu_budget.py stop
  ```
- [ ] 3. Sync all experiment results, session logs, and ROCm notes:
  ```bash
  bash scripts/gpu/04_sync_results.sh
  ```
- [ ] 4. Check git status to ensure zero untracked results:
  ```bash
  git status
  ```
- [ ] 5. **Log into AMD Developer Cloud Console**:
  - Navigate to Instances / Droplets.
  - Select the running MI300X instance.
  - Click **Destroy / Terminate**.
- [ ] 6. Refresh the console page to confirm the instance state is `TERMINATED` / deleted.
- [ ] 7. **HARD STOP DATE: Ensure all resources are permanently terminated by October 23, 2026.**
