# ROCm Running Notes & Troubleshooting Log

This document serves as an operational journal for deploying vLLM on AMD ROCm (MI300X, gfx942).
Any kernel issues, flag changes, driver compatibility quirks, or quantization behavior must be documented here.

## Reference Environment
- **Accelerator**: AMD Instinct™ MI300X (192 GB HBM3)
- **ROCm Version**: 6.2+
- **vLLM Image**: `rocm/vllm:rocm6.2_mi300_ubuntu22.04_py3.10_vllm_0.6.3` (VERIFY tag)
- **Driver devices**: `/dev/kfd`, `/dev/dri`

## Debugging Cap Policy
- Debugging is strictly capped at **30 minutes per issue**. If a model fails to load or FP8 is unsupported for a specific architecture, log the finding below and move to the next matrix configuration.

## Logged Issues & Findings

| Date | Model / Config | Issue Encountered | Root Cause | Resolution / Workaround | Status |
|---|---|---|---|---|---|
| 2026-10-17 | Scaffold setup | Initial ROCm image validation | Verification required for official tags | Set default recommended image in .env.example | Resolved |
| | | | | | |
