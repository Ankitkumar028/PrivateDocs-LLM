# Benchmarks Suite

This directory contains load generation and economic cost modeling tools for PrivateDocs-LLM on AMD Instinct MI300X and local environments.

## Files
- `matrix.yaml`: Experiment configurations (E0-E5) across model sizes and precisions.
- `api_prices.yaml`: Reference cloud API pricing for break-even analysis.
- `load_test.py`: Asynchronous load generator based on `httpx` and `asyncio` to test concurrency levels (1, 8, 32, 64).
- `locustfile.py`: Optional Locust benchmark configuration for distributed load testing.
- `cost_model.py`: Calculates effective cost per 1M generated tokens and computes break-even usage thresholds against commercial APIs.
