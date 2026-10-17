"""Asynchronous load tester using httpx and asyncio with warm-up exclusion."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> dict:
    start_time = time.perf_counter()
    status_code = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    success = False
    error = None

    try:
        response = await client.post(url, headers=headers, json=payload, timeout=90.0)
        status_code = response.status_code
        if response.status_code == 200:
            data = response.json()
            prompt_tokens = data.get("prompt_tokens", 0)
            completion_tokens = data.get("completion_tokens", 0)
            total_tokens = data.get("total_tokens", data.get("tokens", 0))
            success = True
        else:
            error = f"HTTP {response.status_code}: {response.text[:120]}"
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start_time
    return {
        "latency_sec": elapsed,
        "status_code": status_code,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "success": success,
        "error": error,
    }


async def run_benchmark(
    base_url: str,
    api_key: str,
    concurrency: int,
    total_requests: int,
    warmup_requests: int,
    question: str,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/ask"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload = {"question": question, "top_k": 3}

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        # Phase 1: Warm-up (excluded from benchmark metrics)
        if warmup_requests > 0:
            print(f"Executing {warmup_requests} warm-up request(s) (excluded from metrics)...")
            for i in range(warmup_requests):
                res = await send_request(client, url, headers, payload)
                if not res["success"]:
                    print(f"  Warm-up request {i + 1} warning: {res['error']}")
                else:
                    print(f"  Warm-up request {i + 1} succeeded ({res['latency_sec']:.2f}s)")

        # Phase 2: Timed Benchmark
        sem = asyncio.Semaphore(concurrency)

        async def worker() -> dict:
            async with sem:
                return await send_request(client, url, headers, payload)

        print(f"Starting benchmark: {total_requests} requests at concurrency={concurrency}...")
        overall_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - overall_start

    latencies = [r["latency_sec"] for r in results if r["success"]]
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results if r["success"])
    total_completion_tokens = sum(r["completion_tokens"] for r in results if r["success"])
    total_tokens = sum(r["total_tokens"] for r in results if r["success"])
    success_count = sum(1 for r in results if r["success"])
    error_count = total_requests - success_count

    summary = {
        "concurrency": concurrency,
        "warmup_requests_excluded": warmup_requests,
        "total_requests_measured": total_requests,
        "success_count": success_count,
        "error_count": error_count,
        "total_duration_sec": round(total_time, 3),
        "requests_per_sec": round(success_count / total_time, 2) if total_time > 0 else 0,
        "prompt_tokens_per_sec": round(total_prompt_tokens / total_time, 2) if total_time > 0 else 0,
        "completion_tokens_per_sec": round(total_completion_tokens / total_time, 2) if total_time > 0 else 0,
        "total_tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        "latency_p50_sec": round(statistics.median(latencies), 3) if latencies else 0,
        "latency_p95_sec": round(
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0),
            3,
        ),
        "latency_p99_sec": round(
            statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else (max(latencies) if latencies else 0),
            3,
        ),
        "latency_mean_sec": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_min_sec": round(min(latencies), 3) if latencies else 0,
        "latency_max_sec": round(max(latencies), 3) if latencies else 0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run async load test against /v1/ask with warm-up exclusion.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key", default="user-secret-dev-key-change-in-prod", help="API Key")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency level")
    parser.add_argument("--requests", type=int, default=16, help="Total measured requests")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warm-up requests to exclude")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.requests < 50:
        print(
            f"\n[WARNING] Only {args.requests} measured requests requested (< 50). "
            "Load test statistical significance (particularly p95 and p99 tail latencies) "
            "requires at least 50 measured requests.\n",
            file=sys.stderr,
        )

    results = asyncio.run(
        run_benchmark(
            base_url=args.url,
            api_key=args.key,
            concurrency=args.concurrency,
            total_requests=args.requests,
            warmup_requests=args.warmup,
            question="What are the key financial risk factors described?",
        )
    )
    print(json.dumps(results, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
