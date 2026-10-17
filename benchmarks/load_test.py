"""Asynchronous load tester using httpx and asyncio."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
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
    tokens = 0
    success = False
    error = None

    try:
        response = await client.post(url, headers=headers, json=payload, timeout=60.0)
        status_code = response.status_code
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("tokens", 0)
            success = True
        else:
            error = f"HTTP {response.status_code}: {response.text[:100]}"
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start_time
    return {
        "latency_sec": elapsed,
        "status_code": status_code,
        "tokens": tokens,
        "success": success,
        "error": error,
    }


async def run_benchmark(
    base_url: str,
    api_key: str,
    concurrency: int,
    total_requests: int,
    question: str,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/ask"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload = {"question": question, "top_k": 3}

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def worker() -> dict:
            async with sem:
                return await send_request(client, url, headers, payload)

        overall_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - overall_start

    latencies = [r["latency_sec"] for r in results if r["success"]]
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    success_count = sum(1 for r in results if r["success"])
    error_count = total_requests - success_count

    summary = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "success_count": success_count,
        "error_count": error_count,
        "total_duration_sec": total_time,
        "requests_per_sec": success_count / total_time if total_time > 0 else 0,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "latency_p50_sec": statistics.median(latencies) if latencies else 0,
        "latency_p95_sec": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0)
        ),
        "latency_mean_sec": statistics.mean(latencies) if latencies else 0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run async load test against /v1/ask.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key", default="user-secret-dev-key-change-in-prod", help="API Key")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency level")
    parser.add_argument("--requests", type=int, default=32, help="Total requests")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = asyncio.run(
        run_benchmark(
            base_url=args.url,
            api_key=args.key,
            concurrency=args.concurrency,
            total_requests=args.requests,
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
