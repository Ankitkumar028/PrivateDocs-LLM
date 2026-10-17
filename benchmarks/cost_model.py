"""Cost modeling tool for self-hosted LLM vs commercial APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_api_prices(prices_file: Path | None = None) -> dict:
    """Load reference API pricing from YAML."""
    if prices_file is None:
        prices_file = Path(__file__).parent / "api_prices.yaml"
    if not prices_file.exists():
        return {}
    with open(prices_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def calculate_cost_per_million(
    hourly_gpu_rate: float,
    tokens_per_second: float,
) -> float:
    """Calculate cost per 1M generated tokens.

    formula: (hourly_rate / tokens_per_hour) * 1_000_000
    """
    if tokens_per_second <= 0:
        return float("inf")
    tokens_per_hour = tokens_per_second * 3600.0
    return (hourly_gpu_rate / tokens_per_hour) * 1_000_000.0


def calculate_breakeven_hours(
    daily_gpu_cost: float,
    api_cost_per_million: float,
    daily_tokens: float,
) -> float:
    """Calculate break-even threshold comparing fixed GPU cost to API usage."""
    api_daily_cost = (daily_tokens / 1_000_000.0) * api_cost_per_million
    return api_daily_cost - daily_gpu_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate self-hosted LLM cost per 1M tokens.")
    parser.add_argument("--rate", type=float, default=1.99, help="Hourly GPU rate (default: $1.99)")
    parser.add_argument("--tok-per-sec", type=float, default=450.0, help="Sustained tokens/sec")
    args = parser.parse_args()

    cost_per_m = calculate_cost_per_million(args.rate, args.tok_per_sec)
    print(f"Hourly GPU Rate: ${args.rate:.2f}/hr")
    print(f"Throughput: {args.tok_per_sec:.1f} tokens/sec ({args.tok_per_sec * 3600:,.0f} tok/hr)")
    print(f"Effective Cost per 1M Output Tokens: ${cost_per_m:.4f}")


if __name__ == "__main__":
    main()
