"""Comprehensive cost modeling tool comparing self-hosted LLM inference vs commercial cloud APIs."""

from __future__ import annotations

import argparse
import json
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


def compare_costs(
    hourly_gpu_rate: float,
    tokens_per_second: float,
    daily_token_volumes: list[int] | None = None,
    prices_file: Path | None = None,
) -> dict:
    """Generate side-by-side cost comparison across token volumes."""
    if daily_token_volumes is None:
        daily_token_volumes = [100_000, 500_000, 2_000_000, 10_000_000, 50_000_000]

    api_pricing = load_api_prices(prices_file)
    self_hosted_cost_per_m = calculate_cost_per_million(hourly_gpu_rate, tokens_per_second)
    daily_gpu_cost = hourly_gpu_rate * 24.0
    monthly_gpu_cost = daily_gpu_cost * 30.0

    comparisons = []
    for vol in daily_token_volumes:
        vol_m = vol / 1_000_000.0
        row: dict[str, object] = {
            "daily_tokens": vol,
            "monthly_tokens": vol * 30,
            "self_hosted_monthly_usd": round(monthly_gpu_cost, 2),
            "cloud_apis": {},
        }

        # Compare with known models
        providers = api_pricing.get("providers", {})
        for provider, models in providers.items():
            for model_name, rates in models.items():
                in_rate = rates.get("input_per_million", 0.0)
                out_rate = rates.get("output_per_million", 0.0)
                # Assume 3:1 input-to-output ratio typical in RAG workflows
                blended_rate = (0.75 * in_rate) + (0.25 * out_rate)
                daily_cost = vol_m * blended_rate
                monthly_cost = daily_cost * 30.0
                diff = monthly_cost - monthly_gpu_cost

                row["cloud_apis"][f"{provider}/{model_name}"] = {
                    "monthly_cost_usd": round(monthly_cost, 2),
                    "self_hosted_savings_usd": round(diff, 2),
                    "cheaper_option": "self_hosted" if diff > 0 else "cloud_api",
                }
        comparisons.append(row)

    return {
        "self_hosted_params": {
            "hourly_gpu_rate_usd": hourly_gpu_rate,
            "tokens_per_second": tokens_per_second,
            "cost_per_million_tokens_usd": round(self_hosted_cost_per_m, 4),
            "fixed_daily_gpu_cost_usd": round(daily_gpu_cost, 2),
            "fixed_monthly_gpu_cost_usd": round(monthly_gpu_cost, 2),
        },
        "volume_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate self-hosted LLM cost per 1M tokens vs commercial APIs.")
    parser.add_argument("--rate", type=float, default=1.99, help="Hourly GPU rate (default: $1.99)")
    parser.add_argument("--tok-per-sec", type=float, default=450.0, help="Sustained tokens/sec (default: 450)")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = compare_costs(args.rate, args.tok_per_sec)

    print("\n" + "=" * 70)
    print("PRIVATE DOCS LLM COST MODEL (SELF-HOSTED VS COMMERCIAL APIS)")
    print("=" * 70)
    p = results["self_hosted_params"]
    print(f"Hourly GPU Rate:               ${p['hourly_gpu_rate_usd']:.2f}/hr")
    print(f"Sustained Throughput:          {p['tokens_per_second']:.1f} tokens/sec")
    print(f"Self-Hosted Cost per 1M Tok:   ${p['cost_per_million_tokens_usd']:.4f}")
    print(f"Fixed Dedicated Cost:          ${p['fixed_daily_gpu_cost_usd']:.2f}/day | ${p['fixed_monthly_gpu_cost_usd']:.2f}/month")
    print("-" * 70)
    print(f"{'Daily Volume':14} | {'Self-Hosted/Mo':15} | {'GPT-4o/Mo':13} | {'GPT-4o-mini/Mo':15} | {'Claude-Sonnet/Mo':16}")
    print("-" * 70)

    for c in results["volume_comparisons"]:
        vol_str = f"{c['daily_tokens']:,}"
        sh_cost = f"${c['self_hosted_monthly_usd']:.2f}"
        g4o = f"${c['cloud_apis'].get('openai/gpt-4o', {}).get('monthly_cost_usd', 0):.2f}"
        g4m = f"${c['cloud_apis'].get('openai/gpt-4o-mini', {}).get('monthly_cost_usd', 0):.2f}"
        c3s = f"${c['cloud_apis'].get('anthropic/claude-3-5-sonnet', {}).get('monthly_cost_usd', 0):.2f}"
        print(f"{vol_str:14} | {sh_cost:15} | {g4o:13} | {g4m:15} | {c3s:16}")
    print("=" * 70)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved cost model output to {out_p}")


if __name__ == "__main__":
    main()
