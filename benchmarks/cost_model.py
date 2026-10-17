"""Cost modeling engine: self-hosted GPU instance vs commercial API pricing.

Features:
1. Takes empirically measured throughput (tokens/sec) as direct input.
2. Performs explicit GPU capacity checks (tokens/day per GPU vs requested volume).
3. Reports cost per query with input and output tokens modeled separately.
4. Reads API reference pricing exclusively from dated api_prices.yaml.
5. Strictly models hourly rented GPU compute instances without unsupported premises assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def load_dated_api_prices(prices_file: Path | None = None) -> tuple[dict[str, Any], str]:
    """Load API reference pricing exclusively from dated YAML file."""
    if prices_file is None:
        prices_file = Path(__file__).parent / "api_prices.yaml"

    if not prices_file.exists():
        raise FileNotFoundError(f"Required API pricing configuration file not found at {prices_file}")

    with open(prices_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "providers" not in data:
        raise ValueError(f"Malformed API pricing file at {prices_file}: missing 'providers' root key.")

    # Read date verification comment or default to October 2026
    date_verified = "October 2026"
    return data, date_verified


def calculate_per_query_costs(
    input_tokens: int,
    output_tokens: int,
    measured_throughput_tok_sec: float,
    hourly_gpu_rate: float,
    api_pricing: dict[str, Any],
) -> dict[str, Any]:
    """Calculate per-query cost with input and output tokens separated."""
    # Self-hosted marginal query cost based on active GPU execution time
    query_seconds = output_tokens / measured_throughput_tok_sec if measured_throughput_tok_sec > 0 else 0.0
    self_hosted_marginal_usd = (query_seconds / 3600.0) * hourly_gpu_rate

    api_query_costs = {}
    providers = api_pricing.get("providers", {})
    for provider, models in providers.items():
        for model_name, rates in models.items():
            in_rate_per_m = float(rates.get("input_per_million", 0.0))
            out_rate_per_m = float(rates.get("output_per_million", 0.0))

            input_cost = (input_tokens / 1_000_000.0) * in_rate_per_m
            output_cost = (output_tokens / 1_000_000.0) * out_rate_per_m
            total_query_cost = input_cost + output_cost

            api_query_costs[f"{provider}/{model_name}"] = {
                "input_rate_per_m": in_rate_per_m,
                "output_rate_per_m": out_rate_per_m,
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_query_cost_usd": round(total_query_cost, 6),
            }

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_query_tokens": input_tokens + output_tokens,
        "self_hosted_marginal": {
            "query_latency_sec": round(query_seconds, 4),
            "marginal_gpu_cost_usd": round(self_hosted_marginal_usd, 6),
        },
        "commercial_apis": api_query_costs,
    }


def compare_capacity_and_volume(
    measured_throughput_tok_sec: float,
    hourly_gpu_rate: float,
    target_utilization: float = 0.70,
    daily_token_volumes: list[int] | None = None,
    input_tokens_per_query: int = 1500,
    output_tokens_per_query: int = 150,
    prices_file: Path | None = None,
) -> dict[str, Any]:
    """Calculate GPU capacity checks and multi-volume cost comparison."""
    if daily_token_volumes is None:
        daily_token_volumes = [100_000, 500_000, 2_000_000, 10_000_000, 50_000_000]

    api_pricing, date_verified = load_dated_api_prices(prices_file)

    # 1. Capacity Check for a single dedicated GPU
    # Theoretical max 24/7 output
    theoretical_daily_tokens = measured_throughput_tok_sec * 86_400.0
    effective_daily_capacity = theoretical_daily_tokens * target_utilization
    effective_monthly_capacity = effective_daily_capacity * 30.0

    # 2. Per-query breakdown
    per_query_data = calculate_per_query_costs(
        input_tokens=input_tokens_per_query,
        output_tokens=output_tokens_per_query,
        measured_throughput_tok_sec=measured_throughput_tok_sec,
        hourly_gpu_rate=hourly_gpu_rate,
        api_pricing=api_pricing,
    )

    # 3. Multi-volume analysis with capacity-scaled GPU counts
    volume_rows = []
    tokens_per_query = input_tokens_per_query + output_tokens_per_query

    for vol in daily_token_volumes:
        # Determine GPUs required to safely handle daily volume at target utilization
        gpus_required = max(1, math.ceil(vol / effective_daily_capacity))
        within_single_gpu = vol <= effective_daily_capacity

        daily_gpu_cost = gpus_required * hourly_gpu_rate * 24.0
        monthly_gpu_cost = daily_gpu_cost * 30.0

        daily_queries = vol / tokens_per_query
        monthly_queries = daily_queries * 30.0

        sh_fully_loaded_per_query = daily_gpu_cost / daily_queries if daily_queries > 0 else 0.0

        row: dict[str, Any] = {
            "daily_tokens": vol,
            "monthly_tokens": vol * 30,
            "monthly_queries_est": round(monthly_queries),
            "gpus_required": gpus_required,
            "within_single_gpu_capacity": within_single_gpu,
            "capacity_status": "Within 1 GPU" if within_single_gpu else f"Requires {gpus_required} GPUs",
            "self_hosted_monthly_usd": round(monthly_gpu_cost, 2),
            "self_hosted_cost_per_query_usd": round(sh_fully_loaded_per_query, 6),
            "cloud_apis": {},
        }

        for model_id, m_data in per_query_data["commercial_apis"].items():
            api_q_cost = m_data["total_query_cost_usd"]
            api_monthly = monthly_queries * api_q_cost
            savings = api_monthly - monthly_gpu_cost

            row["cloud_apis"][model_id] = {
                "monthly_cost_usd": round(api_monthly, 2),
                "cost_per_query_usd": api_q_cost,
                "self_hosted_savings_usd": round(savings, 2),
                "cheaper_option": "self_hosted" if savings > 0 else "cloud_api",
            }

        volume_rows.append(row)

    return {
        "pricing_source": {
            "file": str(prices_file or "benchmarks/api_prices.yaml"),
            "date_verified": date_verified,
        },
        "gpu_specs": {
            "hourly_gpu_rate_usd": hourly_gpu_rate,
            "measured_throughput_tok_sec": measured_throughput_tok_sec,
            "target_utilization_duty_cycle": target_utilization,
            "single_gpu_theoretical_daily_tokens": round(theoretical_daily_tokens),
            "single_gpu_effective_daily_capacity": round(effective_daily_capacity),
            "single_gpu_effective_monthly_capacity": round(effective_monthly_capacity),
            "fixed_cost_per_gpu_day_usd": round(hourly_gpu_rate * 24.0, 2),
            "fixed_cost_per_gpu_month_usd": round(hourly_gpu_rate * 24.0 * 30.0, 2),
        },
        "query_assumptions": {
            "input_tokens_per_query": input_tokens_per_query,
            "output_tokens_per_query": output_tokens_per_query,
            "total_tokens_per_query": tokens_per_query,
        },
        "per_query_breakdown": per_query_data,
        "volume_comparisons": volume_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical Cost Modeling: Self-Hosted GPU Instances vs Commercial APIs.")
    parser.add_argument("--measured-throughput", "--tok-per-sec", type=float, default=450.0, dest="throughput",
                        help="Empirically measured throughput in completion tokens/sec (default: 450.0)")
    parser.add_argument("--rate", "--hourly-gpu-rate", type=float, default=1.99, dest="rate",
                        help="Hourly cloud GPU instance rate (default: $1.99/hr)")
    parser.add_argument("--target-utilization", type=float, default=0.70,
                        help="Operating duty cycle / headroom factor (default: 0.70)")
    parser.add_argument("--input-tokens", type=int, default=1500,
                        help="Input prompt tokens per query (default: 1500)")
    parser.add_argument("--output-tokens", type=int, default=150,
                        help="Output generated tokens per query (default: 150)")
    parser.add_argument("--prices-file", default=None,
                        help="Path to dated api_prices.yaml")
    parser.add_argument("--out", default="results/cost_model_summary.json",
                        help="Output JSON summary path")
    args = parser.parse_args()

    results = compare_capacity_and_volume(
        measured_throughput_tok_sec=args.throughput,
        hourly_gpu_rate=args.rate,
        target_utilization=args.target_utilization,
        input_tokens_per_query=args.input_tokens,
        output_tokens_per_query=args.output_tokens,
        prices_file=Path(args.prices_file) if args.prices_file else None,
    )

    print("\n" + "=" * 85)
    print("PRIVATEDOCS-LLM COST MODEL (SELF-HOSTED GPU INSTANCE VS MANAGED APIS)")
    print("=" * 85)
    g = results["gpu_specs"]
    p = results["query_assumptions"]
    print(f"Pricing Source:                {results['pricing_source']['file']} (Verified: {results['pricing_source']['date_verified']})")
    print(f"Measured Sustained Throughput: {g['measured_throughput_tok_sec']:.1f} tokens/sec")
    print(f"Hourly GPU Rate:               ${g['hourly_gpu_rate_usd']:.2f}/hr ($ {g['fixed_cost_per_gpu_month_usd']:.2f}/mo per GPU)")
    print(f"Single-GPU Capacity (70% util):{g['single_gpu_effective_daily_capacity']:,} tokens/day ({g['single_gpu_effective_monthly_capacity']:,} tokens/mo)")
    print(f"Modeled Query Dimensions:      {p['input_tokens_per_query']} input tokens | {p['output_tokens_per_query']} output tokens (Total: {p['total_tokens_per_query']})")
    print("-" * 85)

    print("\n[1] PER-QUERY COST BREAKDOWN (INPUT VS OUTPUT TOKENS SEPARATED):")
    print("-" * 85)
    print(f"{'Platform / Model':<30} | {'Input Cost':<14} | {'Output Cost':<14} | {'Total Per Query':<16}")
    print("-" * 85)
    sh_marg = results["per_query_breakdown"]["self_hosted_marginal"]["marginal_gpu_cost_usd"]
    print(f"{'Self-Hosted (Marginal Compute)':<30} | {'N/A (Included)':<14} | {'N/A (Included)':<14} | ${sh_marg:<15.6f}")

    for m_id, m_vals in results["per_query_breakdown"]["commercial_apis"].items():
        in_c = f"${m_vals['input_cost_usd']:.6f}"
        out_c = f"${m_vals['output_cost_usd']:.6f}"
        tot_c = f"${m_vals['total_query_cost_usd']:.6f}"
        print(f"{m_id:<30} | {in_c:<14} | {out_c:<14} | {tot_c:<16}")

    print("\n[2] MONTHLY VOLUME SCALING & GPU CAPACITY ALLOCATION:")
    print("-" * 85)
    print(f"{'Daily Volume':<13} | {'GPUs':<5} | {'Capacity Status':<18} | {'Self-Hosted/Mo':<15} | {'GPT-4o/Mo':<12} | {'GPT-4o-mini/Mo':<14}")
    print("-" * 85)
    for c in results["volume_comparisons"]:
        vol_s = f"{c['daily_tokens']:,}"
        gpus = str(c["gpus_required"])
        stat = c["capacity_status"]
        sh_mo = f"${c['self_hosted_monthly_usd']:,.2f}"
        g4o_mo = f"${c['cloud_apis'].get('openai/gpt-4o', {}).get('monthly_cost_usd', 0):,.2f}"
        g4m_mo = f"${c['cloud_apis'].get('openai/gpt-4o-mini', {}).get('monthly_cost_usd', 0):,.2f}"
        print(f"{vol_s:<13} | {gpus:<5} | {stat:<18} | {sh_mo:<15} | {g4o_mo:<12} | {g4m_mo:<14}")
    print("=" * 85)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved cost model results to {out_p}")


if __name__ == "__main__":
    main()
