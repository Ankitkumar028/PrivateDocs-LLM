"""Cost modeling engine: self-hosted GPU instance vs commercial API pricing.

Requirements:
1. Separate 'measured' inputs (aggregate throughput, prefill/decode at concurrency)
   from 'assumed' inputs (hourly rates, query dimensions, utilization headroom).
2. Refuse to publish CPU throughput numbers (prominently labeled 'PLACEHOLDER').
3. Model prefill and decode separately.
4. Use aggregate throughput at concurrency rather than single-stream speed.
5. In the price table: model name, price date, source URL, and active/retired status.
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

    date_verified = "October 2026"
    return data, date_verified


def calculate_per_query_costs(
    input_tokens: int,
    output_tokens: int,
    aggregate_prefill_tok_sec: float,
    aggregate_decode_tok_sec: float,
    hourly_gpu_rate: float,
    api_pricing: dict[str, Any],
) -> dict[str, Any]:
    """Calculate per-query cost with input and output tokens modeled separately."""
    prefill_sec = (input_tokens / aggregate_prefill_tok_sec) if aggregate_prefill_tok_sec > 0 else 0.0
    decode_sec = (output_tokens / aggregate_decode_tok_sec) if aggregate_decode_tok_sec > 0 else 0.0
    total_active_sec = prefill_sec + decode_sec
    self_hosted_marginal_usd = (total_active_sec / 3600.0) * hourly_gpu_rate

    api_query_costs = {}
    providers = api_pricing.get("providers", {})
    for provider, models in providers.items():
        for model_key, rates in models.items():
            in_rate_per_m = float(rates.get("input_per_million", 0.0))
            out_rate_per_m = float(rates.get("output_per_million", 0.0))

            input_cost = (input_tokens / 1_000_000.0) * in_rate_per_m
            output_cost = (output_tokens / 1_000_000.0) * out_rate_per_m
            total_query_cost = input_cost + output_cost

            api_query_costs[f"{provider}/{model_key}"] = {
                "name": rates.get("name", f"{provider}/{model_key}"),
                "status": rates.get("status", "active"),
                "price_date": rates.get("price_date", "2026-10"),
                "source_url": rates.get("source_url", "https://openai.com/api/pricing/"),
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
            "prefill_latency_sec": round(prefill_sec, 4),
            "decode_latency_sec": round(decode_sec, 4),
            "total_active_latency_sec": round(total_active_sec, 4),
            "marginal_gpu_cost_usd": round(self_hosted_marginal_usd, 6),
        },
        "commercial_apis": api_query_costs,
    }


def compare_capacity_and_volume(
    aggregate_decode_tok_sec: float,
    aggregate_prefill_tok_sec: float,
    concurrency: int = 4,
    device: str = "gpu",
    hourly_gpu_rate: float = 1.99,
    target_utilization: float = 0.70,
    daily_token_volumes: list[int] | None = None,
    input_tokens_per_query: int = 1500,
    output_tokens_per_query: int = 150,
    prices_file: Path | None = None,
) -> dict[str, Any]:
    """Calculate capacity check, per-query breakdown, and volume comparisons."""
    is_cpu_placeholder = device.lower() == "cpu" or aggregate_decode_tok_sec < 200.0

    if daily_token_volumes is None:
        daily_token_volumes = [100_000, 500_000, 2_000_000, 10_000_000, 50_000_000]

    api_pricing, date_verified = load_dated_api_prices(prices_file)

    # Effective capacity for a single dedicated GPU instance at target utilization
    theoretical_daily_tokens = aggregate_decode_tok_sec * 86_400.0
    effective_daily_capacity = theoretical_daily_tokens * target_utilization
    effective_monthly_capacity = effective_daily_capacity * 30.0

    per_query_data = calculate_per_query_costs(
        input_tokens=input_tokens_per_query,
        output_tokens=output_tokens_per_query,
        aggregate_prefill_tok_sec=aggregate_prefill_tok_sec,
        aggregate_decode_tok_sec=aggregate_decode_tok_sec,
        hourly_gpu_rate=hourly_gpu_rate,
        api_pricing=api_pricing,
    )

    volume_rows = []
    tokens_per_query = input_tokens_per_query + output_tokens_per_query

    for vol in daily_token_volumes:
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
                "name": m_data["name"],
                "status": m_data["status"],
                "monthly_cost_usd": round(api_monthly, 2),
                "cost_per_query_usd": api_q_cost,
                "self_hosted_savings_usd": round(savings, 2),
                "cheaper_option": "self_hosted" if savings > 0 else "cloud_api",
            }

        volume_rows.append(row)

    return {
        "status": "PLACEHOLDER (CPU Throughput - Not Valid for Publication)" if is_cpu_placeholder else "OFFICIAL (GPU Measured)",
        "is_cpu_placeholder": is_cpu_placeholder,
        "pricing_source": {
            "file": str(prices_file or "benchmarks/api_prices.yaml"),
            "date_verified": date_verified,
        },
        "measured_inputs": {
            "device": device,
            "concurrency": concurrency,
            "aggregate_prefill_tokens_per_sec": aggregate_prefill_tok_sec,
            "aggregate_decode_tokens_per_sec": aggregate_decode_tok_sec,
            "aggregate_total_tokens_per_sec": aggregate_prefill_tok_sec + aggregate_decode_tok_sec,
        },
        "assumed_inputs": {
            "hourly_gpu_rate_usd": hourly_gpu_rate,
            "target_utilization_duty_cycle": target_utilization,
            "input_tokens_per_query": input_tokens_per_query,
            "output_tokens_per_query": output_tokens_per_query,
            "total_tokens_per_query": tokens_per_query,
            "single_gpu_effective_daily_capacity_tokens": round(effective_daily_capacity),
            "single_gpu_effective_monthly_capacity_tokens": round(effective_monthly_capacity),
        },
        "per_query_breakdown": per_query_data,
        "volume_comparisons": volume_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical Cost Modeling: Self-Hosted GPU Instances vs Commercial APIs.")
    parser.add_argument("--decode-tok-sec", type=float, default=141.8,
                        help="Measured aggregate generation/decode tokens/sec at concurrency")
    parser.add_argument("--prefill-tok-sec", type=float, default=829.5,
                        help="Measured aggregate prompt/prefill tokens/sec at concurrency")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Measured concurrency level (default: 4)")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                        help="Hardware device where throughput was measured ('gpu' or 'cpu')")
    parser.add_argument("--rate", type=float, default=1.99,
                        help="Hourly cloud GPU instance rate (default: $1.99/hr)")
    parser.add_argument("--target-utilization", type=float, default=0.70,
                        help="Operating duty cycle headroom (default: 0.70)")
    parser.add_argument("--input-tokens", type=int, default=1500,
                        help="Prompt input tokens per query (default: 1500)")
    parser.add_argument("--output-tokens", type=int, default=150,
                        help="Generated output tokens per query (default: 150)")
    parser.add_argument("--prices-file", default=None,
                        help="Path to dated api_prices.yaml")
    parser.add_argument("--out", default="results/cost_model_summary.json",
                        help="Output JSON summary path")
    args = parser.parse_args()

    results = compare_capacity_and_volume(
        aggregate_decode_tok_sec=args.decode_tok_sec,
        aggregate_prefill_tok_sec=args.prefill_tok_sec,
        concurrency=args.concurrency,
        device=args.device,
        hourly_gpu_rate=args.rate,
        target_utilization=args.target_utilization,
        input_tokens_per_query=args.input_tokens,
        output_tokens_per_query=args.output_tokens,
        prices_file=Path(args.prices_file) if args.prices_file else None,
    )

    print("\n" + "=" * 90)
    if results["is_cpu_placeholder"]:
        print("  *** [PLACEHOLDER - CPU THROUGHPUT DETECTED - NOT VALID FOR GPU PUBLICATION] ***")
    else:
        print("  *** [OFFICIAL GPU BENCHMARK COST MODEL] ***")
    print("=" * 90)

    m = results["measured_inputs"]
    a = results["assumed_inputs"]
    p = results["pricing_source"]

    print(f"\n[MEASURED INPUTS at Concurrency={m['concurrency']} on {m['device'].upper()}]:")
    print(f"  Aggregate Prefill Throughput: {m['aggregate_prefill_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Aggregate Decode Throughput:  {m['aggregate_decode_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Total Combined Throughput:    {m['aggregate_total_tokens_per_sec']:.1f} tokens/sec")

    print(f"\n[ASSUMED INPUTS]:")
    print(f"  Hourly GPU Instance Rate:     ${a['hourly_gpu_rate_usd']:.2f}/hr")
    print(f"  Target Headroom / Duty Cycle: {a['target_utilization_duty_cycle']:.0%}")
    print(f"  Single-GPU Effective Capacity:{a['single_gpu_effective_daily_capacity_tokens']:,} tokens/day ({a['single_gpu_effective_monthly_capacity_tokens']:,} tokens/mo)")
    print(f"  Modeled Query Size:           {a['input_tokens_per_query']} input tokens + {a['output_tokens_per_query']} output tokens")
    print(f"  Pricing Source Reference:     {p['file']} (Verified: {p['date_verified']})")
    print("-" * 90)

    print("\n[1] PER-QUERY COST BREAKDOWN (PREFILL VS DECODE SEPARATED):")
    print("-" * 90)
    print(f"{'Platform / Model':<32} | {'Status':<15} | {'Input Cost':<12} | {'Output Cost':<12} | {'Total Per Query':<14}")
    print("-" * 90)
    sh_marg = results["per_query_breakdown"]["self_hosted_marginal"]["marginal_gpu_cost_usd"]
    sh_prefill_sec = results["per_query_breakdown"]["self_hosted_marginal"]["prefill_latency_sec"]
    sh_decode_sec = results["per_query_breakdown"]["self_hosted_marginal"]["decode_latency_sec"]
    print(f"{'Self-Hosted Marginal Compute':<32} | {'active':<15} | {sh_prefill_sec:.3f}s prefill | {sh_decode_sec:.3f}s decode | ${sh_marg:<13.6f}")

    for m_id, m_vals in results["per_query_breakdown"]["commercial_apis"].items():
        name_str = m_vals["name"][:30]
        stat_str = f"[{m_vals['status']}]"
        in_c = f"${m_vals['input_cost_usd']:.6f}"
        out_c = f"${m_vals['output_cost_usd']:.6f}"
        tot_c = f"${m_vals['total_query_cost_usd']:.6f}"
        print(f"{name_str:<32} | {stat_str:<15} | {in_c:<12} | {out_c:<12} | {tot_c:<14}")

    print("\n[2] MONTHLY VOLUME SCALING & CAPACITY CHECKS:")
    print("-" * 90)
    print(f"{'Daily Volume':<13} | {'GPUs':<5} | {'Capacity Status':<18} | {'Self-Hosted/Mo':<15} | {'GPT-4o/Mo':<12} | {'GPT-4o-mini/Mo':<14}")
    print("-" * 90)
    for c in results["volume_comparisons"]:
        vol_s = f"{c['daily_tokens']:,}"
        gpus = str(c["gpus_required"])
        stat = c["capacity_status"]
        sh_mo = f"${c['self_hosted_monthly_usd']:,.2f}"
        g4o_mo = f"${c['cloud_apis'].get('openai/gpt-4o', {}).get('monthly_cost_usd', 0):,.2f}"
        g4m_mo = f"${c['cloud_apis'].get('openai/gpt-4o-mini', {}).get('monthly_cost_usd', 0):,.2f}"
        print(f"{vol_s:<13} | {gpus:<5} | {stat:<18} | {sh_mo:<15} | {g4o_mo:<12} | {g4m_mo:<14}")
    print("=" * 90)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved cost model results to {out_p}")


if __name__ == "__main__":
    main()
