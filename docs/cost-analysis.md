# Cost Analysis: Self-Hosted MI300X vs. Commercial Cloud APIs

## 1. Cost Modeling Formulation

The effective cost per 1M output tokens on self-hosted infrastructure is calculated as:

$$\text{Cost per 1M Tokens} = \left(\frac{\text{Hourly GPU Cost}}{\text{Throughput in Tokens/Hour}}\right) \times 1{,}000{,}000$$

Where:
- Hourly GPU Cost for AMD Instinct™ MI300X (192 GB) = **$1.99 / hr**.
- Throughput = Aggregate generated output tokens per hour across all concurrent worker requests.

## 2. API Baseline Prices (Dated: October 2026)

| Provider / Model | Input ($/1M tokens) | Output ($/1M tokens) | Blend (75% in / 25% out) |
|---|---|---|---|
| OpenAI GPT-4o | $2.50 | $10.00 | $4.38 |
| OpenAI GPT-4o-mini | $0.15 | $0.60 | $0.26 |
| Anthropic Claude 3.5 Sonnet | $3.00 | $15.00 | $6.00 |
| Anthropic Claude 3.5 Haiku | $0.80 | $4.00 | $1.60 |

## 3. Break-Even Analysis Table

| Model Class & Precision | Sustained Output Tokens/Sec | Total Tokens / Hour | Self-Hosted Cost / 1M Tokens | Break-Even vs GPT-4o-mini | Break-Even vs Claude 3.5 Sonnet |
|---|---|---|---|---|---|
| 8B BF16 | ~450 tok/s | 1,620,000 | ~$1.23 / 1M | ~5.2 hrs/day sustained | Cheaper at all volumes |
| 8B FP8 | ~850 tok/s | 3,060,000 | ~$0.65 / 1M | ~2.5 hrs/day sustained | Cheaper at all volumes |
| 30B BF16 | ~220 tok/s | 792,000 | ~$2.51 / 1M | Premium over mini | Break-even at 2.4 hrs/day |
| 70B BF16 | ~110 tok/s | 396,000 | ~$5.03 / 1M | Premium over mini | Break-even at 4.8 hrs/day |
| 70B FP8 | ~210 tok/s | 756,000 | ~$2.63 / 1M | Premium over mini | Break-even at 2.6 hrs/day |

*(Values will be updated with empirical empirical benchmark data from Phase C/D).*
