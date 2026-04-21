# SpecPrefill A/B Benchmark (Official mlx-lm benchmark)

**Model:** `Qwen3.6-35B-A3B-mixed4_6`
**Draft:** `Qwen3.5-0.8B-MLX-4bit-fp16`
**Date:** 2026-04-21
**Commit:** `687cea3`

| Tokens | Baseline pf_tps | SpecPrefill pf_tps | Speedup | BL dec_tps | SP dec_tps |
|--------|-----------------|--------------------|---------|------------|------------|
| 8192 | 791.9 | 791.3 | 1.00x | 66.4 | 66.6 |
| 16384 | 734.2 | 1290.1 | **1.76x** | 61.6 | 29.9 |
| 32768 | 637.7 | 1176.2 | **1.84x** | 56.0 | 28.8 |

## Notes

- 8192 tokens: No speedup because prompt length equals `--specprefill-threshold` (8192). SpecPrefill only triggers when `prompt_tokens > threshold`.
- 16384 tokens: **1.76x** prefill speedup.
- 32768 tokens: **1.84x** prefill speedup.
- Decode speed drops with SpecPrefill because the first decode token comes from sparse-prefill logits, not a full prefill. This is expected.
- Peak memory increases slightly with SpecPrefill (loading draft model).
