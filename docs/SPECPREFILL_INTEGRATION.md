# SpecPrefill Integration for mlx-lm

## Overview

This folder contains a port of **SpecPrefill** (Speculative Prefill) to the standard
`mlx-lm` package (v0.31.2). SpecPrefill reduces TTFT (Time-To-First-Token) on long
prompts by using a small draft model to score token importance, then sparse-prefilling
the target model with only the most important token chunks.

Reference: [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token
Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)

## Files

| File | Description |
|------|-------------|
| `specprefill.py` | Core algorithm: `score_tokens`, `select_chunks`, `sparse_prefill`, `cleanup_rope` |
| `generate_specprefill.patch` | Patch for `mlx_lm/generate.py` — adds specprefill params to `generate_step` and `stream_generate` |
| `server_specprefill.patch` | Patch for `mlx_lm/server.py` — adds CLI args, model loading, and integration in `_serve_single` |
| `specprefill-pr180/` | Original vllm-mlx PR #180 reference implementation (28454 bytes) |
| `specprefill_paper.pdf` | Full paper PDF |

## What Changed

### 1. `mlx_lm/specprefill.py` (new file)

Copied from `patches/specprefill.py` into the installed `mlx_lm` package.

Key functions:
- `score_tokens(draft_model, tokens, n_lookahead=8, ...)` — prefill draft, run lookahead decode, capture queries, compute attention-based importance scores
- `select_chunks(importance, keep_pct=0.3, chunk_size=32)` — chunk-level top-k selection with avg-pool smoothing
- `sparse_prefill(model, tokens, selected_indices, cache, ...)` — prefill target model with only selected tokens, preserving original positions via `_PositionMappedRoPE` and `_OffsetAdjustedRoPE`
- `cleanup_rope(model)` — restore original RoPE after generation

### 2. `mlx_lm/generate.py`

Added to `generate_step` signature:
```python
specprefill_model: Optional[nn.Module] = None,
specprefill_keep_pct: float = 0.3,
specprefill_threshold: int = 8192,
specprefill_lookahead: int = 8,
specprefill_pool_kernel: int = 13,
```

When `specprefill_model` is provided and prompt length > threshold:
1. Runs `score_tokens` with the draft model
2. Runs `select_chunks` to get selected token indices
3. Runs `sparse_prefill` on the target model (populates cache, returns logits)
4. Samples first decode token from sparse-prefill logits
5. Skips normal prefill loop
6. Continues with standard decode loop

`stream_generate` wraps the generation loop in `try/finally` to call `cleanup_rope` if specprefill was used.

### 3. `mlx_lm/server.py`

New CLI args:
```bash
--specprefill-model PATH          # Small model for scoring (e.g. Qwen3.5-0.8B)
--specprefill-keep-pct 0.3        # Fraction of tokens to keep
--specprefill-threshold 8192      # Min prompt length to trigger
--specprefill-lookahead 8         # Lookahead decode steps
--specprefill-pool-kernel 13      # Smoothing kernel for attention scores
```

`ModelProvider` loads `specprefill_model` alongside the main model.

`_serve_single` passes specprefill kwargs to `stream_generate` when the model is loaded.

## Usage

### Server

```bash
cd ~/mlx-env && source bin/activate
python -m mlx_lm server \
  --model /Users/shivam94/.cache/huggingface/hub/Qwen3.6-35B-A3B-mixed4_6 \
  --host 127.0.0.1 --port 8000 \
  --kv-bits 8,4 --kv-group-size 64,32 \
  --prefill-step-size 2048 \
  --specprefill-model /Users/shivam94/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16 \
  --specprefill-keep-pct 0.3 \
  --specprefill-threshold 8192 \
  --specprefill-lookahead 8
```

### Programmatic (local generation)

```python
from mlx_lm import load, stream_generate
from mlx_lm.specprefill import cleanup_rope

model, tokenizer = load("Qwen3.6-35B-A3B-mixed4_6")
draft_model, _ = load("Qwen3.5-0.8B-MLX-4bit-fp16")

for response in stream_generate(
    model, tokenizer,
    prompt="Very long prompt...",
    specprefill_model=draft_model,
    specprefill_keep_pct=0.3,
    specprefill_threshold=8192,
    specprefill_lookahead=8,
):
    print(response.text, end="")
```

## Architecture Notes

### RoPE Position Mapping

Selected tokens keep their original absolute positions. After sparse-prefilling N tokens
from a total prompt of M, cache.offset = N but decode needs position M. The
`_OffsetAdjustedRoPE` wrapper adds (M - N) to each RoPE offset call during decode.

### Hybrid Model Handling (Qwen3.5)

Qwen3.5 has both full-attention layers (KVCache) and GatedDeltaNet linear attention
layers (ArraysCache). `sparse_prefill` processes all tokens through GatedDeltaNet
layers (required for correct recurrent state) but only selected tokens through
full-attention layers. This is lossy but acceptable per the paper.

### Early EOS in Lookahead

The vLLM reference implementation stops lookahead early when EOS is hit. The current
MLX port always runs the full `n_lookahead` steps. This is a minor optimization gap.

## Known Limitations

1. **No input_embeddings support** — specprefill is skipped if `input_embeddings` is provided
2. **No per-request API overrides** — specprefill params are CLI-only currently
3. **No batched mode support** — only single-request `_serve_single` path is integrated
4. **Cleanup on exception** — `cleanup_rope` runs in `finally`, but if the process crashes mid-generation, RoPE wrappers may persist

## Benchmarks

See `bench_kvsplit_k8v4.txt` and `benchmark-plan.md` for performance baselines.
Expected TTFT reduction: ~3-7x on prompts > 8K tokens (depending on keep_pct).

## Patch Application

To apply to a fresh mlx-lm installation:

```bash
cd ~/mlx-env/lib/python3.12/site-packages/mlx_lm
cp ~/mlx-env/docs/mlx-lm/patches/specprefill.py .
patch -p2 < ~/mlx-env/docs/mlx-lm/patches/generate_specprefill.patch
patch -p2 < ~/mlx-env/docs/mlx-lm/patches/server_specprefill.patch
```

## References

- Paper: https://arxiv.org/abs/2502.02789
- Original vLLM implementation: https://github.com/Jingyu6/speculative_prefill
- vllm-mlx PR #180: waybarrios/vllm-mlx (merged Mar 2026)
