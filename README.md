# SpecPrefill for mlx-lm

Speculative Prefill (SpecPrefill) implementation for [mlx-lm](https://github.com/ml-explore/mlx-lm), ported from the [vllm-mlx PR #180](https://github.com/waybarrios/vllm-mlx/pull/180) reference.

Reduces TTFT (Time-To-First-Token) on long prompts by using a small draft model to identify important tokens, then sparse-prefilling the target model with only those tokens.

**Paper:** [Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)

---

## Benchmark Results

**Hardware:** Apple M1 Max 64GB  
**Target:** Qwen3.6-35B-A3B-mixed4_6  
**Draft:** Qwen3.5-0.8B-MLX-4bit-fp16  
**KV Cache:** K8/V4 asymmetric quantization

| Config | TTFT | pf_tps | Speedup | Quality |
|--------|------|--------|---------|---------|
| Baseline | 23.1s | 706 | 1.00x | -- |
| keep=0.5 | 18.1s | 905 | 1.28x | Preserved |
| keep=0.3 | 12.6s | 1298 | 1.83x | Preserved |
| keep=0.1 | 7.4s | 2226 | **3.11x** | Preserved |

At `keep_pct=0.1`, output is bit-identical to baseline for this prompt. See [benchmarks/](benchmarks/) for full reproduction scripts.

---

## Installation

### Prerequisites

- Python 3.10+
- `mlx-lm` >= 0.31.2
- A draft model (e.g. Qwen3.5-0.8B, Llama-3.2-1B)

```bash
pip install mlx-lm
```

### Step 1: Download This Repo

```bash
git clone https://github.com/shivam94/specprefill-mlx-lm.git
cd specprefill-mlx-lm
```

### Step 2: Apply Patches to Your mlx-lm Installation

```bash
# Find where mlx-lm is installed
MLX_LM=$(python -c "import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))")
echo "mlx-lm is at: $MLX_LM"

# Copy the core SpecPrefill algorithm
cp patches/specprefill.py "$MLX_LM/"

# Apply the integration patches
cd "$MLX_LM"
patch -p2 < ~/specprefill-mlx-lm/patches/generate_specprefill.patch
patch -p2 < ~/specprefill-mlx-lm/patches/cli_generate_specprefill.patch
patch -p2 < ~/specprefill-mlx-lm/patches/server_specprefill.patch

# Optional: patch benchmark.py for built-in benchmarking
patch -p2 < ~/specprefill-mlx-lm/patches/benchmark_specprefill.patch

# Verify nothing is broken
python -m py_compile specprefill.py generate.py server.py benchmark.py
```

If `patch` fails due to version mismatch, see [Manual Integration](#manual-integration) below.

### Step 3: Download a Draft Model

```bash
# Option A: Qwen3.5-0.8B (recommended, small and fast)
huggingface-cli download mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16 \
    --local-dir ~/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-fp16

# Option B: Llama-3.2-1B (if you use Llama target models)
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit \
    --local-dir ~/.cache/huggingface/hub/Llama-3.2-1B-Instruct-4bit
```

The draft model should be the same architecture family as your target model (Qwen for Qwen, Llama for Llama).

### Step 4: Verify It Works

```bash
python -m mlx_lm benchmark \
  --model Qwen3.6-35B-A3B-mixed4_6 \
  --prompt-tokens 16384 \
  --generation-tokens 20 \
  --specprefill-model ~/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-fp16 \
  --specprefill-keep-pct 0.3
```

If you see higher `prompt_tps` than without `--specprefill-model`, it's working.

---

### Manual Integration

If `patch` fails (e.g. mlx-lm version mismatch), integrate manually:

1. Copy `patches/specprefill.py` into your `mlx_lm/` directory.
2. In `generate.py`, add these 5 kwargs to `generate_step()` and `stream_generate()`:
   ```python
   specprefill_model: Optional[nn.Module] = None,
   specprefill_keep_pct: float = 0.3,
   specprefill_threshold: int = 8192,
   specprefill_lookahead: int = 8,
   specprefill_pool_kernel: int = 13,
   ```
3. In `generate_step()`, replace the normal prefill loop with:
   ```python
   use_specprefill = (
       specprefill_model is not None
       and total_prompt_tokens > specprefill_threshold
       and input_embeddings is None
   )
   if use_specprefill:
       # run score_tokens, select_chunks, sparse_prefill
   else:
       # normal prefill loop
   ```
4. In `server.py`, add `--specprefill-model` CLI arg and pass it to `stream_generate()`.

See `docs/SPECPREFILL_INTEGRATION.md` for full code snippets.

---

## Usage

### Server Mode

```bash
python -m mlx_lm server \
  --model Qwen3.6-35B-A3B-mixed4_6 \
  --host 127.0.0.1 --port 8000 \
  --kv-bits 8,4 --kv-group-size 64,32 \
  --prefill-step-size 2048 \
  --specprefill-model Qwen3.5-0.8B-MLX-4bit-fp16 \
  --specprefill-keep-pct 0.3 \
  --specprefill-threshold 8192 \
  --specprefill-lookahead 8
```

### CLI Generation

```bash
python -m mlx_lm generate \
  --model Qwen3.6-35B-A3B-mixed4_6 \
  --prompt "Very long prompt..." \
  --specprefill-model Qwen3.5-0.8B-MLX-4bit-fp16 \
  --specprefill-keep-pct 0.3
```

### Programmatic

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

---

## Benchmark Reproduction

```bash
cd benchmarks

# A/B comparison (baseline vs specprefill)
./run_ab_kv_benchmark.sh

# Keep_pct sweep (0.1, 0.3, 0.5)
python bench_keeppct_sweep.py

# Official benchmark module wrapper
python bench_specprefill_official.py \
  --model Qwen3.6-35B-A3B-mixed4_6 \
  --draft-model Qwen3.5-0.8B-MLX-4bit-fp16
```

---

## Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `keep_pct` | 0.3 | 0.1 - 0.5 | Fraction of tokens to keep. 0.1 = max speedup, 0.3 = safe default |
| `threshold` | 8192 | 4096 - 16384 | Min prompt length to trigger SpecPrefill |
| `lookahead` | 8 | 4 - 16 | Draft model decode steps for scoring |
| `pool_kernel` | 13 | 0 (off) - 21 | Smoothing kernel for attention scores |

From the paper: chunk_size=32 is fixed. `keep_pct=0.1` is their default for max TTFT reduction.

---

## Files

```
patches/
  specprefill.py                  # Core algorithm (742 lines)
  generate_specprefill.patch      # generate.py integration
  cli_generate_specprefill.patch  # generate.py CLI args
  server_specprefill.patch        # server.py integration
  benchmark_specprefill.patch     # benchmark.py integration

benchmarks/
  bench_keeppct_sweep.py          # Hyperparameter sweep
  bench_specprefill_official.py   # Official benchmark wrapper
  run_ab_kv_benchmark.sh          # Shell A/B benchmark
  bench_specprefill_official_results.md  # Our results

docs/
  SPECPREFILL_INTEGRATION.md      # Architecture docs
  PATCH_INVENTORY.md              # Prior patches (thinking models, etc.)
```

---

## Architecture

See [docs/SPECPREFILL_INTEGRATION.md](docs/SPECPREFILL_INTEGRATION.md) for:
- How RoPE position mapping works
- Hybrid model handling (Qwen3.5 GatedDeltaNet layers)
- Sparse prefill vs standard speculative decoding differences

---

## Prior Patches

This repo also documents other mlx-lm modifications:
- `preserve_thinking` for Qwen3.x chat templates
- `checkpoint_caching` for KV cache trimming
- PR #1102 memory optimization (documented, not applied)

See [docs/PATCH_INVENTORY.md](docs/PATCH_INVENTORY.md).

---

## License

Apache-2.0 (matches mlx-lm and the original SpecPrefill paper).
