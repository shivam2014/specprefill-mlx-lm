# Patch Inventory: mlx-lm Modifications

## Current State

All modifications tracked in git at `~/mlx-env/lib/python3.12/site-packages/mlx_lm/.git`
Baseline: `b13ea73` = vanilla mlx-lm v0.31.2

---

## Prior Patches (Applied Before SpecPrefill Work)

| Patch | File | Status | Description |
|-------|------|--------|-------------|
| 01_preserve_thinking | `server.py` | **APPLIED** | Qwen3.x `preserve_thinking=True` for stable cache keys |
| 02_checkpoint_caching | `server.py` | **APPLIED** | Trim KV cache before `<think>` token for cache hits |
| pr-1102-memory-opt | `convert.py` | NOT APPLIED | Per-layer eval during quantization (documented only) |

### Verify Applied Patches

```bash
# Check preserve_thinking
grep -n "preserve_thinking" ~/mlx-env/lib/python3.12/site-packages/mlx_lm/server.py
# Output: line 541

# Check checkpoint_caching
grep -n "think_start_id" ~/mlx-env/lib/python3.12/site-packages/mlx_lm/server.py
# Output: lines 561, 601, 651

# Check PR-1102
grep -n "_eval_quantized_per_layer" ~/mlx-env/lib/python3.12/site-packages/mlx_lm/convert.py
# Output: none (not applied)
```

---

## SpecPrefill Patches (Our Work)

| Commit | File(s) | Description |
|--------|---------|-------------|
| `258df86` | `specprefill.py`, `generate.py`, `server.py` | Core specprefill integration (from vllm-mlx PR #180) |
| `cd59af2` | `generate.py` | Fix sampler input shape (1D vs 2D) |
| `8941996` | `generate.py` | Add `--specprefill-*` CLI args to `mlx_lm generate` |
| `687cea3` | `benchmark.py` | Add `--specprefill-*` args to `mlx_lm benchmark` |
| `fc4ba24` | `benchmark.py` | Add `--kv-bits`, `--kv-group-size` to benchmark |
| `6814251` | docs | Document KV benchmark results |

---

## Benchmark Scripts (NOT for upstream PR)

| Script | Description | Keep Separate |
|--------|-------------|---------------|
| `bench_ab.py` | Cross-commit A/B framework | Example/tools repo |
| `bench_specprefill_ab.py` | Same-commit toggle benchmark | Example/tools repo |
| `bench_specprefill_official.py` | Wrapper around built-in benchmark | Example/tools repo |

---

## Clean PR Structure

For upstream mlx-lm PR, submit ONLY:

```
mlx_lm/specprefill.py          (new file, 742 lines)
mlx_lm/generate.py             (+~140 lines)
mlx_lm/server.py               (+~40 lines, specprefill ONLY)
mlx_lm/benchmark.py            (+~70 lines)
```

**EXCLUDE from specprefill PR:**
- `preserve_thinking` logic (separate PR for thinking models)
- `checkpoint_caching` logic (separate PR for thinking models)
- `bench_ab.py`, `bench_specprefill_ab.py` (benchmark tools)

---

## How to Extract Clean PR

```bash
cd ~/mlx-env/lib/python3.12/site-packages/mlx_lm

# Create clean branch from baseline
git checkout -b specprefill-pr b13ea73

# Cherry-pick ONLY specprefill commits
git cherry-pick 258df86  # core
git cherry-pick cd59af2  # sampler fix
git cherry-pick 8941996  # CLI args
git cherry-pick 687cea3  # benchmark support
git cherry-pick fc4ba24  # kv-bits in benchmark

# Remove thinking-model patches from server.py if they leaked in
# (258df86 might include them since they were already in your server.py)
```

**Note**: Your `server.py` at `258df86` includes both specprefill AND thinking patches. You need to manually separate them for a clean PR.
