#!/usr/bin/env python3
"""
Sweep keep_pct values for SpecPrefill on 35B model.
Loads model once, runs baseline + 3 keep_pct configs.
2 trials each (first is warmup, dropped if >=3; here we keep both since only 2).
"""

import time
import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = "/Users/shivam94/.cache/huggingface/hub/Qwen3.6-35B-A3B-mixed4_6"
DRAFT = "/Users/shivam94/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16"
PROMPT_LEN = 16384
GEN_TOKENS = 20
NUM_TRIALS = 2
KEEP_PCTS = [0.1, 0.3, 0.5]
KV_BITS = (8, 4)
KV_GROUP_SIZE = (64, 32)
PREFILL_STEP = 2048

PROMPT_CHUNK = (
    "The history of artificial intelligence began in the 1950s with the "
    "foundational work of Alan Turing and the Dartmouth Conference. "
    "Machine learning, a subset of AI, enables systems to learn from data. "
    "Deep learning uses neural networks with many layers to model complex patterns. "
    "Transformers, introduced in 2017, revolutionized natural language processing. "
    "Large language models like GPT and Llama can generate human-like text. "
    "Quantization reduces model size by using lower precision weights. "
    "KV cache compression speeds up inference by reducing memory bandwidth. "
)


def make_prompt(tokenizer, n_tokens):
    toks = tokenizer.encode(PROMPT_CHUNK)
    repeats = (n_tokens // len(toks)) + 1
    long_toks = (toks * repeats)[:n_tokens]
    text = tokenizer.decode(long_toks)
    return tokenizer.encode(text)


def run_trial(model, tokenizer, prompt_tokens, draft_model=None, keep_pct=None, seed=42):
    mx.random.seed(seed)
    prompt_mx = mx.array(prompt_tokens)

    # Warmup
    for _ in stream_generate(model, tokenizer, prompt_mx[:10], max_tokens=2):
        pass
    mx.clear_cache()

    tic = time.perf_counter()
    kwargs = {
        "max_tokens": GEN_TOKENS,
        "prefill_step_size": PREFILL_STEP,
        "kv_bits": KV_BITS,
        "kv_group_size": KV_GROUP_SIZE,
    }
    if draft_model is not None:
        kwargs.update({
            "specprefill_model": draft_model,
            "specprefill_keep_pct": keep_pct,
            "specprefill_threshold": 8192,
            "specprefill_lookahead": 8,
            "specprefill_pool_kernel": 13,
        })

    results = []
    for response in stream_generate(model, tokenizer, prompt_mx, **kwargs):
        if response.prompt_tps is not None and not results:
            ttft = time.perf_counter() - tic
            prompt_tps = response.prompt_tps
        results.append(response)

    total = time.perf_counter() - tic
    decode_toks = len(results) - 1
    decode_tps = decode_toks / (total - ttft) if (total - ttft) > 0 else 0
    return ttft, prompt_tps, decode_tps


def avg_trials(trials):
    return sum(t[0] for t in trials) / len(trials), \
           sum(t[1] for t in trials) / len(trials), \
           sum(t[2] for t in trials) / len(trials)


print("=" * 70)
print("SpecPrefill keep_pct sweep")
print("=" * 70)
print(f"Model: {MODEL}")
print(f"Draft: {DRAFT}")
print(f"Prompt: {PROMPT_LEN} tokens, Gen: {GEN_TOKENS} tokens")
print(f"Trials: {NUM_TRIALS} (keeping both, no warmup drop)")
print(f"KV: K{KV_BITS[0]}/V{KV_BITS[1]}")
print()

print("Loading models...")
t0 = time.time()
target, tokenizer = load(MODEL)
draft, _ = load(DRAFT)
print(f"Loaded in {time.time()-t0:.1f}s\n")

prompt_tokens = make_prompt(tokenizer, PROMPT_LEN)

# Baseline
print("Running BASELINE...")
baseline_trials = []
for t in range(NUM_TRIALS):
    ttft, pt, dt = run_trial(target, tokenizer, prompt_tokens, seed=42+t)
    baseline_trials.append((ttft, pt, dt))
    print(f"  T{t+1}: TTFT={ttft:.3f}s  pf_tps={pt:.0f}")
bl_ttft, bl_pt, bl_dt = avg_trials(baseline_trials)
print(f"  AVG: TTFT={bl_ttft:.3f}s  pf_tps={bl_pt:.0f}\n")

# SpecPrefill sweep
results = []
for kp in KEEP_PCTS:
    print(f"Running SPECPREFILL keep_pct={kp}...")
    trials = []
    for t in range(NUM_TRIALS):
        ttft, pt, dt = run_trial(target, tokenizer, prompt_tokens, draft_model=draft, keep_pct=kp, seed=42+t)
        trials.append((ttft, pt, dt))
        print(f"  T{t+1}: TTFT={ttft:.3f}s  pf_tps={pt:.0f}")
    avg_ttft, avg_pt, avg_dt = avg_trials(trials)
    speedup = bl_ttft / avg_ttft if avg_ttft > 0 else 0
    print(f"  AVG: TTFT={avg_ttft:.3f}s  pf_tps={avg_pt:.0f}  speedup={speedup:.2f}x\n")
    results.append({"keep_pct": kp, "ttft": avg_ttft, "pt": avg_pt, "speedup": speedup})

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Config':>12} | {'TTFT (s)':>10} | {'pf_tps':>10} | {'Speedup':>8}")
print("-" * 55)
print(f"{'Baseline':>12} | {bl_ttft:>10.3f} | {bl_pt:>10.0f} | {'1.00x':>8}")
for r in results:
    print(f"{'keep=' + str(r['keep_pct']):>12} | {r['ttft']:>10.3f} | {r['pt']:>10.0f} | {r['speedup']:>7.2f}x")
