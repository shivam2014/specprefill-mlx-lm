#!/usr/bin/env python3
"""
A/B benchmark using mlx-lm's OFFICIAL benchmark module.

Compares baseline vs SpecPrefill on SAME commit by toggling
--specprefill-model in the built-in benchmark.

Usage:
    cd ~/mlx-env && source bin/activate
    python docs/mlx-lm/patches/bench_specprefill_official.py \
        --model /Users/shivam94/.cache/huggingface/hub/Qwen3.6-35B-A3B-mixed4_6 \
        --draft-model /Users/shivam94/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16 \
        --prompt-lengths 8192,16384,32768 \
        --generation-tokens 20 \
        --num-trials 3 \
        --output bench_specprefill_official

Output:
    bench_specprefill_official.json
    bench_specprefill_official.md
"""

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Any

# Import the official benchmark
from mlx_lm.benchmark import setup_arg_parser
from mlx_lm import load


def run_benchmark(args_list: List[str]) -> Dict[str, Any]:
    """Run mlx_lm benchmark with given args, capture output, parse results."""
    import mlx.core as mx
    from mlx_lm.benchmark import main as benchmark_main

    # Parse args
    parser = setup_arg_parser()
    args = parser.parse_args(args_list)

    # Run benchmark and capture output
    old_argv = sys.argv
    sys.argv = ["benchmark"] + args_list

    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        try:
            benchmark_main()
        except SystemExit:
            pass

    sys.argv = old_argv
    output = f.getvalue()

    # Parse "Averages: prompt_tps=X, generation_tps=Y, peak_memory=Z" line
    avg_line = None
    for line in output.strip().split('\n'):
        if line.startswith("Averages:"):
            avg_line = line
            break

    if avg_line is None:
        return {"error": "No averages found", "raw_output": output}

    # Parse key=value pairs
    result = {"raw_output": output}
    parts = avg_line.replace("Averages: ", "").split(", ")
    for part in parts:
        if "=" in part:
            k, v = part.split("=")
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v

    return result


def main():
    parser = argparse.ArgumentParser(description="SpecPrefill A/B using official benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt-lengths", default="8192,16384,32768")
    parser.add_argument("--generation-tokens", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", default="bench_specprefill_official")
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.prompt_lengths.split(",")]

    print("=" * 70)
    print("SpecPrefill A/B Benchmark (using mlx-lm official benchmark)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Draft: {args.draft_model}")
    print(f"Prompt lengths: {lengths}")
    print(f"Generation tokens: {args.generation_tokens}")
    print(f"Trials: {args.num_trials}")
    print("")

    results = []

    for n in lengths:
        print(f"\n{'='*60}")
        print(f"Prompt length: {n} tokens")
        print(f"{'='*60}")

        # Baseline
        print("Running BASELINE...")
        base_args = [
            "--model", args.model,
            "--prompt-tokens", str(n),
            "--generation-tokens", str(args.generation_tokens),
            "--num-trials", str(args.num_trials),
            "--prefill-step-size", str(args.prefill_step_size),
        ]
        base_result = run_benchmark(base_args)
        print(f"  Baseline: {base_result.get('prompt_tps', 0):.1f} prompt_tps, "
              f"{base_result.get('generation_tps', 0):.1f} generation_tps")

        # SpecPrefill
        print("Running SPECPREFILL...")
        spec_args = base_args + [
            "--specprefill-model", args.draft_model,
            "--specprefill-keep-pct", "0.3",
            "--specprefill-threshold", "8192",
            "--specprefill-lookahead", "8",
        ]
        spec_result = run_benchmark(spec_args)
        print(f"  SpecPrefill: {spec_result.get('prompt_tps', 0):.1f} prompt_tps, "
              f"{spec_result.get('generation_tps', 0):.1f} generation_tps")

        speedup = (spec_result.get('prompt_tps', 0) / base_result.get('prompt_tps', 1)
                   if base_result.get('prompt_tps', 0) > 0 else 0)

        results.append({
            "prompt_len": n,
            "baseline": base_result,
            "specprefill": spec_result,
            "speedup": speedup,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Tokens':>8} | {'BL pf_tps':>10} {'SP pf_tps':>10} {'Speedup':>8} | "
          f"{'BL dec_tps':>10} {'SP dec_tps':>10}")
    print("-" * 70)
    for r in results:
        bl = r["baseline"]
        sp = r["specprefill"]
        print(f"{r['prompt_len']:>8} | "
              f"{bl.get('prompt_tps', 0):>10.1f} {sp.get('prompt_tps', 0):>10.1f} {r['speedup']:>7.2f}x | "
              f"{bl.get('generation_tps', 0):>10.1f} {sp.get('generation_tps', 0):>10.1f}")

    # Save
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    md_path = Path(args.output).with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# SpecPrefill A/B Benchmark (Official mlx-lm benchmark)\n\n")
        f.write(f"**Model:** `{args.model}`\n")
        f.write(f"**Draft:** `{args.draft_model}`\n\n")
        f.write("| Tokens | Baseline pf_tps | SpecPrefill pf_tps | Speedup |\n")
        f.write("|--------|-----------------|--------------------|---------|\n")
        for r in results:
            f.write(f"| {r['prompt_len']} | {r['baseline'].get('prompt_tps', 0):.1f} | "
                    f"{r['specprefill'].get('prompt_tps', 0):.1f} | {r['speedup']:.2f}x |\n")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
