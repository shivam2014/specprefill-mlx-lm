#!/bin/bash
# A/B benchmark with KV cache quantization (matching server flags)
set -e

MODEL="/Users/shivam94/.cache/huggingface/hub/Qwen3.6-35B-A3B-mixed4_6"
DRAFT="/Users/shivam94/.cache/huggingface/hub/mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16"
OUTDIR="/Users/shivam94/mlx-env/docs/mlx-lm/patches"

COMMON_ARGS="--model $MODEL --generation-tokens 20 --num-trials 3 --prefill-step-size 2048 --kv-bits 8,4 --kv-group-size 64,32"

PYTHON="/Users/shivam94/mlx-env/bin/python"

echo "============================================================"
echo "SpecPrefill A/B with KV quantization"
echo "============================================================"

for N in 8192 16384 32768; do
    echo ""
    echo "Prompt length: $N tokens"
    echo "---"

    echo "BASELINE..."
    $PYTHON -m mlx_lm benchmark $COMMON_ARGS --prompt-tokens $N 2>&1 | tee "$OUTDIR/bench_kv_${N}_baseline.txt"

    echo "SPECPREFILL..."
    $PYTHON -m mlx_lm benchmark $COMMON_ARGS --prompt-tokens $N \
        --specprefill-model "$DRAFT" \
        --specprefill-keep-pct 0.3 \
        --specprefill-threshold 8192 \
        --specprefill-lookahead 8 2>&1 | tee "$OUTDIR/bench_kv_${N}_specprefill.txt"
done

echo ""
echo "Done. Results in $OUTDIR/bench_kv_*.txt"
