#!/usr/bin/env bash
# run_all.sh — one entry point for every experiment in docs/PLAN.md.
#
#   ./src/run_all.sh smoke                 # 1 doc, 2 langs, ~1 min, validates the pipeline
#   ./src/run_all.sh sweep                 # full quality sweep (PLAN E2)
#   ./src/run_all.sh needle                # retrieval probe (E4)
#   ./src/run_all.sh gen                   # 16-step generation (E5)
#   ./src/run_all.sh recency               # protect-last control (E6)
#   ./src/run_all.sh serve                 # vLLM serving sweep (E7, CUDA only)
#   ./src/run_all.sh analyze               # tables + figures (E8)
#
# Override anything with environment variables, e.g.
#   MODEL=Qwen/Qwen2.5-3B-Instruct DTYPE=float16 DEVICE=cuda DOCS=60 ./src/run_all.sh sweep

set -euo pipefail

PY=${PY:-python}
DEVICE=${DEVICE:-cpu}
MODEL=${MODEL:-meta-llama/Llama-3.2-1B-Instruct}
DTYPE=${DTYPE:-float32}
LANGS=${LANGS:-"english nepali hindi bengali tamil amharic"}
DOCS=${DOCS:-60}
SENTS=${SENTS:-10}
MAXTOK=${MAXTOK:-4096}
RATIOS=${RATIOS:-"0.3 0.6 0.8"}
TAG=${TAG:-}
SLUG=${SLUG:-$(basename "$MODEL")}
OUTDIR=${OUTDIR:-out}

mkdir -p "$OUTDIR" "$OUTDIR/tables" "$OUTDIR/figures"

case "${1:-}" in
  smoke)
    $PY src/make_data.py
    $PY src/kv_compress.py --selftest --device "$DEVICE" --dtype "$DTYPE"
    $PY src/kv_compress.py --device "$DEVICE" --dtype "$DTYPE" \
        --docs 1 --sents 3 --max-tokens 256 --ratios 0.3 \
        --methods evict_snapkv merge_snapkv --langs english nepali \
        --equal-memory --out "$OUTDIR/smoke.csv" --tag "$TAG"
    ;;

  sweep)
    $PY src/kv_compress.py --device "$DEVICE" --dtype "$DTYPE" \
        --langs $LANGS --docs "$DOCS" --sents "$SENTS" --max-tokens "$MAXTOK" \
        --ratios $RATIOS --equal-memory --append \
        --out "$OUTDIR/compress_${SLUG}.csv" --tag "$TAG"
    ;;

  needle)
    $PY src/kv_compress.py --device "$DEVICE" --dtype "$DTYPE" \
        --langs $LANGS --docs "$DOCS" --sents "$SENTS" --max-tokens "$MAXTOK" \
        --ratios $RATIOS --equal-memory --append --needle \
        --out "$OUTDIR/needle_${SLUG}.csv" --tag "$TAG"
    ;;

  gen)
    $PY src/kv_compress.py --device "$DEVICE" --dtype "$DTYPE" \
        --langs $LANGS --docs "${GEN_DOCS:-20}" --sents "$SENTS" --max-tokens "$MAXTOK" \
        --ratios 0.3 0.6 0.8 --equal-memory --append --gen-steps 16 \
        --methods evict_snapkv evict_snapkv_word merge_snapkv merge_entropy \
        --out "$OUTDIR/gen_${SLUG}.csv" --tag "$TAG"
    ;;

  recency)
    $PY src/kv_compress.py --device "$DEVICE" --dtype "$DTYPE" \
        --langs $LANGS --docs "${REC_DOCS:-20}" --sents "$SENTS" --max-tokens "$MAXTOK" \
        --ratios 0.3 0.6 --equal-memory --append --protect-last 32 \
        --out "$OUTDIR/recency_${SLUG}.csv" --tag "$TAG"
    ;;

  serve)
    $PY src/serve_bench.py --mode offline --model "$MODEL" --dtype "${SERVE_DTYPE:-float16}" \
        --langs $LANGS --docs "${SERVE_DOCS:-16}" --batch-sizes 1 2 4 8 16 32 64 \
        --out "$OUTDIR/serve_${SLUG}.csv"
    ;;

  analyze)
    $PY src/analyze.py --inputs "$OUTDIR/compress_*.csv" \
        --needle "$OUTDIR"/needle_*.csv --gen "$OUTDIR"/gen_*.csv \
        --serve "$OUTDIR"/serve_*.csv --outdir "$OUTDIR"
    ;;

  *)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
