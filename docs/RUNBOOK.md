# RUNBOOK — how to run everything

Companion to `docs/PLAN.md` (what to measure) and `docs/PAPER.md` (what to write).
Every command below is copy-pasteable from the repository root.

---

## 0. TL;DR

```bash
# local sanity check (~2 min, Apple Silicon)
HF_HUB_OFFLINE=1 PY=llm_env/bin/python DEVICE=mps ./src/run_all.sh smoke

# data + fertility table
llm_env/bin/python src/make_data.py --tokenizer meta-llama/Llama-3.2-1B-Instruct

# full quality sweep (GPU; see §5 for sharding)
MODEL=meta-llama/Llama-3.2-1B-Instruct DTYPE=float32 DEVICE=cuda DOCS=60 \
  ./src/run_all.sh sweep

# analysis + figures
llm_env/bin/python src/analyze.py --inputs "out/compress_*.csv" --outdir out
```

---

## 1. Environments

| Host | Interpreter | Notes |
|---|---|---|
| Mac (M1 Pro, MPS) | `llm_env/bin/python` | torch 2.15.0, transformers 5.22.1, pandas 3.1.0. **`.venv` is a vLLM install and has no pandas — do not use it for these scripts.** |
| Kaggle (2×T4) | system `python` | 16 GB per T4, ~30 GPU-h/week, 9 h per committed run, no bf16 (Turing). |

The code is validated against `transformers==5.22.1`. On any new host, the first
thing to run is the parity test:

```bash
python src/kv_compress.py --selftest --device cuda --dtype float32
# expect: max |manual - HF| logit difference < 1e-3
```

If that fails, **do not run experiments** — the hand-written forward pass no longer
matches Hugging Face and every number downstream is wrong.

Local model cache already contains `meta-llama/Llama-3.2-1B-Instruct` (2.3 GB).
Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on the Mac to avoid network calls.

---

## 2. Git

The working repo is **diverged** from `origin/main` (15 ahead, 5 behind), so do not
push `main`. Put this work on its own branch:

```bash
git status                     # review: README.MD is modified, several files untracked
git checkout -b paper/scale-up
git add src docs data/flores200/parallel_6lang.csv data/flores200/fertility_*.csv
git commit -m "Six-language equal-memory KV compression study: data, sweep, analysis"
git push -u origin paper/scale-up
```

Kaggle clones that branch (§9). Keep `out/*.csv` committed too — they are small and
they are the evidence.

---

## 3. Data

```bash
python src/make_data.py                                   # download + align 6 languages
python src/make_data.py --tokenizer meta-llama/Llama-3.2-1B-Instruct
python src/make_data.py --tokenizer Qwen/Qwen2.5-3B-Instruct
```

Outputs:

| File | Contents |
|---|---|
| `data/flores200/parallel_6lang.csv` | 2,009 aligned sentences, one column per language |
| `data/flores200/raw/*` | original FLORES files (cached, re-downloadable) |
| `data/flores200/fertility_<model>.csv` | tokens/sentence, expansion vs English, equal-memory ratio |

Fertility is **per tokenizer**. Never pool the Llama and Qwen fertility tables.

---

## 4. Quality sweep (the core experiment)

Local smoke test — one document, two languages, two policies:

```bash
HF_HUB_OFFLINE=1 PY=llm_env/bin/python DEVICE=mps ./src/run_all.sh smoke
```

Full run on one GPU:

```bash
python src/kv_compress.py \
  --device cuda --dtype float32 \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --langs english nepali hindi bengali tamil amharic \
  --docs 60 --sents 10 --max-tokens 4096 \
  --ratios 0.3 0.6 0.8 --equal-memory \
  --out out/compress_llama1b.csv
```

What the flags do:

| Flag | Why it exists |
|---|---|
| `--equal-memory` | adds each language's exact $r_\ell = 1 - P_{en}/P_\ell$ to the sweep, so no interpolation is needed for the headline number |
| `--doc-start` | first document index; used for sharding across sessions |
| `--append` | merge into an existing CSV and de-duplicate, so a killed run resumes |
| `--max-tokens 4096` | safety cap only. It must exceed the longest Tamil/Amharic document; a binding cap truncates only the high-fertility languages and biases $r_\ell$ downwards. Every row records `truncated` and `n_tokens_full`. |
| `--protect-last 32` | recency control (PLAN E6) |
| `--dtype float16` | only for Qwen-3B on T4; keep Llama-1B in FP32 |
| `--tag` | overrides the model tag written to every row |

Output columns worth knowing: `delta_nll` (per-token), `ratio_kind`
(`grid` or `equal_mem`), `ctx_len`, `ref_ctx_len`, `equal_mem_ratio`, `top1_agree`.

### Sharding across Kaggle sessions

60 documents × 6 languages × 10 policies × 3 ratios is ~2 T4-hours, comfortably
inside one 9-hour session. If you want to parallelise across the two T4s or across
accounts, give each run a disjoint document range and its own file:

```bash
# session A
python src/kv_compress.py --docs 20 --doc-start 0  ... --out out/compress_llama1b_s0.csv
# session B
python src/kv_compress.py --docs 20 --doc-start 20 ... --out out/compress_llama1b_s1.csv
# session C
python src/kv_compress.py --docs 20 --doc-start 40 ... --out out/compress_llama1b_s2.csv
```

Analysis globs `out/compress_*.csv`, so shards merge automatically.

---

## 5. Retrieval, generation, recency control

```bash
python src/kv_compress.py --device cuda --langs english nepali hindi bengali tamil amharic \
  --docs 40 --needle --equal-memory --out out/needle_llama1b.csv

python src/kv_compress.py --device cuda --langs english nepali hindi bengali tamil amharic \
  --docs 20 --gen-steps 16 --equal-memory \
  --methods evict_snapkv evict_snapkv_word merge_snapkv merge_entropy \
  --out out/gen_llama1b.csv

python src/kv_compress.py --device cuda --langs english nepali hindi bengali tamil amharic \
  --docs 20 --ratios 0.3 0.6 --protect-last 32 --equal-memory \
  --out out/recency_llama1b.csv
```

`--gen-steps` is non-incremental (16 full forwards per configuration). Restrict it
to the four policies above and 20 documents, or it will dominate the GPU budget.

---

## 6. Serving measurement

Three modes, in order of preference:

```bash
# 1) analytic capacity — no GPU, validates prompt sets
python src/serve_bench.py --mode tokens --model meta-llama/Llama-3.2-1B-Instruct

# 2) vLLM with PagedAttention (the number the paper wants)
VLLM_ATTENTION_BACKEND=XFORMERS python src/serve_bench.py --mode offline \
  --model meta-llama/Llama-3.2-1B-Instruct --dtype float16 \
  --batch-sizes 1 2 4 8 16 32 64 --out out/serve_llama1b.csv

# 3) fallback if vLLM has no sm_75 backend on the T4
python src/serve_bench.py --mode hf --model meta-llama/Llama-3.2-1B-Instruct \
  --dtype float16 --out out/serve_llama1b_hf.csv
```

Prompts are matched on **content** (same ten sentences), never truncated to equal
token counts — that would delete the effect being measured. `--prompt-tokens N`
exists only as a latency-matched control.

If mode 3 is used, the paper must say "eager attention, contiguous KV cache; no
PagedAttention", and the capacity claim rests on the analytic table plus the
measured throughput curve.

---

## 7. Analysis

```bash
python src/analyze.py \
  --inputs "out/compress_*.csv" \
  --needle out/needle_llama1b.csv \
  --gen out/gen_llama1b.csv \
  --outdir out
```

Produces:

| Path | Contents |
|---|---|
| `out/tables/expansion.csv` | per-model token expansion and $r_\ell$ |
| `out/tables/equal_memory.csv` | ΔPPL %, bootstrap CI, top-1 agreement at equal memory |
| `out/tables/regression.csv` | ΔPPL % vs log(expansion) slope, $R^2$, p |
| `out/tables/paired_evict_vs_merge.csv` | Wilcoxon + Holm for the operator test |
| `out/tables/needle_recall.csv` | exact recall per language/policy/ratio |
| `out/tables/gen_prefix.csv` | fraction of the uncompressed greedy prefix reproduced |
| `out/tables/serve.csv` | max batch without OOM, throughput, TTFT/TPOT, VRAM, preemptions |
| `out/figures/fig1…fig5` | PNG + PDF, 300 dpi |

Sanity checks before believing anything:

- `expansion.csv` should reproduce §3 of `PLAN.md` for Llama (2.5×–7.6×).
- `equal_memory.csv` should have one row per (model, policy, language).
- If `regression.csv` is empty, you have fewer than three non-English languages
  in the input — check the glob.

---

## 8. Figures and manuscript

Figures are generated by `analyze.py`. The manuscript outline, section word
budgets, figure captions and the claims ledger live in `docs/PAPER.md`.

Build order for the 3-day draft:

1. Freeze `out/tables/*.csv`.
2. Write §3 (method) and §4 (results) from the tables — never from memory.
3. Write §1–2 last, so the intro promises exactly what §4 delivers.
4. Re-read the abstract against `equal_memory.csv` line by line.

---

## 9. Kaggle recipe

1. **Push the branch** (§2). The repo is public, so cloning needs no token.
2. New Notebook → *File → Import Notebook* → `kaggle/kv_sweep.ipynb`, or paste the
   cells from that file.
3. Settings → Accelerator **GPU T4 ×2**, Internet **On**.
4. Add-ons → Secrets → add `HF_TOKEN` (needed for the gated Llama weights).
5. Set the config cell: `SHARD`, `DOC_START`, `DOCS`, `MODEL`, `DTYPE`.
6. *Save Version → Save & Run All (Commit)*. A 9-hour cap applies; the sweep
   checkpoints after every document, so a timeout costs at most one document.
7. Download `out/*.csv` from the version's Output tab, drop them into the local
   `out/`, run `analyze.py`.

Only one account is needed: the whole matrix is ~6 T4-hours against a 30 h/week
quota. Pooling quota across accounts risks the accounts and saves nothing.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pandas` | ran with `.venv` | use `llm_env/bin/python` |
| `fp16 ... requires --device cuda` | MPS/CPU has no fp16 path | `--dtype float32` |
| `CUDA out of memory` on Qwen-3B | FP32 weights (12.4 GB) + activations | `--dtype float16` |
| `return_offsets_mapping` KeyError | slow tokenizer | the script forces `use_fast=True`; check the model id |
| vLLM: no supported attention backend | T4 is sm_75 | `VLLM_ATTENTION_BACKEND=XFORMERS`, else `--mode hf` |
| Kaggle run killed at 9 h | session cap | smaller `--docs`, `--append`, resume |
| `--append` duplicates rows | key columns changed between runs | keep `--tag`, `--sents`, `--protect-last` identical |
| `[warn] TRUNCATED` | `--max-tokens` below the Tamil/Amharic document length | raise it to 4096+; the run is invalid for that language otherwise |
| Selftest diff ≫ 1e-3 | transformers version drift | pin `transformers==5.22.1` |

---

## 11. Time ledger (one T4)

| Step | Estimate |
|---|---|
| `make_data.py` (6 langs, 2 tokenizers) | 5 min |
| E2 sweep, 60 docs, Llama-1B FP32 | ~3 h |
| E3 sweep, 60 docs, Qwen-3B FP16 | ~1.5 h |
| E4 needle, 40 docs | ~1.5 h |
| E5 generation, 20 docs | ~2 h |
| E6 recency control | ~45 min |
| E7 serving | ~1 h |
| Analysis + figures | 5 min |
| **Total** | **~9–10 h** |

Estimates assume 10-sentence documents at `--max-tokens 4096`, i.e. contexts of
~220 (EN) to ~2,100 (TA/AM) tokens. Shard E2 into three 20-document runs so no
single session approaches the 9-hour cap.
