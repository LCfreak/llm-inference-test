# START HERE — what to do, in order

Everything is built and pushed. The branch `paper/scale-up` is live and public, so
Kaggle can clone it with no token. What is left is GPU time and writing.

Estimated total: **2 working days of GPU + 1 day of writing** for the arXiv draft.

---

## Step 0 — today, 5 minutes

Nothing to run. Confirm you can see the branch:

<https://github.com/LCfreak/llm-inference-test/tree/paper/scale-up>

If it is there, the code, data and docs are published and reproducible.

---

## Step 1 — Kaggle, one shard (≈40 min)

This is the run that proves the whole pipeline works on the real card. Do **one**
shard before committing to the full matrix.

1. <https://www.kaggle.com/notebooks> → **New Notebook** → *File → Import Notebook*
   → upload `kaggle/kv_sweep.ipynb` from your local checkout.
2. Right panel:
   - **Accelerator: GPU T4 ×2**
   - **Internet: On**
   - **Add-ons → Secrets → Add** a secret named `HF_TOKEN` (your Hugging Face
     read token; the Llama weights are gated).
3. In the `CONFIG` cell, leave everything as-is (`DOC_START=0`, `DOCS=20`,
   Llama-1B, fp32).
4. **Save Version → Save & Run All (Commit)**.

### What you must see

| Cell | Expected |
|---|---|
| Diagnostics | `sm_75`, two T4s, ~15.6 GB each |
| Parity gate | `max |manual - HF| logit difference: < 1e-3` then `PARITY OK` |
| Data | `wrote data/flores200/parallel_6lang.csv (2009 rows)` |
| Sweep | `doc 0 done (…s, N rows)` … `doc 19 done` |
| Sweep log | **no `[warn] … TRUNCATED` line** |

If you see `TRUNCATED`, the run is invalid for that language — raise `MAX_TOKENS`
to 4096 (already the default) and check you did not override it.

Download `out/compress_llama1b_s0.csv` from the version's **Output** tab.

---

## Step 2 — the rest of the quality sweep (≈4 h, can overlap)

Same notebook, change `CONFIG` each time, one version per shard:

| # | MODEL | DTYPE | DOC_START | DOCS | OUT |
|---|---|---|---:|---:|---|
| 1 | `meta-llama/Llama-3.2-1B-Instruct` | `float32` | 20 | 20 | `out/compress_llama1b_s1.csv` |
| 2 | `meta-llama/Llama-3.2-1B-Instruct` | `float32` | 40 | 20 | `out/compress_llama1b_s2.csv` |
| 3 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 0 | 20 | `out/compress_qwen3b_s0.csv` |
| 4 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 20 | 20 | `out/compress_qwen3b_s1.csv` |
| 5 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 40 | 20 | `out/compress_qwen3b_s2.csv` |

Qwen-3B must be `float16` — fp32 weights are 12.4 GB before activations and will OOM.

While these run, do Step 4 (writing). Do not babysit the queue.

---

## Step 3 — the three small probes (≈4 h total)

Same notebook; set `NEEDLE=True`, `GEN_STEPS=16`, or `PROTECT_LAST=32` in `CONFIG`.

| Run | CONFIG changes | OUT | Answers |
|---|---|---|---|
| Retrieval | `NEEDLE=True`, `DOCS=40`, `DOC_START=0` | `out/needle_llama1b.csv` | does compression keep a planted fact |
| Generation | `GEN_STEPS=16`, `DOCS=20`, `METHODS=["evict_snapkv","evict_snapkv_word","merge_snapkv","merge_entropy"]` | `out/gen_llama1b.csv` | does the greedy trajectory survive |
| Recency | `PROTECT_LAST=32`, `RATIOS=[0.3,0.6]`, `DOCS=20` | `out/recency_llama1b.csv` | is merging only failing on recent tokens |

The recency run is the one that defends the central operator claim. Do not drop it.

---

## Step 4 — write, starting now (no GPU needed)

Open `docs/PAPER.md`. The section budget and the claims ledger are there. Order:

1. **§3 Method** — you already know it: equal-memory ratio, the 10 policies, the
   leakage-free prefill→compress→score protocol.
2. **§4.1 Fertility** — Table 1 is already final (`out/tables/expansion.csv`,
   `data/flores200/fertility_*.csv`). Write it today, while shards run.
3. **§4.2–4.4** — wait for Step 2, then read `out/tables/equal_memory.csv` and
   `out/tables/regression.csv` and write from those numbers only.
4. **§1–§2 last**, so the intro promises exactly what §4 delivers.

Rule: every number in the abstract must be traced to a row in `out/tables/`. If it
is not there, it does not go in the abstract.

---

## Step 5 — analysis (5 min, local)

```bash
cd ~/Desktop/kv/inference
llm_env/bin/python src/analyze.py \
  --inputs "out/compress_*.csv" \
  --needle "out/needle_*.csv" --gen "out/gen_*.csv" --serve "out/serve_*.csv" \
  --fertility data/flores200/fertility_*.csv \
  --outdir out
```

Read the `[warn] … truncated` line first. Then `out/tables/*.csv` and
`out/figures/fig1…5`.

---

## Step 6 — serving (≈1 h, optional for v1)

```bash
VLLM_ATTENTION_BACKEND=XFORMERS python src/serve_bench.py --mode offline \
  --model meta-llama/Llama-3.2-1B-Instruct --dtype float16 \
  --batch-sizes 1 2 4 8 16 32 64 --out out/serve_llama1b.csv
```

If vLLM refuses on sm_75, use `--mode hf` and say so in the paper
("eager attention, contiguous KV cache; no PagedAttention"). If neither runs, cut
§4.6 and keep the analytic capacity table — labelled *analytic*, not measured.

---

## What not to do

- **Do not pool Kaggle accounts.** The matrix is ~10 T4-hours against 30 h/week.
  Multi-accounting risks the accounts and saves nothing.
- **Do not skip the parity gate.** A wrong forward pass invalidates every number.
- **Do not cite the n=4 pilot.** It is a pipeline check, not a result.
- **Do not write "100% recall" or "2.6–4.0×".** Those were the old README's numbers;
  the measured range is 2.5–7.7× and the recall probe is n=8, p=0.50.
