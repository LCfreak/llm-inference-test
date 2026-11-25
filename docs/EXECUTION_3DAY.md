# EXECUTION — 3 days, 24 h/day

Read this top to bottom once. Then work down the checkboxes. Nothing here requires
you to make a judgement call; the calls are already made and written down.

Companion files: `docs/PLAN.md` (why each experiment exists), `docs/RUNBOOK.md`
(flags and troubleshooting), `docs/PAPER.md` (manuscript outline + claims ledger).

---

## PART 0 — What I did to your repo

### Git

You were on `main`, 15 commits ahead / 5 behind `origin/main`. I did **not** touch
`main` and did **not** push to it. I created one branch and pushed it:

```
paper/scale-up   →  github.com/LCfreak/llm-inference-test/tree/paper/scale-up
```

Four commits on it. Kaggle clones this branch; no token needed (public repo).

### Files I added (all new, nothing of yours deleted or overwritten)

| File | Lines | What it is |
|---|---:|---|
| `src/kv_compress.py` | ~830 | Your `measure/pre_test/kv_compress.py`, generalized: 6 languages, 12 policies, `--dtype`, `--doc-start`, `--append`, `--equal-memory`, `--fixed-budget`, truncation accounting |
| `src/make_data.py` | 132 | Downloads FLORES-200 dev+devtest for 6 languages, aligns them, writes the parallel CSV and the fertility table |
| `src/analyze.py` | ~520 | All statistics and all figures. Equal-memory interpolation, bootstrap CIs, Holm, regression, fixed-budget table |
| `src/serve_bench.py` | 299 | GPU serving: analytic capacity (`tokens`), vLLM (`offline`), HF fallback (`hf`) |
| `src/run_all.sh` | 87 | One entry point: `smoke / sweep / needle / gen / recency / serve / analyze` |
| `kaggle/kv_sweep.ipynb` | 10 cells | The notebook you will run. Diagnostics → parity gate → data → sweep → export |
| `docs/PLAN.md` | 190 | Scope freeze, claim, experiments E1–E9, timeline, risks |
| `docs/RUNBOOK.md` | 285 | Every command, sharding, Kaggle recipe, troubleshooting, time ledger |
| `docs/PAPER.md` | 185 | Title options, abstract skeleton, section budget, figure map, claims ledger C1–C10 |
| `docs/START_HERE.md` | 150 | Short checklist version of this file |
| `data/flores200/parallel_6lang.csv` | 2009 rows | The corpus. English + Nepali + Hindi + Bengali + Tamil + Amharic, aligned sentence by sentence |
| `data/flores200/fertility_*.csv` | 7 rows | Tokens/sentence, expansion vs English, equal-memory ratio, per tokenizer |

### Files I modified

Only `.gitignore` (added `data/flores200/raw/`, `out/test_*/`, `__pycache__/`).

### Files I did NOT touch

`README.MD`, `analyze_compress.py`, `measure/*`, `models/*`, `requirements.txt`,
`llm_env/`. Your uncommitted work there is exactly as you left it.

### What is proven to work (I ran all of this)

| Check | Result |
|---|---|
| Hand-written forward pass vs Hugging Face | `max |manual − HF| = 6.71e-05` |
| 4-doc × 4-language sweep | 180 rows, all tables and figures produced |
| `--append` resume | 32 rows, 0 duplicates |
| Needle probe | runs, writes `exact` per row |
| Generation probe | runs, writes `gen_prefix_frac` |
| Data build | 997 + 1012 = 2009 aligned sentences |
| Notebook | all 10 cells compile |
| Two new baselines + fixed budget | verified on a 2-doc × 3-language run |

---

## PART 1 — What is done vs what is missing

**Done:** code, data, docs, notebook, local validation, the fertility result.

**Missing: the GPU numbers.** Everything after that is writing. There is no
experiment left that needs a design decision.

The single real result you already have:

| Language | tokens/sentence | vs English | must remove to match English memory |
|---|---:|---:|---:|
| English | 26.36 | 1.00× | 0% |
| Nepali | 68.22 | 2.59× | 61% |
| Hindi | 66.67 | 2.53× | 60% |
| Bengali | 152.26 | 5.78× | 83% |
| Tamil | 201.19 | 7.63× | 87% |
| Amharic | 201.65 | 7.65× | 87% |

---

## PART 2 — Day 1: get every GPU number (≈9 h of wall clock, ~5 h of your attention)

Run shards back to back. Do **not** wait and watch; start writing §3 after the
first shard finishes.

### 2.1 First shard — the only one that can go wrong (40 min)

1. <https://www.kaggle.com/notebooks> → **New Notebook** → *File → Import Notebook*
   → `kaggle/kv_sweep.ipynb`.
2. Right panel: **Accelerator: GPU T4 ×2** · **Internet: On** ·
   **Add-ons → Secrets → Add** `HF_TOKEN` (Hugging Face read token).
3. `CONFIG` cell: leave everything as it is.
4. **Save Version → Save & Run All (Commit)**.

Expected, in this order:

| Cell | Must print |
|---|---|
| Diagnostics | `sm_75`, two GPUs, ~15.6 GB each |
| Parity gate | `max |manual - HF| logit difference: < 1e-3` then `PARITY OK` |
| Data | `wrote data/flores200/parallel_6lang.csv (2009 rows, 6 languages)` |
| Sweep | `doc 0 done …` through `doc 19 done` |
| Sweep | **no `[warn] … TRUNCATED`** |

If the parity gate fails, stop and read the message it prints — it tells you
whether to pin versions or report drift. Do not run anything downstream.

### 2.2 The remaining five shards (≈3.5 h, unattended)

One Kaggle version per row. Change only what the table says.

| # | MODEL | DTYPE | DOC_START | DOCS | OUT |
|---|---|---|---:|---:|---|
| 2 | `meta-llama/Llama-3.2-1B-Instruct` | `float32` | 20 | 20 | `out/compress_llama1b_s1.csv` |
| 3 | `meta-llama/Llama-3.2-1B-Instruct` | `float32` | 40 | 20 | `out/compress_llama1b_s2.csv` |
| 4 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 0 | 20 | `out/compress_qwen3b_s0.csv` |
| 5 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 20 | 20 | `out/compress_qwen3b_s1.csv` |
| 6 | `Qwen/Qwen2.5-3B-Instruct` | `float16` | 40 | 20 | `out/compress_qwen3b_s2.csv` |

Qwen **must** be `float16`. fp32 weights are 12.4 GB before activations and OOM.

While these run, write §3 (Method). It needs no data.

### 2.3 The three probes (≈4 h, unattended)

| # | CONFIG changes | OUT |
|---|---|---|
| 7 | `NEEDLE=True`, `DOCS=40`, `DOC_START=0` | `out/needle_llama1b.csv` |
| 8 | `GEN_STEPS=16`, `DOCS=20`, `METHODS=["evict_snapkv","evict_snapkv_word","merge_snapkv","evict_vk"]` | `out/gen_llama1b.csv` |
| 9 | `PROTECT_LAST=32`, `RATIOS=[0.3,0.6]`, `DOCS=20` | `out/recency_llama1b.csv` |

Run 9 before 8 if you are short of time. It defends the central claim; 8 is
supporting.

### 2.4 Serving (≈1 h, optional for v1)

Run in a Kaggle notebook with a plain cell:

```bash
VLLM_ATTENTION_BACKEND=XFORMERS python src/serve_bench.py --mode offline \
  --model meta-llama/Llama-3.2-1B-Instruct --dtype float16 \
  --batch-sizes 1 2 4 8 16 32 64 --out out/serve_llama1b.csv
```

If vLLM refuses on sm_75, use `--mode hf`. If both fail, cut §4.6 and keep the
analytic table, labelled **analytic**.

### 2.5 Bring the numbers home (10 min)

Download every CSV from each version's **Output** tab into
`~/Desktop/kv/inference/out/`, then:

```bash
cd ~/Desktop/kv/inference
llm_env/bin/python src/analyze.py \
  --inputs "out/compress_*.csv" \
  --needle "out/needle_*.csv" --gen "out/gen_*.csv" --serve "out/serve_*.csv" \
  --fertility data/flores200/fertility_*.csv \
  --outdir out
```

Read in this order: the truncation warning (must be absent) → `out/tables/expansion.csv`
→ `equal_memory.csv` → `fixed_budget.csv` → `regression.csv` → `out/figures/`.

**Day 1 is done when those six files exist and no row is truncated.**

---

## PART 3 — Day 2: write the paper from the tables

Rules that keep you out of trouble:

1. Never type a number that is not in `out/tables/`.
2. Write §3 and §4 first, §1–§2 last.
3. Every abstract sentence must point at a table row or a figure.
4. If a result is not significant, say "not significant" and give the CI. Do not
   round it into a trend.

Order and time boxes:

| Hours | Section | Source |
|---|---|---|
| 0–2 | §3 Method: equal-memory ratio, 12 policies, leakage-free scoring, operator control | `docs/PAPER.md` §4 |
| 2–5 | §4.1 Fertility + §4.2 quality at equal memory | `expansion.csv`, `equal_memory.csv`, Fig. 1–3 |
| 5–6 | §4.3 regression, §4.4 operator isolation | `regression.csv`, `paired_evict_vs_merge.csv`, Fig. 4 |
| 6–7 | §4.5 retrieval + generation | `needle_recall.csv`, `gen_prefix.csv`, Fig. 5 |
| 7–8 | §4.6 serving | `serve.csv` or the analytic table |
| 8–10 | §4.7 the fixed-budget audit | `fixed_budget.csv`, Fig. 6 |
| 10–13 | §5 Discussion, §6 Threats | `docs/PAPER.md` §8 |
| 13–16 | §1 Intro, §2 Related work | `docs/PAPER.md` §7 |
| 16–18 | Abstract, title, figures' captions | every number traced |
| 18–20 | Read it as a hostile reviewer, fix | claims ledger |

### The four sentences the paper must be able to defend

1. "At a fixed token budget of 1024, English is not compressed at all while Tamil
   loses 52% of its cache." — arithmetic, from `fixed_budget.csv`.
2. "At equal memory, the damage grows with fertility." — `regression.csv` slope + CI.
3. "With identical selection scores, eviction and merging differ by X points." —
   `paired_evict_vs_merge.csv`, Holm-corrected.
4. "Attention-free selection changes the cross-language ranking by Y." —
   `equal_memory.csv`, rows `evict_vk` / `evict_keynorm`.

If any of the four comes back flat, that *is* the finding; report it as such.

---

## PART 4 — Day 3: ship

| Hours | Task |
|---|---|
| 0–2 | Fix the figures: axis labels, one message per figure, captions carry model/dtype/hardware/n |
| 2–4 | Abstract + intro rewrite against the final tables |
| 4–6 | Claims-ledger audit: every `Pending` becomes `Measured` or is deleted from the text |
| 6–8 | LaTeX: ACL template, anonymised, 8 pages, limitations section |
| 8–9 | Anonymity check: no repo URL, no acknowledgements, no author names in PDF metadata |
| 9–10 | Upload to arXiv (cs.CL, cross-list cs.LG), post v1 |
| 10–12 | Buffer |

`docs/PAPER.md` §9 has the ARR checklist for the January cycle. arXiv v1 now,
ARR submission on ~4 Jan.

---

## PART 5 — How to make it stronger with the hours you have

Ranked by result-per-hour. Do them in this order if time appears.

| Rank | Addition | Cost | Why it matters |
|---|---|---|---|
| 1 | **Fixed-budget audit** (`--fixed-budget 1024`) | already in the sweep, 0 extra hours | Turns "your comparison is unfair" into a measured number. Nobody has published this for KV compression |
| 2 | **Two attention-free baselines** (`evict_vk`, `evict_keynorm`) | already in the sweep | Makes the paper speak to the systems literature instead of past it |
| 3 | **Native-speaker check** of the 6 needle sentences | 30 min, no GPU | Removes the biggest "you machine-translated your probe" objection |
| 4 | **100 docs instead of 60** | +2 T4-hours | Tightens every CI by ~30% |
| 5 | **One qualitative example per language** (a compressed vs uncompressed generation) | 1 h, no GPU | Reviewers remember examples; tables get skimmed |
| 6 | **Recency control** (`--protect-last 32`) | 45 min | Kills the "merging only fails because it eats recent tokens" confound |

### Do not do these

- vLLM kernel work, Triton, block-table patches. Different paper, systems venue.
- 7B/8B models. T4 will OOM or take all day; fertility is a tokenizer property.
- More languages. Six already spans 2.5×–7.7×.
- Human evaluation, LongBench, multi-turn. Not in scope, not in three days.
- Pooling Kaggle accounts. ~10 T4-hours against 30 h/week. It saves nothing and
  risks the accounts.

---

## PART 6 — If something breaks

| Symptom | Do this |
|---|---|
| Parity gate fails | Read the message it prints. Pin `transformers==5.22.1` and restart the session; if that fails, run `--device cpu --docs 1` to prove the code is intact, then report drift |
| `[warn] TRUNCATED` | `MAX_TOKENS` must be ≥4096. The run is invalid for that language |
| CUDA OOM on Qwen | `DTYPE="float16"`, `DOCS` down to 10 |
| Kaggle kills at 9 h | `--append` is already on; re-run the same version, it resumes |
| vLLM: no supported backend | `VLLM_ATTENTION_BACKEND=XFORMERS`, else `--mode hf` |
| `ModuleNotFoundError: pandas` locally | you used `.venv`; use `llm_env/bin/python` |
| `analyze.py` prints an empty regression | fewer than 3 non-English languages in the glob |
| Abstract number has no table row | delete the sentence |

---

## PART 7 — The one thing to do in the next ten minutes

Open Kaggle, import `kaggle/kv_sweep.ipynb`, set T4 ×2 + Internet + `HF_TOKEN`,
and commit the default config.

Everything else in this document waits on that single 40-minute run.
