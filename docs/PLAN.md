# PLAN — from a two-language CPU study to a six-language GPU paper

Status: **scope frozen 2026-11-24**. This document is the contract for what gets
measured, what gets claimed, and what gets cut. If an experiment is not in §5, it
does not go in the paper.

---

## 1. Where the project actually stands

| Asset | State |
|---|---|
| `src/kv_compress.py` | Working. 10 policies, hand-written Llama forward pass, leakage-free prefill → compress → score protocol, `--selftest` parity vs Hugging Face. |
| `src/make_data.py` | **New.** FLORES-200 dev+devtest, 6 languages, aligned wide CSV + per-model fertility table. |
| `src/analyze.py` | **New.** Equal-memory interpolation, bootstrap CIs, Holm correction, dose–response regression, five figures. |
| `src/serve_bench.py` | **New.** Analytic KV capacity + vLLM concurrency sweep (GPU). |
| Existing results | `measure/compress_results.csv` (n=8), `gen_results_16.csv` (n=10), needle n=8. All Llama-3.2-1B, FP32, CPU/MPS, English+Nepali. |
| Missing before this plan | The input corpus was not committed; only two languages; no GPU execution; no equal-memory statistics. |

The old README's headline ("2.5× for Nepali") is **the least interesting number in
the new data**. See §3.

---

## 2. The claim

> Tokenizer fertility determines how much of a language's KV cache must be
> discarded to reach parity with English memory. Under that equal-memory
> constraint, query-aware eviction preserves quality while vector merging does
> not, and the damage grows monotonically with fertility across scripts.

Three contributions, in order of importance:

1. **Protocol.** Equal-memory evaluation, $r_\ell = 1 - 1/R_\ell$, instead of
   equal-compression-ratio. Most KV-compression papers evaluate at a fixed ratio,
   which silently asks high-fertility languages to discard far more content.
2. **Dose–response.** Six languages spanning $R = 2.5$–$7.6$ under one tokenizer,
   plus a second tokenizer. The effect is continuous in fertility, not a
   Devanagari curiosity.
3. **Operator isolation.** `merge_snapkv` uses the *same* SnapKV selection scores
   as `evict_snapkv` but merges instead of evicting. That separates *what to
   select* from *what to do with it* — the causal test for the post-RoPE merging
   hypothesis.

---

## 3. The measured fertility gradient (local, Llama-3.2 tokenizer)

FLORES-200 dev+devtest, 2,009 aligned sentences, `src/make_data.py`:

| Language | tokens / sentence | vs English | equal-memory removal $r_\ell$ |
|---|---:|---:|---:|
| English | 26.36 | 1.00× | 0% |
| Nepali | 68.22 | 2.59× | 61% |
| Hindi | 66.67 | 2.53× | 60% |
| Bengali | 152.26 | 5.78× | 83% |
| Tamil | 201.19 | 7.63× | 87% |
| Amharic | 201.65 | 7.65× | 87% |

Analytic KV capacity at a 2 GB budget, 10-sentence prompts (`serve_bench.py --mode tokens`):

| | English | Nepali | Hindi | Bengali | Tamil | Amharic |
|---|---:|---:|---:|---:|---:|---:|
| max concurrent requests | 209 | 81 | 82 | 37 | 29 | 29 |

**Consequence for the paper:** the title must not say "Devanagari". The interesting
range is 2.5×–7.6×, and the hardest case is Tamil/Amharic at 87% removal, where
every merging method is expected to fail outright. That is the experiment that
makes the paper general.

### Pilot signal (n = 4 docs, 3 languages — do not cite)

`evict_snapkv` vs `merge_snapkv` at equal memory, short documents:

| | Nepali | Hindi | Bengali |
|---|---:|---:|---:|
| `evict_snapkv` | +8.0% | +1.4% | +6.6% |
| `merge_snapkv` | +11.5% | +0.4% | +6.3% |

Nothing is significant at this sample size, but the direction matters: merging with
*good* scores is nowhere near the 27–47% collapse seen with entropy/random/fixed
merging. If that holds at n=60, the finding is "**selection explains most of the
gap; the operator adds a smaller penalty**", not "merging is broken". Write
contribution 3 to report whichever the data shows — the control exists precisely so
the post-RoPE hypothesis cannot be asserted without evidence.

---

## 4. Scope freeze

**Languages:** English, Nepali, Hindi, Bengali, Tamil, Amharic. No additions.

**Models:**
- `meta-llama/Llama-3.2-1B-Instruct`, FP32 — baseline, parity-verified.
- `Qwen/Qwen2.5-3B-Instruct`, FP16 — second tokenizer, same code path.
- Optional if Blackwell arrives: `Qwen/Qwen2.5-32B-Instruct` or `Llama-3.1-8B`, serving only.

**Policies:** the 10 already implemented. No new policies.

**Hardware:** 1×T4 16 GB (Kaggle) for quality; same for serving. Report the
hardware tier honestly. No A100 claims.

**Out of scope, explicitly:** vLLM kernel/BlockSpaceManager integration, Triton
kernels, multi-turn benchmarks, LongBench, 70B models, human evaluation,
tensor-parallel scaling.

---

## 5. Experiment matrix

| # | Experiment | Command | Answers | Cost (T4) |
|---|---|---|---|---|
| E1 | Fertility table, 6 langs, 2 tokenizers | `make_data.py --tokenizer ...` | RQ1: how big is the penalty, per tokenizer | minutes |
| E2 | Quality sweep, Llama-1B FP32, 60 docs | `kv_compress.py --docs 60 --equal-memory` | RQ2/RQ3: quality at equal memory | ~2 h |
| E3 | Quality sweep, Qwen-3B FP16, 60 docs | same, `--model Qwen/... --dtype float16` | RQ5: does it replicate across tokenizers | ~1 h |
| E4 | Needle probe, 40 docs | `--needle --docs 40` | RQ2: long-range recall under compression | ~1 h |
| E5 | Generation, 20 docs, 16 steps | `--gen-steps 16` | RQ4: trajectory stability | ~1.5 h |
| E6 | Recency control | `--protect-last 32` | Confound: is merging failing only because it eats recent tokens | ~30 min |
| E7 | Serving sweep, vLLM, 6 langs | `serve_bench.py --mode offline` | RQ1: measured capacity, TTFT/TPOT, preemptions | ~1 h |
| E8 | Bootstrap + regression + figures | `analyze.py` | All | minutes |

Total ≈ 6 T4-hours. One Kaggle account (30 h/week) is enough; do not spread runs
across accounts.

**E6 is not optional.** The current report already flags "recent-token protection
confound" as a threat. It is one flag and it defends the central operator claim.

---

## 6. Timeline

### 3 days — arXiv v1

| Day | Deliverable |
|---|---|
| **Tue 24 Nov** | Code frozen (done). Local smoke test. Kaggle shard 0 running. Draft §1–3. |
| **Wed 25 Nov** | E2–E6 complete; E7 running. Draft §4 (results) against real numbers. |
| **Thu 26 Nov** | E8, figures, §5–7, internal read-through. **arXiv v1 posted Fri 27.** |

### To the real deadline

| Date | Milestone |
|---|---|
| **Tue 15 Dec** | arXiv v2: 60→100 docs, Qwen results, serving figure, appendix. |
| **~Mon 4 Jan 2027** | **ARR submission → ACL 2027.** Main target. |
| **~Fri 22 Jan 2027** | ICML 2027 fallback (needs an ML-theory framing; less natural). |
| **~30 Mar 2027** | COLM 2027 fallback. |
| any time | TMLR (rolling) as the safety net. |

No peer-reviewed venue closes before 15 Dec. MLSys 2027 (30 Oct) and the Oct ARR
cycle (NAACL/COLING) are already gone.

---

## 7. Risks, ranked

1. **Kaggle friction, not compute.** 9-hour sessions, no SSH, 20 GB disk. Mitigation:
   per-document CSV checkpointing (already in the script), one shard per run,
   `--append` to resume.
2. **FP16 numerics.** T4 has no bf16. Mitigation: FP32 for Llama-1B (5 GB, fits),
   FP16 only for Qwen-3B, softmax forced to FP32 in `attend()`.
3. **Transformer version drift.** The code is validated on `transformers 5.22.1` /
   `torch 2.15`. Kaggle's image differs. Mitigation: pin versions in the notebook;
   `--selftest` must pass before any run counts.
4. **Missing native-speaker check.** The needle sentences for Hindi/Bengali/Tamil/
   Amharic are machine-written. Mitigation: have a speaker verify before the paper
   claims per-language recall; otherwise report recall only for EN/NE and mark the
   rest exploratory.
5. **Overclaiming.** The old README says "restores 100% recall" and "2.6–4.0×".
   The new numbers are 2.5–7.6× and the recall result is $n=8$, $p=0.50$. Rewrite
   every claim against §3 and the new CSVs.

---

## 8. Definition of done (arXiv v1)

- [ ] `python src/kv_compress.py --selftest` passes on the run host.
- [ ] `data/flores200/parallel_6lang.csv` committed; `make_data.py` reproduces it.
- [ ] E2–E6 CSVs committed under `out/`.
- [ ] `analyze.py` reproduces every number in the abstract from those CSVs.
- [ ] Every abstract number appears in a table or figure.
- [ ] Claims ledger has no "Hypothesis" row presented as a result.
- [ ] Hardware, dtype, n, and seeds stated in the caption of every table.
- [ ] Repo has a one-command reproduction path (`docs/RUNBOOK.md` §4).
