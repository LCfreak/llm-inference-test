# Language-Fair LLM Serving: The Systems Cost of Tokenizer Fertility and KV-Cache Compression for Devanagari

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Type: technical report](https://img.shields.io/badge/type-technical_report-informational.svg)]()
[![Model: Llama 3.2 1B](https://img.shields.io/badge/model-Llama_3.2_1B-lightgrey.svg)]()
[![Status: Active Research](https://img.shields.io/badge/status-active_research-green.svg)]()

**Ashutosh Bhattarai** (Pulchowk Campus, IOE, Tribhuvan University) · bhattaraiaasu1@gmail.com

> **Status.** Preliminary results: one model (Llama 3.2 1B), one language pair (English-Nepali), 8-10 documents per
> language, CPU / Apple Silicon. Every claim is labeled by how strongly the data supports it (Section 6).
> Feedback and collaboration are welcome (Section 9).

---

## At a glance

| | |
|---|---|
| **Problem** | Nepali needs ~2.6x the tokens of English, so ~2.6x the KV-cache memory per request |
| **Question** | Can KV-cache compression recover that cost without hurting quality, and which compression action works? |
| **Finding 1** | At English-level memory (61% of entries removed), SnapKV-style eviction costs Nepali ~3.5% perplexity; every merging variant tested costs 27-47% |
| **Finding 2** | Query-aware eviction keeps a planted secret through 60% compression in both languages (8/8); word-aligned eviction is suggestive at 80% (8/8 vs 6/8, not significant) |
| **Not shown (yet)** | That RoPE causes the merging failure; that word alignment helps generation; any GPU or vLLM result |
| **Try it** | `python kv_compress.py --selftest` (checks the hand-written forward pass against Hugging Face) |

**Contents:** [Abstract](#abstract) · [Motivation](#1-motivation) · [Setup](#2-setup) · [Results](#3-results) ·
[Discussion](#4-discussion) · [Threats to validity](#5-threats-to-validity) · [Claims ledger](#6-claims-ledger) ·
[Roadmap](#7-roadmap) · [Reproducibility](#8-reproducibility) · [Feedback](#9-feedback-wanted) ·
[Related work](#10-related-work-verify-each-citation-against-the-paper) · [Citation](#citation)

---

## Abstract

BPE tokenizers trained on English-dominated text split Devanagari words into many byte-level pieces, so the same
content costs about 2.6x more tokens in Nepali than in English. In an LLM serving system, token count sets KV-cache
memory per request, prefill compute and concurrency, a cost that per-token benchmarks hide. We ask whether KV-cache
compression can recover it. Using a leakage-free protocol (prefill, compress the cache, score only the
continuation), we compare compression methods at equal memory. At the memory footprint of English, SnapKV-style
eviction keeps Nepali within about 3.5% perplexity (top-1 next-token agreement 0.84), whereas every merging variant
tested loses 27-47%, and merging hurts Nepali more than English. In a long-range retrieval probe, query-aware
eviction keeps a planted secret through 60% compression in both languages. A word-aligned variant (keep or evict whole
words) recalled the secret in 8/8 Nepali documents at 80% compression versus 6/8 for token-level SnapKV, but this is
not statistically established, and it gives no benefit on KL divergence, top-1 agreement or 16-step generation.
Several confounds remain open and are listed explicitly.

## 1. Motivation

```
more tokens per request -> larger KV cache -> fewer concurrent users
                        -> more prefill compute -> higher time-to-first-token
```

Standard serving metrics (tokens/s, latency per token) are language-blind. We propose reading throughput per unit
of content: *effective semantic throughput* = raw token throughput / token expansion ratio, where the token
expansion ratio is tokens(Nepali) / tokens(English) for paired sentences.

**Questions.**
1. **RQ1 (cost).** How large is the cost and where does it appear (memory, prefill, concurrency)?
2. **RQ2 (recovery).** At equal memory (Nepali compressed so its cache matches English uncompressed), how much quality
   is lost, and does the compression action (evict vs merge) matter?
3. **RQ3 (selection).** Does attention entropy identify redundant tokens? Does respecting word boundaries help for a
   script whose words fragment into several tokens?

## 2. Setup

| Item | Choice |
|---|---|
| Model | Llama 3.2 1B Instruct (fp32, eager attention) |
| Data | FLORES-200 paired English/Nepali sentences; documents are 10 consecutive sentences |
| Protocol | Prefill the context uncompressed; compress each layer's KV cache; score only continuation tokens against the compressed cache (no information leakage) |
| Compression | Same ratio in every layer; the first token (attention sink) is never compressed; merged entries get a log(m) attention-mass bonus |
| Metrics | Perplexity change; KL divergence and top-1 agreement with the uncompressed model; exact recall of a planted secret; 16-step free-running generation (prefix match) |
| Verification | The hand-written forward pass matches Hugging Face (max logit difference ~2e-7): `python kv_compress.py --selftest` |

**Which study used which methods.** The perplexity study (Section 3.2, 8 documents) compared eight methods and
predates word-aligned eviction. The retrieval probe (3.3, 8 documents) and the generation run (3.4, 10 documents)
add `evict_snapkv_word` and use different document sets, so context lengths differ (about 220 vs 566 tokens in
the perplexity study, 305 vs 770 in the retrieval probe including the planted fact, 212 vs 545 in the generation run).

| Method | Action | Selection rule |
|---|---|---|
| `evict_snapkv` | evict | SnapKV-style: attention from an observation window at the end of the prompt, pooled (first-stage family of RocketKV; no adaptive pooling, no decode-time sparse attention) |
| `evict_snapkv_word` | evict | Same scores aggregated per word (max over fragments); whole words kept or evicted |
| `evict_h2o` | evict | Accumulated attention (H2O-style) |
| `evict_window` | evict | First token + most recent tokens |
| `merge_entropy` | merge | Runs of lowest attention-entropy tokens (normalized by position) |
| `merge_random`, `merge_fixed` | merge | Random tokens / equal-size windows (controls) |
| `merge_sim` | merge | Greedy merging of adjacent most-similar K/V |
| `merge_word` | merge | Collapse fragments of the same word |
| `merge_snapkv` | merge | Same scores as `evict_snapkv`, but merge instead of evict (control, not yet run) |

## 3. Results

### 3.1 The cost (RQ1)

Nepali needs about **2.6x** the tokens of English (30 paired FLORES sentences, Llama 3.2 tokenizer: 27.4 vs 68.6
tokens per sentence; the full-set re-run is pending). KV memory per request scales with this ratio.

![Serving capacity benchmark](research_serving_capacity_benchmark.png)
*Fig. 1. llama.cpp serving-capacity profiling (CPU / Apple Silicon): with adaptive context sizing under a 2 GB KV
budget, roughly 415 English, 258 Hindi and 250 Nepali concurrent users fit. The gain over static allocation comes
from right-sizing the context, not from anything language-specific; the English-Nepali gap follows from the token
ratio.*

### 3.2 Equal-memory perplexity (RQ2, RQ3; 8 documents)

Matching English memory means removing **61%** of Nepali's KV entries. Nepali results at that point:

| Method | Perplexity change | Top-1 agreement |
|---|---|---|
| `evict_snapkv` | +3.5% | 0.84 |
| `evict_window` | +5.2% | 0.82 |
| `evict_h2o` | +16.9% | 0.73 |
| `merge_entropy` | +27.5% | 0.72 |
| `merge_random` | +28.3% | 0.71 |
| `merge_word` | +30.9% | 0.71 |
| `merge_sim` | +32.4% | 0.73 |
| `merge_fixed` | +46.8% | 0.66 |

![Perplexity vs KV entries removed](compress_curves.png)
*Fig. 2. Perplexity change vs fraction of KV entries removed. The dashed line marks the 61% equal-memory point.*

- Eviction beats every merging variant by a wide margin. `evict_snapkv` and `evict_window` are **not** distinguishable
  at this sample size (paired p = 0.38; SnapKV better on 6/8 documents); one document shows +15% under SnapKV.
- Merging hurts Nepali **more** than English at equal compression (p = 0.008 for fixed, random, similarity and word
  merging), the opposite of the hypothesis that fragment tokens are redundant.
- Attention entropy is not a redundancy signal: `merge_entropy` is no better than `merge_random` in Nepali and worse
  than random in English (p <= 0.04).

### 3.3 Long-range retrieval probe (RQ2, RQ3; 8 documents)

A random 4-digit secret ("The secret code is 4721." / a Nepali equivalent) is planted ~10% into the document and the
question is placed at the end of the context, so the observation window sees it. The uncompressed model recalls the
secret in 8/8 documents in both languages.

| Exact recall (of 8) | English 30% | 60% | 80% | Nepali 30% | 60% | 80% |
|---|---|---|---|---|---|---|
| `evict_snapkv` | 8 | 8 | 4 | 8 | 8 | 6 |
| `evict_snapkv_word` | 8 | 8 | 5 | 8 | 8 | 8 |
| `evict_window`, `evict_h2o` | 0 | 0 | 0 | 0 | 0 | 0 |
| `merge_entropy` | 2 | 0 | 0 | 8 | 2 | 0 |
| `merge_word` | 0 | 0 | 0 | 8 | 0 | 0 |
| `merge_random` | 1 | 0 | 0 | 5 | 1 | 0 |
| `merge_sim` | 1 | 0 | 0 | 2 | 0 | 0 |
| `merge_fixed` | 0 | 0 | 0 | 1 | 0 | 0 |

Cache entries kept (mean): English 213 / 122 / 61 and Nepali 539 / 308 / 154 at 30% / 60% / 80% removal, out of
305 and 770 context tokens. Answer-token NLL change in Nepali at 80%: `evict_snapkv` +0.123, `evict_snapkv_word` -0.046.

- Query-aware selection is what preserves the secret: SnapKV-style methods keep it to 60% compression in both
  languages; query-agnostic eviction loses it.
- **Reading "100% at 80%" correctly.** It is 8/8 documents (consistent with a true rate anywhere above about 63%).
  Nepali's longer context leaves more absolute entries at the same ratio (154 vs 61), and English also scores 8/8 when
  it keeps 122 entries, so this is an absolute-budget effect, not evidence of a Nepali-specific advantage.
- **The probe is easy.** The question nearly repeats the planted sentence and sits in the observation window. Harder
  variants (paraphrased question, distractor codes, several depths, longer contexts) are planned.
- **Not a fair merge-vs-evict comparison:** only the SnapKV methods see the question. The control `merge_snapkv` (same
  scores, merge instead of evict) is the next run.
- Word-aligned vs token-level SnapKV at 80%: Nepali 8/8 vs 6/8, English 5/8 vs 4/8. The word-aligned variant never
  lost (2 wins, 0 losses in Nepali; sign test p = 0.50), so this is suggestive only. At 60% the two are identical.
- Unexplained: at 30% compression, entropy-guided and word merging keep the secret in 8/8 Nepali documents but not in
  English, the opposite direction from the perplexity result.

### 3.4 Teacher-forced agreement and 16-step generation (10 documents)

Prefix match is the fraction of 16 greedy tokens before the first divergence from the uncompressed output.

| Method | Nepali prefix match (30 / 60 / 80%) | English prefix match (30 / 60 / 80%) | KL at 60% (Nepali / English) |
|---|---|---|---|
| `evict_snapkv_word` | 0.32 / 0.25 / 0.16 | 0.74 / 0.49 / 0.42 | 0.0746 / 0.023 |
| `evict_snapkv` | 0.30 / 0.24 / 0.16 | 0.79 / 0.47 / 0.43 | 0.0718 / 0.024 |
| `evict_window` | 0.28 / 0.21 / 0.14 | 0.73 / 0.46 / 0.38 | 0.0831 / 0.039 |
| `merge_entropy` | 0.15 / 0.09 / 0.05 | 0.28 / 0.08 / 0.04 | 0.2121 / 0.174 |

- Word-aligned and token-level SnapKV are indistinguishable on prefix match, KL and top-1 agreement across both
  languages and all ratios; where a difference is nominally significant (KL, Nepali 80%, p = 0.05), word alignment is
  slightly *worse*. This does **not** support a generation-stability advantage for word alignment.
- Nepali output diverges from the uncompressed output earlier than English at the same ratio, and even at 30%
  removal Nepali prefix match is only about 0.30. Because near-tied predictions in uncertain text can flip from tiny
  numerical noise, the prefix-match metric's noise floor must be measured before it is used to compare methods.
- `merge_entropy` is clearly worse than all eviction methods in both languages.

## 4. Discussion

1. **A language-dependent cost is real and large, and eviction recovers most of it at modest perplexity cost** in this
   setup. Perplexity is a proxy; top-1 agreement of 0.84 means about 1 in 6 predictions changes.
2. **Merging averaged key/value vectors did poorly here**, and more so for Nepali. A candidate explanation is that
   averaging post-RoPE keys mixes position-dependent rotations, but that mechanism is untested; the planned controls
   (`merge_snapkv`, merging only older context, pre-RoPE merging) separate the action from the selection rule.
3. **The Devanagari-specific idea** (do not split words when evicting) has a plausible rationale and a suggestive
   retrieval signal, but no benefit on the other metrics. It needs more documents, harder probes, a second language
   and a second tokenizer before it can be called an effect.

## 5. Threats to validity

- 8-10 documents per language; many paired tests with no multiplicity correction; treat p-values as indicative.
- One 1B model, short contexts (~200-800 tokens), which favors recency; one language pair; CPU only.
- The equal-memory perplexity study did not include word-aligned eviction.
- The merging methods also collapsed recent tokens, which the eviction baselines protect (control pending).
- The retrieval probe uses one fact type, one planting position, and a near-verbatim question; query-aware methods see
  the question. Query-aware compression needs the question at compression time, which fails for multi-turn use.
- Perplexity, top-1 agreement and prefix match are proxies; no downstream task has been run.
- The SnapKV baseline is first-stage-style only; selection is shared across heads within a layer.

## 6. Claims ledger

| Claim | Evidence | Status |
|---|---|---|
| Nepali needs ~2.6x the tokens of English | 30 FLORES pairs; document sets of 10 sentences | Measured (full-set re-run pending) |
| SnapKV-style eviction brings Nepali to English-level memory for ~+3.5% perplexity | n = 8, one document at +15% | Measured, small sample |
| All merging variants lose 27-47% at 61% | n = 8 | Measured |
| Merging hurts Nepali more than English | p = 0.008, n = 8 | Measured, small sample |
| Entropy does not identify redundant tokens | worse than random in English (p <= 0.04) | Measured, small sample |
| Query-aware eviction keeps a planted secret to 60% compression | 8/8 in both languages | Measured; easy probe, confounded by query access |
| Word-aligned eviction beats token-level SnapKV at 80% retrieval | 8/8 vs 6/8 (Nepali), p = 0.50 | Suggestive |
| Word alignment improves KL, agreement or generation | no difference; KL nominally worse | Not supported |
| RoPE phase interference explains the merging failure | none yet | Hypothesis |
| Behavior in vLLM / PagedAttention, on GPUs, in other models or languages | none | Not tested |

## 7. Roadmap

| # | Experiment | Command |
|---|---|---|
| 1 | Merge vs evict with identical SnapKV scores | `python kv_compress.py --needle --methods evict_snapkv merge_snapkv evict_snapkv_word merge_random --ratios 0.3 0.6 0.8 --docs 30 --out needle_control.csv` |
| 2 | Equal-memory perplexity including word-aligned eviction, 30 documents | `python kv_compress.py --methods evict_snapkv merge_snapkv evict_snapkv_word evict_window --ratios 0.3 0.6 0.8 --docs 30 --out compress_control.csv` |
| 3 | Merge only older context (protect last 32 tokens) | `python kv_compress.py --protect-last 32 --docs 12 --ratios 0.3 0.6 --out compress_pl32.csv` |
| 4 | Harder retrieval probe: paraphrased question, distractors, several depths, longer contexts | planned |
| 5 | Noise floor for prefix match (perturb K/V slightly, rerun) | planned |
| 6 | Second language (Hindi), second model, downstream task | planned |
| 7 | NVIDIA GPU replication, profiling, vLLM comparison | planned |
| 8 | Custom kernel (Triton / CUDA) for segment-aware top-k | conditional: only if profiling shows the selection step matters |

## 8. Reproducibility

```bash
pip install torch transformers pandas scipy matplotlib tqdm
python kv_compress.py --selftest                                   # forward pass vs Hugging Face
python kv_compress.py --docs 8 --ratios 0.3 0.6 0.8 --out compress_results.csv
python kv_compress.py --needle --needle-pos 0.1 --docs 8 --ratios 0.3 0.6 0.8 --out needle_results.csv
python kv_compress.py --docs 10 --ratios 0.3 0.6 0.8 --methods evict_snapkv_word evict_snapkv evict_window merge_entropy \
    --gen-steps 16 --out gen_results_16.csv
python analyze_compress.py compress_results.csv                    # tables + compress_curves.png
python analyze_needle.py needle_results.csv
```

```text
llm-inference-test/
├── kv_compress.py            # prefill, leakage-free scoring, compression methods, needle probe, generation
├── analyze_compress.py       # paired tests, equal-memory interpolation, figure
├── analyze_needle.py         # retrieval-probe analysis
├── compress_results.csv, needle_results.csv, gen_results_16.csv
├── compress_curves.png, research_serving_capacity_benchmark.png
└── llama_cpp/                # serving-capacity profiling scripts
```

## 9. Feedback wanted

- Is there an existing result on merge-vs-evict at fixed selection scores for non-Latin scripts?
- Is word-level (or morpheme-level) eviction worth pursuing in engines such as vLLM or llama.cpp, or does a
  tokenizer-level fix make it moot?
- Which benchmark would convince you that a compressed cache is "equivalent" for a low-resource language?

## 10. Related work (verify each citation against the paper)

Kwon et al. 2023, *PagedAttention* (vLLM) · Xiao et al. 2023, *StreamingLLM* · Zhang et al. 2023, *H2O* ·
Li et al. 2024, *SnapKV* · Liu et al. 2023, *Scissorhands* · Liu et al. 2024, *KIVI* ·
Behnam et al., ICML 2025, *RocketKV* · Agrawal et al., OSDI 2024, *Sarathi-Serve* ·
Petrov et al. 2023, *Language Model Tokenizers Introduce Unfairness Between Languages*

## Citation

```bibtex
@misc{bhattarai2026languagefair,
  author = {Bhattarai, Ashutosh},
  title  = {Language-Fair LLM Serving: The Systems Cost of Tokenizer Fertility and KV-Cache Compression for Devanagari},
  year   = {2026},
  howpublished = {\url{https://github.com/LCfreak/llm-inference-test}}
}
```
