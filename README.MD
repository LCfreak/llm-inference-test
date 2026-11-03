# Language-Fair LLM Serving: The Systems Cost of Tokenizer Fertility

This repository contains profiling scripts, benchmarks, and architectural investigations addressing memory bottlenecks in LLM inference caused by **tokenization inequality**.

> How much more does it cost to serve Nepali than English, where does that cost show up in an inference system (KV-cache memory, prefill compute, concurrency), and how much of it can KV-cache compression recover without hurting quality?

**Status:** Active research, preliminary results (8 documents per language, Llama 3.2 1B, Apple Silicon / CPU). NVIDIA GPU replication is planned, not done.

---

## 🚀 The Observation: The Fertility Gap

BPE tokenizers learn their merges from English-dominated corpora, meaning Devanagari text often falls back to byte-level pieces. For the exact same semantic payload, high-fertility scripts consume significantly more tokens:

```text
more tokens per request -> larger KV cache -> fewer concurrent users
                        -> more prefill compute -> higher time-to-first-token
```

Currently, multi-tenant serving systems allocate KV cache based on fixed sequence lengths. Per-token benchmarks (tokens/s, latency) hide the cost of tokenization inequality. A language-aware metric is *effective semantic throughput* = token throughput / fertility.

## 📊 Result 1: The Cost is About 2.6x

Paired FLORES-200 sentences evaluated with the Llama 3.2 tokenizer reveal that Nepali requires about **2.6x** the tokens of English (a 30-sentence sample averaged 27.4 English vs. 68.6 Nepali tokens per sentence). In the compression study, documents of ~10 sentences average 220 English vs. 566 Nepali tokens.

Under static KV cache allocation, this elevated fertility rapidly fragments memory blocks, leading to early OOM (Out of Memory) errors and dropped sessions in multi-tenant environments. KV memory per request and concurrency scale directly with this ratio; static-vs-adaptive context allocation is a right-sizing effect, not a language-specific one.

*(Note: The raw profiling scripts are located in the `src/profiling` directory.)*

## 🏗️ Result 2: Eviction Recovers the Cost, Merging Does Not (Equal-Memory Study)

This project explores dynamic, runtime context compression to maintain concurrency without rigid prompt-time budgets.

**Setup:** Prefill the context normally, compress its KV cache, then score the continuation tokens against the compressed cache to prevent information leakage. "Equal memory" means Nepali was compressed by 61% so its cache matched English's uncompressed footprint.

| Method (Nepali, 61% of KV entries removed) | Perplexity change | Next-token agreement |
|---|---|---|
| SnapKV-style eviction | +3.5% | 0.84 |
| Recency-window eviction | +5.2% | 0.82 |
| Accumulated-attention eviction (H2O-style) | +16.9% | 0.73 |
| Entropy-guided merging | +27.5% | 0.72 |
| Random merging | +28.3% | 0.71 |
| Word-boundary merging | +30.9% | 0.71 |
| Similarity merging | +32.4% | 0.73 |
| Fixed-window merging | +46.8% | 0.66 |

**Key Takeaways:**
*   **Eviction beats merging by a wide margin:** SnapKV and the plain recency window are not distinguishable at this sample size (paired p = 0.38); one document shows +15%, meaning it is "usable", not "lossless".
*   **Merging hurts Nepali more than English:** At equal compression (p = 0.008 for fixed, random, similarity, and word merging; n = 8). This contradicts the original hypothesis that high-fertility tokens are redundant. The byte-level fragments of a Devanagari word appear not to be interchangeable.
*   **Attention entropy is not a redundancy signal:** Entropy-guided merging is no better than random in Nepali and *worse* than random in English (p <= 0.04).
*   **Confounds still open:** Merging variants also collapsed the most recent tokens (which eviction baselines protect), and the contexts are short (~566 tokens), favoring recency.

### Earlier Inconclusive Experiment
An initial in-place merging test (30 sentence pairs, layers 4-10) showed 1-3% perplexity change in both languages, but the cache never shrank, there was no control, and averaging inside a block leaked future tokens. It is superseded by the equal-memory study above.

## 💻 Next Experiments & Hardware Roadmap

As an independent research project, development is proceeding incrementally.

| # | Experiment | Command |
|---|---|---|
| 1 | Control: merge only the older context, protect the last 32 tokens | `python kv_compress.py --protect-last 32 --docs 12 --ratios 0.3 0.6 --out compress_pl32.csv` |
| 2 | Word-aligned eviction (whole words kept or evicted) + more documents | `python kv_compress.py --methods evict_snapkv evict_snapkv_word evict_window --ratios 0.3 0.6 0.8 --docs 30 --out compress_evict.csv` |
| 3 | Long-range retrieval probe (needle) | `python kv_compress.py --needle --methods evict_snapkv evict_snapkv_word evict_window merge_word --ratios 0.3 0.6 0.8 --docs 20 --out needle_results.csv` |
| 4 | Second language (Hindi) and second model; generation check (`--gen-steps 16`) | planned |
| 5 | NVIDIA GPU replication and profiling; vLLM comparison | planned |
| 6 | CUDA/MPS kernel | conditional: eviction's selection step is a cheap top-k, profile first |

*Analysis:* `python analyze_compress.py <csv>` and `python analyze_needle.py needle_results.csv`.
*Self-test (hand-written forward pass vs Hugging Face):* `python kv_compress.py --selftest`.

## Limitations
*   One small model family (1B), short contexts, one language pair, CPU / Apple Silicon only.
*   8 documents per language, many paired tests, no multiplicity correction; treat p-values as indicative.
*   Perplexity and top-1 agreement are proxies; downstream-task quality is not yet measured.
*   Averaging post-RoPE keys is an approximation; selection is shared across heads within a layer.
*   SnapKV-style baseline implements the first-stage family of RocketKV only (no adaptive pooling, no decode-time sparse attention).

## Related Work
*   Kwon et al., 2023, *PagedAttention* (vLLM); Dao et al., 2022, *FlashAttention*
*   Xiao et al., 2023, *StreamingLLM*; Zhang et al., 2023, *H2O*; Li et al., 2024, *SnapKV*
*   Liu et al., 2023, *Scissorhands*; Liu et al., 2024, *KIVI*
*   Behnam et al., ICML 2025, *RocketKV* (Tumanov senior author); Agrawal et al., OSDI 2024, *Sarathi-Serve*
*   Petrov et al., 2023, *Language Model Tokenizers Introduce Unfairness Between Languages*

---
*An independent research project by Ashutosh Bhattarai (Pulchowk Campus, Institute of Engineering) aimed at bridging the gap between multilingual LLM usability and backend systems efficiency.*