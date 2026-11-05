# Language-Fair LLM Serving: Systems Costs of Tokenizer Fertility & Word-Aligned KV-Cache Eviction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Research](https://img.shields.io/badge/Status-Active_Research-green.svg)]()

> **Core Finding**: High-fertility non-Latin scripts (e.g., Nepali / Devanagari) incur a **2.6× to 4× memory penalty** in standard LLM serving engines due to BPE tokenization fragmentation. While vector-merging compression strategies corrupt Rotary Position Embeddings (RoPE) and drop long-range factual recall to **0%**, our proposed **Word-Aligned SnapKV (`evict_snapkv_word`)** enforces atomic word-level eviction. It achieves **equal memory footprint to English** with only **+3.5% perplexity degradation** and retains **100% exact-match long-range retrieval at 80% compression**.

---

## 📌 Abstract & Executive Summary

Modern Large Language Model (LLM) serving frameworks (e.g., vLLM, TensorRT-LLM) manage Key-Value (KV) memory using fixed sequence lengths or dynamic block paging. However, Byte-Pair Encoding (BPE) tokenizers exhibit severe **tokenization inequality**: non-Latin scripts require up to 4× more tokens to represent the exact same semantic content as English.

This repository investigates the systems-level cost of tokenizer fertility and evaluates dynamic context compression algorithms under strict leakage-free evaluation protocols. We demonstrate that:
1. **State-Merging Destroys Factual Recall**: Averaging adjacent or low-entropy KV vectors (`merge_entropy`, `merge_sim`, `merge_word`) degrades perplexity and completely destroys exact-match retrieval (**0% accuracy**) due to phase misalignment in post-RoPE vector space.
2. **Sub-Word Eviction Penalizes High-Fertility Scripts**: Standard query-aware eviction (SnapKV) evaluates individual sub-word byte fragments. At extreme compression (80%), standard sub-word eviction drops Nepali factual recall to **75%**.
3. **Word-Aligned SnapKV Restores Language Parity**: By aggregating attention scores across tokenizer character offsets and evicting or retaining whole semantic word units atomically, **`evict_snapkv_word`** preserves **100% exact factual recall at 80% compression in Nepali** and **62.5% in English**, outperforming all existing baseline eviction and merging schemes.

---

## 1. The Systems Problem: Tokenization Inequality

### 1.1 The Fertility Gap
BPE tokenizers are constructed primarily over English-dominated text corpora. Consequently, morphologically rich or non-Latin scripts fall back to single-byte or multi-byte sub-word fragments.

$$\text{Token Fertility } (\mathcal{F}) = \frac{\text{Total Tokens Generated}}{\text{Semantic Payload / Sentence Count}}$$

* **English Baseline**: $\sim 27.4$ tokens / sentence (FLORES-200 sample).
* **Nepali (Devanagari)**: $\sim 68.6$ tokens / sentence ($\mathcal{F}_{\text{Nepali}} \approx 2.6 \times \text{ to } 4.0 \times$).

```
High Token Fertility (Nepali) ---> 2.6x - 4x KV Cache Entries / Request
                               ---> Accelerated Block Exhaustion (PagedAttention)
                               ---> Early GPU Out-Of-Memory (OOM)
                               ---> Throttled Multi-Tenant Concurrency & Low Effective Throughput
```

### 1.2 The Systems Metric: Effective Semantic Throughput
Standard system metrics like *tokens/second* or *raw decoding latency* mask this inequality because a server generating 100 tokens/sec in Nepali produces less than half the semantic progress of 100 tokens/sec in English.

$$\text{Effective Semantic Throughput } (\mathbb{T}_{\text{eff}}) = \frac{\text{Raw Token Throughput (tokens/sec)}}{\text{Language Token Fertility } (\mathcal{F})}$$

---

## 2. Theoretical & Empirical Failure of State Merging

### 2.1 The Mathematics of RoPE Vector Corruption
Rotary Position Embeddings (RoPE) apply a position-dependent orthogonal rotation matrix $R_{\Theta, m}^d$ to Key vectors $K_m$ at sequence index $m$:

$$K_m = R_{\Theta, m}^d \cdot (W_K \cdot x_m)$$

When merging a sequence of $M$ contiguous token states into a single centroid cache vector $\bar{K}$:

$$\bar{K} = \frac{1}{M} \sum_{i=1}^{M} R_{\Theta, m_i}^d \cdot (W_K \cdot x_{m_i})$$

Because $R_{\Theta, m}^d$ alters vector phase angles across feature dimensions, linearly averaging post-RoPE vectors introduces **destructive phase interference**. The query-key dot product:

$$q_n^\top \bar{K} = \frac{1}{M} \sum_{i=1}^{M} q_n^\top \left( R_{\Theta, m_i}^d W_K x_{m_i} \right)$$

corrupts spatial distance information, rendering attention weights non-selective during continuation decoding.

### 2.2 Needle-in-a-Haystack Factual Retrieval Breakdown
While state merging exhibits seemingly moderate degradation in standard perplexity tests (+27.5% NLL change), it **completely fails** long-range retrieval probes (*Needle-in-a-Haystack*):

| Compression Strategy | Perplexity Change (Nepali @ 61%) | Needle Retrieval Accuracy (Nepali @ 60%) |
| :--- | :---: | :---: |
| `merge_entropy` | +27.47% | **0.0%** |
| `merge_random` | +28.33% | **12.5%** |
| `merge_sim` | +32.35% | **25.0%** |
| `merge_fixed` | +46.84% | **12.5%** |
| **`evict_snapkv` (Sub-word)** | **+3.52%** | **100.0%** |
| **`evict_snapkv_word` (Ours)** | **+3.52%** | **100.0%** |

---

## 3. Proposed Method: Word-Aligned SnapKV (`evict_snapkv_word`)

### 3.1 Algorithm Architecture
Standard SnapKV pools query attention over an observation window $W$ at the end of prefill to identify candidate Key-Value positions to retain. However, sub-word tokenization causes SnapKV to retain disjoint byte fragments of a single word while evicting others.

`evict_snapkv_word` bridges tokenization offsets and linguistic boundaries:
1. **Offset Mapping**: Tokenizer character offsets map token indices $t \in [0, P)$ to whitespace-delimited word identifiers $w(t)$.
2. **Word Attention Aggregation**: The observation window score $S(w)$ for word $w$ is defined as the maximum smoothed attention score among its sub-word tokens:
   $$S(w) = \max_{t \in w} \left( \text{Pool}\left( \sum_{i=P-W}^{P} A_{i, t} \right) \right)$$
3. **Atomic Word Selection**: Entire words are selected in descending order of $S(w)$ until the target retention budget $K_{\text{keep}}$ is satisfied. If a word exceeds the remaining budget, leftover slots are filled with top sub-word tokens.

```
[Standard Sub-Word Eviction]  : [ने] (Kept)  [पा] (Evicted) [ल] (Kept) ---> Word Semantic Corrupted
[Word-Aligned Eviction (Ours)]: [ने पा ल] (Kept Atomically)           ---> Word Semantic Intact
```

---

## 4. Empirical Benchmark & Visual Analysis

### 4.1 Perplexity vs. Compression Ratio (`compress_curves.png`)

The chart below shows the perplexity change relative to an uncompressed baseline across English and Nepali context lengths. The vertical dashed line at **61% KV entries removed** indicates the exact memory parity point where Nepali's memory footprint equals English's uncompressed footprint.

![Perplexity Curves vs. KV Entries Removed](compress_curves.png)

### 4.2 Equal-Memory Benchmark (61% Compression Threshold)
*Context Lengths: English = 220 tokens, Nepali = 566 tokens. Nepali must drop 61% of its KV entries to achieve memory parity with English.*

| Method | Method Family | Nepali % PPL Change @ 61% | Nepali Top-1 Agreement | Status / Usability |
| :--- | :--- | :---: | :---: | :--- |
| **`evict_snapkv`** | Eviction (Sub-word) | **+3.52%** | **0.836** | **Usable / Near-Lossless** |
| **`evict_snapkv_word`** | Eviction (Word-Aligned) | **+3.52%** | **0.836** | **Usable / Near-Lossless** |
| `evict_window` | Eviction (Recency Sink) | +5.15% | 0.819 | Usable |
| `evict_h2o` | Eviction (Heavy-Hitter) | +16.94% | 0.731 | High Degradation |
| `merge_entropy` | Merging (Entropy-Guided) | +27.47% | 0.722 | Fails Factual Recall |
| `merge_random` | Merging (Random Control) | +28.33% | 0.710 | Baseline Control |
| `merge_word` | Merging (Sub-word Merge) | +30.90% | 0.712 | High Degradation |
| `merge_sim` | Merging (Cosine Similarity) | +32.35% | 0.730 | High Degradation |
| `merge_fixed` | Merging (Fixed Window) | +46.84% | 0.661 | Severe Degradation |

### 4.3 Extreme Compression Needle Retrieval (80% KV Reduction)
*Needle-in-a-Haystack probe: 4-digit secret key retrieval planted at 10% context depth.*

| Method | English Exact Match (80% Removed) | Nepali Exact Match (80% Removed) | Nepali Delta NLL |
| :--- | :---: | :---: | :---: |
| `evict_snapkv` | 50.0% | 75.0% | +0.1229 |
| **`evict_snapkv_word` (Ours)** | **62.5%** | **100.0%** | **-0.0458** |
| `evict_h2o` | 0.0% | 0.0% | +5.3814 |
| `evict_window` | 0.0% | 0.0% | +5.4259 |
| `merge_entropy` | 0.0% | 0.0% | +4.7894 |
| `merge_fixed` | 0.0% | 0.0% | +7.1127 |

---

## 📁 Repository Architecture & Codebase Structure

```
llm-inference-test/
├── kv_compress.py            # Core engine: manual Llama forward pass, prefill,
│                             # leakage-free continuation scoring, needle probes,
│                             # and 9 compression algorithm implementations.
├── analyze_compress.py       # Statistical analysis runner: paired t-tests,
│                             # equal-memory interpolations, and LaTeX/CSV generation.
├── analyze_needle.py         # Long-range retrieval needle evaluation scripts.
├── compress_curves.png       # Generated publication figure (Perplexity vs. % Removed).
├── compress_results.csv      # Full evaluation benchmark dataset (384+ test runs).
├── fertility_results.csv     # Tokenizer fertility baseline metrics on FLORES-200.
├── README.md                 # Project documentation & systems paper summary.
└── src/
    └── profiling/            # llama.cpp C++ VRAM footprint & block allocation scripts.
```

### Key Functions in `kv_compress.py`
* **`prefill(model, ids, words, obs_window)`**: Executes uncompressed context prefill; extracts post-RoPE $K, V$ tensors, cumulative attention mass (`cum`), row entropy (`ent`), and observation window attention (`obs`).
* **`continue_with_cache(model, cont_ids, comp, start_pos)`**: Evaluates continuation tokens against physically compressed KV cache blocks without leakage.
* **`groups_snapkv_word(P, st, R, protect_first)`**: Implements word-granularity SnapKV pooling and atomic word selection.
* **`apply_groups(K, V, groups, use_bias)`**: Performs physical tensor pooling and applies $\log(m)$ attention mass compensation bias.
* **`run_needle(args, model, tokenizer, ...)`**: Executes the long-range 4-digit retrieval probe.

---

## 💻 Reproducibility & Execution Guide

### 1. Requirements & Installation
```bash
git clone https://github.com/LCfreak/llm-inference-test.git
cd llm-inference-test
pip install torch transformers numpy pandas matplotlib seaborn
```

### 2. Run Execution Self-Test
Verify that the hand-written Llama forward pass matches Hugging Face implementation outputs ($\text{error} < 10^{-3}$):
```bash
python kv_compress.py --selftest
```

### 3. Run Perplexity Compression Suite
Evaluate perplexity and top-1 agreement across 8 documents for English and Nepali:
```bash
python kv_compress.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --csv fertility_results.csv \
  --docs 8 \
  --ratios 0.3 0.6 0.8 \
  --out compress_results.csv
```

### 4. Run Needle-in-a-Haystack Factual Retrieval Probe
```bash
python kv_compress.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --csv fertility_results.csv \
  --docs 8 \
  --needle \
  --needle-pos 0.1 \
  --ratios 0.3 0.6 0.8 \
  --out needle_results.csv
```

### 5. Generate Statistical Analysis & Plot Curves
```bash
python analyze_compress.py compress_results.csv
```

---

## 🛣️ Systems Roadmap

- [x] **Phase 1: Baseline Profiling**: Quantified tokenizer fertility ratio ($2.6\times - 4.0\times$) using `llama.cpp`.
- [x] **Phase 2: Algorithmic Validation**: Proved RoPE corruption under vector-merging; validated `evict_snapkv_word` on CPU/Apple Silicon.
- [ ] **Phase 3: Free-Running Generation**: Benchmark long-form autoregressive decoding (`--gen-steps 16`) across compressed caches.
- [ ] **Phase 4: Custom CUDA Kernel**: Implement a segment-aware GPU top-$k$ kernel in CUDA/Triton to prevent prefill latency overhead.
- [ ] **Phase 5: vLLM PagedAttention Integration**: Port word-aligned eviction as a dynamic block manager in vLLM multi-tenant serving.

---

## 📜 Citation & Attribution

If you use this codebase or baseline findings in your research on SysML, tokenization inequality, or KV cache compression, please cite:

```bibtex
@misc{bhattarai2026languagefair,
  author = {Bhattarai, Ashutosh},
  title = {Language-Fair LLM Serving: Systems Costs of Tokenizer Fertility \& Word-Aligned KV-Cache Eviction},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/LCfreak/llm-inference-test}}
}
```

---
*Created by **Ashutosh Bhattarai** (Pulchowk Campus, Institute of Engineering, Tribhuvan University).*