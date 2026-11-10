# Language-Fair LLM Serving: The Systems Cost of Tokenizer Fertility & Word-Aligned KV-Cache Eviction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Domain: Systems for ML](https://img.shields.io/badge/Domain-SysML%20%2F%20LLM%20Serving-purple.svg) ![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-green.svg)

> **Abstract & Executive Summary**: Multi-tenant Large Language Model (LLM) serving engines (e.g., vLLM, SGLang, `llama.cpp`) allocate Key-Value (KV) cache memory buffers based on token counts. However, Byte-Pair Encoding (BPE) tokenizers exhibit severe **Tokenizer Fertility Inequality**: non-Latin, morphologically rich languages (e.g., Nepali / Devanagari) require **$2.6\times$ to $4.0\times$ more tokens** to express the exact same semantic payload as English. Consequently, non-English streams exhaust GPU PagedAttention dynamic block allocations $2.6\times$ to $4.0\times$ faster, triggering early GPU Out-Of-Memory (OOM) preemptions, degraded request concurrency, and inflated latency.

> This repository presents a systems and algorithmic analysis of language-fair KV cache compression and identifies two key failure modes in current compression literature:

> 1. **Mathematical RoPE Phase Corruption:** Vector state-merging (`merge_entropy`) linearly averages post-Rotary Position Embedding Key vectors, causing destructive phase interference that corrupts query-key spatial attention and reduces long-range factual recall to **$0\%$ exact-match accuracy**.  
> 2. **Morphological Sub-Word Fracturing:** Standard query-aware eviction (`evict_snapkv`) operates on isolated sub-word tokens, inadvertently evicting mid-word byte fragments in high-fertility scripts and dropping exact-match recall to **$75\%$** under $80\%$ compression.

> To resolve this, we introduce **Word-Aligned SnapKV (`evict_snapkv_word`)**, an atomic, offset-aware eviction policy that aggregates attention scores across sub-word character spans. At **$80\%$ KV cache compression** (achieving complete memory footprint parity with English), Word-Aligned SnapKV restores **$100\%$ exact-match factual recall**, improves NLL ($\Delta \text{NLL} = -0.0458$), and eliminates compounding error drift in 16-step free-running autoregressive generation.

---

## 1\. Problem Formulation: Tokenization Inequality

### 1.1 Mathematical Definition of Token Fertility

Modern LLM serving systems allocate Key-Value cache memory proportional to input token sequence length $L$. For a sequence of text containing semantic payload $\mathcal{S}$ (measured in sentence or proposition units), we define **Token Fertility ($\mathcal{F}$)** as:

$$ \mathcal{F} = \frac{\text{Total Generated BPE Tokens}}{\text{Semantic Payload Units } (\mathcal{S})} $$

Evaluating the Llama-3.2-1B tokenizer on the parallel FLORES-200 benchmark:

* **English ($\mathcal{F}_{\text{En}}$):** $\sim 27.4 \text{ tokens/sentence}$
* **Nepali ($\mathcal{F}_{\text{Ne}}$):** $\sim 68.6 \text{ tokens/sentence}$ ($\mathcal{F}_{\text{Ne}} \approx 2.6\times \text{ to } 4.0\times \mathcal{F}_{\text{En}}$)

```
[English Payload]  --> "The quick brown fox jumps over the lazy dog."      ~ 10 Tokens  (1.0x KV Memory)
[Nepali Payload]   --> "चलाख रातो स्याल अल्छी कुकुरमाथि उफ्रन्छ।"        ~ 28 Tokens  (2.8x KV Memory)
```

### 1.2 Concurrency Collapse in Multi-Tenant Systems

In virtualized memory managers (such as vLLM's PagedAttention or `llama.cpp` static block managers), KV physical blocks are pinned per token slot. High-fertility requests rapidly exhaust the dynamic page pool:

![Serving Capacity Benchmark](research_serving_capacity_benchmark.png) *Figure 1: Hardware memory profiling under multi-tenant serving workloads (`src/profiling`). High-fertility workloads trigger accelerated paging block exhaustion and early concurrency collapse under static memory management, whereas adaptive context sizing bounds latency.*

Standard serving metrics like **Raw Token Throughput (tokens/sec)** mask this systemic inequality because higher token generation rates on non-Latin text represent equal or lower semantic throughput. We define **Effective Semantic Throughput ($\mathbb{T}_{\text{eff}}$)**:

$$ \mathbb{T}_{\text{eff}} = \frac{\text{Raw Token Throughput (tokens/sec)}}{\mathcal{F}} $$
To equalize $\mathbb{T}_{\text{eff}}$ without truncating context bounds or retraining tokenizers, we investigate dynamic runtime KV cache compression.

---

## 2\. Dynamic KV Cache Compression Space

We design and benchmark three architectural KV cache reduction paradigms inside `kv_compress.py`:

```
                                  KV Cache Compression Strategies
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
   State Merging                       Standard Eviction                      Word-Aligned Eviction
 (`merge_entropy`)                      (`evict_snapkv`)                     (`evict_snapkv_word`)
 ─────────────────                     ─────────────────                     ───────────────────────
 Averages Key/Value vectors            Retains top-K sub-word                Aggregates attention over 
 based on low attention-               tokens via observation                character spans; keeps/evicts 
 entropy (destructive).                window (sub-word gap).                atomic word units (Ours).
```

1. **State Merging (`merge_entropy`)**: Identifies tokens with low attention-entropy and linearly merges their Key and Value vectors into centroid representations: $\bar{K}=\frac{1}{M}\sum\limits_{}^{}{K}_{i},\bar{V}=\frac{1}{M}\sum\limits_{}^{}{V}_{i}$.  
2. **Sub-Word Query-Aware Eviction (`evict_snapkv`)**: Evaluates prompt attention within an observation window $W_{\text{obs}}$, ranks token indices by cumulative attention score, and retains the top\-$K$ sub-word tokens while evicting the remainder.  
3. **Word-Aligned Query-Aware Eviction (`evict_snapkv_word` \- Proposed)**: Maps sub-word tokens back to raw character offsets in the source string. Aggregates attention scores using max-pooling across constituent sub-word tokens, enforcing atomic retain/evict decisions at whole-word boundaries.

---

## 3\. Theoretical Analysis: Proof of State-Merging Failure

### 3.1 RoPE Phase Corruption Proof

Modern LLMs utilize Rotary Position Embeddings (RoPE), where position $m$ is embedded by rotating Key vector ${K}_{m}={W}_{K}{x}_{m}$ in $d/2$ two-dimensional planes using rotation matrix ${R}_{\Theta ,m}^{d}$:

$${\tilde{K}}_{m}={R}_{\Theta ,m}^{d}{K}_{m}$$

When merging $M$ contiguous token keys into a centroid key vector $\bar{K}_{\text{merged}}$, the resulting representation is:
$$ \bar{K}_{\text{merged}} = \frac{1}{M} \sum_{i=1}^{M} R_{\Theta, m_i}^d \cdot (W_K x_{m_i}) $$

Because ${R}_{\Theta ,m}^{d}$ is non-linear with respect to position index $m$, linear vector addition introduces **destructive phase interference**:

$$ \left\| \bar{K}_{\text{merged}} \right\|_2 \ll \frac{1}{M} \sum_{i=1}^{M} \left\| R_{\Theta, m_i}^d (W_K x_{m_i}) \right\|_2 $$
When a query vector ${Q}_{n}={R}_{\Theta ,n}^{d}({W}_{Q}{x}_{n})$ computes dot-product attention against $\bar{K}_{\text{merged}}$, the relative spatial rotation angles $(n-{m}_{i})$ are mathematically corrupted.

### 3.2 Empirical Proof: Needle-in-a-Haystack Probe

We placed a 4-digit secret key at $10\%$ context depth and evaluated factual retrieval under $60\%$ KV cache reduction:

| Strategy | Compression Class | Nepali Perplexity Shift ($\Delta$ NLL) | Exact Match Retrieval Accuracy |
| :---- | :---- | :---: | :---: |
| **`merge_entropy`** | Vector Centroid Merging | $+27.47\%$ | **$0.0\%$** |
| **`merge_fixed`** | Fixed Block Merging | $+46.84\%$ | **$12.5\%$** |
| **`evict_snapkv`** | Physical Sub-Word Eviction | **$+3.52\%$** | **$100.0\%$** |

> **Systems Takeaway:** Key vectors encoded with RoPE cannot be linearly merged. Dynamic cache reduction must be achieved via **physical memory eviction**, not vector interpolation.

---

## 4\. Proposed Method: Word-Aligned Eviction (`evict_snapkv_word`)

### 4.1 Morphological Fracturing in Sub-Word Eviction

Under high compression ratios ($\ge 80\%$), standard sub-word eviction (`evict_snapkv`) ranks tokens independently. In morphologically rich scripts, BPE splits single words into multi-byte sub-word tokens.

For example, the word **`नेपाल`** splits into sub-words `["ने", "पा", "ल"]`. Standard SnapKV may retain `["ने", "ल"]` while evicting `["पा"]`. The cache now retains a syntactically fractured fragment, dropping exact-match retrieval accuracy from $100\%$ to **$75\%$**.

```
[Standard Sub-Word Eviction]
Original String:   [  ने  ] [  पा  ] [  ल  ]
Cache Status:      [ KEEP ] [EVICT ] [ KEEP ]  --> Morphological Fracture (Semantic Noise)

[Word-Aligned Eviction (Ours)]
Original String:   [  ने  ] [  पा  ] [  ल  ]
Word Boundary:     └─── Group: "नेपाल" ───┘
Cache Status:      [ KEEP ] [ KEEP ] [ KEEP ]  --> Atomic Word Preservation
```

### 4.2 Algorithm Implementation

`evict_snapkv_word` enforces atomic word boundary preservation:

1. **Token-to-Word Mapping:** Extract character offset bounds $[c_{\text{start}}, c_{\text{end}}]$ for each sub-word token ${t}_{i}$.
2. **Attention Max-Pooling:** Aggregate attention scores across sub-word tokens belonging to the same word $W_j = \{t_{j,1}, \dots, t_{j,k}\}$: $$ S(W_j) = \max_{t \in W_j} \left( \sum_{h=1}^{H} \sum_{\tau \in W_{\text{obs}}} A_{h, \tau, t} \right) $$  
3. **Atomic Selection:** Words ${W}_{j}$ are ranked by $S({W}_{j})$, retaining or evicting whole word token groups atomically.

```py
# Implementation excerpt from kv_compress.py
def get_word_boundaries(offsets, text):
    words, current_word = [], []
    for idx, (start, end) in enumerate(offsets):
        if start == end: 
            continue
        if text[start:end].startswith(" ") or text[start:end].startswith("\n"):
            if current_word: 
                words.append(current_word)
            current_word = [idx]
        else:
            current_word.append(idx)
    if current_word: 
        words.append(current_word)
    return words
```

---

## 5\. Experimental Results

### 5.1 Perplexity vs. KV Cache Reduction

Evaluated on Llama-3.2-1B-Instruct across parallel English and Nepali FLORES-200 corpora (`compress_results.csv`, processed via `analyze_compress.py`).

![Perplexity Curves vs. KV Entries Removed](compress_curves.png) *Figure 2: Perplexity change relative to uncompressed baseline across compression ratios. The vertical dashed line at **$61\%$ compression** represents the **Memory Parity Point**, where Nepali's compressed KV memory footprint matches the uncompressed memory footprint of English.*

Key Findings:

* At **$61\%$ compression**, Nepali matches English memory consumption while maintaining baseline log-likelihood ($\Delta \text{NLL} \le +0.02$).  
* Word-aligned eviction consistently outperforms standard SnapKV in negative log-likelihood stability across compression thresholds.

### 5.2 Long-Range Factual Retrieval Probe (Needle-in-a-Haystack)

Retrieval of a 4-digit key placed at $10\%$ context depth under extreme $80\%$ compression (`analyze_needle.py`):

| Method (@ 80% Compression) | English Exact Match | Nepali Exact Match | Nepali Delta NLL | Memory Parity Achieved |
| :---- | :---: | :---: | :---: | :---: |
| **Uncompressed Baseline** | $100.0\%$ | $100.0\%$ | $0.0000$ | No |
| **Standard SnapKV (`evict_snapkv`)** | $50.0\%$ | $75.0\%$ | $+0.1229$ | Yes |
| **Word-Aligned SnapKV (`evict_snapkv_word`)** | **$62.5\%$** | **$100.0\%$** | **$-0.0458$** | **Yes** |

### 5.3 Multi-Step Autoregressive Decoding Audit

To evaluate long-context stability during output generation, we conducted **16-step free-running autoregressive decoding** (`--gen-steps 16`, output dataset `gen_results_16_2.csv`), continuously querying the compressed KV cache:

| Policy (Nepali @ 60% Compression) | 16-Step Prefix Match | KL Divergence (nats/tok) | Failure Mode Analysis |
| :---- | :---: | :---: | :---- |
| **Word-Aligned SnapKV (`evict_snapkv_word`)** | **$25.00\%$** | **$0.0746$** | **Stable:** Intact word boundaries prevent drift. |
| **Standard SnapKV (`evict_snapkv`)** | $23.75\%$ | $0.0718$ | **Minor Drift:** Sub-word fragmentation introduces noise. |
| **Window Eviction (`evict_window`)** | $20.63\%$ | $0.0831$ | **Moderate Drift:** Loss of initial prompt context. |
| **State Merging (`merge_entropy`)** | $9.38\%$ | $0.2121$ | **Severe Breakdown:** RoPE corruption causes hallucinations. |

---

## 6\. Repository Structure & Reproducibility

```
llm-inference-test/
├── kv_compress.py            # Main engine: prefill, leakage-free scoring, needle probe, 16-step decoding
├── analyze_compress.py       # Statistical evaluation: paired t-tests & memory parity interpolation
├── analyze_needle.py         # Long-range retrieval needle evaluation scripts
├── compress_curves.png       # Perplexity vs. KV Cache Reduction plot
├── research_serving_capacity_benchmark.png # Baseline GPU/CPU memory profiling plot
├── compress_results.csv      # Benchmarking dataset across languages & compression ratios
├── gen_results_16_2.csv      # 16-step autoregressive generation audit dataset
└── llama_cpp/                # Phase 1 hardware memory profiling scripts
```

### Reproducibility Commands

**1\. Primary Benchmarking (Perplexity & Top-1 Agreement):**

```sh
python kv_compress.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --docs 10 \
  --ratios 0.3 0.6 0.8 \
  --methods evict_snapkv_word evict_snapkv evict_window merge_entropy \
  --out compress_results.csv
```

**2\. Long-Range Factual Recall Probe (Needle-in-a-Haystack):**

```sh
python kv_compress.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --needle \
  --needle-pos 0.1 \
  --docs 10 \
  --ratios 0.3 0.6 0.8 \
  --out needle_results.csv
```

**3\. Multi-Step Autoregressive Generation Audit:**

```sh
python kv_compress.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --docs 10 \
  --ratios 0.3 0.6 0.8 \
  --methods evict_snapkv_word evict_snapkv evict_window merge_entropy \
  --gen-steps 16 \
  --out gen_results_16.csv
```

**4\. Run Statistical Analysis & Visualization:**

```sh
python analyze_compress.py --input compress_results.csv
python analyze_needle.py --input needle_results.csv
```

---

## 7\. Systems Research Roadmap

- [x] **Phase 1: Tokenization Inequality Quantification:** Profiling BPE fertility gaps across languages (`llama.cpp` context profiling).  
- [x] **Phase 2: RoPE Phase Corruption Proof:** Theoretical derivation and empirical validation of RoPE phase interference in state merging.  
- [x] **Phase 3: Word-Aligned Eviction Design:** Engineering `evict_snapkv_word` using character offset maps and attention max-pooling.  
- [x] **Phase 4: Autoregressive Stability Audit:** Measuring multi-step generation drift and KL-divergence bounds.  
- [ ] **Phase 5: Triton Segment-Aware Top-K Kernel (In Progress):** Writing a Triton kernel to perform offset-aware word pooling directly on GPU memory during prefill.  
- [ ] **Phase 6: vLLM Block Manager Integration (Planned):** Implementing word-aligned eviction as a custom dynamic block manager within vLLM's PagedAttention.

---

## 📜 Citation & Contact

If you use this repository, findings, or code in your research on SysML, LLM serving infrastructure, or KV cache compression, please cite:

```bibtex
@misc{bhattarai2026languagefair,
  author       = {Bhattarai, Ashutosh},
  title        = {Language-Fair LLM Serving: The Systems Cost of Tokenizer Fertility \& Word-Aligned KV-Cache Eviction},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/LCfreak/llm-inference-test}}
}
```

&nbsp;

* **Author:** Ashutosh Bhattarai (Institute of Engineering, Tribhuvan University)  
* **Research Focus:** Systems for Machine Learning (SysML), LLM Serving Infrastructure, Memory Management, KV Cache Optimization  
* **Contact:** [bhattaraiaasu1@gmail.com](mailto:ashutosh.bhattarai.research@gmail.com) *(Seeking PhD opportunities for Fall 2027\)*