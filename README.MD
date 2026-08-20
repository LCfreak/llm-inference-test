# Hardware-Aware LLM Inference Under Non-Uniform Token Distributions

> Investigating how tokenizer fertility disparities across languages create measurable systems-level bottlenecks in LLM inference — with a focus on memory, concurrency, and hardware-aware mitigation strategies.

**Status:** Active experimental research  
**Domain:** AI Systems · LLM Inference · KV Cache · Memory Systems · GPU Architecture  
**Hardware:** Apple Silicon (ARM64) → NVIDIA CUDA (in progress)

---

## The Core Observation

Most LLM inference systems assume token sequences are roughly uniform in length across inputs. They are not.

Languages using scripts such as Devanagari (Nepali, Hindi, Sanskrit) are tokenized at significantly higher fertility by standard BPE tokenizers — not because of linguistic complexity, but because of a structural artifact: BPE merge rules are learned from training corpora that are overwhelmingly English. Devanagari characters frequently fall back to byte-level representations, producing 4–5x more tokens for semantically equivalent content.

This is not primarily an NLP problem. It is a **systems problem with an NLP root cause**.

```
Non-uniform token fertility
         ↓
Non-uniform KV cache pressure
         ↓
Non-uniform memory allocation per request
         ↓
Reduced serving concurrency for affected languages
         ↓
Structural infrastructure inequity at scale
```

The question this project investigates: **can the inference system account for token distribution non-uniformity to recover lost concurrency and memory efficiency?**

---

## Key Results (Preliminary)

All results measured on Apple Silicon M-series (ARM64, unified memory) using llama.cpp with Llama 3.2-1B and Qwen 2.5-0.5B.

### Tokenization Fertility Disparity

| Language | Tokens per 100 chars (short) | Tokens per 100 chars (long) | Fertility ratio vs English |
|----------|------------------------------|------------------------------|---------------------------|
| English  | ~27                          | ~19                          | 1.0× (baseline)           |
| Hindi    | ~60                          | ~53                          | ~2.9×                     |
| Nepali   | ~64                          | ~57                          | ~3.2×                     |

### KV Cache Memory Per Request

| Allocation Strategy | English | Hindi | Nepali |
|---------------------|---------|-------|--------|
| Static n_ctx=512    | ~40 MB  | ~40 MB | ~40 MB |
| Static n_ctx=2048   | ~150 MB | ~150 MB | ~150 MB |
| **Adaptive Context**| **~3 MB** | **~8 MB** | **~8 MB** |

Adaptive context allocation reduces per-request KV cache footprint by **5–15× vs static baselines** while preserving generation quality.

### Concurrent Serving Capacity (2 GB KV Budget)

| Strategy            | English users | Hindi users | Nepali users |
|---------------------|--------------|-------------|--------------|
| Static n_ctx=512    | ~63          | ~63         | ~63          |
| Static n_ctx=2048   | ~13          | ~13         | ~13          |
| **Adaptive Context**| **~415**     | **~258**    | **~250**     |

Under a fixed 2 GB KV cache budget, adaptive allocation serves **4–6× more concurrent users** than static allocation strategies — with the benefit distributed more equitably across language groups than static approaches.

> **Finding:** Per-request latency appears equal across languages (~170ms), masking the real cost. The disparity becomes visible only at serving scale, where token fertility differences translate directly into reduced concurrency capacity.

---

## Why This Framing Matters

Standard inference benchmarks report:

```
tokens/second   latency   GPU utilization
```

These metrics are computed over tokens — not semantic units. When a language requires 3× more tokens to express the same content, token-level benchmarks systematically underreport the effective cost of serving that language.

This project introduces the concept of **effective semantic throughput**:

```
Effective Semantic Throughput = Token Throughput / Token Fertility
```

Under this metric, the serving cost gap between English and Nepali becomes explicit rather than hidden.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Request                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Language / Script Detector                     │
│  Devanagari Unicode (U+0900–U+097F) ratio analysis          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Token Fertility Estimator                      │
│  Tokenize with minimal context → measure chars/token        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           Adaptive KV Cache Context Allocator               │
│  n_ctx = ceil((token_count + output_budget) × buffer)       │
│  buffer: 1.5× Devanagari · 1.2× Latin                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Engine                          │
│  llama.cpp (ARM64) → vLLM → custom CUDA (planned)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Measurement & Analysis                      │
│  TTFT · TPOT · Memory · Concurrency · Semantic Throughput   │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiments

### Completed

**Experiment 1 — Tokenizer Fertility Characterization**  
Measured token fertility across English, Hindi, and Nepali for short, medium, and long prompts across two model families (Llama 3.2-1B, Qwen 2.5-0.5B). Fertility ratios are consistent across model families, confirming the effect is tokenizer-driven, not model-specific.

**Experiment 2 — KV Cache Memory Under Static vs Adaptive Allocation**  
Compared per-request KV cache footprint under three allocation strategies. Adaptive allocation reduces memory footprint by 5–15× with no degradation in generation quality.

**Experiment 3 — Concurrent Serving Capacity**  
Measured maximum concurrent users under a fixed 2 GB KV cache budget. Adaptive allocation recovers 4–6× concurrency vs static baselines across all language groups.

**Experiment 4 — Serving Capacity by Prompt Length**  
Measured how concurrency capacity degrades as prompt length increases. English degrades more slowly than Hindi/Nepali under static allocation; adaptive allocation narrows this gap.

---

### In Progress

**Experiment 5 — Multi-Model Fertility Comparison**  
Extend fertility measurements to mBERT, XLM-R, Mistral, Gemma 2, and language-specific models (Namaste-LLaMA, Airavata). Hypothesis: models trained with larger multilingual vocabularies show smaller fertility gaps for Devanagari.

**Experiment 6 — vLLM PagedAttention Behavior**  
Test whether vLLM's page-based KV cache allocation handles high-fertility languages differently than llama.cpp's contiguous allocation. PagedAttention uses fixed-size "pages" — the interaction with variable fertility distributions is unexplored.

**Experiment 7 — TTFT / TPOT Decomposition**  
Decompose total inference time into prefill (TTFT) and decode (TPOT) components across languages. Hypothesis: prefill cost scales with token count while decode cost is roughly constant, making the fertility gap visible in TTFT but not TPOT.

---

### Planned

**Experiment 8 — KV Cache Eviction Under High Fertility**  

Investigate whether high-fertility inputs cause more frequent KV cache eviction under memory pressure, and whether eviction patterns differ across language groups.

**Experiment 9 — Custom CUDA Kernel for Adaptive KV Fetch**  

Profile KV cache memory access patterns during attention computation for high-fertility vs low-fertility sequences. Investigate whether vectorized memory access (`float4`) and fertility-aware scheduling reduce memory bandwidth consumption for Devanagari inputs on NVIDIA hardware.

**Experiment 10 — Cache Compression Evaluation**  
Benchmark KV cache compression strategies (quantized KV, token pruning, sliding window attention) specifically for high-fertility language inputs. Determine which compression approach preserves generation quality best for Devanagari scripts.

**Experiment 11 — Paged Attention Block Size Sensitivity**  
Test whether larger page sizes in vLLM's PagedAttention reduce fragmentation for high-fertility inputs, potentially recovering concurrency without requiring adaptive allocation at the application layer.

**Experiment 12 — NVIDIA GPU Replication**  
Replicate all Apple Silicon experiments on an NVIDIA A100/H100 instance. Determine whether unified memory architecture (Apple Silicon) produces fundamentally different fertility-related bottlenecks compared to discrete GPU memory (NVIDIA).

---

## Roadmap

```
Phase 1 — Characterization (current)
  ✅ Tokenizer fertility measurement
  ✅ KV cache memory under static vs adaptive allocation
  ✅ Concurrent serving capacity
  🔄 Multi-model fertility comparison
  🔄 TTFT / TPOT decomposition

Phase 2 — Engine-Level Investigation
  📋 vLLM PagedAttention behavior
  📋 KV cache eviction analysis
  📋 Cache compression evaluation (quantized KV, token pruning)
  📋 Paged attention block size sensitivity

Phase 3 — GPU / CUDA Investigation
  📋 NVIDIA GPU replication of Phase 1 results
  📋 CUDA memory access profiling (nsight compute)
  📋 Custom adaptive KV fetch kernel
  📋 Fertility-aware attention scheduling

Phase 4 — Optimization Evaluation
  📋 End-to-end benchmark: baseline vs adaptive vs compressed
  📋 Hardware-specific optimization recommendations
  📋 Inference serving guidelines for multilingual deployments
```

---

## Repository Structure

```
.
├── README.md
├── benchmarks/
│   ├── tokenizer/          # fertility measurement scripts
│   ├── inference/          # single-request latency benchmarks
│   └── concurrency/        # multi-user serving experiments
│
├── experiments/
│   ├── apple_silicon/      # ARM64 / llama.cpp experiments
│   └── cuda/               # NVIDIA GPU experiments (planned)
│
├── kernels/
│   └── cuda/               # custom CUDA kernels (planned)
│
├── results/
│   ├── raw/                # raw benchmark outputs
│   └── figures/            # generated plots
│
├── adaptive_kv_cache.py    # adaptive context allocation system
├── benchmark.py            # tokenizer fertility benchmark
├── throughput.py           # concurrent serving measurement
└── requirements.txt
```

---

## Background and Related Work

This project builds directly on foundational work in LLM inference optimization:

- **PagedAttention / vLLM** (Kwon et al., 2023) — paged KV cache management for efficient memory use
- **FlashAttention** (Dao et al., 2022) — IO-aware attention for GPU memory efficiency
- **RocketKV** (Tumanov et al., ICML 2025) — two-stage KV cache compression for long-context inference
- **StreamingLLM** — attention sinks and sliding window for infinite-length generation
- **Sarathi-Serve** (Tumanov et al., OSDI 2024) — chunked prefill scheduling for inference efficiency

The contribution of this project is not to replace these approaches but to characterize **how non-uniform token fertility interacts with existing KV cache strategies** — a dimension that existing work has not explicitly studied for low-resource language scripts.

---

## What This Project Is Not Claiming

- This is **not** a claim that adaptive context allocation is production-ready
- This is **not** a claim that the measured effects generalize to all hardware configurations without further testing
- All results on this page are from **Apple Silicon experiments only** — GPU replication is planned but not yet complete
- Performance claims will only be made after comparison against reproducible baselines on identical hardware

---

## Open Questions

1. Does token fertility disparity become more or less pronounced under quantization (INT4 vs INT8)?
2. Does vLLM's PagedAttention implicitly mitigate fertility-related fragmentation, or does it amplify it?
3. Is the fertility gap primarily a vocabulary coverage problem (solvable by tokenizer retraining) or a memory architecture problem (requiring hardware-level mitigation)?
4. Does the effect differ between models trained with byte-fallback tokenization vs character-level Devanagari tokens?
5. At what concurrency level does fertility disparity become the dominant bottleneck vs other inference system components?

---

## Contributing

This is an active experimental project. Contributions are welcome in:

- Tokenizer benchmarking and dataset preparation
- Inference measurement and profiling
- Apple Silicon / NVIDIA experimental replication
- vLLM and TGI integration experiments
- CUDA kernel development
- Visualization and analysis tooling

To contribute, open an issue describing the experiment or component you want to work on.

---

## License

MIT — see LICENSE file.

---

*Project by Ashutosh Bhattarai · Pulchowk Campus, Institute of Engineering, Nepal*  
*Exploring the intersection of tokenizer design and inference systems for global AI deployment*

