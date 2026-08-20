<!-- # Exploring Hardware-Aware LLM Inference Under Tokenizer Disparity

> An experimental systems project investigating how differences in tokenization—particularly for low-resource languages such as Nepali—may affect LLM inference efficiency, memory usage, and concurrent serving.

**Status:** Early-stage / Experimental
**Domain:** AI Systems · LLM Inference · Memory Systems · Low-Resource NLP

---

## Overview

Large Language Models do not necessarily represent the same semantic content using the same number of tokens across languages.

For some languages, including Nepali and other languages written in scripts such as Devanagari, a sentence can be represented using substantially more tokens than an equivalent English sentence.

This project explores a systems-level question:

> **When a language requires substantially more tokens to represent the same semantic content, how much does that difference matter to the underlying inference system?**

Rather than treating tokenizer efficiency purely as an NLP problem, this project looks at the possible interaction between:

```text
Language
   ↓
Tokenization
   ↓
Sequence Length
   ↓
Attention / KV Cache
   ↓
Memory Traffic
   ↓
Hardware Utilization
   ↓
Inference Throughput & Concurrency
```

The goal at this stage is **not to claim a final solution**, but to build experiments that can determine where tokenizer disparity actually becomes a systems bottleneck.

---

## Why This Project?

Most LLM benchmarking is reported using metrics such as:

* tokens/second
* latency
* model throughput
* GPU utilization

However, tokens are not necessarily equivalent to semantic units across languages.

For example, if two semantically equivalent inputs require:

```text
English       → 20 tokens
Nepali        → 70 tokens
```

then comparing only tokens/second can hide an important part of the actual system cost.

The additional tokens can potentially affect:

* context length
* KV-cache memory consumption
* prefill computation
* decoding memory traffic
* maximum concurrent requests
* cache behavior
* memory bandwidth utilization

This project therefore explores whether **tokenizer disparity can be studied as an end-to-end systems problem rather than only an NLP problem**.

---

# Current Hypothesis

The working hypothesis is:

> **Higher token fertility can increase the computational and memory cost of LLM inference, particularly in memory-constrained and highly concurrent serving environments.**

A simplified relationship is:

```text
Higher Token Fertility
        ↓
Longer Token Sequence
        ↓
Larger KV Cache
        ↓
Higher Memory Traffic
        ↓
Lower Maximum Concurrency
        ↓
Potentially Lower Effective Throughput
```

This is a hypothesis being investigated through experiments, not a final conclusion.

---

# What This Project Is Exploring

The project currently has several related investigation areas.

### 1. Tokenizer Disparity

Measure how different tokenizers represent equivalent content across languages.

Particular attention is given to:

* Nepali
* English
* Devanagari text
* low-resource language inputs

A basic metric of interest is token fertility:

```text
Fertility = Number of Tokens / Semantic Unit
```

The exact definition of a "semantic unit" will need to be established carefully during experimentation.

---

### 2. KV-Cache Memory

Longer sequences require more KV-cache storage during inference.

A simplified representation is:

```text
KV Memory ∝
Batch Size × Sequence Length ×
Number of Layers × Number of Heads ×
Head Dimension × Bytes per Element
```

Therefore, if tokenization increases sequence length, the corresponding memory requirement can also increase.

One of the project's questions is:

> How significant is this additional memory cost in realistic serving workloads?

---

### 3. Concurrent Inference

The effect may become more visible when multiple users are generating responses simultaneously.

Instead of measuring only:

```text
Single Request
      ↓
Latency
```

the project intends to investigate:

```text
1 request
4 requests
8 requests
16 requests
32 requests
64 requests
      ↓
Memory usage
TTFT
TPOT
Throughput
OOM behaviour
```

This can help determine whether tokenizer disparity has a meaningful effect on **multi-tenant inference**.

---

# Experimental Hardware

The project is interested in comparing different execution environments.

## Apple Silicon

Apple Silicon is particularly interesting because CPU/GPU resources share unified memory.

The initial direction is to investigate whether longer token sequences increase memory pressure and reduce effective inference efficiency on memory-constrained local machines.

Potential measurements include:

* memory bandwidth
* memory usage
* inference latency
* tokens/sec
* context length
* concurrent request capacity

---

## NVIDIA CUDA

A second direction is investigating GPU execution.

Potential areas include:

* KV-cache memory access
* attention kernels
* memory bandwidth
* kernel execution time
* CUDA profiling
* different KV-cache layouts
* possible adaptive memory allocation strategies

The project may eventually include custom CUDA experiments, but these are currently **future implementation areas rather than completed components**.

---

# Possible System-Level Optimization

One direction being explored is **adaptive context / KV-cache management**.

The basic idea is:

```text
                Tokenizer
                   │
                   ▼
           Token Distribution
                   │
                   ▼
        Estimate Sequence Density
                   │
                   ▼
        Adaptive KV-Cache Layout
                   │
                   ▼
        Hardware-Aware Execution
```

Instead of assuming that every language produces similar token distributions, an inference system could potentially account for differences in token density when managing memory.

This is an area for experimentation rather than a finalized architecture.

---

# Metrics

The project will initially focus on measurable systems-level metrics.

### Time To First Token — TTFT

Measures the latency before the first generated token.

Useful for studying the cost of processing longer input sequences.

### Time Per Output Token — TPOT

Measures decoding performance after generation begins.

This is particularly useful when investigating memory-bandwidth-related bottlenecks.

### Throughput

```text
Tokens / second
```

Standard inference throughput measurement.

### Memory Consumption

Measure:

```text
Model Memory
+
KV Cache
+
Runtime Overhead
```

### Concurrent Throughput

Measure throughput as the number of simultaneous requests increases.

### Effective Semantic Throughput

A possible experimental metric is to normalize throughput against tokenization differences.

For example:

```text
Effective Semantic Throughput
≈
Token Throughput / Token Fertility
```

The exact formulation is still experimental and may change as the benchmark methodology develops.

---

# Experimental Design

The project will gradually build toward a benchmark pipeline such as:

```text
                ┌─────────────────────┐
                │   Language Dataset  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      Tokenizer      │
                └──────────┬──────────┘
                           │
                    Token Statistics
                           │
                           ▼
                ┌─────────────────────┐
                │     LLM Inference   │
                │       Engine        │
                └──────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Latency       Memory       Throughput
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                ┌─────────────────────┐
                │ System-Level        │
                │ Analysis             │
                └─────────────────────┘
```

Later experiments may include concurrent request generation using an inference-serving framework and a load generator.

---

# Possible Experiments

The repository will evolve incrementally.

## Experiment 1 — Tokenizer Comparison

Compare token counts for equivalent text across languages.

```text
Input
 ↓
Tokenizer A
 ↓
Token count

Input
 ↓
Tokenizer B
 ↓
Token count
```

### Goal

Establish whether meaningful tokenizer disparity exists for the selected datasets.

---

## Experiment 2 — Sequence Length vs Memory

Measure how increasing input sequence length affects:

* KV-cache memory
* inference latency
* throughput

### Goal

Determine whether the expected relationship between token count and memory usage appears in practice.

---

## Experiment 3 — Concurrent Requests

Run increasing numbers of simultaneous requests.

```text
Concurrency
1 → 2 → 4 → 8 → 16 → 32 → 64
```

Measure:

* TTFT
* TPOT
* throughput
* memory consumption
* OOM threshold

### Goal

Determine whether tokenizer disparity becomes more significant under concurrency.

---

## Experiment 4 — Hardware Comparison

Where hardware is available, compare execution characteristics across:

```text
Apple Silicon
      vs
NVIDIA CUDA
```

The comparison will focus on system behaviour rather than attempting to declare one architecture universally better.

---

## Experiment 5 — Adaptive KV-Cache Strategies

If earlier experiments demonstrate a meaningful bottleneck, investigate whether adaptive memory management can reduce the observed overhead.

Possible directions include:

* different KV-cache block sizes
* dynamic allocation strategies
* sequence-aware memory management
* cache paging strategies
* kernel-level memory-access optimizations

---

# CUDA Exploration

A future component may investigate custom CUDA kernels for memory-intensive inference operations.

For example, vectorized memory access could be explored using CUDA types such as `float4`.

A conceptual prototype might look like:

```cpp
__global__ void adaptive_kv_fetch_kernel(
    const float4* __restrict__ kv_cache,
    float4* __restrict__ output_buffer,
    const int* __restrict__ token_fertility_offsets,
    int head_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float4 cache_element = kv_cache[tid];

    // Future:
    // investigate memory access patterns,
    // cache behaviour and attention computation.
}
```

This code is currently a **prototype direction**, not a production attention implementation.

Performance claims will only be made after profiling and comparison against appropriate baselines.

---

# Baselines

An important part of the project is avoiding comparisons against an undefined or unfair baseline.

Where possible, experiments should compare:

```text
Baseline inference implementation
             vs
Experimental implementation
```

while keeping as many variables constant as possible:

* model
* quantization
* prompt
* generation parameters
* hardware
* batch size
* context length
* software versions

This is important because inference performance can be affected by many factors unrelated to tokenization.

---

# Current Status

This project is currently in the **exploration and experimental design stage**.

### Currently defined

* [x] Problem area
* [x] Systems-level hypothesis
* [x] Initial architecture
* [x] Candidate metrics
* [x] Possible benchmark methodology
* [x] Potential optimization directions

### To be implemented

* [ ] Establish tokenizer benchmark dataset
* [ ] Measure token fertility across languages
* [ ] Build reproducible inference benchmark
* [ ] Measure KV-cache memory
* [ ] Measure TTFT and TPOT
* [ ] Test increasing concurrency
* [ ] Profile memory behaviour
* [ ] Establish baseline results
* [ ] Investigate adaptive KV-cache strategies
* [ ] Explore CUDA kernels if justified by profiling
* [ ] Compare optimization against baseline

---

# Repository Structure

The repository is expected to evolve roughly toward:

```text
.
├── README.md
├── benchmarks/
│   ├── tokenizer/
│   ├── inference/
│   └── concurrency/
│
├── experiments/
│   ├── apple_silicon/
│   └── cuda/
│
├── kernels/
│   └── cuda/
│
├── scripts/
│   ├── benchmark.py
│   └── load_test.py
│
├── results/
│   ├── raw/
│   └── figures/
│
├── docs/
│   └── experiments/
│
└── requirements.txt
```

The structure is intentionally flexible and will change as the project develops.

---

# Existing Work vs This Project

This project builds on existing work in several areas, including:

* LLM inference optimization
* KV-cache management
* memory-efficient attention
* tokenizer design
* GPU kernel optimization
* inference serving systems

The purpose of this repository is **not to claim ownership of these underlying techniques**.

Instead, the project explores how these existing ideas might be combined or evaluated specifically in the context of **non-uniform token distributions across languages**.

Potentially relevant systems and techniques include:

* PagedAttention
* FlashAttention
* StreamingLLM
* inference engines such as vLLM and TGI
* CUDA-based kernel optimization

Specific implementations and experiments will be credited to their respective projects.

---

# What I Am Trying to Find Out

The central questions are intentionally simple:

1. **How much does tokenizer disparity actually affect inference performance?**
2. **Does the effect become larger under concurrent workloads?**
3. **Is the primary limitation computation, memory capacity, or memory bandwidth?**
4. **Does the effect differ significantly between hardware architectures?**
5. **Can better memory management reduce the overhead?**
6. **At what point does a language-specific tokenization issue become a systems-level problem?**

The project will let experimental results—not assumptions—answer these questions.

---

# Future Directions

Depending on the results of the initial experiments, future work may explore:

* language-aware KV-cache allocation
* adaptive PagedAttention block sizing
* tokenizer-aware scheduling
* memory-aware request batching
* custom CUDA kernels
* CPU/Apple Silicon optimization
* quantization-aware memory management
* low-resource language inference benchmarking
* more comprehensive multilingual benchmarks

Some of these directions may ultimately prove unnecessary. The project will use profiling and benchmarking to decide which ones are worth pursuing.

---

# Contributions

At this stage, contributions are welcome in areas such as:

* tokenizer benchmarking
* dataset preparation
* inference benchmarking
* profiling
* Apple Silicon experiments
* CUDA experiments
* KV-cache analysis
* visualization
* benchmark methodology

If you are interested in contributing, feel free to open an issue describing the experiment or improvement you would like to work on.

---

# Disclaimer

This repository represents an **ongoing experimental project**.

The hypotheses, proposed architectures, metrics, and optimization ideas are subject to change as experiments are performed.

Numbers or relationships mentioned in early documentation should not be interpreted as established results unless they are accompanied by reproducible benchmark data.

The primary goal at this stage is to build a reliable experimental foundation and determine whether the underlying hypothesis is supported by evidence.

---

# License

To be decided. -->


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

