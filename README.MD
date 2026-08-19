# Exploring Hardware-Aware LLM Inference Under Tokenizer Disparity

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

To be decided.
