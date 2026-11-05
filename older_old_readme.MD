# Dynamic KV-Cache Management for High-Fertility Workloads

This repository contains the profiling scripts, benchmarks, and architectural designs for addressing memory bottlenecks in LLM inference caused by **tokenization inequality**. 

Currently, multi-tenant serving systems allocate KV cache based on fixed sequence lengths. However, high-fertility scripts (like Devanagari) consume 3–4x more tokens than English for the exact same semantic payload. This disproportionately exhausts paging blocks, causes severe memory fragmentation, and directly throttles serving concurrency. 

This project explores dynamic, runtime context compression to maintain concurrency without rigid prompt-time budgets.

## 🚀 Phase 1: Baseline Profiling & Adaptive Context Sizing (Completed)

In Phase 1, we quantified the systems-level cost of tokenization inequality using `llama.cpp` on Apple Silicon CPUs (moving to MPS/CUDA in Phase 2). We evaluated standard static KV allocation against a prompt-time adaptive context-sizing scheme.

### Setup
*   **Models Evaluated:** `Llama-3.2-1B`, `Qwen2.5-0.5B`
*   **Framework:** `llama.cpp`
*   **Workloads:** English (Baseline) vs. Nepali/Devanagari (High-Fertility Stress Test)

### Key Findings
1.  **The Fertility Gap:** Devanagari inputs exhibit a **3–4x higher BPE token fertility** compared to English.
2.  **Concurrency Collapse:** Under static KV cache allocation, this elevated fertility rapidly fragments memory blocks, leading to early OOM (Out of Memory) errors and dropped sessions in multi-tenant environments.
3.  **Adaptive Recovery:** By implementing a simple adaptive context-sizing scheme at prompt time, we recovered **4–6x more concurrent users** and achieved **5–15x memory savings** under a strictly fixed memory budget.

*(Note: The raw profiling scripts for Phase 1 are located in the `src/profiling` directory.)*

## 🏗️ Phase 2: Entropy-Based KV Merging Architecture (In Design)

While Phase 1 proved that reducing the context footprint drastically improves concurrency, prompt-time adaptive sizing is a one-shot allocation. It cannot accommodate conversational sessions that grow dynamically across multiple turns, causing it to break down for real multi-turn serving.

Phase 2 focuses on an active runtime intervention: **Entropy-Based KV Merging**. 

Instead of relying on brute-force block evictions or fixed upfront budgets, this architecture is being designed to:
*   Evaluate layer-wise attention scores during generation.
*   Identify low-entropy, high-fertility sub-words (frequent in Devanagari and other non-Latin scripts).
*   Selectively compress and merge these tokens dynamically *per turn*.

This distills the context into its most salient semantic units, providing a more flexible and bounded memory primitive for multi-turn multi-tenant serving.

## 💻 Current Status & Hardware Roadmap
As an independent research project, development is proceeding incrementally.
*   [x] **Phase 1:** Initial memory profiling and adaptive context baselines executed on Apple Silicon CPU.
*   [ ] **Phase 2 (Active):** Finalizing the algorithmic architecture for layer-wise entropy merging.
*   [ ] **Phase 3 (Upcoming):** Porting evaluations to MPS and CUDA to validate fragmentation behavior, memory bounds, and throughput scaling on GPU hardware.

## About
This is an independent research project by Ashutosh Bhattarai (Pulchowk Campus, Institute of Engineering) aimed at bridging the gap between multilingual LLM usability and backend systems efficiency.
