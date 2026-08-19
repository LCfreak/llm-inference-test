# from llama_cpp import Llama
# import time
# import psutil
# import os

# MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

# def measure_kv_cache_pressure(prompt, label, context_lengths=[64, 128, 256, 512]):
#     results = []
    
#     for ctx_len in context_lengths:
#         llm = Llama(model_path=MODEL_PATH, 
#                    n_ctx=ctx_len, 
#                    verbose=False)
        
#         tokens = llm.tokenize(prompt.encode())
        
#         if len(tokens) >= ctx_len:
#             continue
            
#         process = psutil.Process(os.getpid())
#         mem_before = process.memory_info().rss / 1024 / 1024
        
#         start = time.perf_counter()
#         output = llm(prompt, max_tokens=20)
#         elapsed = time.perf_counter() - start
        
#         mem_after = process.memory_info().rss / 1024 / 1024
        
#         results.append({
#             "label": label,
#             "context_length": ctx_len,
#             "input_tokens": len(tokens),
#             "memory_mb": round(mem_after - mem_before, 2),
#             "inference_ms": round(elapsed * 1000, 1),
#             "memory_per_token_kb": round(
#                 (mem_after - mem_before) * 1024 / len(tokens), 2)
#         })
        
#         del llm
    
#     return results

# english = "Explain how machine learning works."
# nepali = "मेसिन लर्निङ कसरी काम गर्छ भनेर बुझाउनुहोस्।"

# print("English KV cache pressure:")
# for r in measure_kv_cache_pressure(english, "English"):
#     print(f"  ctx={r['context_length']}: "
#           f"{r['input_tokens']} tokens, "
#           f"{r['memory_mb']}MB, "
#           f"{r['memory_per_token_kb']}KB/token, "
#           f"{r['inference_ms']}ms")

# print("\nNepali KV cache pressure:")
# for r in measure_kv_cache_pressure(nepali, "Nepali"):
#     print(f"  ctx={r['context_length']}: "
#           f"{r['input_tokens']} tokens, "
#           f"{r['memory_mb']}MB, "
#           f"{r['memory_per_token_kb']}KB/token, "
#           f"{r['inference_ms']}ms") 

"""
KV-cache / memory pressure benchmark across English, Hindi, Nepali.

Restructured from the original per-prompt version:
  - Model is loaded ONCE per context_length (not once per prompt per
    context_length). For 45 prompts x 4 context lengths that's the
    difference between 4 model loads and 180. Model-load time is
    substantial on CPU and was previously leaking into your memory/latency
    measurements as noise unrelated to the actual inference.
  - One untimed warmup inference per (context_length) after loading, so the
    first *measured* prompt at each context length isn't penalized by
    lazy allocation / cold KV-cache buffers.
  - psutil RSS delta is still noisy for single-shot measurements (subject to
    GC, page cache, OS scheduling) -- treat memory_mb as directional, not
    precise. If you need tighter numbers later, run each (lang, ctx) cell
    multiple times and take the median, or use tracemalloc / resource.getrusage
    for peak RSS instead of process RSS deltas.

Exports: kv_cache_results.csv (one row per prompt x context_length that fit)
"""

import csv
import os
import time

import psutil
from llama_cpp import Llama

from parallel_prompts import PROMPTS, LENGTH_BUCKETS

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"
CONTEXT_LENGTHS = [64, 128, 256, 512]
MAX_TOKENS = 20
OUT_CSV = "kv_cache_results.csv"


def run_for_context_length(ctx_len, all_prompts):
    """Load the model once at this ctx_len, run every prompt that fits."""
    llm = Llama(model_path=MODEL_PATH, n_ctx=ctx_len, verbose=False)
    process = psutil.Process(os.getpid())

    # untimed warmup to prime allocators / KV cache buffers at this ctx size
    llm("Warm up.", max_tokens=5)

    rows = []
    for lang, prompt, bucket, idx in all_prompts:
        tokens = llm.tokenize(prompt.encode())
        if len(tokens) >= ctx_len:
            continue  # prompt doesn't fit this context window, skip

        mem_before = process.memory_info().rss / 1024 / 1024
        start = time.perf_counter()
        llm(prompt, max_tokens=MAX_TOKENS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        mem_after = process.memory_info().rss / 1024 / 1024

        mem_delta = mem_after - mem_before
        rows.append({
            "language": lang,
            "prompt_idx": idx,
            "length_bucket": bucket,
            "context_length": ctx_len,
            "input_tokens": len(tokens),
            "memory_delta_mb": round(mem_delta, 3),
            "memory_per_token_kb": round(mem_delta * 1024 / len(tokens), 3) if len(tokens) else None,
            "inference_ms": round(elapsed_ms, 2),
        })

    del llm
    return rows


def main():
    # flatten (language, prompt, bucket, idx) tuples once, reused per ctx_len
    all_prompts = []
    for lang, prompts in PROMPTS.items():
        for idx, (prompt, bucket) in enumerate(zip(prompts, LENGTH_BUCKETS)):
            all_prompts.append((lang, prompt, bucket, idx))

    all_rows = []
    for ctx_len in CONTEXT_LENGTHS:
        print(f"Running context_length={ctx_len} ({len(all_prompts)} prompts)...")
        rows = run_for_context_length(ctx_len, all_prompts)
        print(f"  {len(rows)}/{len(all_prompts)} prompts fit within ctx={ctx_len}")
        all_rows.extend(rows)

    if not all_rows:
        print("No results collected -- check MODEL_PATH / context lengths.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()