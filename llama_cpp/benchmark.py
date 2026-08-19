# # benchmark.py
# from llama_cpp import Llama
# import time
# import json

# MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

# def benchmark(prompt, label, runs=5):
#     llm = Llama(model_path=MODEL_PATH, n_ctx=512, verbose=False)
    
#     # Tokenize and count
#     tokens = llm.tokenize(prompt.encode())
    
#     # Time inference
#     times = []
#     for _ in range(runs):
#         start = time.perf_counter()
#         out = llm(prompt, max_tokens=30, echo=False)
#         elapsed = time.perf_counter() - start
#         times.append(elapsed)
    
#     times.sort()
#     median_time = times[len(times)//2]
    
#     return {
#         "label": label,
#         "prompt_chars": len(prompt),
#         "token_count": len(tokens),
#         "chars_per_token": round(len(prompt)/len(tokens), 2),
#         "median_inference_ms": round(median_time * 1000, 1),
#         "tokens_generated_per_sec": round(30/median_time, 1)
#     }

# # Test prompts — same meaning, different languages
# prompts = [
#     ("Explain machine learning in simple terms.", "English"),
#     ("मशीन लर्निंग को सरल शब्दों में समझाइए।", "Hindi"),
#     ("मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।", "Nepali"),
# ]

# results = []
# for prompt, label in prompts:
#     print(f"Benchmarking {label}...")
#     r = benchmark(prompt, label)
#     results.append(r)
#     print(json.dumps(r, indent=2, ensure_ascii=False))

# print("\n=== SUMMARY ===")
# english_chars_per_token = results[0]["chars_per_token"]
# for r in results:
#     ratio = round(english_chars_per_token / r["chars_per_token"], 2)
#     print(f"{r['label']}: {r['chars_per_token']} chars/token "
#           f"({ratio}x less efficient than English)") 


"""
Latency + chars-per-token efficiency benchmark, English vs Hindi vs Nepali.

Restructured from the original:
  - Model loaded ONCE (n_ctx=512) and reused across all 45 prompts, instead
    of once per prompt. Original did this already for a 3-prompt run; at
    45 prompts a per-prompt reload would dominate wall time.
  - Kept your median-of-N approach (median is more robust to occasional
    OS-scheduling latency spikes than mean), but now stores all N raw run
    times per prompt too, so you can compute variance / do a proper
    significance test (e.g. Mann-Whitney U) across languages later instead
    of just comparing medians.
  - One untimed warmup call before the timed runs for each prompt, since
    the very first call after model load is reliably slower.

Exports: benchmark_results.csv (one row per prompt, with raw run times
in a semicolon-separated column for later stats)
"""

import csv
import time

from llama_cpp import Llama

from parallel_prompts import PROMPTS, LENGTH_BUCKETS

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"
N_CTX = 512
RUNS_PER_PROMPT = 5
MAX_TOKENS = 30
OUT_CSV = "benchmark_results.csv"


def benchmark_prompt(llm, prompt, runs=RUNS_PER_PROMPT):
    tokens = llm.tokenize(prompt.encode())

    llm(prompt, max_tokens=MAX_TOKENS, echo=False)  # untimed warmup

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        llm(prompt, max_tokens=MAX_TOKENS, echo=False)
        times.append(time.perf_counter() - start)

    times_sorted = sorted(times)
    median_time = times_sorted[len(times_sorted) // 2]

    return {
        "prompt_chars": len(prompt),
        "token_count": len(tokens),
        "chars_per_token": round(len(prompt) / len(tokens), 3),
        "median_inference_ms": round(median_time * 1000, 2),
        "min_inference_ms": round(times_sorted[0] * 1000, 2),
        "max_inference_ms": round(times_sorted[-1] * 1000, 2),
        "tokens_generated_per_sec": round(MAX_TOKENS / median_time, 2),
        "raw_run_times_ms": ";".join(str(round(t * 1000, 2)) for t in times),
    }


def main():
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, verbose=False)

    rows = []
    for lang, prompts in PROMPTS.items():
        print(f"Benchmarking {lang} ({len(prompts)} prompts, "
              f"{RUNS_PER_PROMPT} runs each)...")
        for idx, (prompt, bucket) in enumerate(zip(prompts, LENGTH_BUCKETS)):
            result = benchmark_prompt(llm, prompt)
            result.update({"language": lang, "prompt_idx": idx, "length_bucket": bucket})
            rows.append(result)

    fieldnames = ["language", "prompt_idx", "length_bucket", "prompt_chars",
                  "token_count", "chars_per_token", "median_inference_ms",
                  "min_inference_ms", "max_inference_ms",
                  "tokens_generated_per_sec", "raw_run_times_ms"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    print("\n=== QUICK SUMMARY (mean chars/token by language) ===")
    for lang in PROMPTS:
        lang_rows = [r for r in rows if r["language"] == lang]
        mean_cpt = sum(r["chars_per_token"] for r in lang_rows) / len(lang_rows)
        print(f"{lang}: {mean_cpt:.3f} chars/token")


if __name__ == "__main__":
    main()