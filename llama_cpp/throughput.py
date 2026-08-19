# # import concurrent.futures
# # import time
# # from llama_cpp import Llama

# # MODEL_PATH = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# # def simulate_throughput(prompts, label, n_concurrent=5):
# #     """Simulate multiple concurrent inference requests"""
    
# #     def single_inference(prompt):
# #         llm = Llama(model_path=MODEL_PATH, 
# #                    n_ctx=256, n_threads=2, verbose=False)
# #         start = time.perf_counter()
# #         output = llm(prompt, max_tokens=20)
# #         elapsed = time.perf_counter() - start
# #         tokens = len(llm.tokenize(prompt.encode()))
# #         return {"tokens": tokens, "ms": elapsed * 1000}
    
# #     start_wall = time.perf_counter()
    
# #     with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
# #         futures = [executor.submit(single_inference, p) for p in prompts]
# #         results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
# #     wall_time = (time.perf_counter() - start_wall) * 1000
# #     total_tokens = sum(r["tokens"] for r in results)
    
# #     return {
# #         "label": label,
# #         "concurrent_requests": n_concurrent,
# #         "total_tokens_processed": total_tokens,
# #         "wall_time_ms": round(wall_time, 1),
# #         "requests_per_second": round(n_concurrent / (wall_time/1000), 2),
# #         "tokens_per_second": round(total_tokens / (wall_time/1000), 1)
# #     }

# # english_prompts = [
# #     "Explain machine learning briefly."
# # ] * 5

# # nepali_prompts = [
# #     "मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।"
# # ] * 5

# # print("Throughput comparison (5 concurrent requests):")
# # print()
# # eng = simulate_throughput(english_prompts, "English")
# # nep = simulate_throughput(nepali_prompts, "Nepali")

# # for r in [eng, nep]:
# #     print(f"{r['label']}:")
# #     print(f"  Total tokens processed: {r['total_tokens_processed']}")
# #     print(f"  Wall time: {r['wall_time_ms']}ms")
# #     print(f"  Requests/sec: {r['requests_per_second']}")
# #     print(f"  Tokens/sec: {r['tokens_per_second']}")

# # print(f"\nTo serve same number of Nepali vs English requests:")
# # print(f"Nepali requires {round(nep['total_tokens_processed']/eng['total_tokens_processed'], 2)}x more compute")
# # print(f"Effective throughput ratio: {round(eng['requests_per_second']/nep['requests_per_second'], 2)}x fewer Nepali users served per unit time")


# import time
# from llama_cpp import Llama

# MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

# def simulate_throughput_sequential(prompts, label):
#     """
#     Simulate batch throughput by running N requests sequentially
#     and measuring total wall time — honest single-instance measurement
#     """
#     llm = Llama(model_path=MODEL_PATH,
#                 n_ctx=256,
#                 n_threads=4,
#                 verbose=False)
    
#     results = []
#     total_start = time.perf_counter()
    
#     for prompt in prompts:
#         tokens = llm.tokenize(prompt.encode())
#         req_start = time.perf_counter()
#         output = llm(prompt, max_tokens=20)
#         req_elapsed = (time.perf_counter() - req_start) * 1000
#         results.append({
#             "tokens": len(tokens),
#             "ms": req_elapsed
#         })
    
#     total_wall = (time.perf_counter() - total_start) * 1000
#     total_tokens = sum(r["tokens"] for r in results)
#     n = len(prompts)
    
#     print(f"\n{label} — {n} sequential requests:")
#     print(f"  Total tokens processed: {total_tokens}")
#     print(f"  Total wall time: {round(total_wall, 1)}ms")
#     print(f"  Avg latency per request: {round(total_wall/n, 1)}ms")
#     print(f"  Throughput: {round(n/(total_wall/1000), 2)} requests/sec")
#     print(f"  Token throughput: {round(total_tokens/(total_wall/1000), 1)} tokens/sec")
    
#     return {
#         "label": label,
#         "total_tokens": total_tokens,
#         "total_wall_ms": total_wall,
#         "requests_per_sec": n / (total_wall/1000),
#         "tokens_per_sec": total_tokens / (total_wall/1000)
#     }

# # 10 requests each language
# english_prompts = ["Explain machine learning briefly."] * 10
# hindi_prompts = ["मशीन लर्निंग को सरल शब्दों में समझाइए।"] * 10
# nepali_prompts = ["मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।"] * 10

# eng = simulate_throughput_sequential(english_prompts, "English")
# hin = simulate_throughput_sequential(hindi_prompts, "Hindi")
# nep = simulate_throughput_sequential(nepali_prompts, "Nepali")

# print("\n=== INFRASTRUCTURE COST COMPARISON ===")
# print(f"To serve 1000 users with equivalent content:")
# print(f"English tokens needed:  {round(eng['total_tokens']/10 * 1000):,}")
# print(f"Hindi tokens needed:    {round(hin['total_tokens']/10 * 1000):,}")
# print(f"Nepali tokens needed:   {round(nep['total_tokens']/10 * 1000):,}")
# print(f"\nNepali compute overhead vs English: "
#       f"{round(nep['total_tokens']/eng['total_tokens'], 2)}x")
# print(f"Hindi compute overhead vs English:  "
#       f"{round(hin['total_tokens']/eng['total_tokens'], 2)}x")
# print(f"\nEffective user capacity on fixed compute:")
# print(f"English users served per unit: 1.00x (baseline)")
# print(f"Hindi users served per unit:   "
#       f"{round(eng['tokens_per_sec']/hin['tokens_per_sec'], 2)}x")
# print(f"Nepali users served per unit:  "
#       f"{round(eng['tokens_per_sec']/nep['tokens_per_sec'], 2)}x")


"""
Multi-prompt tokenization / throughput benchmark (English vs Hindi vs Nepali).

Runs each of the 15 parallel prompts per language individually (not just
aggregate sums), so you get a per-sentence sample size large enough for:
  - mean / median / std of tokens-per-prompt and latency-per-prompt
  - within-language variance (needed for error bars / significance tests)
  - within-length-bucket breakdown (short/medium/long), to check the
    tokenization premium isn't just a length artifact

Outputs:
  results_raw.csv   -- one row per (language, prompt) with tokens & latency
  results_summary.json -- aggregated stats consumed by plot_results.py

Run this on the machine with llama_cpp + your model installed.
plot_results.py can then be run separately (even without llama_cpp)
on results_summary.json / results_raw.csv.
"""

import csv
import json
import statistics
import time

from llama_cpp import Llama
from parallel_prompts import PROMPTS, LENGTH_BUCKETS

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"
N_CTX = 256
N_THREADS = 4
MAX_TOKENS = 20
N_WARMUP = 2  # untimed warmup calls to avoid cold-start bias in first measurement


def run_language(llm, lang, prompts, buckets):
    rows = []
    # warmup (discarded) -- avoids first-call overhead skewing prompt 0
    for p in prompts[:N_WARMUP]:
        llm(p, max_tokens=MAX_TOKENS)

    for i, (prompt, bucket) in enumerate(zip(prompts, buckets)):
        input_tokens = llm.tokenize(prompt.encode())
        start = time.perf_counter()
        output = llm(prompt, max_tokens=MAX_TOKENS)
        elapsed_ms = (time.perf_counter() - start) * 1000

        rows.append({
            "language": lang,
            "prompt_idx": i,
            "length_bucket": bucket,
            "input_tokens": len(input_tokens),
            "latency_ms": round(elapsed_ms, 3),
            "chars": len(prompt),
            "tokens_per_char": round(len(input_tokens) / max(len(prompt), 1), 4),
        })
    return rows


def summarize(rows):
    """Aggregate per-language and per-(language, bucket) stats."""
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)

    summary = {}
    for lang, lang_rows in by_lang.items():
        tokens = [r["input_tokens"] for r in lang_rows]
        latency = [r["latency_ms"] for r in lang_rows]
        tpc = [r["tokens_per_char"] for r in lang_rows]

        bucket_stats = {}
        for bucket in ("short", "medium", "long"):
            b_tokens = [r["input_tokens"] for r in lang_rows if r["length_bucket"] == bucket]
            if b_tokens:
                bucket_stats[bucket] = {
                    "mean_tokens": round(statistics.mean(b_tokens), 2),
                    "std_tokens": round(statistics.pstdev(b_tokens), 2) if len(b_tokens) > 1 else 0.0,
                    "n": len(b_tokens),
                }

        summary[lang] = {
            "n_prompts": len(lang_rows),
            "total_tokens": sum(tokens),
            "mean_tokens_per_prompt": round(statistics.mean(tokens), 3),
            "std_tokens_per_prompt": round(statistics.pstdev(tokens), 3) if len(tokens) > 1 else 0.0,
            "median_tokens_per_prompt": statistics.median(tokens),
            "mean_latency_ms": round(statistics.mean(latency), 3),
            "std_latency_ms": round(statistics.pstdev(latency), 3) if len(latency) > 1 else 0.0,
            "mean_tokens_per_char": round(statistics.mean(tpc), 4),
            "tokens_per_sec": round(sum(tokens) / (sum(latency) / 1000), 2),
            "by_length_bucket": bucket_stats,
            "raw_tokens": tokens,       # kept for box plots / significance tests
            "raw_latency_ms": latency,
        }

    # relative overhead vs English (the reference / baseline language)
    if "English" in summary:
        eng_mean = summary["English"]["mean_tokens_per_prompt"]
        eng_tps = summary["English"]["tokens_per_sec"]
        for lang, s in summary.items():
            s["token_overhead_vs_english"] = round(s["mean_tokens_per_prompt"] / eng_mean, 3)
            s["relative_capacity_vs_english"] = round(s["tokens_per_sec"] and eng_tps / s["tokens_per_sec"], 3) if s["tokens_per_sec"] else None

    return summary


def main():
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)

    all_rows = []
    for lang, prompts in PROMPTS.items():
        print(f"Running {lang} ({len(prompts)} prompts)...")
        rows = run_language(llm, lang, prompts, LENGTH_BUCKETS)
        all_rows.extend(rows)

    # write raw per-prompt CSV
    with open("results_raw.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print("Wrote results_raw.csv")

    summary = summarize(all_rows)
    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Wrote results_summary.json")

    print("\n=== SUMMARY ===")
    for lang, s in summary.items():
        print(f"{lang}: mean_tokens={s['mean_tokens_per_prompt']} "
              f"(+/-{s['std_tokens_per_prompt']}), "
              f"overhead_vs_english={s.get('token_overhead_vs_english')}x, "
              f"tokens/sec={s['tokens_per_sec']}")


if __name__ == "__main__":
    main()