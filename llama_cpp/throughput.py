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