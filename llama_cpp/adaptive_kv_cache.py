import csv
import math
import time

from llama_cpp import Llama

from parallel_prompts import PROMPTS, LENGTH_BUCKETS

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"
MAX_TOKENS = 50
TOKENIZER_CTX = 64  # only used for tokenization, not real inference
OUT_CSV = "adaptive_ctx_results.csv"


def detect_script(prompt):
    if not prompt:
        return "unknown"
    count = sum(1 for ch in prompt if "\u0900" <= ch <= "\u097F")
    return "devanagari" if count / len(prompt) >= 0.3 else "latin"


def adaptive_ctx(token_count, chars_per_token):
    # chars_per_token < 3.0 => inefficient tokenization (Devanagari territory)
    # English sits ~5.0, Hindi/Nepali much lower; threshold separates them
    if chars_per_token < 3.0:
        return math.ceil((token_count + 50) * 1.5)  # larger buffer for Devanagari
    return math.ceil((token_count + 50) * 1.2)       # standard buffer for Latin


def run_inference(prompt, n_ctx):
    llm = Llama(model_path=MODEL_PATH, n_ctx=n_ctx, verbose=False)
    start = time.perf_counter()
    llm(prompt, max_tokens=MAX_TOKENS)
    elapsed_ms = (time.perf_counter() - start) * 1000
    del llm
    return elapsed_ms


def main():
    tokenizer_llm = Llama(model_path=MODEL_PATH, n_ctx=TOKENIZER_CTX, verbose=False)

    rows = []
    for lang, prompts in PROMPTS.items():
        print(f"Processing {lang} ({len(prompts)} prompts)...")
        for idx, (prompt, bucket) in enumerate(zip(prompts, LENGTH_BUCKETS)):
            detected_script = detect_script(prompt)
            token_count = len(tokenizer_llm.tokenize(prompt.encode()))
            chars_per_token = round(len(prompt) / token_count, 3) if token_count else None
            n_ctx = adaptive_ctx(token_count, chars_per_token)

            latency_ms = run_inference(prompt, n_ctx)

            rows.append({
                "language": lang,
                "prompt_idx": idx,
                "length_bucket": bucket,
                "detected_script": detected_script,
                "prompt_chars": len(prompt),
                "token_count": token_count,
                "chars_per_token": chars_per_token,
                "adaptive_n_ctx": n_ctx,
                "latency_ms": round(latency_ms, 2),
            })

    del tokenizer_llm

    fieldnames = ["language", "prompt_idx", "length_bucket", "detected_script",
                  "prompt_chars", "token_count", "chars_per_token",
                  "adaptive_n_ctx", "latency_ms"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    print("\n=== QUICK SUMMARY (mean adaptive_n_ctx by language) ===")
    for lang in PROMPTS:
        lang_rows = [r for r in rows if r["language"] == lang]
        mean_ctx = sum(r["adaptive_n_ctx"] for r in lang_rows) / len(lang_rows)
        print(f"{lang}: mean adaptive_n_ctx = {mean_ctx:.1f}")


if __name__ == "__main__":
    main()