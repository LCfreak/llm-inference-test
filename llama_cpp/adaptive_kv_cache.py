# from llama_cpp import Llama
# import math
# import time

# MODEL_PATH = "/Users/ashutoshbhattarai/Desktop/learning/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

# def detect_lang(prompt):
#     count = sum(1 for char in prompt if '\u0900' <= char <= '\u097F')
#     return "devanagari" if count/len(prompt) >= 0.3 else "latin"

# def get_token_count(prompt):
#     temp = Llama(model_path=MODEL_PATH, n_ctx=64, verbose=False)
#     return len(temp.tokenize(prompt.encode()))

# def adaptive_ctx(token_count, chars_per_token):
#     # chars_per_token < 3.0 means inefficient tokenization (Devanagari territory)
#     # English sits at ~5.0, Nepali at ~1.0, threshold at 3.0 cleanly separates
#     if chars_per_token < 3.0:
#         return math.ceil((token_count + 50) * 1.5)  # larger buffer for Devanagari
#     return math.ceil((token_count + 50) * 1.2)      # standard buffer for Latin

# def run_inference(prompt, n_ctx):
#     llm = Llama(model_path=MODEL_PATH, n_ctx=n_ctx, verbose=False)
#     start = time.perf_counter()
#     output = llm(prompt, max_tokens=50)
#     elapsed = (time.perf_counter() - start) * 1000
#     return output, elapsed

# def process(prompt, label):
#     lang = detect_lang(prompt)
#     token_count = get_token_count(prompt)
#     chars_per_token = round(len(prompt) / token_count, 2)
#     n_ctx = adaptive_ctx(token_count, chars_per_token)
#     output, latency_ms = run_inference(prompt, n_ctx)
    
#     print(f"\n{label}")
#     print(f"  Language detected : {lang}")
#     print(f"  Tokens            : {token_count}")
#     print(f"  Chars/token       : {chars_per_token}")
#     print(f"  Adaptive n_ctx    : {n_ctx}")
#     print(f"  Latency           : {latency_ms:.1f}ms")

# def main():
#     prompts = [
#         ("Explain machine learning in simple terms.", "English"),
#         ("मशीन लर्निंग को सरल शब्दों में समझाइए।", "Hindi"),
#         ("मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।", "Nepali"),
#     ]
#     for prompt, label in prompts:
#         process(prompt, label)

# if __name__ == "__main__":
#     main() 

"""
Adaptive n_ctx sizing benchmark -- your "optimization" script, expanded.

Idea being tested: since Devanagari-script prompts tokenize to ~3-4x more
tokens than English for the same meaning, giving them a proportionally
larger context buffer (1.5x vs 1.2x) should avoid truncation/overflow
without over-allocating for every request by default.

Restructured from the original:
  - Tokenization no longer reloads a fresh Llama per prompt. Token count
    doesn't depend on n_ctx, so ONE small shared "tokenizer" model
    (n_ctx=64) is loaded once and reused for all 45 prompts' token counts.
    The original called `Llama(...)` fresh inside get_token_count() for
    every prompt just to tokenize -- that's a full model load wasted on
    something that doesn't need it.
  - The actual INFERENCE model still has to be reloaded per prompt, because
    n_ctx is genuinely different per prompt by design (that's the thing
    you're testing). This is unavoidable given the adaptive-ctx approach --
    flagging it so it's a known cost, not a bug, when you see wall time.
  - detect_lang's div-by-zero on an empty prompt is guarded.
  - language label from your dataset is also recorded per row (in addition
    to the char-ratio-based `detected_script`), so you can separately check
    whether char-based script detection agrees with the ground-truth
    language label -- useful if you ever feed in code-switched text.

Exports: adaptive_ctx_results.csv
"""

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