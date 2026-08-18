from llama_cpp import Llama
import math
import time

MODEL_PATH = "/Users/ashutoshbhattarai/Desktop/learning/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def detect_lang(prompt):
    count = sum(1 for char in prompt if '\u0900' <= char <= '\u097F')
    return "devanagari" if count/len(prompt) >= 0.3 else "latin"

def get_token_count(prompt):
    temp = Llama(model_path=MODEL_PATH, n_ctx=64, verbose=False)
    return len(temp.tokenize(prompt.encode()))

def adaptive_ctx(token_count, chars_per_token):
    # chars_per_token < 3.0 means inefficient tokenization (Devanagari territory)
    # English sits at ~5.0, Nepali at ~1.0, threshold at 3.0 cleanly separates
    if chars_per_token < 3.0:
        return math.ceil((token_count + 50) * 1.5)  # larger buffer for Devanagari
    return math.ceil((token_count + 50) * 1.2)      # standard buffer for Latin

def run_inference(prompt, n_ctx):
    llm = Llama(model_path=MODEL_PATH, n_ctx=n_ctx, verbose=False)
    start = time.perf_counter()
    output = llm(prompt, max_tokens=50)
    elapsed = (time.perf_counter() - start) * 1000
    return output, elapsed

def process(prompt, label):
    lang = detect_lang(prompt)
    token_count = get_token_count(prompt)
    chars_per_token = round(len(prompt) / token_count, 2)
    n_ctx = adaptive_ctx(token_count, chars_per_token)
    output, latency_ms = run_inference(prompt, n_ctx)
    
    print(f"\n{label}")
    print(f"  Language detected : {lang}")
    print(f"  Tokens            : {token_count}")
    print(f"  Chars/token       : {chars_per_token}")
    print(f"  Adaptive n_ctx    : {n_ctx}")
    print(f"  Latency           : {latency_ms:.1f}ms")

def main():
    prompts = [
        ("Explain machine learning in simple terms.", "English"),
        ("मशीन लर्निंग को सरल शब्दों में समझाइए।", "Hindi"),
        ("मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।", "Nepali"),
    ]
    for prompt, label in prompts:
        process(prompt, label)

if __name__ == "__main__":
    main()