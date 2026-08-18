from llama_cpp import Llama
import time
import resource

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def measure_clean(prompt, label, n_ctx=512, runs=5):
    llm = Llama(model_path=MODEL_PATH,
                n_ctx=n_ctx,
                n_threads=4,
                verbose=False)
    
    tokens = llm.tokenize(prompt.encode())
    token_count = len(tokens)
    
    # Warmup run — eliminates cold start noise
    llm(prompt, max_tokens=20)
    
    # Actual timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = llm(prompt, max_tokens=20)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        times.append(elapsed_ms)
    
    times.sort()
    
    # Peak memory via resource module — more accurate than psutil RSS
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    
    return {
        "label": label,
        "token_count": token_count,
        "chars_per_token": round(len(prompt) / token_count, 2),
        "p50_ms": round(times[len(times)//2], 1),
        "p95_ms": round(times[int(len(times)*0.95)], 1),
        "min_ms": round(times[0], 1),
        "peak_memory_mb": round(peak_kb / 1024, 1),
        "ms_per_token": round(times[len(times)//2] / token_count, 2)
    }

# Test across languages with same semantic content
test_cases = [
    ("Explain how machine learning works in simple terms.", "English"),
    ("मशीन लर्निंग कैसे काम करता है इसे सरल शब्दों में समझाइए।", "Hindi"),
    ("मेसिन लर्निङ कसरी काम गर्छ भनेर सरल शब्दमा बुझाउनुहोस्।", "Nepali"),
]

print(f"{'Language':<10} {'Tokens':>6} {'Chars/Tok':>9} "
      f"{'P50 ms':>8} {'P95 ms':>8} {'ms/token':>9}")
print("-" * 60)

results = []
for prompt, label in test_cases:
    r = measure_clean(prompt, label)
    results.append(r)
    print(f"{r['label']:<10} {r['token_count']:>6} "
          f"{r['chars_per_token']:>9} {r['p50_ms']:>8} "
          f"{r['p95_ms']:>8} {r['ms_per_token']:>9}")

# Key finding calculation
english_mpt = results[0]['ms_per_token']
print(f"\n=== KEY FINDING ===")
for r in results:
    overhead = round(r['ms_per_token'] / english_mpt, 2)
    print(f"{r['label']}: {r['ms_per_token']}ms/token "
          f"({overhead}x vs English)")