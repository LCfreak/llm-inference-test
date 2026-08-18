# benchmark.py
from llama_cpp import Llama
import time
import json

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def benchmark(prompt, label, runs=5):
    llm = Llama(model_path=MODEL_PATH, n_ctx=512, verbose=False)
    
    # Tokenize and count
    tokens = llm.tokenize(prompt.encode())
    
    # Time inference
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = llm(prompt, max_tokens=30, echo=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    times.sort()
    median_time = times[len(times)//2]
    
    return {
        "label": label,
        "prompt_chars": len(prompt),
        "token_count": len(tokens),
        "chars_per_token": round(len(prompt)/len(tokens), 2),
        "median_inference_ms": round(median_time * 1000, 1),
        "tokens_generated_per_sec": round(30/median_time, 1)
    }

# Test prompts — same meaning, different languages
prompts = [
    ("Explain machine learning in simple terms.", "English"),
    ("मशीन लर्निंग को सरल शब्दों में समझाइए।", "Hindi"),
    ("मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।", "Nepali"),
]

results = []
for prompt, label in prompts:
    print(f"Benchmarking {label}...")
    r = benchmark(prompt, label)
    results.append(r)
    print(json.dumps(r, indent=2, ensure_ascii=False))

print("\n=== SUMMARY ===")
english_chars_per_token = results[0]["chars_per_token"]
for r in results:
    ratio = round(english_chars_per_token / r["chars_per_token"], 2)
    print(f"{r['label']}: {r['chars_per_token']} chars/token "
          f"({ratio}x less efficient than English)")