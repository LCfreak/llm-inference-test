from llama_cpp import Llama
import time
import psutil
import os

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def measure_kv_cache_pressure(prompt, label, context_lengths=[64, 128, 256, 512]):
    results = []
    
    for ctx_len in context_lengths:
        llm = Llama(model_path=MODEL_PATH, 
                   n_ctx=ctx_len, 
                   verbose=False)
        
        tokens = llm.tokenize(prompt.encode())
        
        if len(tokens) >= ctx_len:
            continue
            
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start = time.perf_counter()
        output = llm(prompt, max_tokens=20)
        elapsed = time.perf_counter() - start
        
        mem_after = process.memory_info().rss / 1024 / 1024
        
        results.append({
            "label": label,
            "context_length": ctx_len,
            "input_tokens": len(tokens),
            "memory_mb": round(mem_after - mem_before, 2),
            "inference_ms": round(elapsed * 1000, 1),
            "memory_per_token_kb": round(
                (mem_after - mem_before) * 1024 / len(tokens), 2)
        })
        
        del llm
    
    return results

english = "Explain how machine learning works."
nepali = "मेसिन लर्निङ कसरी काम गर्छ भनेर बुझाउनुहोस्।"

print("English KV cache pressure:")
for r in measure_kv_cache_pressure(english, "English"):
    print(f"  ctx={r['context_length']}: "
          f"{r['input_tokens']} tokens, "
          f"{r['memory_mb']}MB, "
          f"{r['memory_per_token_kb']}KB/token, "
          f"{r['inference_ms']}ms")

print("\nNepali KV cache pressure:")
for r in measure_kv_cache_pressure(nepali, "Nepali"):
    print(f"  ctx={r['context_length']}: "
          f"{r['input_tokens']} tokens, "
          f"{r['memory_mb']}MB, "
          f"{r['memory_per_token_kb']}KB/token, "
          f"{r['inference_ms']}ms")