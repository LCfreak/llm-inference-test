# import concurrent.futures
# import time
# from llama_cpp import Llama

# MODEL_PATH = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# def simulate_throughput(prompts, label, n_concurrent=5):
#     """Simulate multiple concurrent inference requests"""
    
#     def single_inference(prompt):
#         llm = Llama(model_path=MODEL_PATH, 
#                    n_ctx=256, n_threads=2, verbose=False)
#         start = time.perf_counter()
#         output = llm(prompt, max_tokens=20)
#         elapsed = time.perf_counter() - start
#         tokens = len(llm.tokenize(prompt.encode()))
#         return {"tokens": tokens, "ms": elapsed * 1000}
    
#     start_wall = time.perf_counter()
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
#         futures = [executor.submit(single_inference, p) for p in prompts]
#         results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
#     wall_time = (time.perf_counter() - start_wall) * 1000
#     total_tokens = sum(r["tokens"] for r in results)
    
#     return {
#         "label": label,
#         "concurrent_requests": n_concurrent,
#         "total_tokens_processed": total_tokens,
#         "wall_time_ms": round(wall_time, 1),
#         "requests_per_second": round(n_concurrent / (wall_time/1000), 2),
#         "tokens_per_second": round(total_tokens / (wall_time/1000), 1)
#     }

# english_prompts = [
#     "Explain machine learning briefly."
# ] * 5

# nepali_prompts = [
#     "मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।"
# ] * 5

# print("Throughput comparison (5 concurrent requests):")
# print()
# eng = simulate_throughput(english_prompts, "English")
# nep = simulate_throughput(nepali_prompts, "Nepali")

# for r in [eng, nep]:
#     print(f"{r['label']}:")
#     print(f"  Total tokens processed: {r['total_tokens_processed']}")
#     print(f"  Wall time: {r['wall_time_ms']}ms")
#     print(f"  Requests/sec: {r['requests_per_second']}")
#     print(f"  Tokens/sec: {r['tokens_per_second']}")

# print(f"\nTo serve same number of Nepali vs English requests:")
# print(f"Nepali requires {round(nep['total_tokens_processed']/eng['total_tokens_processed'], 2)}x more compute")
# print(f"Effective throughput ratio: {round(eng['requests_per_second']/nep['requests_per_second'], 2)}x fewer Nepali users served per unit time")


import time
from llama_cpp import Llama

MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def simulate_throughput_sequential(prompts, label):
    """
    Simulate batch throughput by running N requests sequentially
    and measuring total wall time — honest single-instance measurement
    """
    llm = Llama(model_path=MODEL_PATH,
                n_ctx=256,
                n_threads=4,
                verbose=False)
    
    results = []
    total_start = time.perf_counter()
    
    for prompt in prompts:
        tokens = llm.tokenize(prompt.encode())
        req_start = time.perf_counter()
        output = llm(prompt, max_tokens=20)
        req_elapsed = (time.perf_counter() - req_start) * 1000
        results.append({
            "tokens": len(tokens),
            "ms": req_elapsed
        })
    
    total_wall = (time.perf_counter() - total_start) * 1000
    total_tokens = sum(r["tokens"] for r in results)
    n = len(prompts)
    
    print(f"\n{label} — {n} sequential requests:")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Total wall time: {round(total_wall, 1)}ms")
    print(f"  Avg latency per request: {round(total_wall/n, 1)}ms")
    print(f"  Throughput: {round(n/(total_wall/1000), 2)} requests/sec")
    print(f"  Token throughput: {round(total_tokens/(total_wall/1000), 1)} tokens/sec")
    
    return {
        "label": label,
        "total_tokens": total_tokens,
        "total_wall_ms": total_wall,
        "requests_per_sec": n / (total_wall/1000),
        "tokens_per_sec": total_tokens / (total_wall/1000)
    }

# 10 requests each language
english_prompts = ["Explain machine learning briefly."] * 10
hindi_prompts = ["मशीन लर्निंग को सरल शब्दों में समझाइए।"] * 10
nepali_prompts = ["मेसिन लर्निङलाई सरल भाषामा बुझाउनुहोस्।"] * 10

eng = simulate_throughput_sequential(english_prompts, "English")
hin = simulate_throughput_sequential(hindi_prompts, "Hindi")
nep = simulate_throughput_sequential(nepali_prompts, "Nepali")

print("\n=== INFRASTRUCTURE COST COMPARISON ===")
print(f"To serve 1000 users with equivalent content:")
print(f"English tokens needed:  {round(eng['total_tokens']/10 * 1000):,}")
print(f"Hindi tokens needed:    {round(hin['total_tokens']/10 * 1000):,}")
print(f"Nepali tokens needed:   {round(nep['total_tokens']/10 * 1000):,}")
print(f"\nNepali compute overhead vs English: "
      f"{round(nep['total_tokens']/eng['total_tokens'], 2)}x")
print(f"Hindi compute overhead vs English:  "
      f"{round(hin['total_tokens']/eng['total_tokens'], 2)}x")
print(f"\nEffective user capacity on fixed compute:")
print(f"English users served per unit: 1.00x (baseline)")
print(f"Hindi users served per unit:   "
      f"{round(eng['tokens_per_sec']/hin['tokens_per_sec'], 2)}x")
print(f"Nepali users served per unit:  "
      f"{round(eng['tokens_per_sec']/nep['tokens_per_sec'], 2)}x")