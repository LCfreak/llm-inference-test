"""
serve_bench.py — GPU serving measurement: how much concurrency does each language
cost at a fixed VRAM budget?

This is the experiment the current report marks "Not Tested": token-entry reduction
is a simulation until it is measured on a real GPU with a real paged-attention engine.

Two modes
---------
  --mode tokens   (no GPU needed) token counts per language + the analytic capacity
                  implied by a fixed KV budget. Use this on a laptop to sanity-check
                  the prompt sets before spending GPU hours.

  --mode offline  (CUDA + vLLM) load one model with a fixed gpu_memory_utilization
                  and ramp the number of concurrent requests. Reports throughput,
                  per-request latency, TTFT/TPOT and preemptions per language.

TTFT/TPOT are measured the cheap way: a one-token run gives prefill latency (TTFT),
a 64-token run gives (t64 - t1) / 63 per output token.

Usage
-----
  python src/serve_bench.py --mode tokens --model meta-llama/Llama-3.2-1B-Instruct
  python src/serve_bench.py --mode offline --model meta-llama/Llama-3.2-1B-Instruct \
      --batch-sizes 1 2 4 8 16 32 64 --out out/serve_llama1b.csv
"""

import argparse
import logging
import os
import time

import pandas as pd


def build_prompts(csv_path, langs, docs, sents, tokenizer=None, max_tokens=None, cap=None):
    """One prompt per document = `sents` aligned sentences.

    Prompts are matched on CONTENT (same sentences), never on token count: truncating
    each language to the same number of tokens would delete the fertility effect that
    this experiment is trying to measure. `max_tokens` is an optional latency-matched
    control; `cap` is only a safety limit for max_model_len.
    """
    df = pd.read_csv(csv_path).dropna(subset=langs)
    out = {}
    for lang in langs:
        texts = []
        for d in range(docs):
            chunk = df.iloc[d * sents:(d + 1) * sents]
            if len(chunk) < sents:
                break
            text = " ".join(chunk[lang].astype(str))
            if tokenizer is not None:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if max_tokens:
                    ids = ids[:max_tokens]
                if cap:
                    ids = ids[:cap]
                text = tokenizer.decode(ids)
            texts.append(text)
        out[lang] = texts
    return out


def analytic_capacity(prompts, tokenizer, cfg, kv_budget_gb):
    """Max concurrent sequences at a fixed KV budget (no measurement, just arithmetic)."""
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    bytes_per_token = 2 * n_layers * n_kv * hd * 2          # K and V, fp16
    rows = []
    for lang, texts in prompts.items():
        lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        mean = sum(lens) / len(lens)
        rows.append({"lang": lang, "n_prompts": len(lens),
                     "tokens_mean": mean, "tokens_min": min(lens), "tokens_max": max(lens),
                     "kv_mb_per_request": mean * bytes_per_token / 1e6,
                     "max_concurrent_at_budget": kv_budget_gb * 1e9 / (mean * bytes_per_token)})
    return pd.DataFrame(rows)


class _PreemptionCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.n = 0

    def emit(self, record):
        try:
            if "preempt" in record.getMessage().lower():
                self.n += 1
        except Exception:
            pass


def run_offline(args):
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
    from vllm import LLM, SamplingParams
    import torch
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cfg = AutoConfig.from_pretrained(args.model)
    prompts = build_prompts(args.csv, args.langs, args.docs, args.sents, tok, args.prompt_tokens,
                            cap=args.max_model_len - args.out_tokens - 32)

    print(analytic_capacity(prompts, tok, cfg, args.kv_budget_gb).to_string(index=False))

    llm = LLM(model=args.model, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len,
              max_num_seqs=max(args.batch_sizes),
              enforce_eager=args.enforce_eager,
              trust_remote_code=True)

    num_blocks = None
    for path in ("llm_engine.vllm_config.cache_config", "llm_engine.cache_config"):
        obj = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            num_blocks = getattr(obj, "num_gpu_blocks", None)
            if num_blocks:
                break
        except Exception:
            continue
    print(f"num_gpu_blocks={num_blocks}")

    counter = _PreemptionCounter()
    logging.getLogger("vllm").addHandler(counter)
    logging.getLogger("vllm").setLevel(logging.INFO)

    def gen(texts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(texts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, outs

    rows = []
    # warmup
    gen(prompts["english"][:1], 8)

    for lang in args.langs:
        base = prompts[lang]
        # TTFT / TPOT at batch 1
        _, out1 = gen(base[:1], 1)
        t1, _ = gen(base[:1], 1)
        t64, _ = gen(base[:1], 64)
        ttft = t1
        tpot = max(0.0, (t64 - t1) / 63)
        print(f"[{lang}] ttft={ttft*1e3:.1f} ms  tpot={tpot*1e3:.2f} ms")

        for b in args.batch_sizes:
            texts = (base * ((b // len(base)) + 1))[:b]
            torch.cuda.reset_peak_memory_stats()
            counter.n = 0
            wall, outs = gen(texts, args.out_tokens)
            n_out = sum(len(o.outputs[0].token_ids) for o in outs)
            rows.append({
                "model": args.model, "dtype": args.dtype, "lang": lang, "batch": b,
                "gpu_util": args.gpu_util, "max_model_len": args.max_model_len,
                "num_gpu_blocks": num_blocks,
                "prompt_tokens_mean": sum(len(tok(t, add_special_tokens=False)["input_ids"])
                                          for t in texts) / len(texts),
                "wall_s": wall, "out_tokens": n_out,
                "throughput_tok_s": n_out / wall,
                "latency_s_per_request": wall,
                "ttft_s": ttft, "tpot_s": tpot,
                "preemptions": counter.n,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            })
            print(f"  batch={b:3d}  {n_out/wall:8.1f} tok/s  "
                  f"{wall*1e3/max(1,n_out):6.2f} ms/tok  preempt={counter.n}")
        pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


def run_hf(args):
    """Fallback when vLLM will not run on the card (e.g. T4 + a vLLM build without
    an sm_75 attention backend). Measures real GPU throughput and latency with eager
    attention and a contiguous KV cache — no PagedAttention, so report it as such.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(args.model)
    prompts = build_prompts(args.csv, args.langs, args.docs, args.sents, tok, args.prompt_tokens,
                            cap=args.max_model_len - args.out_tokens - 32)
    print(analytic_capacity(prompts, tok, cfg, args.kv_budget_gb).to_string(index=False))

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager").to("cuda").eval()

    def gen(texts, n):
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, out, enc

    rows = []
    gen(prompts["english"][:1], 4)                      # warmup
    for lang in args.langs:
        base = prompts[lang]
        t1, _, _ = gen(base[:1], 1)
        t64, _, _ = gen(base[:1], 64)
        ttft, tpot = t1, max(0.0, (t64 - t1) / 63)
        print(f"[{lang}] ttft={ttft*1e3:.1f} ms  tpot={tpot*1e3:.2f} ms")
        for b in args.batch_sizes:
            texts = (base * ((b // len(base)) + 1))[:b]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                wall, out, enc = gen(texts, args.out_tokens)
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={b:3d}  OOM")
                rows.append({"model": args.model, "dtype": args.dtype, "lang": lang,
                             "batch": b, "oom": True, "gpu_util": args.gpu_util,
                             "max_model_len": args.max_model_len,
                             "num_gpu_blocks": None, "prompt_tokens_mean": None,
                             "wall_s": None, "out_tokens": None, "throughput_tok_s": None,
                             "latency_s_per_request": None, "ttft_s": ttft,
                             "tpot_s": tpot, "preemptions": None,
                             "peak_vram_gb": None})
                torch.cuda.empty_cache()
                break
            n_out = (out.shape[1] - enc["input_ids"].shape[1]) * b
            rows.append({
                "model": args.model, "dtype": args.dtype, "lang": lang, "batch": b,
                "oom": False, "gpu_util": None, "max_model_len": args.max_model_len,
                "num_gpu_blocks": None,
                "prompt_tokens_mean": enc["input_ids"].shape[1],
                "wall_s": wall, "out_tokens": n_out,
                "throughput_tok_s": n_out / wall,
                "latency_s_per_request": wall,
                "ttft_s": ttft, "tpot_s": tpot, "preemptions": 0,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            })
            print(f"  batch={b:3d}  {n_out/wall:8.1f} tok/s  "
                  f"{torch.cuda.max_memory_allocated()/1e9:5.2f} GB")
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


def run_tokens(args):
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cfg = AutoConfig.from_pretrained(args.model)
    prompts = build_prompts(args.csv, args.langs, args.docs, args.sents, tok, args.prompt_tokens,
                            cap=args.max_model_len - args.out_tokens - 32)
    tab = analytic_capacity(prompts, tok, cfg, args.kv_budget_gb)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nKV budget {args.kv_budget_gb:.1f} GB; capacity = budget / (tokens x bytes/token).")
    if args.out:
        tab.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tokens", choices=["tokens", "offline", "hf"])
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--csv", default="data/flores200/parallel_6lang.csv")
    ap.add_argument("--langs", nargs="+",
                    default=["english", "nepali", "hindi", "bengali", "tamil", "amharic"])
    ap.add_argument("--docs", type=int, default=16, help="distinct prompts per language")
    ap.add_argument("--sents", type=int, default=10)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="0 = no truncation (content-matched prompts); >0 is a latency-matched control")
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-budget-gb", type=float, default=2.0)
    ap.add_argument("--attention-backend", default="XFORMERS",
                    help="T4/Turing needs XFORMERS; leave default on Ampere+ to override")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "offline":
        run_offline(args)
    elif args.mode == "hf":
        run_hf(args)
    else:
        run_tokens(args)
