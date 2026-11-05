# """
# Test whether merging low-attention-entropy KV blocks (at chosen intermediate
# layers) preserves perplexity, for Nepali vs English sentences.

# Requires the HF (non-GGUF) Llama 3.2 weights and eager attention, since we
# need real attention weight tensors (output_attentions=True) and we hook the
# k_proj/v_proj submodules directly.
# """

# import torch
# import pandas as pd
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or a local path

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float32,
#     attn_implementation="eager",
# )
# model.eval()


# # ---------- 1. Per-position attention entropy at a given layer ----------

# @torch.no_grad()
# def position_entropy_at_layer(attentions, layer_idx):
#     """attentions: tuple from model(..., output_attentions=True).attentions
#     Returns a 1D tensor of length seq_len: entropy of each query position's
#     attention row, averaged across heads, at the given layer."""
#     attn = attentions[layer_idx][0]          # (n_heads, seq, seq)
#     p = attn.clamp(min=1e-12)
#     row_entropy = -(p * p.log2()).sum(dim=-1)  # (n_heads, seq)
#     return row_entropy.mean(dim=0)             # (seq,)


# # ---------- 2. Turn low-entropy positions into contiguous merge blocks ----

# def low_entropy_blocks(entropy, percentile=25, min_block_len=2):
#     """Returns a list of (start, end) index pairs (end exclusive) marking
#     contiguous runs of positions whose entropy falls below the given
#     percentile for this sequence."""
#     threshold = torch.quantile(entropy, percentile / 100.0)
#     is_low = (entropy < threshold).tolist()

#     blocks = []
#     start = None
#     for i, low in enumerate(is_low + [False]):  # sentinel to flush last run
#         if low and start is None:
#             start = i
#         elif not low and start is not None:
#             if i - start >= min_block_len:
#                 blocks.append((start, i))
#             start = None
#     return blocks


# # ---------- 3. Hooks that average K or V within each block, in place -----

# def make_merge_hook(blocks):
#     def hook(module, inputs, output):
#         out = output.clone()
#         for start, end in blocks:
#             avg = out[:, start:end, :].mean(dim=1, keepdim=True)
#             out[:, start:end, :] = avg
#         return out
#     return hook


# def register_merge_hooks(model, layer_idx, blocks):
#     layer = model.model.layers[layer_idx].self_attn
#     handles = [
#         layer.k_proj.register_forward_hook(make_merge_hook(blocks)),
#         layer.v_proj.register_forward_hook(make_merge_hook(blocks)),
#     ]
#     return handles


# # ---------- 4. Perplexity for one sentence, baseline vs merged -----------

# @torch.no_grad()
# def perplexity(input_ids):
#     out = model(input_ids=input_ids, labels=input_ids)
#     return torch.exp(out.loss).item()


# @torch.no_grad()
# def merge_experiment(text, target_layer, percentile=25, min_block_len=2):
#     inputs = tokenizer(text, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     if input_ids.shape[1] < 4:
#         return None  # too short to be meaningful

#     # Pass 1: baseline, and get attentions to decide what to merge
#     out = model(input_ids=input_ids, labels=input_ids, output_attentions=True)
#     baseline_ppl = torch.exp(out.loss).item()

#     entropy = position_entropy_at_layer(out.attentions, target_layer)
#     blocks = low_entropy_blocks(entropy, percentile, min_block_len)

#     if not blocks:
#         return {
#             "baseline_ppl": baseline_ppl,
#             "merged_ppl": baseline_ppl,
#             "n_blocks": 0,
#             "n_merged_tokens": 0,
#             "seq_len": input_ids.shape[1],
#         }

#     # Pass 2: merged
#     handles = register_merge_hooks(model, target_layer, blocks)
#     try:
#         merged_ppl = perplexity(input_ids)
#     finally:
#         for h in handles:
#             h.remove()

#     return {
#         "baseline_ppl": baseline_ppl,
#         "merged_ppl": merged_ppl,
#         "n_blocks": len(blocks),
#         "n_merged_tokens": sum(e - s for s, e in blocks),
#         "seq_len": input_ids.shape[1],
#     }


# # ---------- 5. Example usage against the generated sentence pairs --------
# # Run this file directly to execute the experiment. Set TARGET_LAYER to one
# # of the model's layer indices before running.
# if __name__ == "__main__":
#     TARGET_LAYER = 6
#     df = pd.read_csv("fertility_results.csv")
    
#     LAYERS = [4,6,8,10]

#     results = []
#     for _, row in df.iterrows():
#         for lang in ["english", "nepali"]:
#             for layer in LAYERS:
#                 r = merge_experiment(row[lang], layer, percentile=25, min_block_len=2)
#                 if r is not None:
#                     r.update(lang=lang, layer= layer,
#                              token_expansion_ratio=row["token_expansion_ratio"])
#                     results.append(r)



#     res_df = pd.DataFrame(results)
#     res_df["ppl_delta"] = res_df["merged_ppl"] - res_df["baseline_ppl"]
#     res_df.to_csv("merge_results.csv", index=False)
#     print(res_df.groupby(["layer", "lang"])["ppl_delta"].describe())



"""
Test whether merging low-attention-entropy KV blocks (at chosen intermediate
layers) preserves perplexity, for Nepali vs English sentences.

Requires the HF (non-GGUF) Llama 3.2 weights and eager attention, since we
need real attention weight tensors (output_attentions=True) and we hook the
k_proj/v_proj submodules directly.
"""

import torch
import pandas as pd
from tqdm import tqdm  # pip install tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or a local path

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()


# ---------- 1. Per-position attention entropy at a given layer ----------

@torch.no_grad()
def position_entropy_at_layer(attentions, layer_idx):
    attn = attentions[layer_idx][0]            # (n_heads, seq, seq)
    p = attn.clamp(min=1e-12)
    row_entropy = -(p * p.log2()).sum(dim=-1)  # (n_heads, seq)
    return row_entropy.mean(dim=0)             # (seq,)


# ---------- 2. Contiguous low-entropy runs -> merge blocks ----------------

def low_entropy_blocks(entropy, percentile=25, min_block_len=2):
    threshold = torch.quantile(entropy, percentile / 100.0)
    is_low = (entropy < threshold).tolist()
    is_low[0] = False  # never merge the BOS / attention-sink position

    blocks = []
    start = None
    for i, low in enumerate(is_low + [False]):  # sentinel flushes last run
        if low and start is None:
            start = i
        elif not low and start is not None:
            if i - start >= min_block_len:
                blocks.append((start, i))
            start = None
    return blocks


# ---------- 3. Hooks that average K / V within each block -----------------

def make_merge_hook(blocks):
    def hook(module, inputs, output):
        out = output.clone()
        for start, end in blocks:
            avg = out[:, start:end, :].mean(dim=1, keepdim=True)
            out[:, start:end, :] = avg
        return out
    return hook


def register_merge_hooks(model, layer_idx, blocks):
    layer = model.model.layers[layer_idx].self_attn
    return [
        layer.k_proj.register_forward_hook(make_merge_hook(blocks)),
        layer.v_proj.register_forward_hook(make_merge_hook(blocks)),
    ]


# ---------- 4. Perplexity: baseline vs merged ------------------------------

@torch.no_grad()
def perplexity(input_ids):
    out = model(input_ids=input_ids, labels=input_ids)
    return torch.exp(out.loss).item()


@torch.no_grad()
def merge_experiment_layers(text, layers, percentile=25, min_block_len=2):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] < 4:
        return []

    # ONE baseline pass yields attentions for every layer
    out = model(input_ids=input_ids, labels=input_ids, output_attentions=True)
    baseline_ppl = torch.exp(out.loss).item()

    rows = []
    for layer in layers:
        entropy = position_entropy_at_layer(out.attentions, layer)
        blocks = low_entropy_blocks(entropy, percentile, min_block_len)

        merged_ppl = baseline_ppl
        if blocks:
            handles = register_merge_hooks(model, layer, blocks)
            try:
                merged_ppl = perplexity(input_ids)
            finally:
                for h in handles:
                    h.remove()

        rows.append({
            "layer": layer,
            "baseline_ppl": baseline_ppl,
            "merged_ppl": merged_ppl,
            "n_blocks": len(blocks),
            "n_merged_tokens": sum(e - s for s, e in blocks),
            "seq_len": input_ids.shape[1],
        })
    return rows


# ---------- 5. Driver -------------------------------------------------------

if __name__ == "__main__":
    N_PAIRS = 30            # start small; raise once it works
    LAYERS = [4, 6, 8, 10]

    df = pd.read_csv("fertility_results.csv").sample(N_PAIRS, random_state=0)

    results = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        for lang in ["english", "nepali"]:
            for r in merge_experiment_layers(row[lang], LAYERS):
                r.update(lang=lang,
                         token_expansion_ratio=row["token_expansion_ratio"])
                results.append(r)

        if (i + 1) % 10 == 0:  # checkpoint so work is never lost
            pd.DataFrame(results).to_csv("merge_results_partial.csv", index=False)

    res_df = pd.DataFrame(results)
    res_df["ppl_delta"] = res_df["merged_ppl"] - res_df["baseline_ppl"]
    res_df.to_csv("merge_results.csv", index=False)
    print(res_df.groupby(["layer", "lang"])["ppl_delta"].describe())