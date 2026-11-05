import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or a local path to the HF (non-GGUF) weights

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",   # required — SDPA/flash paths won't return real attention weights
)
model.eval()

@torch.no_grad()
def layer_head_entropy(text):
    """Returns a list of length n_layers: mean attention-row entropy (bits) for that layer,
    averaged across heads and query positions."""
    inputs = tokenizer(text, return_tensors="pt")
    out = model(**inputs, output_attentions=True)

    per_layer = []
    for layer_attn in out.attentions:          # tuple, len = n_layers
        attn = layer_attn[0]                   # (n_heads, seq, seq) — batch size 1
        p = attn.clamp(min=1e-12)
        row_entropy = -(p * p.log2()).sum(dim=-1)   # (n_heads, seq) — one H per row
        per_layer.append(row_entropy.mean().item())
    return per_layer


# Apply to your existing dataframe
df["english_layer_entropy"] = df["english"].apply(layer_head_entropy)
df["nepali_layer_entropy"]  = df["nepali"].apply(layer_head_entropy)

# Quick correlation check against your fertility columns, per layer
import numpy as np
from scipy.stats import pearsonr

n_layers = len(df["nepali_layer_entropy"].iloc[0])
for layer in range(n_layers):
    nepali_H = df["nepali_layer_entropy"].apply(lambda l: l[layer])
    r, p = pearsonr(df["token_expansion_ratio"], nepali_H)
    print(f"layer {layer:2d}: r={r:+.3f}  p={p:.3g}")