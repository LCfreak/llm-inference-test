"""
make_data.py — build the six-language parallel corpus for the fertility study.

Downloads FLORES-200 dev / devtest for English, Nepali, Hindi, Bengali, Tamil and
Amharic, aligns them line-by-line, and writes one wide CSV with a column per
language. The compression sweep reads that CSV directly (--csv).

Usage
-----
  python src/make_data.py                          # download + build the corpus
  python src/make_data.py --split devtest          # devtest only (default: both)
  python src/make_data.py --tokenizer meta-llama/Llama-3.2-1B-Instruct
                                                   # also write per-model fertility

Why per-model fertility: token counts are a property of the tokenizer, not the
language. The same six languages have different expansion ratios under Llama-3.2,
Qwen2.5 or Gemma, so fertility must be reported per model and never pooled.
"""

import argparse
import os
import urllib.request

import pandas as pd

BASE = ("https://raw.githubusercontent.com/common-parallel-corpora/"
        "common-parallel-corpora/master/data/common-parallel-corpora")

# language name -> FLORES-200 code
LANGS = {
    "english": "eng_Latn",
    "nepali": "npi_Deva",
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "amharic": "amh_Ethi",
}

SPLITS = {"dev": "flores-200-dev", "devtest": "flores-200-devtest"}


def download(code, split, out_dir):
    """Fetch one FLORES file; skip if it is already on disk."""
    ext = split
    url = f"{BASE}/{SPLITS[split]}/{code}.{ext}"
    path = os.path.join(out_dir, f"{code}.{ext}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "kv-fertility/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    print(f"  downloaded {code}.{ext}  ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build(splits, out_dir, out_csv):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    frames = []
    for split in splits:
        print(f"[{split}]")
        cols = {}
        for name, code in LANGS.items():
            cols[name] = read_lines(download(code, split, raw_dir))
        n = {k: len(v) for k, v in cols.items()}
        if len(set(n.values())) != 1:
            raise SystemExit(f"{split}: misaligned line counts {n}")
        df = pd.DataFrame(cols)
        df.insert(0, "split", split)
        # keep only lines where every language is non-empty
        df = df[(df[list(LANGS)] != "").all(axis=1)].reset_index(drop=True)
        print(f"  {len(df)} aligned sentences")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "id", range(len(out)))
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nwrote {out_csv}  ({len(out)} rows, {len(LANGS)} languages)")
    return out


def fertility(df, model_name, out_dir):
    """Token counts and expansion ratio under one model's tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    stats = {}
    for name in LANGS:
        counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in df[name]]
        stats[name] = pd.Series(counts)
    stats = pd.DataFrame(stats)
    ref = stats["english"].mean()
    table = pd.DataFrame({
        "lang": list(LANGS),
        "tokens_per_sentence": [stats[l].mean() for l in LANGS],
        "tokens_per_word": [stats[l].sum() / df[l].astype(str).str.split().str.len().sum()
                            for l in LANGS],
    })
    table["expansion_vs_english"] = table["tokens_per_sentence"] / ref
    table["equal_memory_ratio"] = 1 - 1 / table["expansion_vs_english"]

    slug = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fertility_{slug}.csv")
    table.to_csv(path, index=False)

    print(f"\nFertility under {model_name}")
    print(table.to_string(index=False,
                          float_format=lambda x: f"{x:6.3f}"))
    print(f"\nwrote {path}")
    print("equal_memory_ratio = fraction of that language's KV entries that must be "
          "removed to match uncompressed English memory.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/flores200")
    ap.add_argument("--out", default="data/flores200/parallel_6lang.csv")
    ap.add_argument("--split", nargs="+", default=["dev", "devtest"], choices=list(SPLITS))
    ap.add_argument("--tokenizer", default=None,
                    help="HF model id; also write fertility_<model>.csv")
    args = ap.parse_args()

    df = build(args.split, args.out_dir, args.out)
    if args.tokenizer:
        fertility(df, args.tokenizer, args.out_dir)
