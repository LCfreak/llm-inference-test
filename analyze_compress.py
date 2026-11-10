# """
# analyze_compress.py
# ===================
# Summarise compress_results.csv produced by kv_compress.py.

#   python analyze_compress.py compress_results.csv

# Prints these things and saves compress_curves.png:
#   1. Quality loss (delta NLL, % perplexity) per method / compression level / language
#   2. Paired Nepali-minus-English difference at equal compression
#   3. THE CONTROLS: does entropy-guided or word-boundary merging beat random / fixed / similarity /\n     SnapKV-style eviction?
#   4. Equal-memory comparison: Nepali compressed to the size of English UNcompressed
# """

# import sys

# import matplotlib
# import numpy as np
# import pandas as pd
# from scipy import stats

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# path = sys.argv[1] if len(sys.argv) > 1 else "compress_results.csv"
# df = pd.read_csv(path)
# df["pct_ppl"] = (np.exp(df["delta_nll"]) - 1) * 100
# n_docs = df["doc"].nunique()
# print(f"{path}: {len(df)} rows, {n_docs} documents per language\n")
# if n_docs < 15:
#     print("NOTE: fewer than ~15 documents, treat p-values as indicative only.\n")

# pd.set_option("display.width", 200)

# # ---- 1. quality loss ------------------------------------------------------
# print("=== 1. Mean % perplexity change vs uncompressed (lower is better) ===")
# print(df.pivot_table(index=["method", "ratio"], columns="lang",
#                      values="pct_ppl", aggfunc="mean").round(2), "\n")
# print("=== 1b. Next-token agreement with the uncompressed model (higher is better) ===")
# print(df.pivot_table(index=["method", "ratio"], columns="lang",
#                      values="top1_agree", aggfunc="mean").round(3), "\n")
# print("=== 1c. KL divergence from the uncompressed model, nats/token (lower is better) ===")
# print(df.pivot_table(index=["method", "ratio"], columns="lang",
#                      values="kl", aggfunc="mean").round(4), "\n")
# if df["gen_prefix_frac"].notna().any():
#     print("=== 1d. Free-running greedy generation: fraction of tokens before first divergence ===")
#     print(df.pivot_table(index=["method", "ratio"], columns="lang",
#                          values="gen_prefix_frac", aggfunc="mean").round(3), "\n")


# # ---- helpers for paired tests ---------------------------------------------
# def paired(a, b):
#     """Paired comparison of two aligned Series; returns mean diff and p-value."""
#     d = (a - b).dropna()
#     if len(d) < 3 or d.std() == 0:
#         return d.mean() if len(d) else np.nan, np.nan, len(d)
#     return d.mean(), stats.wilcoxon(d).pvalue if len(d) >= 6 else stats.ttest_1samp(d, 0).pvalue, len(d)


# # ---- 2. Nepali minus English at equal compression --------------------------
# print("=== 2. Nepali minus English, delta NLL (negative = Nepali tolerates compression better) ===")
# wide = df.pivot_table(index=["method", "ratio", "doc"], columns="lang", values="delta_nll")
# rows = []
# for (m, r), g in wide.groupby(level=[0, 1]):
#     mean, p, n = paired(g["nepali"], g["english"])
#     rows.append({"method": m, "ratio": r, "ne_minus_en": round(mean, 4), "p": round(p, 3), "n": n})
# print(pd.DataFrame(rows).to_string(index=False), "\n")

# # ---- 3. controls and head-to-head comparisons ------------------------------
# print("=== 3. Head-to-head at equal compression, delta NLL (negative = FIRST method is better) ===")
# pairs = [("merge_entropy", "merge_random"),     # does entropy-guided selection beat chance?
#          ("merge_word", "merge_random"),        # does word-boundary merging beat chance?
#          ("merge_word", "merge_entropy"),       # fertility-aware vs entropy-guided
#          ("merge_entropy", "merge_fixed"), ("merge_entropy", "merge_sim"),
#          ("merge_word", "evict_snapkv"),        # vs the RocketKV first-stage family
#          ("merge_entropy", "evict_snapkv"), ("merge_entropy", "evict_h2o")]
# idx = ["lang", "ratio", "doc"]
# rows = []
# for first, second in pairs:
#     if first not in df.method.unique() or second not in df.method.unique():
#         continue
#     x = df[df.method == first].set_index(idx)["delta_nll"]
#     y = df[df.method == second].set_index(idx)["delta_nll"]
#     for (lang, r), _ in x.groupby(level=[0, 1]):
#         mean, p, n = paired(x.loc[lang, r], y.loc[lang, r])
#         rows.append({"first": first, "second": second, "lang": lang, "ratio": r,
#                      "diff": round(mean, 4), "p": round(p, 3), "n": n})
# print(pd.DataFrame(rows).to_string(index=False), "\n")

# # ---- 4. equal-memory comparison ---------------------------------------------
# ctx = df.groupby("lang")["ctx_len"].mean()
# r_eq = 1 - ctx["english"] / ctx["nepali"]
# print("=== 4. Equal-memory comparison ===")
# print(f"Mean context: English {ctx['english']:.0f} tokens, Nepali {ctx['nepali']:.0f} tokens "
#       f"-> Nepali must drop {r_eq:.0%} of its KV entries to match English's uncompressed size.")
# rows = []
# for m in df.method.unique():
#     s = (df[(df.method == m) & (df.lang == "nepali")]
#          .groupby("ratio")[["kv_saved_total", "pct_ppl"]].mean().sort_values("kv_saved_total"))
#     xs = np.r_[0, s.kv_saved_total.values]
#     ys = np.r_[0, s.pct_ppl.values]
#     inside = xs.max() >= r_eq
#     s2 = (df[(df.method == m) & (df.lang == "nepali")]
#           .groupby("ratio")[["kv_saved_total", "top1_agree"]].mean().sort_values("kv_saved_total"))
#     t1 = np.r_[1.0, s2.top1_agree.values]
#     rows.append({"method": m, "nepali_pct_ppl_at_equal_memory":
#                  round(float(np.interp(r_eq, xs, ys)), 2) if inside else np.nan,
#                  "nepali_top1_agree_at_equal_memory":
#                  round(float(np.interp(r_eq, np.r_[0, s2.kv_saved_total.values], t1)), 3) if inside else np.nan,
#                  "note": "" if inside else f"grid stops at {xs.max():.0%}; add a larger --ratios value"})
# print(pd.DataFrame(rows).to_string(index=False))
# print("\nRough reading guide (my own thresholds, not a standard): <2% = effectively lossless, "
#       "2-5% = usable, >5% = noticeable loss.\n")

# # ---- plot -------------------------------------------------------------------
# fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
# for ax, lang in zip(axes, ["english", "nepali"]):
#     for m in df.method.unique():
#         s = (df[(df.method == m) & (df.lang == lang)]
#              .groupby("ratio")[["kv_saved_total", "pct_ppl"]].mean().sort_values("kv_saved_total"))
#         ax.plot(np.r_[0, s.kv_saved_total] * 100, np.r_[0, s.pct_ppl], marker="o", label=m)
#     if lang == "nepali":
#         ax.axvline(r_eq * 100, ls="--", c="grey")
#         ax.text(r_eq * 100, ax.get_ylim()[1] * 0.9, " equal memory\n to English", fontsize=8)
#     ax.set_title(lang.capitalize())
#     ax.set_xlabel("KV entries removed (% of total cache)")
#     ax.grid(alpha=0.3)
# axes[0].set_ylabel("Perplexity change vs uncompressed (%)")
# axes[1].legend(fontsize=8)
# plt.tight_layout()
# plt.savefig("compress_curves.png", dpi=160)
# print("saved compress_curves.png")

"""
analyze_compress.py
===================
Summarise compress_results.csv produced by kv_compress.py.

  python analyze_compress.py compress_results.csv

Prints these things and saves compress_curves.png:
  1. Quality loss (delta NLL, % perplexity) per method / compression level / language
  2. Paired Nepali-minus-English difference at equal compression
  3. THE CONTROLS: does entropy-guided or word-boundary merging beat random / fixed / similarity /\n     SnapKV-style eviction?
  4. Equal-memory comparison: Nepali compressed to the size of English UNcompressed
"""

import sys

import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "compress_results.csv"
df = pd.read_csv(path)
df["pct_ppl"] = (np.exp(df["delta_nll"]) - 1) * 100
n_docs = df["doc"].nunique()
print(f"{path}: {len(df)} rows, {n_docs} documents per language\n")
if n_docs < 15:
    print("NOTE: fewer than ~15 documents, treat p-values as indicative only.\n")

pd.set_option("display.width", 200)

# ---- 1. quality loss ------------------------------------------------------
print("=== 1. Mean % perplexity change vs uncompressed (lower is better) ===")
print(df.pivot_table(index=["method", "ratio"], columns="lang",
                     values="pct_ppl", aggfunc="mean").round(2), "\n")
print("=== 1b. Next-token agreement with the uncompressed model (higher is better) ===")
print(df.pivot_table(index=["method", "ratio"], columns="lang",
                     values="top1_agree", aggfunc="mean").round(3), "\n")
print("=== 1c. KL divergence from the uncompressed model, nats/token (lower is better) ===")
print(df.pivot_table(index=["method", "ratio"], columns="lang",
                     values="kl", aggfunc="mean").round(4), "\n")
if df["gen_prefix_frac"].notna().any():
    print("=== 1d. Free-running greedy generation: fraction of tokens before first divergence ===")
    print(df.pivot_table(index=["method", "ratio"], columns="lang",
                         values="gen_prefix_frac", aggfunc="mean").round(3), "\n")


# ---- helpers for paired tests ---------------------------------------------
def paired(a, b):
    """Paired comparison of two aligned Series; returns mean diff and p-value."""
    d = (a - b).dropna()
    if len(d) < 3 or d.std() == 0:
        return d.mean() if len(d) else np.nan, np.nan, len(d)
    return d.mean(), stats.wilcoxon(d).pvalue if len(d) >= 6 else stats.ttest_1samp(d, 0).pvalue, len(d)


# ---- 2. Nepali minus English at equal compression --------------------------
print("=== 2. Nepali minus English, delta NLL (negative = Nepali tolerates compression better) ===")
wide = df.pivot_table(index=["method", "ratio", "doc"], columns="lang", values="delta_nll")
rows = []
for (m, r), g in wide.groupby(level=[0, 1]):
    mean, p, n = paired(g["nepali"], g["english"])
    rows.append({"method": m, "ratio": r, "ne_minus_en": round(mean, 4), "p": round(p, 3), "n": n})
print(pd.DataFrame(rows).to_string(index=False), "\n")

# ---- 3. controls and head-to-head comparisons ------------------------------
print("=== 3. Head-to-head at equal compression, delta NLL (negative = FIRST method is better) ===")
pairs = [("merge_entropy", "merge_random"),     # does entropy-guided selection beat chance?
         ("merge_word", "merge_random"),        # does word-boundary merging beat chance?
         ("merge_word", "merge_entropy"),       # fertility-aware vs entropy-guided
         ("merge_entropy", "merge_fixed"), ("merge_entropy", "merge_sim"),
         ("evict_snapkv", "evict_window"),      # does SnapKV scoring beat a plain recency window?
         ("evict_snapkv_word", "evict_snapkv"), # does word-aligned eviction beat token-level SnapKV?
         ("evict_snapkv_word", "evict_window"),
         ("merge_word", "evict_snapkv"),        # vs the RocketKV first-stage family
         ("merge_entropy", "evict_snapkv"), ("merge_entropy", "evict_h2o")]
idx = ["lang", "ratio", "doc"]
rows = []
for first, second in pairs:
    if first not in df.method.unique() or second not in df.method.unique():
        continue
    x = df[df.method == first].set_index(idx)["delta_nll"]
    y = df[df.method == second].set_index(idx)["delta_nll"]
    for (lang, r), _ in x.groupby(level=[0, 1]):
        mean, p, n = paired(x.loc[lang, r], y.loc[lang, r])
        rows.append({"first": first, "second": second, "lang": lang, "ratio": r,
                     "diff": round(mean, 4), "p": round(p, 3), "n": n})
print(pd.DataFrame(rows).to_string(index=False), "\n")

# ---- 4. equal-memory comparison ---------------------------------------------
ctx = df.groupby("lang")["ctx_len"].mean()
r_eq = 1 - ctx["english"] / ctx["nepali"]
print("=== 4. Equal-memory comparison ===")
print(f"Mean context: English {ctx['english']:.0f} tokens, Nepali {ctx['nepali']:.0f} tokens "
      f"-> Nepali must drop {r_eq:.0%} of its KV entries to match English's uncompressed size.")
rows = []
for m in df.method.unique():
    s = (df[(df.method == m) & (df.lang == "nepali")]
         .groupby("ratio")[["kv_saved_total", "pct_ppl"]].mean().sort_values("kv_saved_total"))
    xs = np.r_[0, s.kv_saved_total.values]
    ys = np.r_[0, s.pct_ppl.values]
    inside = xs.max() >= r_eq
    s2 = (df[(df.method == m) & (df.lang == "nepali")]
          .groupby("ratio")[["kv_saved_total", "top1_agree"]].mean().sort_values("kv_saved_total"))
    t1 = np.r_[1.0, s2.top1_agree.values]
    rows.append({"method": m, "nepali_pct_ppl_at_equal_memory":
                 round(float(np.interp(r_eq, xs, ys)), 2) if inside else np.nan,
                 "nepali_top1_agree_at_equal_memory":
                 round(float(np.interp(r_eq, np.r_[0, s2.kv_saved_total.values], t1)), 3) if inside else np.nan,
                 "note": "" if inside else f"grid stops at {xs.max():.0%}; add a larger --ratios value"})
print(pd.DataFrame(rows).to_string(index=False))
print("\nRough reading guide (my own thresholds, not a standard): <2% = effectively lossless, "
      "2-5% = usable, >5% = noticeable loss.\n")

# ---- plot -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, lang in zip(axes, ["english", "nepali"]):
    for m in df.method.unique():
        s = (df[(df.method == m) & (df.lang == lang)]
             .groupby("ratio")[["kv_saved_total", "pct_ppl"]].mean().sort_values("kv_saved_total"))
        ax.plot(np.r_[0, s.kv_saved_total] * 100, np.r_[0, s.pct_ppl], marker="o", label=m)
    if lang == "nepali":
        ax.axvline(r_eq * 100, ls="--", c="grey")
        ax.text(r_eq * 100, ax.get_ylim()[1] * 0.9, " equal memory\n to English", fontsize=8)
    ax.set_title(lang.capitalize())
    ax.set_xlabel("KV entries removed (% of total cache)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Perplexity change vs uncompressed (%)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("compress_curves.png", dpi=160)
print("saved compress_curves.png")




