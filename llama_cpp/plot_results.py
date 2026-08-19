# """
# Research-grade plots for the English/Hindi/Nepali tokenization-premium benchmark.

# Reads results_summary.json (+ optionally results_raw.csv) produced by
# benchmark_multi.py and produces:

#   1. fig1_mean_tokens_per_prompt.png   -- bar + error bars (std), n=15/lang
#   2. fig2_token_distribution_box.png   -- box plot of per-prompt token counts
#   3. fig3_tokens_per_char.png          -- normalized fertility (tokens/char)
#   4. fig4_length_bucket_breakdown.png  -- grouped bars by short/medium/long
#   5. fig5_throughput_and_overhead.png  -- tokens/sec + relative overhead (twin axes)
#   6. summary_table.png                 -- clean table image for slides/paper

# Does NOT require llama_cpp -- run this after benchmark_multi.py, on any
# machine with matplotlib/numpy/pandas.

# Usage:
#     python plot_results.py --summary results_summary.json --raw results_raw.csv --outdir figures
# """

# import argparse
# import json
# import os

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np

# try:
#     import pandas as pd
# except ImportError:
#     pd = None

# # ---- research-plot styling -------------------------------------------------
# plt.rcParams.update({
#     "figure.dpi": 150,
#     "savefig.dpi": 300,
#     "font.size": 11,
#     "font.family": "sans-serif",
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.grid": True,
#     "grid.alpha": 0.25,
#     "grid.linestyle": "--",
#     "legend.frameon": False,
# })

# LANG_ORDER = ["English", "Hindi", "Nepali"]
# LANG_COLORS = {"English": "#4C72B0", "Hindi": "#DD8452", "Nepali": "#55A868"}
# BUCKET_ORDER = ["short", "medium", "long"]


# def load_data(summary_path, raw_path):
#     with open(summary_path, "r", encoding="utf-8") as f:
#         summary = json.load(f)
#     raw_df = None
#     if raw_path and os.path.exists(raw_path) and pd is not None:
#         raw_df = pd.read_csv(raw_path)
#     langs = [l for l in LANG_ORDER if l in summary]
#     return summary, raw_df, langs


# def fig1_mean_tokens(summary, langs, outdir):
#     means = [summary[l]["mean_tokens_per_prompt"] for l in langs]
#     stds = [summary[l]["std_tokens_per_prompt"] for l in langs]
#     ns = [summary[l]["n_prompts"] for l in langs]

#     fig, ax = plt.subplots(figsize=(5.5, 4.5))
#     x = np.arange(len(langs))
#     bars = ax.bar(x, means, yerr=stds, capsize=5,
#                    color=[LANG_COLORS[l] for l in langs], width=0.6,
#                    edgecolor="black", linewidth=0.6)
#     ax.set_xticks(x)
#     ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(langs, ns)])
#     ax.set_ylabel("Mean input tokens per prompt")
#     ax.set_title("Tokenization overhead across languages\n(parallel prompts, identical semantic content)")
#     for xi, m in zip(x, means):
#         ax.text(xi, m + max(means) * 0.02, f"{m:.1f}", ha="center", fontsize=10)
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "fig1_mean_tokens_per_prompt.png"))
#     plt.close(fig)


# def fig2_token_distribution(summary, langs, outdir):
#     data = [summary[l]["raw_tokens"] for l in langs]
#     fig, ax = plt.subplots(figsize=(5.5, 4.5))
#     bp = ax.boxplot(data, tick_labels=langs, patch_artist=True, widths=0.5,
#                      medianprops={"color": "black"})
#     for patch, l in zip(bp["boxes"], langs):
#         patch.set_facecolor(LANG_COLORS[l])
#         patch.set_alpha(0.75)
#     # overlay raw points (jittered) for transparency about sample size
#     rng = np.random.default_rng(0)
#     for i, (l, vals) in enumerate(zip(langs, data), start=1):
#         jitter = rng.uniform(-0.08, 0.08, size=len(vals))
#         ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color="black",
#                    alpha=0.5, zorder=3)
#     ax.set_ylabel("Input tokens per prompt")
#     ax.set_title("Per-prompt token count distribution (15 prompts/language)")
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "fig2_token_distribution_box.png"))
#     plt.close(fig)


# def fig3_tokens_per_char(summary, langs, outdir):
#     vals = [summary[l]["mean_tokens_per_char"] for l in langs]
#     fig, ax = plt.subplots(figsize=(5.5, 4.5))
#     x = np.arange(len(langs))
#     ax.bar(x, vals, color=[LANG_COLORS[l] for l in langs], width=0.6,
#            edgecolor="black", linewidth=0.6)
#     ax.set_xticks(x)
#     ax.set_xticklabels(langs)
#     ax.set_ylabel("Tokens per character (fertility)")
#     ax.set_title("Tokenizer fertility, length-normalized\n(controls for prompt length differences)")
#     for xi, v in zip(x, vals):
#         ax.text(xi, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=10)
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "fig3_tokens_per_char.png"))
#     plt.close(fig)


# def fig4_length_bucket(summary, langs, outdir):
#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     width = 0.25
#     x = np.arange(len(BUCKET_ORDER))
#     for i, l in enumerate(langs):
#         vals = [summary[l]["by_length_bucket"].get(b, {}).get("mean_tokens", 0) for b in BUCKET_ORDER]
#         errs = [summary[l]["by_length_bucket"].get(b, {}).get("std_tokens", 0) for b in BUCKET_ORDER]
#         ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3,
#                label=l, color=LANG_COLORS[l], edgecolor="black", linewidth=0.5)
#     ax.set_xticks(x)
#     ax.set_xticklabels([b.capitalize() for b in BUCKET_ORDER])
#     ax.set_ylabel("Mean input tokens")
#     ax.set_title("Token overhead by prompt length bucket\n(5 parallel prompts per bucket per language)")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "fig4_length_bucket_breakdown.png"))
#     plt.close(fig)


# def fig5_throughput_overhead(summary, langs, outdir):
#     tps = [summary[l]["tokens_per_sec"] for l in langs]
#     overhead = [summary[l].get("token_overhead_vs_english", 1.0) for l in langs]

#     fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
#     x = np.arange(len(langs))
#     b1 = ax1.bar(x - 0.18, tps, width=0.36, color=[LANG_COLORS[l] for l in langs],
#                  edgecolor="black", linewidth=0.6, label="Tokens/sec (throughput)")
#     ax1.set_ylabel("Tokens / second")
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(langs)

#     ax2 = ax1.twinx()
#     ax2.plot(x + 0.0, overhead, marker="o", color="black", linestyle="--",
#               linewidth=1.5, label="Token overhead vs. English")
#     ax2.set_ylabel("Token overhead vs. English (x)")
#     ax2.grid(False)
#     ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
#     ax1.set_title("Throughput vs. tokenization overhead")
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "fig5_throughput_and_overhead.png"))
#     plt.close(fig)


# def summary_table_image(summary, langs, outdir):
#     cols = ["Language", "n", "Mean tokens/prompt", "Std", "Tokens/char",
#             "Tokens/sec", "Overhead vs EN"]
#     rows = []
#     for l in langs:
#         s = summary[l]
#         rows.append([
#             l, s["n_prompts"], f"{s['mean_tokens_per_prompt']:.2f}",
#             f"{s['std_tokens_per_prompt']:.2f}", f"{s['mean_tokens_per_char']:.4f}",
#             f"{s['tokens_per_sec']:.1f}", f"{s.get('token_overhead_vs_english', 1.0):.2f}x",
#         ])
#     fig, ax = plt.subplots(figsize=(8, 0.6 + 0.45 * len(rows)))
#     ax.axis("off")
#     table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.6)
#     for j in range(len(cols)):
#         table[0, j].set_facecolor("#333333")
#         table[0, j].set_text_props(color="white", weight="bold")
#     fig.tight_layout()
#     fig.savefig(os.path.join(outdir, "summary_table.png"), bbox_inches="tight")
#     plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--summary", default="results_summary.json")
#     parser.add_argument("--raw", default="results_raw.csv")
#     parser.add_argument("--outdir", default="figures")
#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     summary, raw_df, langs = load_data(args.summary, args.raw)

#     fig1_mean_tokens(summary, langs, args.outdir)
#     fig2_token_distribution(summary, langs, args.outdir)
#     fig3_tokens_per_char(summary, langs, args.outdir)
#     fig4_length_bucket(summary, langs, args.outdir)
#     fig5_throughput_overhead(summary, langs, args.outdir)
#     summary_table_image(summary, langs, args.outdir)

#     print(f"Saved 6 figures to ./{args.outdir}/")


# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

df_adaptive = pd.read_csv('adaptive_ctx_results.csv')
df_bench = pd.read_csv('benchmark_results.csv')
df_raw = pd.read_csv('results_raw_throughput.csv')

df = df_adaptive.merge(
    df_bench[['language', 'prompt_idx', 'length_bucket', 'median_inference_ms', 'tokens_generated_per_sec']],
    on=['language', 'prompt_idx', 'length_bucket']
).merge(
    df_raw[['language', 'prompt_idx', 'length_bucket', 'latency_ms']],
    on=['language', 'prompt_idx', 'length_bucket'],
    suffixes=('_adaptive', '_raw')
)

# Constants
BYTES_PER_KV_TOKEN = 64 * 1024 # 64 KB per token for Llama-3 8B FP16
KV_BUDGET_MB = 2048 # 2 GB VRAM budget dedicated to KV-Cache

df['kv_cache_mb_adaptive'] = (df['adaptive_n_ctx'] * BYTES_PER_KV_TOKEN) / (1024 * 1024)
df['concurrent_users_adaptive'] = KV_BUDGET_MB / df['kv_cache_mb_adaptive']
df['concurrent_users_512'] = KV_BUDGET_MB / ((512 * BYTES_PER_KV_TOKEN) / (1024 * 1024))
df['concurrent_users_2048'] = KV_BUDGET_MB / ((2048 * BYTES_PER_KV_TOKEN) / (1024 * 1024))

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# Plot A: Subword Tokenization Overhead (Tokens Per 100 Characters)
df['tokens_per_100_chars'] = 100 / df['chars_per_token']
sns.barplot(
    data=df, x='language', y='tokens_per_100_chars', hue='length_bucket',
    ax=axes[0, 0], palette='Reds_r', edgecolor='black', linewidth=0.5
)
axes[0, 0].set_title('(a) Tokenization Overhead (Tokens per 100 Chars)', fontweight='bold', loc='left')
axes[0, 0].set_ylabel('Required Tokens')
axes[0, 0].set_xlabel('Language')
axes[0, 0].grid(axis='y')

# Plot B: KV-Cache VRAM Footprint per Request
lang_order = ['English', 'Hindi', 'Nepali']
kv_adapt = df.groupby('language')['kv_cache_mb_adaptive'].mean().reindex(lang_order)

x = np.arange(len(lang_order))
width = 0.25

axes[0, 1].bar(x - width, [32.0]*3, width, label='Static n_ctx=512 (32 MB)', color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[0, 1].bar(x, [128.0]*3, width, label='Static n_ctx=2048 (128 MB)', color='#95a5a6', edgecolor='black', linewidth=0.5)
axes[0, 1].bar(x + width, kv_adapt, width, label='Adaptive Context (MB)', color='#2ecc71', edgecolor='black', linewidth=0.5)

axes[0, 1].set_title('(b) Allocated KV Cache Memory per Request', fontweight='bold', loc='left')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(lang_order)
axes[0, 1].set_ylabel('VRAM Allocated (MB)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(axis='y', which='both')

# Plot C: Max Concurrent Served Streams (VRAM Capacity)
users_adapt = df.groupby('language')['concurrent_users_adaptive'].mean().reindex(lang_order)

axes[1, 0].bar(x - width, [64]*3, width, label='Static n_ctx=512', color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[1, 0].bar(x, [16]*3, width, label='Static n_ctx=2048', color='#95a5a6', edgecolor='black', linewidth=0.5)
axes[1, 0].bar(x + width, users_adapt, width, label='Adaptive Context', color='#2ecc71', edgecolor='black', linewidth=0.5)

axes[1, 0].set_title('(c) Concurrent User Serving Capacity (2 GB KV Budget)', fontweight='bold', loc='left')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(lang_order)
axes[1, 0].set_ylabel('Max Concurrent Active Users')
axes[1, 0].legend(loc='upper right')
axes[1, 0].grid(axis='y')

# Plot D: Concurrency Scaling vs Prompt Length
sns.lineplot(
    data=df, x='length_bucket', y='concurrent_users_adaptive', hue='language',
    ax=axes[1, 1], marker='o', linewidth=2, markersize=8
)
axes[1, 1].axhline(y=64, color='#e74c3c', linestyle=':', label='Static 512 Baseline')
axes[1, 1].axhline(y=16, color='#95a5a6', linestyle='--', label='Static 2048 Baseline')
axes[1, 1].set_title('(d) Serving Concurrency Across Prompt Bucket', fontweight='bold', loc='left')
axes[1, 1].set_xlabel('Prompt Length Bucket')
axes[1, 1].set_ylabel('Max Concurrent Users')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('research_serving_capacity_benchmark.png', dpi=300)
print("Saved plot research_serving_capacity_benchmark.png")