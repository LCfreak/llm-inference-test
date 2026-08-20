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