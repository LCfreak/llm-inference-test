"""
analyze_needle.py  -  summarise needle_results.csv from `kv_compress.py --needle`.

  python analyze_needle.py needle_results.csv

Reports how often the compressed cache still lets the model recall a planted 4-digit secret.
If the UNcompressed model can't do it either (exact_base low), the probe is uninformative.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "needle_results.csv")
pd.set_option("display.width", 200)
n = df["doc"].nunique()
print(f"{len(df)} rows, {n} documents per language, protect_last = {df['protect_last'].iloc[0]}\n")

base = df.groupby(["lang", "doc"])[["exact_base", "tok_acc_base"]].first().groupby("lang").mean()
print("=== Uncompressed baseline (sanity check) ===")
print(base.round(2).to_string(), "\n")
if (base["exact_base"] < 0.8).any():
    print("WARNING: the uncompressed model recalls the secret in fewer than 80% of documents for at least one\n"
          "language, so the probe is weak there. Read the compressed numbers relative to the baseline.\n")

print("=== Exact recall of the full secret (higher is better) ===")
print(df.pivot_table(index=["method", "ratio"], columns="lang", values="exact", aggfunc="mean").round(2), "\n")
print("=== Per-token accuracy on the secret ===")
print(df.pivot_table(index=["method", "ratio"], columns="lang", values="tok_acc", aggfunc="mean").round(2), "\n")

ctx = df.groupby("lang")["ctx_len"].mean()
r_eq = 1 - ctx["english"] / ctx["nepali"]
print(f"=== Equal-memory point: Nepali keeps {1 - r_eq:.0%} of its entries (drops {r_eq:.0%}) ===")
rows = []
for m in df.method.unique():
    s = (df[(df.method == m) & (df.lang == "nepali")].groupby("ratio")[["kv_saved_total", "exact"]]
         .mean().sort_values("kv_saved_total"))
    xs, ys = np.r_[0, s.kv_saved_total.values], np.r_[base.loc["nepali", "exact_base"], s.exact.values]
    rows.append({"method": m, "nepali_exact_recall_at_equal_memory":
                 round(float(np.interp(r_eq, xs, ys)), 2) if xs.max() >= r_eq else np.nan})
print(pd.DataFrame(rows).to_string(index=False))