"""
analyze.py — statistics and figures for the language-fair KV-cache study.

Reads the CSVs written by src/kv_compress.py (and optionally src/serve_bench.py),
and produces the tables and figures the manuscript needs.

Everything is organised around one idea: compression must be compared at EQUAL
MEMORY, not at equal compression ratio. For a document whose English context is
P_en tokens and whose language-l context is P_l tokens, the equal-memory ratio is

    r_l = 1 - P_en / P_l

so that (1 - r_l) * P_l = P_en: the compressed non-English cache holds exactly as
many entries as the uncompressed English cache. At a fixed ratio r, a high-fertility
language is being asked to throw away far more semantic content than English, which
is the comparison this script avoids.

Usage
-----
  python src/analyze.py --inputs out/compress_*.csv --outdir out
  python src/analyze.py --inputs out/compress_llama1b.csv out/compress_qwen3b.csv \
                        --needle out/needle_llama1b.csv --gen out/gen_llama1b.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

LANGS = ["english", "nepali", "hindi", "bengali", "tamil", "amharic"]
PALETTE = {
    "english": "#4C72B0", "nepali": "#DD8452", "hindi": "#55A868",
    "bengali": "#C44E52", "tamil": "#8172B3", "amharic": "#937860",
}


# ---------------------------------------------------------------- statistics

def boot_ci(x, n=4000, seed=0, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the mean over documents."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return 1.0


# ---------------------------------------------------------------- data loading

def load(paths, required=True):
    frames = []
    for p in paths:
        for f in sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]:
            if os.path.exists(f):
                frames.append(pd.read_csv(f))
    if not frames:
        if required:
            raise SystemExit(f"no input CSVs found for {paths}")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "model" not in df.columns:
        df["model"] = "model"
    for col in ("ratio", "delta_nll", "ctx_len", "ref_ctx_len", "equal_mem_ratio"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_fertility(paths):
    """Read data/flores200/fertility_<model>.csv files (no sweep needed)."""
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        model = os.path.basename(p).replace("fertility_", "").replace(".csv", "")
        d = d.rename(columns={"expansion_vs_english": "expansion"})
        d["model"] = model
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def expansion_table(df):
    """Per (model, lang): mean token expansion and equal-memory ratio, measured on ctx_len."""
    rows = []
    for (model, lang), g in df.groupby(["model", "lang"]):
        if lang == "english" or "ref_ctx_len" not in g:
            continue
        g = g.dropna(subset=["ctx_len", "ref_ctx_len"]).drop_duplicates(subset=["doc"])
        ratio = (g["ctx_len"] / g["ref_ctx_len"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) == 0:
            continue
        rows.append({"model": model, "lang": lang, "n_docs": len(ratio),
                     "expansion": float(ratio.mean()),
                     "equal_memory_ratio": float(1 - 1 / ratio.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- equal memory

def interpolate_at_equal_memory(df):
    """For every (model, lang, doc, method) evaluate delta_nll at that doc's r_l.

    Uses the grid points already computed, so it works even when --equal-memory was
    not passed. Falls back to the exact equal_mem row when present.
    """
    out = []
    keys = ["model", "lang", "doc", "method", "protect_last"]
    for key, g in df.groupby(keys):
        g = g.dropna(subset=["ratio", "delta_nll"]).sort_values("ratio")
        if len(g) == 0:
            continue
        em = g["equal_mem_ratio"].iloc[0]
        if not np.isfinite(em):
            continue
        exact = g[g.get("ratio_kind", "") == "equal_mem"] if "ratio_kind" in g else g.iloc[0:0]
        if len(exact):
            v = float(exact["delta_nll"].iloc[0])
            top1 = float(exact["top1_agree"].iloc[0]) if "top1_agree" in exact else np.nan
        elif g["ratio"].min() <= em <= g["ratio"].max():
            v = float(np.interp(em, g["ratio"], g["delta_nll"]))
            top1 = float(np.interp(em, g["ratio"], g["top1_agree"])) \
                if "top1_agree" in g else np.nan
        else:
            continue
        out.append({"model": key[0], "lang": key[1], "doc": key[2], "method": key[3],
                    "equal_mem_ratio": em, "delta_nll": v, "top1_agree": top1})
    return pd.DataFrame(out)


def equal_memory_table(em, exp):
    rows = []
    for (model, method, lang), g in em.groupby(["model", "method", "lang"]):
        x = g["delta_nll"].to_numpy()
        lo, hi = boot_ci(x)
        e = exp[(exp.model == model) & (exp.lang == lang)]
        rows.append({
            "model": model, "method": method, "lang": lang,
            "expansion": float(e["expansion"].iloc[0]) if len(e) else np.nan,
            "equal_mem_ratio": float(g["equal_mem_ratio"].mean()),
            "n_docs": int(np.isfinite(x).sum()),
            "delta_nll_pct": 100 * (np.expm1(np.nanmean(x))),
            "ci_lo_pct": 100 * (np.expm1(lo)), "ci_hi_pct": 100 * (np.expm1(hi)),
            "top1_agree": float(np.nanmean(g["top1_agree"])) if g["top1_agree"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def regression(em_tab, exp):
    """Delta PPL at equal memory vs log token expansion, one line per (model, method)."""
    from scipy.stats import linregress
    rows = []
    for (model, method), g in em_tab.groupby(["model", "method"]):
        g = g.dropna(subset=["expansion", "delta_nll_pct"])
        g = g[g.lang != "english"]
        if len(g) < 3:
            continue
        r = linregress(np.log(g["expansion"]), g["delta_nll_pct"])
        rows.append({"model": model, "method": method, "n_langs": len(g),
                     "slope_pct_per_log_expansion": r.slope, "intercept": r.intercept,
                     "r2": r.rvalue ** 2, "p": r.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["model", "slope_pct_per_log_expansion"])
    return out


def paired_tests(em, a, b):
    """Paired per-document comparison of method a vs method b, Holm across languages."""
    rows = []
    for (model, lang), g in em.groupby(["model", "lang"]):
        pa = g[g.method == a].set_index("doc")["delta_nll"]
        pb = g[g.method == b].set_index("doc")["delta_nll"]
        j = pa.index.intersection(pb.index)
        if len(j) < 3:
            continue
        rows.append({"model": model, "lang": lang, "n": len(j),
                     "mean_a_pct": 100 * np.expm1(pa[j].mean()),
                     "mean_b_pct": 100 * np.expm1(pb[j].mean()),
                     "p_raw": wilcoxon_p(pa[j], pb[j])})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_raw"])
    return out


def needle_table(df):
    if "exact" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["model", "lang", "method", "ratio"], as_index=False).agg(
        recall=("exact", "mean"), n=("exact", "size"))
    return g


def gen_table(df):
    """Free-running generation: how much of the uncompressed greedy trajectory survives."""
    if df.empty or "gen_prefix_frac" not in df.columns:
        return pd.DataFrame()
    return (df.dropna(subset=["gen_prefix_frac"])
              .groupby(["model", "lang", "method", "ratio"], as_index=False)
              .agg(gen_prefix_frac=("gen_prefix_frac", "mean"),
                   delta_nll_pct=("delta_nll", lambda x: 100 * np.expm1(np.nanmean(x))),
                   n=("gen_prefix_frac", "size")))


def serve_table(df):
    """Serving sweep summary: capacity and latency per language at the largest batch
    that did not OOM, plus the smallest."""
    if df.empty or "lang" not in df.columns:
        return pd.DataFrame()
    ok = df[df.get("oom", False) != True].copy() if "oom" in df.columns else df.copy()
    if ok.empty:
        return pd.DataFrame()
    ok["batch"] = pd.to_numeric(ok["batch"], errors="coerce")
    rows = []
    for (model, lang), g in ok.groupby(["model", "lang"]):
        g = g.sort_values("batch")
        b1 = g[g.batch == g.batch.min()]
        bmax = g[g.batch == g.batch.max()]
        rows.append({"model": model, "lang": lang,
                     "max_batch_no_oom": int(g.batch.max()),
                     "throughput_tok_s_max": float(g["throughput_tok_s"].max()),
                     "ttft_ms": float(g["ttft_s"].iloc[0]) * 1e3 if "ttft_s" in g else np.nan,
                     "tpot_ms": float(g["tpot_s"].iloc[0]) * 1e3 if "tpot_s" in g else np.nan,
                     "peak_vram_gb": float(g["peak_vram_gb"].max()) if "peak_vram_gb" in g else np.nan,
                     "preemptions_total": int(g["preemptions"].fillna(0).sum())
                     if "preemptions" in g else 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_fertility(exp, outdir):
    plt = _plt()
    models = sorted(exp.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        e = exp[exp.model == model].sort_values("expansion")
        ax.barh(e.lang, e.expansion, color=[PALETTE.get(l, "#777") for l in e.lang])
        for y, (v, r) in enumerate(zip(e.expansion, e.equal_memory_ratio)):
            ax.text(v + 0.08, y, f"{v:.2f}x  (drop {100*r:.0f}%)", va="center", fontsize=8)
        ax.axvline(1, color="k", lw=0.8)
        ax.set_xlabel("tokens vs English")
        ax.set_title(model, fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_fertility.png")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return path


def fig_ppl_curve(df, outdir, method="evict_snapkv"):
    plt = _plt()
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        d = df[(df.model == model) & (df.method == method)]
        for lang in LANGS:
            g = d[d.lang == lang].groupby("ratio")["delta_nll"].mean()
            if len(g) < 2:
                continue
            ax.plot(g.index, 100 * np.expm1(g.values), marker="o", ms=3,
                    color=PALETTE[lang], label=lang)
            if lang != "english":
                em = d[d.lang == lang]["equal_mem_ratio"].mean()
                ax.axvline(em, color=PALETTE[lang], ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel("KV entries removed (fraction)")
        ax.set_ylabel("perplexity increase (%)")
        ax.set_title(f"{model} — {method}", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_ppl_vs_ratio.png")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return path


def fig_equal_memory(em_tab, outdir):
    plt = _plt()
    models = sorted(em_tab.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        d = em_tab[(em_tab.model == model) & (em_tab.lang != "english")]
        for method in ["evict_snapkv", "evict_snapkv_word", "merge_snapkv", "merge_entropy"]:
            g = d[d.method == method].sort_values("expansion")
            if len(g) < 2:
                continue
            ax.plot(g.expansion, g.delta_nll_pct, marker="o", ms=4, label=method)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("token expansion vs English (x)")
        ax.set_ylabel("perplexity increase at equal memory (%)")
        ax.set_title(model, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_equal_memory.png")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return path


def fig_operator(em, outdir, model=None):
    plt = _plt()
    if model is None:
        model = sorted(em.model.unique())[0]
    d = em[(em.model == model) & (em.lang != "english")]
    langs = [l for l in LANGS if l in set(d.lang)]
    a = d[d.method == "evict_snapkv"].groupby("lang")["delta_nll"].mean()
    b = d[d.method == "merge_snapkv"].groupby("lang")["delta_nll"].mean()
    x = np.arange(len(langs)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(x - w / 2, [100 * np.expm1(a.get(l, np.nan)) for l in langs], w, label="evict_snapkv")
    ax.bar(x + w / 2, [100 * np.expm1(b.get(l, np.nan)) for l in langs], w, label="merge_snapkv")
    ax.set_xticks(x); ax.set_xticklabels(langs, rotation=20)
    ax.set_ylabel("perplexity increase at equal memory (%)")
    ax.set_title("same selection, different operator", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_operator.png")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return path


def fig_recall(nt, outdir):
    plt = _plt()
    if nt.empty:
        return None
    models = sorted(nt.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        d = nt[nt.model == model]
        piv = d.pivot_table(index="method", columns=["lang", "ratio"], values="recall")
        im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{l[:3]}\n{r:.2f}" for l, r in piv.columns], fontsize=6)
        ax.set_title(f"{model} — exact recall", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_recall.png")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="compress_*.csv files from kv_compress.py")
    ap.add_argument("--fertility", nargs="*", default=[],
                    help="data/flores200/fertility_*.csv; plots Fig. 1 without a sweep")
    ap.add_argument("--needle", nargs="*", default=[])
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--serve", nargs="*", default=[])
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    tab, fig = os.path.join(args.outdir, "tables"), os.path.join(args.outdir, "figures")

    if args.fertility:
        fexp = load_fertility(args.fertility)
        p = fig_fertility(fexp, fig)
        print("fertility figure:", p)
        if not args.inputs:
            return

    df = load(args.inputs)
    print(f"loaded {len(df)} rows from {df.model.nunique()} model(s): {sorted(df.model.unique())}")

    exp = expansion_table(df)
    exp.to_csv(os.path.join(tab, "expansion.csv"), index=False)

    em = interpolate_at_equal_memory(df)
    em_tab = equal_memory_table(em, exp)
    em_tab.to_csv(os.path.join(tab, "equal_memory.csv"), index=False)

    reg = regression(em_tab, exp)
    reg.to_csv(os.path.join(tab, "regression.csv"), index=False)

    pt = paired_tests(em, "evict_snapkv", "merge_snapkv")
    pt.to_csv(os.path.join(tab, "paired_evict_vs_merge.csv"), index=False)

    nt = pd.DataFrame()
    if args.needle:
        nd = load(args.needle, required=False)
        nt = needle_table(nd)
        nt.to_csv(os.path.join(tab, "needle_recall.csv"), index=False)

    gt = pd.DataFrame()
    if args.gen:
        gd = load(args.gen, required=False)
        gt = gen_table(gd)
        gt.to_csv(os.path.join(tab, "gen_prefix.csv"), index=False)

    st = pd.DataFrame()
    if args.serve:
        sd = load(args.serve, required=False)
        st = serve_table(sd)
        st.to_csv(os.path.join(tab, "serve.csv"), index=False)

    print("\n== token expansion ==")
    print(exp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== equal-memory quality (perplexity increase %) ==")
    show = em_tab[["model", "method", "lang", "equal_mem_ratio", "delta_nll_pct",
                   "ci_lo_pct", "ci_hi_pct", "top1_agree"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== dose-response regression (slope in %PPL per log expansion) ==")
    print(reg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== evict_snapkv vs merge_snapkv (same selection scores) ==")
    print(pt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not nt.empty:
        print("\n== needle recall (exact) ==")
        print(nt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not gt.empty:
        print("\n== generation: fraction of uncompressed greedy prefix reproduced ==")
        print(gt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not st.empty:
        print("\n== serving (measured) ==")
        print(st.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not args.no_figures:
        paths = [fig_fertility(exp, fig), fig_ppl_curve(df, fig),
                 fig_equal_memory(em_tab, fig), fig_operator(em, fig)]
        if not nt.empty:
            paths.append(fig_recall(nt, fig))
        print("\nfigures:")
        for p in paths:
            if p:
                print("  " + p)


if __name__ == "__main__":
    main()
