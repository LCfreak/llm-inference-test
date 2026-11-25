# PAPER — outline, figures, claims ledger

Target: **ARR January 2027 cycle → ACL 2027** (8 pages + limitations/references).
Interim: arXiv v1 on 27 Nov, v2 on 15 Dec.

---

## 1. Title

Working title:

> **Equal-Memory KV Compression: Tokenizer Fertility Predicts Cache Damage Across Scripts**

Alternatives, in order of preference:

1. *The Memory You Cannot Afford: Language-Fair KV-Cache Compression at Equal Memory*
2. *Fertility, Not Frequency: Why Fixed-Ratio KV Compression Is Unfair Across Languages*
3. *Seven Times the Cache: Tokenizer Fertility and KV Compression in Six Scripts*

Do **not** keep "for Devanagari" in the title: Tamil and Amharic are the hardest
cases in the data (7.6×), and Nepali is not the extreme.

---

## 2. Abstract skeleton

Fill every `{{...}}` from `out/tables/*.csv`; never type a number by hand.

> Byte-pair tokenizers trained on English-skewed corpora fragment non-Latin scripts
> into far more tokens than English: on FLORES-200, the Llama-3.2 tokenizer needs
> {{2.5}}× to {{7.6}}× more tokens for the same content across Nepali, Hindi,
> Bengali, Tamil and Amharic. In serving systems, that ratio is a memory ratio, so
> a request in a high-fertility language occupies {{2.5}}–{{7.6}}× the KV cache and
> admits {{1/7.6}}–{{1/2.5}} of the concurrent requests at a fixed budget. KV-cache
> compression is the natural remedy, but it is almost always evaluated at a fixed
> compression ratio, which silently asks high-fertility languages to discard far
> more content than English. We introduce an **equal-memory** protocol:
> compress each language by $r_\ell = 1 - 1/R_\ell$ so that its compressed cache
> holds exactly as many entries as uncompressed English. Across six languages and
> two tokenizers, we find that (i) quality loss at equal memory grows monotonically
> with fertility ({{slope}} %PPL per log-unit expansion, $R^2 = {{r2}}$), (ii)
> query-aware eviction stays within {{x}}% perplexity of the uncompressed model
> while every vector-merging method degrades by {{y}}–{{z}}%, and (iii) with
> identical selection scores, merging instead of evicting accounts for
> {{delta}} points of that gap, isolating the operator from the selection rule.
> On a T4 with PagedAttention, measured concurrency falls from {{c_en}} to
> {{c_am}} requests at a {{budget}} GB KV budget. We release the sweep, the
> protocol and the analysis as a reproducible artifact.

Word budget: 180–220.

---

## 3. Contributions (as they will appear in §1)

1. An equal-memory evaluation protocol for cross-lingual KV compression, with the
   analytic ratio $r_\ell = 1 - 1/R_\ell$.
2. A six-language, two-tokenizer dose–response study showing damage is continuous
   in fertility, not specific to one script.
3. An operator-isolation control (`merge_snapkv`) that separates selection from
   aggregation under identical importance scores.
4. A measured GPU serving penalty under PagedAttention, replacing the previous
   simulation-only claim.

---

## 4. Section outline

| § | Content | Length | Evidence |
|---|---|---|---|
| 1 | Intro: fertility → memory → concurrency; the unfair fixed-ratio baseline | 1 p | Fig. 1 |
| 2 | Related work: PagedAttention/serving, KV eviction/merging, tokenizer fairness | 0.75 p | — |
| 3 | Method: protocol, policies, equal-memory definition, leakage-free scoring | 1.5 p | Alg. 1 |
| 4.1 | Fertility and analytic capacity, six languages, two tokenizers | 0.75 p | Tab. 1, Fig. 1 |
| 4.2 | Quality at equal memory: eviction vs merging | 1 p | Tab. 2, Fig. 2, Fig. 3 |
| 4.3 | Dose–response regression | 0.5 p | Fig. 3 |
| 4.4 | Operator isolation (`merge_snapkv`) | 0.5 p | Fig. 4 |
| 4.5 | Retrieval and generation | 0.75 p | Tab. 3, Fig. 5 |
| 4.6 | GPU serving measurement | 0.75 p | Tab. 4 |
| 5 | Discussion: post-RoPE phase interference, word alignment, block fragmentation | 0.75 p | — |
| 6 | Threats to validity | 0.5 p | — |
| 7 | Conclusion | 0.25 p | — |

---

## 5. Figures and tables

| ID | File | Caption must state |
|---|---|---|
| Fig. 1 | `out/figures/fig1_fertility.png` | tokens vs English per language, per tokenizer, with the equal-memory removal % annotated |
| Fig. 2 | `out/figures/fig2_ppl_vs_ratio.png` | perplexity increase vs fraction of entries removed, one line per language; dotted vertical at each $r_\ell$ |
| Fig. 3 | `out/figures/fig3_equal_memory.png` | perplexity increase **at equal memory** vs log expansion; regression line |
| Fig. 4 | `out/figures/fig4_operator.png` | `evict_snapkv` vs `merge_snapkv` per language, same scores |
| Fig. 5 | `out/figures/fig5_recall.png` | exact recall heatmap, policy × (language, ratio) |
| Fig. 6 | `out/figures/fig6_fixed_budget.png` | ΔPPL at a **fixed token budget** $C$, with the fraction of each cache actually removed printed under every language; English at 0% is the point |
| Tab. 1 | `out/tables/expansion.csv` | model, tokenizer, tokens/sentence, expansion, $r_\ell$ |
| Tab. 2 | `out/tables/equal_memory.csv` | ΔPPL % with 95% bootstrap CI, top-1 agreement, n |
| Tab. 3 | `out/tables/needle_recall.csv` | exact recall, n, and the absolute retained-entry count (not just ratio) |
| Tab. 4 | `out/serve_*.csv` | measured concurrency, throughput, TTFT/TPOT, hardware, dtype |

Every caption: model, dtype, hardware, $n$, number of seeds. If a number is
interpolated rather than measured, say so in the caption.

---

## 6. Claims ledger

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Llama-3.2 needs 2.5–7.6× more tokens for the five non-English languages | `fertility_Llama-3.2-1B-Instruct.csv`, 2,009 sentences | **Measured** |
| C2 | At a 2 GB KV budget, capacity falls from 209 (EN) to 29 (TA/AM) requests | `serve_bench.py --mode tokens` | **Analytic** — must be confirmed by E7 |
| C3 | Quality loss at equal memory grows with fertility | `regression.csv` | **Pending E2/E3** |
| C4 | Query-aware eviction dominates merging at equal memory | `equal_memory.csv` | **Pending E2** |
| C5 | With identical scores, merging is worse than evicting | `paired_evict_vs_merge.csv` | **Pending E2** — pilot (n=4) shows a small, non-significant gap; be ready to report "selection dominates" |
| C6 | Word-aligned eviction helps at high compression | `needle_recall.csv` | **Pending E4**, $n=8$ in old data, $p=0.50$ |
| C7 | Post-RoPE phase interference causes merging failure | C5 + pre/post-RoPE ablation | **Hypothesis** unless C5 is large and the ablation runs |
| C8 | Compression yields real VRAM/latency gains under PagedAttention | E7 | **Pending** |
| C9 | At a fixed 1024-token budget, English is not compressed at all while Tamil loses ~half its cache | `fixed_budget.csv`, E9 | **Pending E2** (arithmetic already exact: $1 - C/P$) |
| C10 | Attention-free selection (`evict_vk`, `evict_keynorm`) changes the cross-language ranking relative to query-aware selection | `equal_memory.csv` rows for the two new baselines | **Pending E2** |

Rules:

- C7 may appear in Discussion as a hypothesis, never in the abstract or conclusion.
- C6 may not be called "restores 100% recall" at $n=8$; report the counts and the
  sign test, and label it suggestive.
- If E7 fails to run, C2 stays "analytic" and §4.6 is cut, not softened.

---

## 7. Related work (verify before submission)

**Serving and memory management.** Kwon et al., *PagedAttention/vLLM*, SOSP 2023.
Agrawal et al., *Sarathi-Serve*, OSDI 2024.

**KV eviction and compression.** Xiao et al., *StreamingLLM*, ICLR 2024. Zhang et
al., *H2O*, NeurIPS 2023. Li et al., *SnapKV*, NeurIPS 2024. Liu et al.,
*Scissorhands*, NeurIPS 2023. Liu et al., *KIVI*, ICML 2024. Behnam et al.,
*RocketKV*, ICML 2025. Chitty-Venkata et al., *PagedEviction*, arXiv:2509.04377,
2025 — the block-aligned, attention-free line whose fixed 256–4096-token budget
is the object of our §4.1 critique, and whose $\|V\|_2/\|K\|_2$ score we adopt as a
baseline. Devoto et al., *A Simple and Effective L2 Norm-Based Strategy*, 2024 —
inverse key norm, our second attention-free baseline.

**Tokenizer fairness and multilingual cost.** Petrov et al., *Language Model
Tokenizers Introduce Unfairness Between Languages*, NeurIPS 2023. Ahia et al.,
*Do All Languages Cost the Same?*, EMNLP 2023. Rust et al., *How Good Is Your
Tokenizer?*, ACL 2021.

**Data and evaluation.** Goyal et al., *FLORES-200*, 2022. Hsieh et al., *RULER*,
2024 (for the retrieval-probe design).

Positioning sentence to reuse: *fixed-ratio KV compression benchmarks compare
languages at unequal memory; we show that this reverses the ranking of methods for
high-fertility scripts.*

---

## 8. Threats to validity (draft)

1. **Model scale.** 1B and 3B only; 7B+ untested. Fertility ratios are tokenizer
   properties and transfer, but the absolute damage at 87% removal may not.
2. **Probe simplicity.** The needle query echoes the planted sentence and sits in
   the observation window; query-aware policies get an advantage that may not
   survive paraphrase. Report it.
3. **Absolute budget vs ratio.** At 87% removal a Tamil cache still holds more
   entries than English at 60%. Always report absolute retained entries alongside
   the ratio.
4. **Recency confound.** Addressed by E6 (`--protect-last 32`); report both.
5. **Hardware.** T4, FP16, 16 GB. PagedAttention block size 16 interacts with
   word-aligned eviction (internal fragmentation) — measure or flag.
6. **Statistics.** Paired tests over documents, Holm across policies; documents are
   not independent (consecutive FLORES sentences), so treat $p$-values as
   descriptive and lean on effect sizes with CIs.

---

## 9. ARR submission checklist

- [ ] ACL style, anonymised, 8 pages excluding limitations/references.
- [ ] Limitations section (required).
- [ ] Responsible NLP checklist submitted within 48 h of the deadline.
- [ ] All authors have OpenReview profiles **with ORCID** (required from Oct 2026).
- [ ] Designated contributor registered at submission.
- [ ] arXiv preprint allowed — post v2 before submitting.
- [ ] Artifact: `src/`, `data/flores200/parallel_6lang.csv`, `out/*.csv`, one-command
      reproduction in `docs/RUNBOOK.md`.
- [ ] Every abstract number traced to a row in `out/tables/`.
