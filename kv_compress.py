
import argparse
import heapq
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

METHODS = ["merge_entropy", "merge_random", "merge_fixed", "merge_sim", "merge_word",
           "evict_h2o", "evict_snapkv", "evict_snapkv_word", "evict_window"]


# --------------------------------------------------------------------------
# Model geometry and a hand-written Llama forward pass
# --------------------------------------------------------------------------

def geometry(model):
    c = model.config
    nH = c.num_attention_heads
    nKV = c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // nH
    return nH, nKV, hd


def rope(model, h, position_ids):
    rot = getattr(model.model, "rotary_emb", None)
    if rot is None:  # very old transformers: rotary lives inside each layer
        rot = model.model.layers[0].self_attn.rotary_emb
    return rot(h, position_ids)


def layer_qkv(model, layer, h, cos, sin):
    nH, nKV, hd = geometry(model)
    a = layer.self_attn
    x = layer.input_layernorm(h)
    B, T, _ = x.shape
    q = a.q_proj(x).view(B, T, nH, hd).transpose(1, 2)
    k = a.k_proj(x).view(B, T, nKV, hd).transpose(1, 2)
    v = a.v_proj(x).view(B, T, nKV, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def attend(model, q, k, v, add_mask):
    """q:(B,nH,Tq,hd)  k,v:(B,nKV,Tk,hd)  add_mask:(Tq,Tk) additive."""
    nH, nKV, hd = geometry(model)
    rep = nH // nKV
    kk = k.repeat_interleave(rep, dim=1)
    vv = v.repeat_interleave(rep, dim=1)
    scores = q @ kk.transpose(-1, -2) / math.sqrt(hd) + add_mask
    w = scores.softmax(dim=-1)
    return w @ vv, w


def finish_layer(layer, h, out):
    B, nH, T, hd = out.shape
    out = out.transpose(1, 2).reshape(B, T, nH * hd)
    h = h + layer.self_attn.o_proj(out)
    return h + layer.mlp(layer.post_attention_layernorm(h))


@torch.no_grad()
def prefill(model, ids, words=None, obs_window=16):
    """Normal (uncompressed) pass over the context. Returns, per layer:
    K, V (1,nKV,P,hd) post-RoPE, accumulated attention per key (P,), and
    position-normalised attention-row entropy per query (P,)."""
    P = ids.shape[1]
    dev = ids.device
    pos = torch.arange(P, device=dev)[None]
    h = model.model.embed_tokens(ids)
    cos, sin = rope(model, h, pos)
    causal = torch.full((P, P), float("-inf"), device=dev).triu(1)
    norm = torch.log2(torch.arange(1, P + 1, device=dev).float())
    norm[0] = 1.0
    stats = []
    for layer in model.model.layers:
        q, k, v = layer_qkv(model, layer, h, cos, sin)
        out, w = attend(model, q, k, v, causal)
        w0 = w[0]                                           # (nH,P,P)
        cum = w0.sum(dim=(0, 1))                            # attention each key receives
        ent = -(w0 * torch.log2(w0.clamp_min(1e-12))).sum(-1).mean(0)   # (P,)
        W = max(1, min(obs_window, P // 4))
        obs = w0[:, -W:, :].sum(dim=(0, 1))                 # attention from the last W queries
        stats.append({"K": k, "V": v, "cum": cum.cpu().numpy(),
                      "ent": (ent / norm).cpu().numpy(),
                      "obs": obs.cpu().numpy(), "obs_w": W, "words": words})
        h = finish_layer(layer, h, out)
    return stats


@torch.no_grad()
def continue_with_cache(model, cont_ids, comp, start_pos):
    """Run continuation tokens against a (possibly compressed) per-layer cache.
    comp[l] = (K (1,nKV,Pc,hd), V, bias (Pc,)). Returns logits (1,C,vocab)."""
    C = cont_ids.shape[1]
    dev = cont_ids.device
    pos = torch.arange(start_pos, start_pos + C, device=dev)[None]
    h = model.model.embed_tokens(cont_ids)
    cos, sin = rope(model, h, pos)
    cm = torch.full((C, C), float("-inf"), device=dev).triu(1)
    for layer, (Kc, Vc, bias) in zip(model.model.layers, comp):
        q, k, v = layer_qkv(model, layer, h, cos, sin)
        Pc = Kc.shape[2]
        mask = torch.zeros(C, Pc + C, device=dev)
        mask[:, :Pc] = bias[None, :]
        mask[:, Pc:] = cm
        out, _ = attend(model, q, torch.cat([Kc, k], 2), torch.cat([Vc, v], 2), mask)
        h = finish_layer(layer, h, out)
    return model.lm_head(model.model.norm(h))


@torch.no_grad()
def greedy_generate(model, comp, first_ids, start_pos, steps):
    """Free-running greedy decoding against a (compressed) cache. Simple but not
    incremental: it re-runs the generated tokens each step (fine for ~16 steps)."""
    seq = first_ids.clone()
    for _ in range(steps):
        lg = continue_with_cache(model, seq, comp, start_pos)
        seq = torch.cat([seq, lg[:, -1:].argmax(-1)], dim=1)
    return seq[0, 1:].tolist()


def prefix_match(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------
# Turning a selection into groups, and groups into a smaller cache
# --------------------------------------------------------------------------
# A "group" is a half-open range (s, e) of original positions. A group of size 1
# is an untouched entry; a group of size m>1 becomes ONE entry (mean of K, mean
# of V). Eviction = keep only some singleton groups.

def groups_identity(P):
    return [(i, i + 1) for i in range(P)]


def groups_fixed(P, R, protect_first):
    """Partition [protect_first, P) into (P - protect_first - R) near-equal windows."""
    n_groups = max(1, P - protect_first - R)
    sizes = np.full(n_groups, (P - protect_first) // n_groups)
    sizes[: (P - protect_first) % n_groups] += 1
    g = [(i, i + 1) for i in range(protect_first)]
    s = protect_first
    for m in sizes:
        g.append((s, s + int(m)))
        s += int(m)
    return g


def _runs(mask):
    return int(np.sum(mask & ~np.concatenate([[False], mask[:-1]])))


def groups_from_set(P, mask):
    """Collapse every contiguous run of True positions into a single group."""
    g, i = [], 0
    while i < P:
        if mask[i]:
            j = i
            while j < P and mask[j]:
                j += 1
            g.append((i, j))
            i = j
        else:
            g.append((i, i + 1))
            i += 1
    return g


def groups_by_priority(P, order, R):
    """order = candidate positions, most-mergeable first. Choose the smallest
    prefix of `order` whose runs collapse to at least R removed entries."""
    def removed(k):
        m = np.zeros(P, bool)
        m[order[:k]] = True
        return int(m.sum()) - _runs(m)

    lo, hi = 0, len(order)
    if removed(hi) < R:
        lo = hi
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if removed(mid) >= R:
                hi = mid
            else:
                lo = mid + 1
    m = np.zeros(P, bool)
    m[order[:lo]] = True
    return groups_from_set(P, m)


def groups_similarity(K, V, R, protect_first):
    """Greedy agglomeration: repeatedly merge the adjacent pair of groups whose
    mean K and mean V are most cosine-similar, until R entries are removed."""
    P = K.shape[1]
    Kf = K.permute(1, 0, 2).reshape(P, -1).float().cpu().numpy()
    Vf = V.permute(1, 0, 2).reshape(P, -1).float().cpu().numpy()
    sK, sV = Kf.copy(), Vf.copy()
    cnt = np.ones(P)
    start = np.arange(P)
    end = np.arange(1, P + 1)
    nxt = np.arange(1, P + 1)
    prv = np.arange(-1, P - 1)
    nxt[-1] = -1
    alive = np.ones(P, bool)
    ver = np.zeros(P, int)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def sim(i, j):
        return 0.5 * (cos(sK[i] / cnt[i], sK[j] / cnt[j]) +
                      cos(sV[i] / cnt[i], sV[j] / cnt[j]))

    heap = []

    def push(i, j):
        if i < protect_first or j < 0 or j < protect_first:
            return
        heapq.heappush(heap, (-sim(i, j), i, j, ver[i], ver[j]))

    for i in range(P - 1):
        push(i, i + 1)

    done = 0
    while done < R and heap:
        _, i, j, vi, vj = heapq.heappop(heap)
        if not (alive[i] and alive[j]) or nxt[i] != j or ver[i] != vi or ver[j] != vj:
            continue
        sK[i] += sK[j]
        sV[i] += sV[j]
        cnt[i] += cnt[j]
        end[i] = end[j]
        alive[j] = False
        nxt[i] = nxt[j]
        if nxt[j] != -1:
            prv[nxt[j]] = i
        ver[i] += 1
        done += 1
        if prv[i] != -1:
            push(prv[i], i)
        if nxt[i] != -1:
            push(i, nxt[i])

    return [(int(start[i]), int(end[i])) for i in range(P) if alive[i]]


def groups_word(P, words, R, protect_first):
    """Collapse the sub-word fragments of the same word. Words that are split into the
    most pieces are collapsed first (largest saving first), until R entries are removed."""
    if words is None:
        return groups_fixed(P, R, protect_first)
    words = np.asarray(words)[:P]
    segs, s = [], 0
    for i in range(1, P + 1):
        if i == P or words[i] != words[s]:
            if i - s >= 2 and s >= protect_first:
                segs.append((s, i))
            s = i
    segs.sort(key=lambda x: -(x[1] - x[0]))               # most fragmented words first
    merge_end, removed = {}, 0
    for s, e in segs:
        need = R - removed
        if need <= 0:
            break
        take = min(e - s - 1, need)
        merge_end[s] = s + take + 1
        removed += take
    g, p = [], 0
    while p < P:
        if p in merge_end:
            g.append((p, merge_end[p]))
            p = merge_end[p]
        else:
            g.append((p, p + 1))
            p += 1
    return g


def groups_snapkv(P, st, R, protect_first, kernel=7):
    """SnapKV-style: score each prompt token by the attention it receives from the last W
    queries (observation window), smooth with average pooling, keep the top tokens plus the
    window itself. Selection is shared across heads in a layer (so it is GQA-compatible)."""
    keep_n = P - R
    W = max(1, min(st["obs_w"], keep_n - protect_first - 1))
    score = np.convolve(st["obs"], np.ones(kernel) / kernel, mode="same")
    cand = np.arange(protect_first, P - W)
    n_top = keep_n - protect_first - W
    top = cand[np.argsort(-score[cand], kind="stable")[:n_top]]
    keep = list(range(protect_first)) + list(top) + list(range(P - W, P))
    return groups_evict(P, keep)


def groups_snapkv_word(P, st, R, protect_first, kernel=7):
    """SnapKV-style selection at WORD granularity: a word's score is the best pooled score
    among its fragments, and whole words are kept or evicted. Remaining slots are filled
    with the best leftover tokens so the budget is met exactly."""
    words = st.get("words")
    if words is None:
        return groups_snapkv(P, st, R, protect_first, kernel)
    words = np.asarray(words)[:P]
    keep_n = P - R
    W = max(1, min(st["obs_w"], keep_n - protect_first - 1))
    score = np.convolve(st["obs"], np.ones(kernel) / kernel, mode="same")
    budget = keep_n - protect_first - W
    cand = np.arange(protect_first, P - W)
    uniq, inv = np.unique(words[cand], return_inverse=True)
    wscore = np.full(len(uniq), -np.inf)
    np.maximum.at(wscore, inv, score[cand])
    keep, used = [], 0
    for k in np.argsort(-wscore, kind="stable"):
        m = cand[inv == k]
        if used + len(m) <= budget:
            keep.extend(m.tolist())
            used += len(m)
    if used < budget:
        rest = np.setdiff1d(cand, np.array(keep, dtype=int))
        keep.extend(rest[np.argsort(-score[rest], kind="stable")[: budget - used]].tolist())
    keep = list(range(protect_first)) + keep + list(range(P - W, P))
    return groups_evict(P, keep)


def slice_stats(st, n):
    """The same layer statistics restricted to the first n context positions."""
    return {"K": st["K"][:, :, :n], "V": st["V"][:, :, :n], "cum": st["cum"][:n],
            "ent": st["ent"][:n], "obs": st["obs"][:n], "obs_w": max(1, min(st["obs_w"], n // 4)),
            "words": None if st.get("words") is None else st["words"][:n]}


def groups_evict(P, keep_positions):
    return [(int(i), int(i) + 1) for i in sorted(keep_positions)]


def build_groups(method, st, R, protect_first, rng):
    P = st["K"].shape[2]
    if R <= 0:
        return groups_identity(P)
    R = min(R, P - protect_first - 1)
    cand = np.arange(protect_first, P)
    if method == "merge_fixed":
        return groups_fixed(P, R, protect_first)
    if method == "merge_entropy":
        return groups_by_priority(P, cand[np.argsort(st["ent"][cand], kind="stable")], R)
    if method == "merge_random":
        return groups_by_priority(P, rng.permutation(cand), R)
    if method == "merge_sim":
        return groups_similarity(st["K"][0], st["V"][0], R, protect_first)
    if method == "merge_word":
        return groups_word(P, st.get("words"), R, protect_first)
    keep_n = P - R
    if method == "evict_snapkv":
        return groups_snapkv(P, st, R, protect_first)
    if method == "evict_snapkv_word":
        return groups_snapkv_word(P, st, R, protect_first)
    if method == "evict_h2o":
        top = cand[np.argsort(-st["cum"][cand], kind="stable")[: keep_n - protect_first]]
        return groups_evict(P, list(range(protect_first)) + list(top))
    if method == "evict_window":
        recent = cand[-(keep_n - protect_first):]
        return groups_evict(P, list(range(protect_first)) + list(recent))
    raise ValueError(method)


def apply_groups(K, V, groups, use_bias=True):
    """K,V (1,nKV,P,hd) -> compressed (1,nKV,Pc,hd) plus per-entry log-mass bias.
    A merged entry stands for m tokens, so its attention logit gets +log(m)
    (otherwise the m tokens' attention mass would collapse to that of one token)."""
    s = torch.tensor([g[0] for g in groups], device=K.device)
    e = torch.tensor([g[1] for g in groups], device=K.device)
    m = (e - s).float()

    def mean(X):
        cs = F.pad(X.cumsum(dim=2), (0, 0, 1, 0))        # (1,nKV,P+1,hd)
        return (cs[:, :, e] - cs[:, :, s]) / m[None, None, :, None]

    bias = torch.log(m) if use_bias else torch.zeros_like(m)
    return mean(K), mean(V), bias


def compress_context(stats, method, ratio, layers_to_compress, protect_first,
                     use_bias, seed, protect_last=0):
    comp, kept = [], []
    for li, st in enumerate(stats):
        P = st["K"].shape[2]
        if li in layers_to_compress and ratio > 0:
            rng = np.random.default_rng((seed, li, int(ratio * 1000)))
            R = int(round(ratio * P))
            if protect_last > 0:   # compress only the older part; the last N tokens stay untouched
                Pp = P - protect_last
                groups = build_groups(method, slice_stats(st, Pp), R, protect_first, rng)
                groups = groups + [(i, i + 1) for i in range(Pp, P)]
            else:
                groups = build_groups(method, st, R, protect_first, rng)
        else:
            groups = groups_identity(P)
        comp.append(apply_groups(st["K"], st["V"], groups, use_bias))
        kept.append(len(groups))
    return comp, kept


# --------------------------------------------------------------------------
# Data and experiment loop
# --------------------------------------------------------------------------

def word_ids_from_offsets(text, offsets):
    """Map every token to the index of the whitespace-delimited word it ends in.
    Special tokens (empty offsets, e.g. BOS) get unique negative ids."""
    wid, w = np.zeros(len(text) + 1, int), -1
    for i, ch in enumerate(text):
        if ch.isspace():
            wid[i] = w + 1
        else:
            if i == 0 or text[i - 1].isspace():
                w += 1
            wid[i] = w
    out = []
    for t, (a, b) in enumerate(offsets):
        out.append(-(t + 1) if b <= a else int(wid[min(b - 1, len(text) - 1)]))
    return np.array(out)


def build_ids(tokenizer, text, max_tokens, device):
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"][:, :max_tokens].to(device)
    offsets = enc["offset_mapping"][0, :max_tokens].tolist()
    return ids, word_ids_from_offsets(text, offsets)


def parse_layers(spec, n_layers):
    if spec == "all":
        return set(range(n_layers))
    a, b = spec.split("-")
    return set(range(int(a), int(b) + 1))


@torch.no_grad()
def selftest(model, device, n_tokens=48):
    """Manual forward == HF forward, and identity compression == no compression."""
    torch.manual_seed(0)
    V = model.config.vocab_size
    ids = torch.randint(5, V, (1, n_tokens), device=device)
    hf = model(input_ids=ids).logits
    ctx = 30
    words = np.repeat(np.arange(ctx // 3 + 1), 3)[:ctx]
    words[0] = -1                                           # BOS is its own "word"
    stats = prefill(model, ids[:, :ctx], words)
    comp, _ = compress_context(stats, "merge_fixed", 0.0, set(range(len(stats))), 1, True, 0)
    lg = continue_with_cache(model, ids[:, ctx:], comp, ctx)
    err = (lg - hf[:, ctx:]).abs().max().item()
    print(f"[selftest] max |manual - HF| logit difference: {err:.2e}")
    assert err < 1e-3, "manual forward does not match Hugging Face; do not trust results"
    for m in METHODS:
        comp, kept = compress_context(stats, m, 0.5, set(range(len(stats))), 1, True, 0)
        assert all(k <= ctx - ctx // 2 + 1 and k >= 2 for k in kept), (m, kept)
        lg = continue_with_cache(model, ids[:, ctx:], comp, ctx)
        assert torch.isfinite(lg).all(), m
    print("[selftest] all methods run; kept entries ~= 50% of context. OK")


NEEDLES = {   # (sentence containing the secret, query that should trigger recall)
    "english": ("The secret code is {c}.", "The secret code is"),
    "nepali": ("\u0917\u094b\u092a\u094d\u092f \u0915\u094b\u0921 {c} \u0939\u094b\u0964",
               "\u0917\u094b\u092a\u094d\u092f \u0915\u094b\u0921"),
}


def run_needle(args, model, tokenizer, device, layers_set, n_layers):
    """Long-range retrieval probe: a 4-digit secret is planted early in a multi-sentence
    document; the query sits at the END of the context (so the observation window sees it),
    and we check whether the compressed cache still lets the model recall the digits."""
    df = pd.read_csv(args.csv).dropna(subset=["english", "nepali"])
    rows = []
    for d in range(args.docs):
        chunk = df.iloc[d * args.sents:(d + 1) * args.sents]
        if len(chunk) < args.sents:
            break
        code = str(1000 + (d * 7919 + 17) % 9000)
        for lang in ["english", "nepali"]:
            tmpl, query = NEEDLES[lang]
            sents = chunk[lang].astype(str).tolist()
            k = max(1, int(args.needle_pos * len(sents)))
            text = " ".join(sents[:k] + [tmpl.format(c=code)] + sents[k:]) + " " + query
            ids, words = build_ids(tokenizer, text, args.max_tokens + 64, device)
            ans = tokenizer(" " + code, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            ctx_ids, last_q = ids[:, :-1], ids[:, -1:]
            P = ctx_ids.shape[1]
            cont_in = torch.cat([last_q, ans[:, :-1]], dim=1)
            stats = prefill(model, ctx_ids, words[:P])

            def score(comp):
                lg = continue_with_cache(model, cont_in, comp, P)[0]
                ok = lg.argmax(-1) == ans[0]
                return F.cross_entropy(lg, ans[0]).item(), float(ok.float().mean()), bool(ok.all())

            base, _ = compress_context(stats, "merge_fixed", 0.0, layers_set, args.protect_first,
                                       True, d, args.protect_last)
            nll0, acc0, ex0 = score(base)
            for method in args.methods:
                for ratio in args.ratios:
                    comp, kept = compress_context(stats, method, ratio, layers_set, args.protect_first,
                                                  not args.no_bias, d, args.protect_last)
                    nll, acc, ex = score(comp)
                    rows.append({"doc": d, "lang": lang, "method": method, "ratio": ratio,
                                 "ctx_len": P, "kept_mean": float(np.mean(kept)),
                                 "kv_saved_total": 1 - float(np.sum(kept)) / (P * n_layers),
                                 "nll_base": nll0, "nll": nll, "tok_acc_base": acc0, "tok_acc": acc,
                                 "exact_base": ex0, "exact": ex, "protect_last": args.protect_last})
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"needle doc {d + 1}/{args.docs} done", flush=True)
    print(f"saved {args.out}")


def run(args):
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager").to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers

    if args.selftest:
        selftest(model, device)
        return

    layers_set = parse_layers(args.layers, n_layers)
    if args.needle:
        return run_needle(args, model, tokenizer, device, layers_set, n_layers)
    df = pd.read_csv(args.csv).dropna(subset=["english", "nepali"])
    rows = []
    t0 = time.time()
    for d in range(args.docs):
        chunk = df.iloc[d * args.sents:(d + 1) * args.sents]
        if len(chunk) < args.sents:
            break
        for lang in ["english", "nepali"]:
            text = " ".join(chunk[lang].astype(str))
            ids, words = build_ids(tokenizer, text, args.max_tokens, device)
            N = ids.shape[1]
            ctx_len = int(N * args.ctx_frac)
            cont_in, targets = ids[:, ctx_len:N - 1], ids[0, ctx_len + 1:N]
            first = ids[:, ctx_len:ctx_len + 1]
            stats = prefill(model, ids[:, :ctx_len], words[:ctx_len])

            base_comp, _ = compress_context(stats, "merge_fixed", 0.0, layers_set,
                                            args.protect_first, True, d, args.protect_last)
            lg_base = continue_with_cache(model, cont_in, base_comp, ctx_len)[0]
            nll_base = F.cross_entropy(lg_base, targets).item()
            logp_base, top1_base = lg_base.log_softmax(-1), lg_base.argmax(-1)
            gen_base = (greedy_generate(model, base_comp, first, ctx_len, args.gen_steps)
                        if args.gen_steps else None)

            for method in args.methods:
                for ratio in args.ratios:
                    comp, kept = compress_context(stats, method, ratio, layers_set,
                                                  args.protect_first, not args.no_bias, d, args.protect_last)
                    lg = continue_with_cache(model, cont_in, comp, ctx_len)[0]
                    nll = F.cross_entropy(lg, targets).item()
                    kl = (logp_base.exp() * (logp_base - lg.log_softmax(-1))).sum(-1).mean().item()
                    top1 = (lg.argmax(-1) == top1_base).float().mean().item()
                    gen_frac = float("nan")
                    if args.gen_steps:
                        g = greedy_generate(model, comp, first, ctx_len, args.gen_steps)
                        gen_frac = prefix_match(g, gen_base) / args.gen_steps
                    rows.append({
                        "doc": d, "lang": lang, "method": method, "ratio": ratio,
                        "ctx_len": ctx_len, "n_scored": len(targets),
                        "kept_mean": float(np.mean(kept)),
                        "kv_saved_total": 1 - float(np.sum(kept)) / (ctx_len * n_layers),
                        "nll_base": nll_base, "nll": nll, "delta_nll": nll - nll_base,
                        "kl": kl, "top1_agree": top1, "gen_prefix_frac": gen_frac,
                        "protect_last": args.protect_last,
                    })
        pd.DataFrame(rows).to_csv(args.out, index=False)       # checkpoint every doc
        print(f"doc {d + 1}/{args.docs} done  ({time.time() - t0:.0f}s)", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--csv", default="fertility_results.csv")
    ap.add_argument("--out", default="compress_results.csv")
    ap.add_argument("--docs", type=int, default=8, help="number of multi-sentence documents")
    ap.add_argument("--sents", type=int, default=10, help="FLORES sentences per document")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ctx-frac", type=float, default=0.75, help="fraction used as cached context")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.6, 0.8])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--layers", default="all", help="'all' or a range like 4-11")
    ap.add_argument("--protect-first", type=int, default=1, help="never merge/evict the first N tokens (BOS sink)")
    ap.add_argument("--protect-last", type=int, default=0, help="never merge/evict the last N context tokens")
    ap.add_argument("--needle", action="store_true", help="run the long-range retrieval probe instead of perplexity")
    ap.add_argument("--needle-pos", type=float, default=0.1, help="where the secret is planted (fraction of the document)")
    ap.add_argument("--no-bias", action="store_true", help="disable log(m) attention-mass compensation")
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--gen-steps", type=int, default=0, help="also test free-running greedy generation for N steps (slow)")
    ap.add_argument("--selftest", action="store_true")
    run(ap.parse_args())