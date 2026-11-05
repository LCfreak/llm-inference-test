// attn_entropy_callback.cpp
//
// A ggml "eval callback" for llama.cpp that captures per-layer, per-head
// attention distributions (the softmax(QK^T) rows) and computes the
// Shannon entropy of each row.
//
// BEFORE THIS WORKS, TWO THINGS MUST BE TRUE:
//
//   1. Flash attention must be OFF.
//      Set llama_context_params.flash_attn = false (or pass -fa 0 to the
//      CLI/server tools). With flash attention on, attention weights are
//      fused into one GPU kernel and never exist as a separate tensor —
//      there is nothing for this callback to read.
//
//   2. The tensor name match below is a guess.
//      Uncomment PRINT_ALL_NAMES, run once on a short test sentence, and
//      look at stderr for a tensor whose name contains something like
//      "soft_max" and whose shape is [n_kv, n_tokens, n_head, ...].
//      Update the strstr() match to whatever you actually see — naming
//      has changed across llama.cpp versions.

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>

// #define PRINT_ALL_NAMES  // uncomment once, to discover the real tensor name

// Where results land: per_tensor_entropy["<tensor name>"][head][token_index] = entropy
struct AttnEntropyStore {
    std::map<std::string, std::vector<std::vector<float>>> per_tensor_entropy;
};

static float row_entropy(const float * p, int n) {
    float h = 0.0f;
    for (int i = 0; i < n; i++) {
        if (p[i] > 0.0f) {
            h -= p[i] * std::log2(p[i]);
        }
    }
    return h;
}

// llama.cpp calls this once per graph node, twice each: first with ask=true
// (you say whether you want the tensor's data), then with ask=false once
// it's computed (and copied to host memory, if you asked for it).
static bool attn_entropy_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * store = static_cast<AttnEntropyStore *>(user_data);

#ifdef PRINT_ALL_NAMES
    if (!ask) {
        fprintf(stderr, "tensor: %-40s  shape=[%lld,%lld,%lld,%lld]\n",
                t->name, (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
    }
    return true; // force everything to host memory while discovering names
#endif

    bool is_target = std::strstr(t->name, "kq_soft_max") != nullptr; // <-- adjust after step 4

    if (ask) {
        return is_target; // only pay the host-copy cost for tensors we care about
    }
    if (!is_target) {
        return true; // not ours, let llama.cpp proceed normally
    }

    // ggml dims are innermost-first:
    //   ne[0] = n_kv      (one softmax row's length -> what entropy is taken over)
    //   ne[1] = n_tokens  (one row per query token in this batch)
    //   ne[2] = n_head
    const int n_kv     = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    const int n_head   = (int) t->ne[2];

    std::vector<float> row(n_kv);
    std::vector<std::vector<float>> entropies(n_head, std::vector<float>(n_tokens));

    for (int h = 0; h < n_head; h++) {
        for (int q = 0; q < n_tokens; q++) {
            size_t offset = ((size_t) h * n_tokens + q) * n_kv * sizeof(float);
            ggml_backend_tensor_get(t, row.data(), offset, n_kv * sizeof(float));
            entropies[h][q] = row_entropy(row.data(), n_kv);
        }
    }

    store->per_tensor_entropy[t->name] = std::move(entropies);
    return true;
}

// --- How to wire this up in your own program ---
//
//   AttnEntropyStore store;
//
//   llama_context_params cparams = llama_context_default_params();
//   cparams.flash_attn        = false;          // required, see note at top
//   cparams.cb_eval           = attn_entropy_cb;
//   cparams.cb_eval_user_data = &store;
//
//   llama_context * ctx = llama_new_context_with_model(model, cparams);
//
//   // Build a llama_batch containing every token of your whole sentence
//   // (Nepali or English), then:
//   llama_decode(ctx, batch);
//
//   // Afterwards, store.per_tensor_entropy["<name-for-layer-N>"][head][token]
//   // gives you the entropy for that layer, head, and token position.
//   // There will be one map entry per layer, since each layer produces its
//   // own attention-softmax tensor with its own name.