#include "tts_transformer.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace qwen3_tts {
namespace {

void reset_scheduler_reserve_state(tts_transformer_state & state) {
    state.sched_reserved = false;
    state.sched_reserve_failed = false;
    state.sched_reserved_ctx = 0;
    state.sched_reserved_prefill_len = 0;
}

} // namespace

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;

    free_tts_kv_cache(state_.cache);

    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_layers;
    reset_scheduler_reserve_state(state_);

    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }

    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);

    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);

        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }

    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_kv_cache() {
    state_.cache.n_used = 0;
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;

    free_tts_kv_cache(state_.code_pred_cache);

    state_.code_pred_cache.n_ctx = n_ctx;
    state_.code_pred_cache.n_used = 0;
    state_.code_pred_cache.head_dim = cfg.code_pred_head_dim;
    state_.code_pred_cache.n_kv_heads = cfg.code_pred_n_key_value_heads;
    state_.code_pred_cache.n_layers = cfg.code_pred_layers;
    reset_scheduler_reserve_state(state_);

    const size_t n_tensors = cfg.code_pred_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    state_.code_pred_cache.ctx = ggml_init(params);
    if (!state_.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }

    state_.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    state_.code_pred_cache.v_cache.resize(cfg.code_pred_layers);

    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        state_.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);

        state_.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }

    state_.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(state_.code_pred_cache.ctx, state_.backend);
    if (!state_.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    state_.code_pred_cache.n_used = 0;
}

void TTSTransformer::maybe_reserve_scheduler_graphs(int32_t prefill_len, int32_t required_ctx) {
    if (!state_.sched) {
        return;
    }
    if (state_.sched_reserve_failed) {
        return;
    }
    if (state_.code_pred_cache.n_ctx < 16) {
        return;
    }

    if (state_.sched_reserved &&
        state_.sched_reserved_ctx >= required_ctx &&
        state_.sched_reserved_prefill_len >= prefill_len) {
        return;
    }

    std::string first_failed_graph;
    auto reserve_graph = [&](struct ggml_cgraph * g, const char * name) -> bool {
        if (!g) {
            if (first_failed_graph.empty()) {
                first_failed_graph = name;
            }
            return false;
        }
        const bool ok = ggml_backend_sched_reserve(state_.sched, g);
        ggml_backend_sched_reset(state_.sched);
        if (!ok && first_failed_graph.empty()) {
            first_failed_graph = name;
        }
        return ok;
    };

    bool ok = true;
    ok &= reserve_graph(build_prefill_forward_graph(prefill_len, 0), "talker prefill");
    ok &= reserve_graph(build_step_graph(std::max<int32_t>(0, required_ctx - 1)), "talker step");
    ok &= reserve_graph(build_code_pred_prefill_graph(), "code predictor prefill");

    for (int step = 1; step < 15; ++step) {
        char name[32];
        snprintf(name, sizeof(name), "code predictor step %d", step);
        ok &= reserve_graph(build_code_pred_step_graph(15, step), name);
    }

    if (ok) {
        state_.sched_reserved = true;
        state_.sched_reserve_failed = false;
        state_.sched_reserved_ctx = required_ctx;
        state_.sched_reserved_prefill_len = prefill_len;
    } else {
        state_.sched_reserved = false;
        state_.sched_reserve_failed = true;
        const char * graph_name = first_failed_graph.empty() ? "unknown graph" : first_failed_graph.c_str();
        fprintf(stderr,
                "  Scheduler reserve failed at %s; disabling reserve warmup and using dynamic graph allocation\n",
                graph_name);
    }
}

} // namespace qwen3_tts
