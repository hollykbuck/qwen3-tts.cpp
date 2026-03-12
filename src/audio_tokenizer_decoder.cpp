#include "audio_tokenizer_decoder.h"

#include <cstdio>

namespace qwen3_tts {

void AudioTokenizerDecoder::release_cached_decode_graph() {
    state_.decode_graph = nullptr;
    state_.decode_positions_tensor = nullptr;
    state_.decode_audio_tensor = nullptr;
    state_.decode_graph_n_frames = 0;
    for (int i = 0; i < 16; ++i) {
        state_.decode_code_tensors[i] = nullptr;
    }
    if (state_.decode_graph_ctx) {
        ggml_free(state_.decode_graph_ctx);
        state_.decode_graph_ctx = nullptr;
    }
}

bool AudioTokenizerDecoder::ensure_cached_decode_graph(int32_t n_frames) {
    if (state_.decode_graph && state_.decode_graph_n_frames == n_frames) {
        return true;
    }

    release_cached_decode_graph();

    state_.decode_graph = build_graph_impl(n_frames, &state_.decode_graph_ctx);
    if (!state_.decode_graph || !state_.decode_graph_ctx) {
        error_msg_ = "Failed to build cached decoder graph";
        release_cached_decode_graph();
        return false;
    }

    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        state_.decode_code_tensors[cb] = ggml_graph_get_tensor(state_.decode_graph, name);
        if (!state_.decode_code_tensors[cb]) {
            error_msg_ = "Failed to find cached decoder input tensor for codebook " + std::to_string(cb);
            release_cached_decode_graph();
            return false;
        }
    }

    state_.decode_positions_tensor = ggml_graph_get_tensor(state_.decode_graph, "positions");
    state_.decode_audio_tensor = ggml_graph_get_tensor(state_.decode_graph, "audio");
    if (!state_.decode_audio_tensor) {
        error_msg_ = "Failed to find cached decoder output tensor";
        release_cached_decode_graph();
        return false;
    }

    state_.decode_graph_n_frames = n_frames;
    return true;
}

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;

    if (!ensure_cached_decode_graph(n_frames)) {
        return false;
    }

    struct ggml_cgraph * gf = state_.decode_graph;
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }

    if ((int32_t) codebook_input_bufs_.size() != cfg.n_codebooks) {
        codebook_input_bufs_.assign(cfg.n_codebooks, {});
    }
    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        codebook_input_bufs_[cb].resize(n_frames);
    }

    for (int f = 0; f < n_frames; ++f) {
        const int32_t * frame_codes = codes + (size_t) f * cfg.n_codebooks;
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            codebook_input_bufs_[cb][f] = frame_codes[cb];
        }
    }

    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        ggml_backend_tensor_set(state_.decode_code_tensors[cb], codebook_input_bufs_[cb].data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

    if ((int32_t) positions_buf_.size() != n_frames) {
        positions_buf_.resize(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions_buf_[i] = i;
        }
    }
    if (state_.decode_positions_tensor) {
        ggml_backend_tensor_set(state_.decode_positions_tensor, positions_buf_.data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    struct ggml_tensor * audio_tensor = state_.decode_audio_tensor;
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

} // namespace qwen3_tts
