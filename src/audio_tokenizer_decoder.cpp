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

} // namespace qwen3_tts
