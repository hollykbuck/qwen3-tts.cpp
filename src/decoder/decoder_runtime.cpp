#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"
#include "ggml.h"

#ifdef QWEN3_TTS_TIMING
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#endif

namespace qwen3_tts {

#ifdef QWEN3_TTS_TIMING
namespace {

bool env_flag_enabled(const char * name) {
    const char * v = nullptr;
#ifdef _WIN32
    char * env_buf = nullptr;
    size_t env_len = 0;
    if (_dupenv_s(&env_buf, &env_len, name) != 0 || !env_buf || env_buf[0] == '\0') {
        free(env_buf);
        return false;
    }
    v = env_buf;
#else
    v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }
#endif
    if (strcmp(v, "0") == 0) {
#ifdef _WIN32
        free(env_buf);
#endif
        return false;
    }
    if (_stricmp(v, "false") == 0 || _stricmp(v, "off") == 0 || _stricmp(v, "no") == 0) {
#ifdef _WIN32
        free(env_buf);
#endif
        return false;
    }
#ifdef _WIN32
    free(env_buf);
#endif
    return true;
}

struct decoder_stage_profile_state {
    using clock = std::chrono::steady_clock;

    clock::time_point chunk_start;
    const char * pending_stage = nullptr;
};

void dump_decoder_backend_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * gf) {
    if (!sched || !gf) {
        return;
    }

    struct backend_bucket {
        ggml_backend_t backend = nullptr;
        int count = 0;
    };

    backend_bucket buckets[8] = {};
    int n_buckets = 0;

    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);

        bool found = false;
        for (int j = 0; j < n_buckets; ++j) {
            if (buckets[j].backend == backend) {
                buckets[j].count++;
                found = true;
                break;
            }
        }
        if (!found && n_buckets < (int) (sizeof(buckets) / sizeof(buckets[0]))) {
            buckets[n_buckets].backend = backend;
            buckets[n_buckets].count = 1;
            n_buckets++;
        }
    }

    fprintf(stderr, "  [decoder-backend] node counts by backend:\n");
    for (int i = 0; i < n_buckets; ++i) {
        const char * name = buckets[i].backend ? ggml_backend_name(buckets[i].backend) : "NULL";
        ggml_backend_dev_t dev = buckets[i].backend ? ggml_backend_get_device(buckets[i].backend) : nullptr;
        const char * dev_name = dev ? ggml_backend_dev_name(dev) : "Unknown";
        fprintf(stderr, "    %-12s device=%-12s nodes=%d\n", name, dev_name, buckets[i].count);
    }

    static const char * stage_names[] = {
        "vq_output",
        "pre_conv_output",
        "pre_tfm_output",
        "pre_tfm_reshaped",
        "upsample_output",
        "dec0_output",
        "dec1_output",
        "dec2_output",
        "dec3_output",
        "dec4_output",
        "dec5_output",
        "dec6_output",
        "audio",
    };

    fprintf(stderr, "  [decoder-backend] stage tensors:\n");
    for (const char * stage_name : stage_names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, stage_name);
        ggml_backend_t backend = t ? ggml_backend_sched_get_tensor_backend(sched, t) : nullptr;
        const char * name = backend ? ggml_backend_name(backend) : "NULL";
        ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
        const char * dev_name = dev ? ggml_backend_dev_name(dev) : "Unknown";
        fprintf(stderr, "    %-16s backend=%-12s device=%s\n", stage_name, name, dev_name);
    }
}

bool decoder_stage_profile_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!t || !user_data) {
        return true;
    }

    auto * state = static_cast<decoder_stage_profile_state *>(user_data);
    const char * name = t->name;
    if (!name || name[0] == '\0') {
        return false;
    }

    static const char * stage_names[] = {
        "vq_output",
        "pre_conv_output",
        "pre_tfm_output",
        "pre_tfm_reshaped",
        "upsample_output",
        "dec0_output",
        "dec1_output",
        "dec2_output",
        "dec3_output",
        "dec4_output",
        "dec5_output",
        "dec6_output",
        "audio",
    };

    bool is_stage_boundary = false;
    for (const char * stage_name : stage_names) {
        if (strcmp(name, stage_name) == 0) {
            is_stage_boundary = true;
            break;
        }
    }

    if (!is_stage_boundary) {
        return false;
    }

    if (ask) {
        state->pending_stage = name;
        return true;
    }

    const auto now = decoder_stage_profile_state::clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(now - state->chunk_start).count();
    fprintf(stderr, "    [decoder-stage] %-16s %.2f ms\n",
            state->pending_stage ? state->pending_stage : name, dt_ms);
    state->chunk_start = now;
    state->pending_stage = nullptr;
    return true;
}

} // namespace
#endif

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;
    auto & codebook_input_bufs = impl_->codebook_input_bufs;
    auto & positions_buf = impl_->positions_buf;

    if (!model.ctx) {
        error_msg = "Model not loaded";
        return false;
    }

    const auto & cfg = model.config;

#ifdef QWEN3_TTS_TIMING
    using clock = std::chrono::steady_clock;
    const auto t_decode_start = clock::now();
    const bool stage_profile_enabled = env_flag_enabled("QWEN3_TTS_DECODER_STAGE_PROFILE");
    decoder_stage_profile_state stage_profile = {};
    stage_profile.chunk_start = t_decode_start;
#endif

    if (!decoder_internal::ops::ensure_cached_decode_graph(*this, n_frames)) {
        return false;
    }

    struct ggml_cgraph * gf = state.decode_graph;

#ifdef QWEN3_TTS_TIMING
    const auto t_graph_ready = clock::now();
#endif

    if (!ggml_backend_sched_alloc_graph(state.sched, gf)) {
        error_msg = "Failed to allocate graph";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    const auto t_alloc_done = clock::now();
    const int n_splits = ggml_backend_sched_get_n_splits(state.sched);
    const bool backend_dump_enabled = env_flag_enabled("QWEN3_TTS_DECODER_BACKEND_DUMP");
#endif

    if ((int32_t) codebook_input_bufs.size() != cfg.n_codebooks) {
        codebook_input_bufs.assign(cfg.n_codebooks, {});
    }
    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        codebook_input_bufs[cb].resize(n_frames);
    }

    for (int f = 0; f < n_frames; ++f) {
        const int32_t * frame_codes = codes + (size_t) f * cfg.n_codebooks;
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            codebook_input_bufs[cb][f] = frame_codes[cb];
        }
    }

#ifdef QWEN3_TTS_TIMING
    const auto t_pack_done = clock::now();
#endif

    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        ggml_backend_tensor_set(state.decode_code_tensors[cb], codebook_input_bufs[cb].data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

    if ((int32_t) positions_buf.size() != n_frames) {
        positions_buf.resize(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions_buf[i] = i;
        }
    }
    if (state.decode_positions_tensor) {
        ggml_backend_tensor_set(state.decode_positions_tensor, positions_buf.data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

#ifdef QWEN3_TTS_TIMING
    const auto t_upload_done = clock::now();
#endif

#ifdef QWEN3_TTS_TIMING
    if (backend_dump_enabled) {
        dump_decoder_backend_assignments(state.sched, gf);
    }
    if (stage_profile_enabled) {
        fprintf(stderr, "  [decoder-stage] stage profiling enabled\n");
        ggml_backend_sched_set_eval_callback(state.sched, decoder_stage_profile_callback, &stage_profile);
    }
#endif

    if (ggml_backend_sched_graph_compute(state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg = "Failed to compute graph";
        ggml_backend_sched_set_eval_callback(state.sched, nullptr, nullptr);
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    ggml_backend_sched_set_eval_callback(state.sched, nullptr, nullptr);

#ifdef QWEN3_TTS_TIMING
    const auto t_compute_done = clock::now();
#endif

    struct ggml_tensor * audio_tensor = state.decode_audio_tensor;
    if (!audio_tensor) {
        error_msg = "Failed to find audio tensor";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));

#ifdef QWEN3_TTS_TIMING
    const auto t_download_done = clock::now();
    auto dt_ms = [](const clock::time_point & a, const clock::time_point & b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    fprintf(stderr,
            "  [decoder] frames=%d samples=%lld splits=%d graph=%.2f ms alloc=%.2f ms "
            "pack=%.2f ms upload=%.2f ms compute=%.2f ms download=%.2f ms total=%.2f ms\n",
            n_frames,
            (long long) n_samples,
            n_splits,
            dt_ms(t_decode_start, t_graph_ready),
            dt_ms(t_graph_ready, t_alloc_done),
            dt_ms(t_alloc_done, t_pack_done),
            dt_ms(t_pack_done, t_upload_done),
            dt_ms(t_upload_done, t_compute_done),
            dt_ms(t_compute_done, t_download_done),
            dt_ms(t_decode_start, t_download_done));
#endif

    ggml_backend_sched_reset(state.sched);

    return true;
}

} // namespace qwen3_tts
