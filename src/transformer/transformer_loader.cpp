#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace qwen3_tts {

void TTSTransformer::unload_model() {
    free_tts_kv_cache(impl_->state.cache);
    free_tts_kv_cache(impl_->state.code_pred_cache);
    free_transformer_model(impl_->model);

    impl_->coreml_code_predictor.unload();
    impl_->use_coreml_code_predictor = false;
    impl_->coreml_code_predictor_path.clear();
    impl_->skip_ggml_code_pred_layers = false;

    if (impl_->state.sched) {
        ggml_backend_sched_free(impl_->state.sched);
        impl_->state.sched = nullptr;
    }
    impl_->state.sched_reserved = false;
    impl_->state.sched_reserve_failed = false;
    impl_->state.sched_reserved_ctx = 0;
    impl_->state.sched_reserved_prefill_len = 0;
    if (impl_->state.backend) {
        release_preferred_backend(impl_->state.backend);
        impl_->state.backend = nullptr;
    }
    if (impl_->state.backend_cpu) {
        ggml_backend_free(impl_->state.backend_cpu);
        impl_->state.backend_cpu = nullptr;
    }

    impl_->state.compute_meta.clear();
    impl_->state.code_pred_mask.clear();
    last_hidden_.clear();
    impl_->embd_row_fp16_scratch.clear();
}

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    impl_->skip_ggml_code_pred_layers = false;
#if defined(__APPLE__)
    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (!coreml_disabled) {
        std::string coreml_path;
        const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
        if (override_env && override_env[0] != '\0') {
            coreml_path = override_env;
        } else {
            size_t slash = model_path.find_last_of("/\\");
            const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
            coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
        }

        struct stat st = {};
        if (stat(coreml_path.c_str(), &st) == 0) {
            impl_->skip_ggml_code_pred_layers = true;
        } else if (use_coreml_env && use_coreml_env[0] != '\0') {
            impl_->skip_ggml_code_pred_layers = true;
        }
    }
#endif

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }

    if (!transformer_internal::ops::parse_config(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!transformer_internal::ops::create_tensors(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!transformer_internal::ops::load_tensor_data(*this, model_path, ctx)) {
        free_transformer_model(impl_->model);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!impl_->skip_ggml_code_pred_layers) {
        const auto & cfg = impl_->model.config;
        const bool projection_required = cfg.hidden_size > cfg.code_pred_hidden_size;
        const bool likely_legacy_1p7 = (cfg.hidden_size > 1024 &&
                                        impl_->model.code_pred_small_to_mtp_weight == nullptr);
        if ((projection_required || likely_legacy_1p7) &&
            impl_->model.code_pred_small_to_mtp_weight == nullptr) {
            error_msg_ =
                "Model is missing code_pred.small_to_mtp projection weights. "
                "Re-convert with the updated scripts/convert_tts_to_gguf.py.";
            free_transformer_model(impl_->model);
            gguf_free(ctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);

    impl_->state.backend = init_preferred_backend("TTSTransformer", &error_msg_);
    if (!impl_->state.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(impl_->state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        impl_->state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!impl_->state.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(impl_->state.backend);
    if (impl_->state.backend_cpu) {
        backends.push_back(impl_->state.backend_cpu);
    }
    impl_->state.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!impl_->state.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    impl_->state.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    impl_->state.code_pred_compute_meta.resize(15);
    for (int i = 0; i < 15; ++i) {
        impl_->state.code_pred_compute_meta[i].resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    }

    if (!transformer_internal::ops::try_init_coreml_code_predictor(*this, model_path)) {
        return false;
    }

    return true;
}

bool transformer_internal::ops::try_init_coreml_code_predictor(TTSTransformer & self, const std::string & model_path) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    (void) model_path;
    impl->use_coreml_code_predictor = false;
    impl->coreml_code_predictor_path.clear();

    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (coreml_disabled) {
        return true;
    }

#if !defined(__APPLE__)
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        fprintf(stderr, "  CoreML code predictor requested but this build is not on Apple platform\n");
    }
    return true;
#else
    std::string coreml_path;
    const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
    if (override_env && override_env[0] != '\0') {
        coreml_path = override_env;
    } else {
        size_t slash = model_path.find_last_of("/\\");
        const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
        coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
    }

    if (!impl->coreml_code_predictor.load(coreml_path, impl->model.config.n_codebooks - 1)) {
        if (impl->skip_ggml_code_pred_layers) {
            error_msg = "CoreML code predictor load failed in strict mode: " + impl->coreml_code_predictor.get_error();
            return false;
        } else {
            fprintf(stderr, "  CoreML code predictor load failed: %s\n",
                    impl->coreml_code_predictor.get_error().c_str());
            fprintf(stderr, "  Falling back to GGML code predictor\n");
            return true;
        }
    }

    impl->use_coreml_code_predictor = true;
    impl->coreml_code_predictor_path = coreml_path;
    fprintf(stderr, "  CoreML code predictor enabled: %s\n", impl->coreml_code_predictor_path.c_str());
    return true;
#endif
}

bool transformer_internal::ops::parse_config(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto get_u32_any = [&](std::initializer_list<const char *> keys, int32_t default_val) -> int32_t {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return (int32_t) gguf_get_val_u32(ctx, idx);
            }
        }
        return default_val;
    };

    auto get_f32_any = [&](std::initializer_list<const char *> keys, float default_val) -> float {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return gguf_get_val_f32(ctx, idx);
            }
        }
        return default_val;
    };

    auto get_str_any = [&](std::initializer_list<const char *> keys, const char * default_val) -> std::string {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0 && gguf_get_kv_type(ctx, idx) == GGUF_TYPE_STRING) {
                const char * s = gguf_get_val_str(ctx, idx);
                if (s && s[0] != '\0') {
                    return std::string(s);
                }
            }
        }
        return std::string(default_val ? default_val : "");
    };

    auto get_bool_any = [&](std::initializer_list<const char *> keys, bool default_val, bool * found) -> bool {
        if (found) {
            *found = false;
        }
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx < 0) {
                continue;
            }

            const enum gguf_type type = gguf_get_kv_type(ctx, idx);
            if (found) {
                *found = true;
            }

            switch (type) {
                case GGUF_TYPE_BOOL:
                    return gguf_get_val_bool(ctx, idx);
                case GGUF_TYPE_UINT8:
                    return gguf_get_val_u8(ctx, idx) != 0;
                case GGUF_TYPE_INT8:
                    return gguf_get_val_i8(ctx, idx) != 0;
                case GGUF_TYPE_UINT16:
                    return gguf_get_val_u16(ctx, idx) != 0;
                case GGUF_TYPE_INT16:
                    return gguf_get_val_i16(ctx, idx) != 0;
                case GGUF_TYPE_UINT32:
                    return gguf_get_val_u32(ctx, idx) != 0;
                case GGUF_TYPE_INT32:
                    return gguf_get_val_i32(ctx, idx) != 0;
                case GGUF_TYPE_UINT64:
                    return gguf_get_val_u64(ctx, idx) != 0;
                case GGUF_TYPE_INT64:
                    return gguf_get_val_i64(ctx, idx) != 0;
                default:
                    fprintf(stderr, "  Warning: ignoring non-numeric metadata key '%s' for boolean parse\n", key);
                    if (found) {
                        *found = false;
                    }
                    break;
            }
        }
        return default_val;
    };

    auto & cfg = impl->model.config;
    cfg.text_vocab_size = get_u32_any({
        "qwen3-tts.text.vocab_size",
        "qwen3-tts.text_vocab_size",
    }, 151936);
    cfg.text_embd_dim = get_u32_any({
        "qwen3-tts.text.embedding_dim",
        "qwen3-tts.text_hidden_size",
    }, 2048);
    cfg.hidden_size = get_u32_any({
        "qwen3-tts.talker.embedding_length",
        "qwen3-tts.embedding_length",
    }, 1024);
    cfg.n_layers = get_u32_any({
        "qwen3-tts.talker.block_count",
        "qwen3-tts.block_count",
    }, 28);
    cfg.n_attention_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count",
        "qwen3-tts.attention.head_count",
    }, 16);
    cfg.n_key_value_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count_kv",
        "qwen3-tts.attention.head_count_kv",
    }, 8);
    cfg.intermediate_size = get_u32_any({
        "qwen3-tts.talker.feed_forward_length",
        "qwen3-tts.feed_forward_length",
    }, 3072);
    cfg.head_dim = get_u32_any({
        "qwen3-tts.talker.attention.key_length",
        "qwen3-tts.attention.key_length",
    }, 128);
    cfg.rms_norm_eps = get_f32_any({
        "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
        "qwen3-tts.attention.layer_norm_rms_epsilon",
    }, 1e-6f);
    cfg.rope_theta = get_f32_any({
        "qwen3-tts.talker.rope.freq_base",
        "qwen3-tts.rope.freq_base",
    }, 1000000.0f);

    cfg.codec_vocab_size = get_u32_any({
        "qwen3-tts.talker.codec_vocab_size",
        "qwen3-tts.vocab_size",
    }, 3072);
    cfg.n_codebooks = get_u32_any({
        "qwen3-tts.talker.num_codebooks",
        "qwen3-tts.num_code_groups",
    }, 16);

    cfg.code_pred_layers = get_u32_any({
        "qwen3-tts.code_pred.layer_count",
        "qwen3-tts.code_predictor.layer_count",
    }, 5);
    cfg.code_pred_vocab_size = get_u32_any({
        "qwen3-tts.code_pred.vocab_size",
        "qwen3-tts.code_predictor.vocab_size",
    }, 2048);
    cfg.code_pred_hidden_size = get_u32_any({
        "qwen3-tts.code_pred.embedding_length",
        "qwen3-tts.code_predictor.embedding_length",
    }, cfg.hidden_size);
    cfg.code_pred_intermediate_size = get_u32_any({
        "qwen3-tts.code_pred.feed_forward_length",
        "qwen3-tts.code_predictor.feed_forward_length",
    }, cfg.intermediate_size);
    cfg.code_pred_n_attention_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count",
        "qwen3-tts.code_predictor.attention.head_count",
    }, cfg.n_attention_heads);
    cfg.code_pred_n_key_value_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count_kv",
        "qwen3-tts.code_predictor.attention.head_count_kv",
    }, cfg.n_key_value_heads);
    cfg.code_pred_head_dim = get_u32_any({
        "qwen3-tts.code_pred.attention.key_length",
        "qwen3-tts.code_predictor.attention.key_length",
    }, cfg.head_dim);
    cfg.code_pred_rms_norm_eps = get_f32_any({
        "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon",
        "qwen3-tts.code_predictor.attention.layer_norm_rms_epsilon",
    }, cfg.rms_norm_eps);
    cfg.code_pred_rope_theta = get_f32_any({
        "qwen3-tts.code_pred.rope.freq_base",
        "qwen3-tts.code_predictor.rope.freq_base",
    }, cfg.rope_theta);

    cfg.codec_pad_id = get_u32_any({
        "qwen3-tts.codec.pad_id",
    }, 2148);
    cfg.codec_bos_id = get_u32_any({
        "qwen3-tts.codec.bos_id",
    }, 2149);
    cfg.codec_eos_id = get_u32_any({
        "qwen3-tts.codec.eos_id",
        "qwen3-tts.codec.eos_token_id",
    }, 2150);

    cfg.tts_bos_token_id = get_u32_any({
        "qwen3-tts.tts_bos_token_id",
        "qwen3-tts.tts.bos_token_id",
        "qwen3-tts.tts.bos_id",
    }, 151672);
    cfg.tts_eos_token_id = get_u32_any({
        "qwen3-tts.tts_eos_token_id",
        "qwen3-tts.tts.eos_token_id",
        "qwen3-tts.tts.eos_id",
    }, 151673);
    cfg.tts_pad_token_id = get_u32_any({
        "qwen3-tts.tts_pad_token_id",
        "qwen3-tts.tts.pad_token_id",
        "qwen3-tts.tts.pad_id",
    }, 151671);

    cfg.codec_think_id = get_u32_any({
        "qwen3-tts.codec.think_id",
        "qwen3-tts.codec_think_id",
    }, 2154);
    cfg.codec_nothink_id = get_u32_any({
        "qwen3-tts.codec.nothink_id",
        "qwen3-tts.codec_nothink_id",
    }, 2155);
    cfg.codec_think_bos_id = get_u32_any({
        "qwen3-tts.codec.think_bos_id",
        "qwen3-tts.codec_think_bos_id",
    }, 2156);
    cfg.codec_think_eos_id = get_u32_any({
        "qwen3-tts.codec.think_eos_id",
        "qwen3-tts.codec_think_eos_id",
    }, 2157);

    cfg.english_language_id = get_u32_any({
        "qwen3-tts.language.english_id",
        "qwen3-tts.codec.language.english_id",
        "qwen3-tts.language_id",
    }, 2050);

    cfg.tts_model_type = transformer_internal::normalize_speaker_name(get_str_any({
        "qwen3-tts.tts_model_type",
    }, "base"));
    cfg.supports_instruction = get_bool_any({
        "qwen3-tts.supports_instruction",
        "qwen3-tts.instruction_supported",
        "qwen3-tts.instruct_supported",
    }, false, &cfg.has_supports_instruction);
    cfg.speaker_id_map.clear();

    fprintf(stderr, "  Codec IDs: pad=%d, bos=%d, eos=%d, think=%d, nothink=%d, think_bos=%d, think_eos=%d\n",
            cfg.codec_pad_id, cfg.codec_bos_id, cfg.codec_eos_id,
            cfg.codec_think_id, cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id);
    fprintf(stderr, "  TTS model type: %s\n", cfg.tts_model_type.c_str());
    if (cfg.has_supports_instruction) {
        fprintf(stderr, "  Metadata supports_instruction: %s\n", cfg.supports_instruction ? "true" : "false");
    }

    int64_t mrope_idx = gguf_find_key(ctx, "qwen3-tts.talker.rope.mrope_section");
    if (mrope_idx < 0) {
        mrope_idx = gguf_find_key(ctx, "qwen3-tts.rope.mrope_section");
    }
    if (mrope_idx >= 0) {
        const int32_t * mrope_data = (const int32_t *) gguf_get_arr_data(ctx, mrope_idx);
        if (mrope_data) {
            for (int i = 0; i < 3; ++i) {
                cfg.mrope_section[i] = mrope_data[i];
            }
            cfg.use_mrope = true;
        }
    }

    int64_t spk_count_idx = gguf_find_key(ctx, "qwen3-tts.speaker.count");
    int32_t spk_count = 0;
    if (spk_count_idx >= 0) {
        const enum gguf_type spk_count_type = gguf_get_kv_type(ctx, spk_count_idx);
        if (spk_count_type == GGUF_TYPE_UINT32) {
            spk_count = (int32_t) gguf_get_val_u32(ctx, spk_count_idx);
        } else if (spk_count_type == GGUF_TYPE_INT32) {
            spk_count = gguf_get_val_i32(ctx, spk_count_idx);
        }
    }
    if (spk_count > 0) {
        for (int32_t i = 0; i < spk_count; ++i) {
            char name_key[64];
            char id_key[64];
            snprintf(name_key, sizeof(name_key), "qwen3-tts.speaker.%d.name", i);
            snprintf(id_key, sizeof(id_key), "qwen3-tts.speaker.%d.id", i);
            int64_t name_idx = gguf_find_key(ctx, name_key);
            int64_t id_idx = gguf_find_key(ctx, id_key);
            if (name_idx < 0 || id_idx < 0) {
                continue;
            }
            if (gguf_get_kv_type(ctx, name_idx) != GGUF_TYPE_STRING) {
                continue;
            }
            const char * raw_name = gguf_get_val_str(ctx, name_idx);
            if (!raw_name || raw_name[0] == '\0') {
                continue;
            }

            int32_t spk_id = -1;
            const enum gguf_type id_type = gguf_get_kv_type(ctx, id_idx);
            if (id_type == GGUF_TYPE_UINT32) {
                spk_id = (int32_t) gguf_get_val_u32(ctx, id_idx);
            } else if (id_type == GGUF_TYPE_INT32) {
                spk_id = gguf_get_val_i32(ctx, id_idx);
            } else {
                continue;
            }
            if (spk_id < 0) {
                continue;
            }

            cfg.speaker_id_map[transformer_internal::normalize_speaker_name(raw_name)] = spk_id;
        }
    }

    if (!cfg.speaker_id_map.empty()) {
        fprintf(stderr, "  CustomVoice speakers loaded: %zu\n", cfg.speaker_id_map.size());
    }

    return true;
}

bool transformer_internal::ops::create_tensors(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = impl->model.config;

    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl->model.ctx = ggml_init(params);
    if (!impl->model.ctx) {
        error_msg = "Failed to create GGML context";
        return false;
    }

    impl->model.layers.resize(cfg.n_layers);
    impl->model.code_pred_layers.resize(cfg.code_pred_layers);
    impl->model.code_pred_embd.resize(cfg.n_codebooks - 1);
    impl->model.code_pred_head.resize(cfg.n_codebooks - 1);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;

        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }

        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.code_pred_hidden_size;
            n_dims = 2;
        } else if (strstr(name, "code_pred.small_to_mtp.bias")) {
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "code_pred.blk.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.code_pred_intermediate_size;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int cb_idx = -1;
            if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.code_pred_hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else {
            continue;
        }

        struct ggml_tensor * tensor = ggml_new_tensor(impl->model.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        impl->model.tensors[name] = tensor;

        if (strstr(name, "talker.text_embd.weight")) {
            impl->model.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            impl->model.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            impl->model.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            impl->model.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            impl->model.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            impl->model.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            impl->model.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            impl->model.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = impl->model.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight")) {
            impl->model.code_pred_small_to_mtp_weight = tensor;
        } else if (strstr(name, "code_pred.small_to_mtp.bias")) {
            impl->model.code_pred_small_to_mtp_bias = tensor;
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = impl->model.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_embd[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_head[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            impl->model.code_pred_output_norm = tensor;
        }
    }

    return true;
}

bool transformer_internal::ops::load_tensor_data(TTSTransformer & self, const std::string & path, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    ggml_backend_t backend = init_preferred_backend("TTSTransformer", &error_msg);
    if (!backend) {
        return false;
    }

    impl->model.buffer = ggml_backend_alloc_ctx_tensors(impl->model.ctx, backend);
    if (!impl->model.buffer) {
        error_msg = "Failed to allocate tensor buffer";
        release_preferred_backend(backend);
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        release_preferred_backend(backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = impl->model.tensors.find(name);
        if (it == impl->model.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

#ifdef _WIN32
        if (_fseeki64(f, (int64_t) data_offset + (int64_t) offset, SEEK_SET) != 0) {
#else
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    release_preferred_backend(backend);

    return true;
}

} // namespace qwen3_tts
