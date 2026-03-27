#include "audio_tokenizer_decoder.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool has_gpu_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        return false;
    }
    ggml_backend_free(backend);
    return true;
}

static bool all_finite(const std::vector<float> & samples) {
    for (float sample : samples) {
        if (!std::isfinite(sample)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * tokenizer_path = "models/qwen3-tts-tokenizer-f16.gguf";
    int stress_frames = 420;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            stress_frames = std::max(1, std::atoi(argv[++i]));
        }
    }

    printf("=== Decoder CUDA Long Regression Test ===\n\n");

    if (!has_gpu_backend()) {
        printf("SKIP: No GPU backend available\n");
        return 0;
    }

    qwen3_tts::AudioTokenizerDecoder decoder;
    if (!decoder.load_model(tokenizer_path)) {
        fprintf(stderr, "FAIL: load_model: %s\n", decoder.get_error().c_str());
        return 1;
    }

    const auto & cfg = decoder.get_config();
    printf("Model loaded: sample_rate=%d, n_codebooks=%d, codebook_size=%d\n",
           cfg.sample_rate, cfg.n_codebooks, cfg.codebook_size);

    std::vector<int32_t> short_codes((size_t) cfg.n_codebooks);
    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        short_codes[cb] = cb % std::max(1, cfg.codebook_size);
    }

    std::vector<float> short_samples;
    if (!decoder.decode(short_codes.data(), 1, short_samples)) {
        fprintf(stderr, "FAIL: 1-frame decode: %s\n", decoder.get_error().c_str());
        return 1;
    }
    if (short_samples.empty() || !all_finite(short_samples)) {
        fprintf(stderr, "FAIL: 1-frame decode produced invalid samples\n");
        return 1;
    }
    printf("Short decode: %zu samples\n", short_samples.size());

    std::vector<int32_t> stress_codes((size_t) stress_frames * cfg.n_codebooks);
    for (int frame = 0; frame < stress_frames; ++frame) {
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            stress_codes[(size_t) frame * cfg.n_codebooks + cb] =
                (frame + cb) % std::max(1, cfg.codebook_size);
        }
    }

    std::vector<float> stress_samples;
    if (!decoder.decode(stress_codes.data(), stress_frames, stress_samples)) {
        fprintf(stderr, "FAIL: %d-frame decode: %s\n", stress_frames, decoder.get_error().c_str());
        return 1;
    }
    if (stress_samples.empty()) {
        fprintf(stderr, "FAIL: %d-frame decode produced no samples\n", stress_frames);
        return 1;
    }
    if (!all_finite(stress_samples)) {
        fprintf(stderr, "FAIL: %d-frame decode produced non-finite samples\n", stress_frames);
        return 1;
    }
    if (stress_samples.size() <= short_samples.size()) {
        fprintf(stderr, "FAIL: long decode did not increase output length (%zu <= %zu)\n",
                stress_samples.size(), short_samples.size());
        return 1;
    }

    printf("Long decode: %d frames -> %zu samples\n", stress_frames, stress_samples.size());
    printf("PASS\n");
    return 0;
}
