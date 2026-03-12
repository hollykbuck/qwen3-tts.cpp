#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace qwen3_tts {

struct pre_tfm_layer;
struct upsample_block;
struct residual_block;
struct decoder_block;
struct audio_decoder_private;

// Audio tokenizer decoder (vocoder) configuration
struct audio_decoder_config {
    int32_t sample_rate = 24000;
    int32_t n_codebooks = 16;           // Total codebooks (1 first + 15 rest)
    int32_t codebook_size = 2048;       // Entries per codebook
    int32_t codebook_dim = 256;         // Embedding dimension per codebook
    int32_t latent_dim = 1024;          // Latent dimension after VQ
    int32_t hidden_dim = 512;           // Pre-transformer hidden dimension
    int32_t n_pre_tfm_layers = 8;       // Pre-transformer layers
    int32_t n_heads = 16;               // Attention heads in pre-transformer
    int32_t ffn_dim = 1024;             // FFN intermediate dimension
    int32_t decoder_dim = 1536;         // Initial decoder dimension
    int32_t upsample_rates[4] = {8, 5, 4, 3};  // Total: 480x upsampling
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// Audio tokenizer decoder (vocoder) class
// Decodes discrete audio codes to waveform
class AudioTokenizerDecoder {
public:
    AudioTokenizerDecoder();
    ~AudioTokenizerDecoder();
    
    // Load model from GGUF file (tokenizer model)
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Decode audio codes to waveform
    // codes: audio codes [n_frames, n_codebooks] as int32_t (row-major)
    // n_frames: number of frames
    // Returns: audio samples normalized to [-1, 1] at 24kHz
    bool decode(const int32_t * codes, int32_t n_frames,
                std::vector<float> & samples);
    
    const audio_decoder_config & get_config() const;
    
    const std::string & get_error() const;
    
private:
    // Build computation graph for decoding
    struct ggml_cgraph * build_graph(int32_t n_frames);
    struct ggml_cgraph * build_graph_impl(int32_t n_frames, struct ggml_context ** graph_ctx_out);
    void release_cached_decode_graph();
    bool ensure_cached_decode_graph(int32_t n_frames);
    
    // Apply Snake activation: x + (1/alpha) * sin^2(alpha * x)
    struct ggml_tensor * apply_snake(struct ggml_context * ctx,
                                      struct ggml_tensor * x,
                                      struct ggml_tensor * alpha,
                                      struct ggml_tensor * beta);
    
    // Apply RMSNorm
    struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct ggml_tensor * w,
                                         float eps);
    
    // Apply pre-transformer layer
    struct ggml_tensor * apply_pre_tfm_layer(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const pre_tfm_layer & layer,
                                              int32_t n_frames,
                                              struct ggml_tensor * positions);
    
    // Apply upsample block (ConvNeXt-style)
    struct ggml_tensor * apply_upsample_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const upsample_block & block,
                                               int block_idx);
    
    // Apply residual block
    struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const residual_block & block);
    
    // Apply decoder block (Snake + ConvTranspose + Residuals)
    struct ggml_tensor * apply_decoder_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const decoder_block & block,
                                               int upsample_rate,
                                               int block_idx);
    
    void normalize_codebooks();

    std::unique_ptr<audio_decoder_private> impl_;
};

} // namespace qwen3_tts
