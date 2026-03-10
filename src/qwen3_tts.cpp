#include "qwen3_tts.h"
#include "pipeline/pipeline_internal.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <limits>

namespace qwen3_tts {
using pipeline_internal::format_bytes;
using pipeline_internal::get_process_memory_snapshot;
using pipeline_internal::get_time_ms;
using pipeline_internal::log_memory_usage;
using pipeline_internal::process_memory_snapshot;
using pipeline_internal::resample_linear;

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    
    if (!params.speaker.empty()) {
        std::vector<float> speaker_embedding;
        if (!transformer_.get_named_speaker_embedding(params.speaker, speaker_embedding)) {
            result.error_msg = "Failed to resolve speaker '" + params.speaker + "': " + transformer_.get_error();
            return result;
        }
        if (params.print_progress) {
            fprintf(stderr, "Using named speaker: %s (%zu floats)\n",
                    params.speaker.c_str(), speaker_embedding.size());
        }
        return synthesize_internal(text, speaker_embedding.data(), params, result);
    }

    if (transformer_.get_config().tts_model_type == "custom_voice") {
        result.error_msg = "CustomVoice model requires --speaker, --reference, or --speaker-embedding";
        return result;
    }

    // Default path: no speaker conditioning.
    return synthesize_internal(text, nullptr, params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            log_memory_usage("voice/after-encoder-load");
        }
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after extraction: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
    }
    
    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

tts_result Qwen3TTS::synthesize_with_speaker_embedding(const std::string & text,
                                                        const std::vector<float> & speaker_embedding,
                                                        const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Using provided speaker embedding: %zu floats\n", speaker_embedding.size());
    }

    result.t_encode_ms = 0;
    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                          std::vector<float> & speaker_embedding,
                                          int64_t * encode_time_ms) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    std::vector<float> ref_samples;
    int ref_sample_rate = 0;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int) ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                (long long) (get_time_ms() - t_encoder_load_start));
        log_memory_usage("voice/after-encoder-load");
    }

    const int64_t t_encode_start = get_time_ms();
    if (!audio_encoder_.encode(ref_samples.data(), (int32_t) ref_samples.size(), speaker_embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after extraction: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        error_msg_ = buf;
        return false;
    }

    if (encode_time_ms) {
        *encode_time_ms = get_time_ms() - t_encode_start;
    }
    return true;
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text
    int64_t t_tokenize_start = get_time_ms();
    // Match Python reference behavior:
    // - main text uses assistant prompt format
    // - optional style instruction is passed separately as instruct tokens
    std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    if (!params.instruction.empty()) {
        instruct_tokens = tokenizer_.encode_instruct(params.instruction);
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    if (!params.instruction.empty() && instruct_tokens.empty()) {
        result.error_msg = "Failed to tokenize instruction";
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        if (!instruct_tokens.empty()) {
            fprintf(stderr, "Instruction tokenized: %zu tokens\n", instruct_tokens.size());
        }
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.clear_kv_cache();

    std::vector<int32_t> speech_codes;
    if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                               speaker_embedding, params.max_audio_tokens, speech_codes,
                               params.language_id, params.repetition_penalty,
                               params.temperature, params.top_k,
                               instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                               (int32_t) instruct_tokens.size())) {
        result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
        return result;
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    sample_memory("synth/after-generate");
    
    int n_codebooks = transformer_.get_config().n_codebooks;
    int n_frames = (int)speech_codes.size() / n_codebooks;
    
    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();
    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    if (!audio_decoder_.decode(speech_codes.data(), n_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

// WAV file loading (16-bit PCM or 32-bit float)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }
    
    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }
    
    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }
    
    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;
    
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;
            
            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 32768.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 2147483648.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                samples.resize(n_samples);
                
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    fclose(f);
                    return false;
                }
                
                // Convert to mono
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[i * num_channels + c];
                    }
                    samples[i] = sum / num_channels;
                }
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }
            
            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// WAV file saving (16-bit PCM at specified sample rate)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    
    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = samples.size() * block_align;
    uint32_t file_size = 36 + data_size;
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        // Clamp to [-1, 1] and convert to int16
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }
    
    fclose(f);
    return true;
}

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool has_json_extension(const std::string & path) {
    const size_t pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return false;
    }
    const std::string ext = to_lower_ascii(path.substr(pos));
    return ext == ".json";
}

static bool parse_embedding_text(const std::string & text, std::vector<float> & embedding) {
    std::string cleaned = text;
    for (char & c : cleaned) {
        if (c == '[' || c == ']' || c == ',' || c == ';') {
            c = ' ';
        }
    }

    std::istringstream iss(cleaned);
    float value = 0.0f;
    embedding.clear();
    while (iss >> value) {
        embedding.push_back(value);
    }
    return !embedding.empty();
}

bool load_speaker_embedding_file(const std::string & path,
                                 std::vector<float> & embedding) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "ERROR: Cannot open speaker embedding file: %s\n", path.c_str());
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    if (data.empty()) {
        fprintf(stderr, "ERROR: Speaker embedding file is empty: %s\n", path.c_str());
        return false;
    }

    if (has_json_extension(path) || data.find('[') != std::string::npos) {
        if (!parse_embedding_text(data, embedding)) {
            fprintf(stderr, "ERROR: Failed to parse speaker embedding JSON/text: %s\n", path.c_str());
            return false;
        }
        return true;
    }

    if (data.size() % sizeof(float) != 0) {
        fprintf(stderr, "ERROR: Speaker embedding binary size is not a multiple of 4 bytes: %s\n", path.c_str());
        return false;
    }

    embedding.resize(data.size() / sizeof(float));
    memcpy(embedding.data(), data.data(), data.size());
    return true;
}

bool save_speaker_embedding_file(const std::string & path,
                                 const std::vector<float> & embedding) {
    if (embedding.empty()) {
        fprintf(stderr, "ERROR: Refusing to save empty speaker embedding\n");
        return false;
    }

    if (has_json_extension(path)) {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) {
            fprintf(stderr, "ERROR: Cannot create speaker embedding JSON file: %s\n", path.c_str());
            return false;
        }
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "[\n";
        for (size_t i = 0; i < embedding.size(); ++i) {
            out << "  " << embedding[i];
            if (i + 1 != embedding.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
        return true;
    }

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create speaker embedding binary file: %s\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char *>(embedding.data()),
              (std::streamsize) (embedding.size() * sizeof(float)));
    return out.good();
}

} // namespace qwen3_tts
