#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <map>

// Helper to check if a tensor should be quantized
bool should_quantize(const std::string & name, enum ggml_type src_type, int n_dims) {
    if (src_type != GGML_TYPE_F32 && src_type != GGML_TYPE_F16) {
        return false; // Already quantized or unsupported
    }
    if (n_dims != 2) {
        return false; // Only quantize 2D matrices (Linear layers)
    }

    // Do not quantize layer norms, embeddings, or heads
    if (name.find("norm") != std::string::npos ||
        name.find("embd") != std::string::npos ||
        name.find("head") != std::string::npos ||
        name.find("codebook") != std::string::npos) {
        return false;
    }

    // Target the heavy transformer blocks in both models
    if (name.find("talker.blk.") != std::string::npos ||
        name.find("code_pred.blk.") != std::string::npos ||
        name.find("talker.text_proj.") != std::string::npos ||
        name.find("code_pred.small_to_mtp.") != std::string::npos) {
        return true;
    }

    return false;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "  Types: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k\n");
        return 1;
    }

    const char * fname_inp = argv[1];
    const char * fname_out = argv[2];
    const char * type_str  = argv[3];

    enum ggml_type qtype = GGML_TYPE_COUNT;
    if (strcmp(type_str, "q4_0") == 0) qtype = GGML_TYPE_Q4_0;
    else if (strcmp(type_str, "q4_1") == 0) qtype = GGML_TYPE_Q4_1;
    else if (strcmp(type_str, "q5_0") == 0) qtype = GGML_TYPE_Q5_0;
    else if (strcmp(type_str, "q5_1") == 0) qtype = GGML_TYPE_Q5_1;
    else if (strcmp(type_str, "q8_0") == 0) qtype = GGML_TYPE_Q8_0;
    else if (strcmp(type_str, "q2_k") == 0) qtype = GGML_TYPE_Q2_K;
    else if (strcmp(type_str, "q3_k") == 0) qtype = GGML_TYPE_Q3_K;
    else if (strcmp(type_str, "q4_k") == 0) qtype = GGML_TYPE_Q4_K;
    else if (strcmp(type_str, "q5_k") == 0) qtype = GGML_TYPE_Q5_K;
    else if (strcmp(type_str, "q6_k") == 0) qtype = GGML_TYPE_Q6_K;
    else {
        fprintf(stderr, "error: invalid quantization type '%s'\n", type_str);
        return 1;
    }

    printf("Input:      %s\n", fname_inp);
    printf("Output:     %s\n", fname_out);
    printf("Quant type: %s (%d)\n", type_str, qtype);

    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx_inp = gguf_init_from_file(fname_inp, params);
    if (!ctx_inp) {
        fprintf(stderr, "error: failed to open '%s'\n", fname_inp);
        return 1;
    }

    struct gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_inp);

    const int n_tensors = (int)gguf_get_n_tensors(ctx_inp);
    printf("Tensors:    %d\n\n", n_tensors);

    // Context for writing
    FILE * fout = nullptr;
#ifdef _WIN32
    fopen_s(&fout, fname_out, "wb");
#else
    fout = fopen(fname_out, "wb");
#endif
    if (!fout) {
        fprintf(stderr, "error: failed to open '%s' for writing\n", fname_out);
        gguf_free(ctx_inp);
        gguf_free(ctx_out);
        ggml_free(ctx_data);
        return 1;
    }

    // Prepare tensors for output
    struct ggml_init_params params_out = {
        /* .mem_size   = */ 1024 * 1024 * 10, // 10MB overhead
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx_out_data = ggml_init(params_out);

    std::vector<std::vector<uint8_t>> tensor_data;
    tensor_data.reserve(n_tensors);

    size_t total_size_inp = 0;
    size_t total_size_out = 0;

    // Buffer for F32 conversion
    std::vector<float> work_f32;

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_inp, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            fprintf(stderr, "error: failed to find tensor '%s' in ggml context\n", name);
            return 1;
        }

        enum ggml_type type = t->type;
        const int n_dims = ggml_n_dims(t);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int j = 0; j < n_dims; ++j) {
            ne[j] = t->ne[j];
        }

        const size_t nelements = ggml_nelements(t);
        size_t size_inp = ggml_nbytes(t);
        total_size_inp += size_inp;

        enum ggml_type new_type = type;
        if (should_quantize(name, type, n_dims)) {
            new_type = qtype;
            
            // Check block size alignment
            if (ne[0] % ggml_blck_size(new_type) != 0) {
                printf("[%4d/%4d] %-48s - F16 (skipped, %lld not divisible by %d)\n", 
                    i, n_tensors, name, (long long)ne[0], ggml_blck_size(new_type));
                new_type = type; // Fallback
            }
        }

        struct ggml_tensor * t_out = ggml_new_tensor(ctx_out_data, new_type, n_dims, ne);
        ggml_set_name(t_out, name);
        gguf_add_tensor(ctx_out, t_out);

        const size_t size_out = ggml_nbytes(t_out);
        total_size_out += size_out;

        std::vector<uint8_t> data(size_out);
        const void * data_inp = t->data;

        if (new_type == type) {
            // No quantization, just copy
            printf("[%4d/%4d] %-48s - %s\n", i, n_tensors, name, ggml_type_name(type));
            memcpy(data.data(), data_inp, size_inp);
        } else {
            // Quantize
            printf("[%4d/%4d] %-48s - %s -> %s\n", i, n_tensors, name, ggml_type_name(type), ggml_type_name(new_type));

            // Convert to F32 first if necessary
            const float * f32_data = nullptr;
            if (type == GGML_TYPE_F32) {
                f32_data = (const float *)data_inp;
            } else if (type == GGML_TYPE_F16) {
                work_f32.resize(nelements);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)data_inp, work_f32.data(), nelements);
                f32_data = work_f32.data();
            } else {
                fprintf(stderr, "error: unsupported source type %d\n", type);
                return 1;
            }

            // Perform quantization
            int64_t n_per_row = ne[0];
            int64_t nrows = ne[1];
            size_t qs = ggml_quantize_chunk(new_type, f32_data, data.data(), 0, nrows, n_per_row, nullptr);
            
            if (qs != size_out) {
                fprintf(stderr, "error: quantization size mismatch (expected %zu, got %zu)\n", size_out, qs);
                return 1;
            }
        }

        tensor_data.push_back(std::move(data));
    }

    // Write metadata
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size);
    gguf_get_meta_data(ctx_out, meta_data.data());
    fwrite(meta_data.data(), 1, meta_size, fout);

    // Write tensors
    for (int i = 0; i < n_tensors; ++i) {
        // Pad for alignment
        size_t offset = ftell(fout);
        size_t pad = GGML_PAD(offset, gguf_get_alignment(ctx_out)) - offset;
        if (pad > 0) {
            std::vector<uint8_t> zeros(pad, 0);
            fwrite(zeros.data(), 1, pad, fout);
        }

        fwrite(tensor_data[i].data(), 1, tensor_data[i].size(), fout);
    }

    fclose(fout);
    gguf_free(ctx_inp);
    gguf_free(ctx_out);
    ggml_free(ctx_data);
    ggml_free(ctx_out_data);

    printf("\nQuantization complete!\n");
    printf("Original size:  %8.2f MB\n", total_size_inp / 1024.0 / 1024.0);
    printf("Quantized size: %8.2f MB\n", total_size_out / 1024.0 / 1024.0);

    return 0;
}
