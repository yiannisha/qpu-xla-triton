#include "ggml.h"
#include "llama.h"
#include "qpu_llama_ffn_island.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double maximum_probability_error_limit = 1.0e-4;
constexpr double total_variation_limit = 1.0e-4;
constexpr double kl_divergence_limit = 1.0e-5;

struct metrics {
    double raw_maximum_absolute_error = 0.0;
    double raw_mean_absolute_error = 0.0;
    double raw_rms_error = 0.0;
    double mean_signed_difference = 0.0;
    double centered_maximum_absolute_error = 0.0;
    double centered_mean_absolute_error = 0.0;
    double centered_rms_error = 0.0;
    double maximum_probability_error = 0.0;
    double total_variation = 0.0;
    double kl_divergence = 0.0;
    int32_t cpu_argmax = -1;
    int32_t candidate_argmax = -1;
    bool finite = true;
};

void quiet_log_callback(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::vector<llama_token> tokenize_exact(
        const llama_vocab * vocab, std::string text, int32_t count) {
    std::vector<llama_token> tokens;
    for (;;) {
        const int32_t required = llama_tokenize(
            vocab, text.c_str(), static_cast<int32_t>(text.size()),
            nullptr, 0, true, true);
        if (required == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("tokenization size overflow");
        }
        const int32_t capacity = required < 0 ? -required : required;
        tokens.resize(static_cast<size_t>(capacity));
        const int32_t produced = llama_tokenize(
            vocab, text.c_str(), static_cast<int32_t>(text.size()),
            tokens.data(), capacity, true, true);
        if (produced < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize(static_cast<size_t>(produced));
        if (produced >= count) {
            tokens.resize(static_cast<size_t>(count));
            return tokens;
        }
        text += text;
    }
}

std::vector<float> decode_last_logits(
        llama_context * context, const std::vector<llama_token> & tokens,
        int32_t vocabulary, bool candidate) {
    setenv("GGML_QPU_FFN_ISLAND", candidate ? "1" : "0", 1);
    llama_memory_clear(llama_get_memory(context), true);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = tokens[static_cast<size_t>(index)];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index + 1 == batch.n_tokens ? 1 : 0;
    }
    const int32_t status = llama_decode(context, batch);
    llama_batch_free(batch);
    if (status != 0) {
        throw std::runtime_error("llama_decode failed with status " +
            std::to_string(status));
    }
    const float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
        throw std::runtime_error("decode returned no final logits");
    }
    return std::vector<float>(logits, logits + vocabulary);
}

double log_sum_exp(const std::vector<float> & values) {
    const double maximum = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values) {
        sum += std::exp(static_cast<double>(value) - maximum);
    }
    return maximum + std::log(sum);
}

metrics compare(
        const std::vector<float> & cpu, const std::vector<float> & candidate) {
    metrics result;
    result.cpu_argmax = static_cast<int32_t>(
        std::max_element(cpu.begin(), cpu.end()) - cpu.begin());
    result.candidate_argmax = static_cast<int32_t>(
        std::max_element(candidate.begin(), candidate.end()) - candidate.begin());
    const double cpu_normalizer = log_sum_exp(cpu);
    const double candidate_normalizer = log_sum_exp(candidate);
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    for (size_t index = 0; index < cpu.size(); ++index) {
        const double left = cpu[index];
        const double right = candidate[index];
        const double error = std::fabs(left - right);
        result.raw_maximum_absolute_error = std::max(
            result.raw_maximum_absolute_error, error);
        absolute_sum += error;
        squared_sum += error * error;
        result.mean_signed_difference += right - left;
        const double log_cpu_probability = left - cpu_normalizer;
        const double log_candidate_probability = right - candidate_normalizer;
        const double cpu_probability = std::exp(log_cpu_probability);
        const double candidate_probability = std::exp(log_candidate_probability);
        const double probability_error = std::fabs(
            cpu_probability - candidate_probability);
        result.maximum_probability_error = std::max(
            result.maximum_probability_error, probability_error);
        result.total_variation += probability_error;
        result.kl_divergence += cpu_probability *
            (log_cpu_probability - log_candidate_probability);
        result.finite = result.finite && std::isfinite(left) &&
            std::isfinite(right);
    }
    result.raw_mean_absolute_error = absolute_sum / cpu.size();
    result.raw_rms_error = std::sqrt(squared_sum / cpu.size());
    result.mean_signed_difference /= cpu.size();
    result.total_variation *= 0.5;
    absolute_sum = 0.0;
    squared_sum = 0.0;
    for (size_t index = 0; index < cpu.size(); ++index) {
        const double error = static_cast<double>(candidate[index]) - cpu[index] -
            result.mean_signed_difference;
        const double absolute_error = std::fabs(error);
        result.centered_maximum_absolute_error = std::max(
            result.centered_maximum_absolute_error, absolute_error);
        absolute_sum += absolute_error;
        squared_sum += error * error;
    }
    result.centered_mean_absolute_error = absolute_sum / cpu.size();
    result.centered_rms_error = std::sqrt(squared_sum / cpu.size());
    result.finite = result.finite &&
        std::isfinite(result.kl_divergence);
    return result;
}

bool passes(const metrics & value) {
    return value.finite &&
        value.maximum_probability_error <= maximum_probability_error_limit &&
        value.total_variation <= total_variation_limit &&
        value.kl_divergence >= -1.0e-12 &&
        value.kl_divergence <= kl_divergence_limit &&
        value.cpu_argmax == value.candidate_argmax;
}

} // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

int main(int argc, char ** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--help") == 0) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return argc == 2 ? 0 : 2;
    }
    try {
        setenv("GGML_QPU_FFN_ISLAND_FRACTION", "0.125", 1);
        // Registration occurs while model weights are loaded. Individual
        // reference decodes disable execution after the resident QPU copies exist.
        setenv("GGML_QPU_FFN_ISLAND", "1", 1);
        setenv("GGML_QPU_FFN_ISLAND_MIN_ROWS", "64", 1);
        setenv("GGML_QPU_FFN_ISLAND_MAX_ROWS", "528", 1);
        setenv("GGML_QPU_FFN_ISLAND_WGS", "24", 1);
        setenv("GGML_QPU_TELEMETRY", "0", 1);
        unsetenv("GGML_QPU_FFN_ISLAND_VERIFY");
        unsetenv("GGML_QPU_FFN_ISLAND_FAIL_STAGE");
        unsetenv("GGML_QPU_FFN_ISLAND_FAIL_LAYER");
        llama_log_set(quiet_log_callback, nullptr);
        llama_backend_init();

        llama_model_params model_parameters = llama_model_default_params();
        model_parameters.n_gpu_layers = 0;
        const llama_model_tensor_buft_override repack_overrides[] = {
            {
                "blk\\.[0-9]+\\.ffn_(gate|up|down)\\.weight",
                ggml_backend_cpu_repack_buffer_type(),
            },
            {nullptr, nullptr},
        };
        model_parameters.tensor_buft_overrides = repack_overrides;
        llama_model * model = llama_model_load_from_file(argv[1], model_parameters);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model");
        }
        llama_context_params context_parameters = llama_context_default_params();
        context_parameters.n_ctx = 512;
        context_parameters.n_batch = 512;
        context_parameters.n_ubatch = 512;
        context_parameters.n_outputs_max = 1;
        context_parameters.n_threads = 4;
        context_parameters.n_threads_batch = 4;
        llama_context * context = llama_init_from_model(model, context_parameters);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("failed to create context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocabulary = llama_vocab_n_tokens(vocab);
        const int32_t layers = llama_model_n_layer(model);
        const std::array<std::string, 3> prompts = {
            "A careful engineer compares complete numerical results before reporting performance. ",
            "During an agent workflow, a tool returns structured observations and the model reasons over them. ",
            "The quick brown fox crosses the quiet laboratory while deterministic instruments record each value. ",
        };
        const std::array<int32_t, 2> row_counts = {129, 257};
        const uint64_t initial_dispatches = ggml_qpu_ffn_island_dispatch_count();
        const uint64_t initial_islands = ggml_qpu_ffn_island_count();
        bool passed = true;
        for (int32_t rows : row_counts) {
            for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
                const std::vector<llama_token> tokens = tokenize_exact(
                    vocab, prompts[prompt_index], rows);
                const std::vector<float> cpu = decode_last_logits(
                    context, tokens, vocabulary, false);
                const std::vector<float> candidate = decode_last_logits(
                    context, tokens, vocabulary, true);
                const metrics observed = compare(cpu, candidate);
                const bool case_passed = passes(observed);
                passed = passed && case_passed;
                std::printf(
                    "{\"kind\":\"qpu-ffn-island-model-shape\","
                    "\"m\":%d,\"prompt_index\":%zu,\"vocabulary\":%d,"
                    "\"raw_maximum_absolute_error\":%.9g,"
                    "\"raw_mean_absolute_error\":%.9g,\"raw_rms_error\":%.9g,"
                    "\"mean_signed_difference\":%.9g,"
                    "\"centered_maximum_absolute_error\":%.9g,"
                    "\"centered_mean_absolute_error\":%.9g,"
                    "\"centered_rms_error\":%.9g,"
                    "\"maximum_probability_error\":%.9g,"
                    "\"total_variation\":%.9g,"
                    "\"kl_divergence_cpu_to_candidate\":%.9g,"
                    "\"cpu_argmax\":%d,\"candidate_argmax\":%d,"
                    "\"argmax_equal\":%s,\"passed\":%s}\n",
                    rows, prompt_index, vocabulary,
                    observed.raw_maximum_absolute_error,
                    observed.raw_mean_absolute_error, observed.raw_rms_error,
                    observed.mean_signed_difference,
                    observed.centered_maximum_absolute_error,
                    observed.centered_mean_absolute_error,
                    observed.centered_rms_error,
                    observed.maximum_probability_error,
                    observed.total_variation,
                    observed.kl_divergence, observed.cpu_argmax,
                    observed.candidate_argmax,
                    observed.cpu_argmax == observed.candidate_argmax ? "true" : "false",
                    case_passed ? "true" : "false");
                std::fflush(stdout);
            }
        }
        const uint64_t dispatches =
            ggml_qpu_ffn_island_dispatch_count() - initial_dispatches;
        const uint64_t islands =
            ggml_qpu_ffn_island_count() - initial_islands;
        const uint64_t expected_islands =
            static_cast<uint64_t>(layers) * row_counts.size() * prompts.size();
        passed = passed && islands == expected_islands && dispatches == 4 * islands;
        std::printf(
            "{\"kind\":\"qpu-ffn-island-model-smoke\","
            "\"cases\":%zu,\"layers\":%d,\"island_count\":%llu,"
            "\"expected_island_count\":%llu,\"dispatch_count\":%llu,"
            "\"limits\":{\"maximum_probability_error\":%.9g,"
            "\"total_variation\":%.9g,"
            "\"kl_divergence_cpu_to_candidate\":%.9g},\"passed\":%s}\n",
            row_counts.size() * prompts.size(), layers,
            static_cast<unsigned long long>(islands),
            static_cast<unsigned long long>(expected_islands),
            static_cast<unsigned long long>(dispatches),
            maximum_probability_error_limit, total_variation_limit,
            kl_divergence_limit, passed ? "true" : "false");

        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qpu_ffn_island_model_smoke: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
}
