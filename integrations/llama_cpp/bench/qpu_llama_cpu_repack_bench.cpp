#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// Internal pinned-llama.cpp entry point used by the production CPU model loader.
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

constexpr uint32_t q4_0_block_elements = 32;
constexpr uint32_t q4_0_block_bytes = 18;
constexpr uint32_t q4_k_block_elements = 256;
constexpr uint32_t q4_k_block_bytes = 144;
constexpr uint32_t q6_k_block_elements = 256;
constexpr uint32_t q6_k_block_bytes = 210;
constexpr uint32_t q8_0_block_elements = 32;
constexpr uint32_t q8_0_block_bytes = 34;

enum class weight_format {
    q4_0,
    q4_k,
    q6_k,
    q8_0,
};

struct options {
    std::string weights_path;
    std::string activation_path;
    std::string output_path;
    weight_format format = weight_format::q4_0;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t rows = 0;
    uint32_t cpu_threads = 1;
    uint32_t warmups = 5;
    uint32_t samples = 31;
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool parse_u32(const char * text, uint32_t & result) {
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(value);
    return true;
}

void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s --weights FILE --activation-f32 FILE --output-bin FILE "
        "[--weight-type q4_0|q4_k|q6_k|q8_0] "
        "--input-columns N --output-columns N --rows 1|4 --cpu-threads N "
        "--warmups N --samples N\n",
        program);
}

bool parse_options(int argc, char ** argv, options & result) {
    options value;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        const std::string name = argv[index];
        const char * argument = argv[index + 1];
        if (name == "--weights") {
            value.weights_path = argument;
        } else if (name == "--activation-f32") {
            value.activation_path = argument;
        } else if (name == "--output-bin") {
            value.output_path = argument;
        } else if (name == "--weight-type") {
            if (std::strcmp(argument, "q4_0") == 0) {
                value.format = weight_format::q4_0;
            } else if (std::strcmp(argument, "q4_k") == 0) {
                value.format = weight_format::q4_k;
            } else if (std::strcmp(argument, "q6_k") == 0) {
                value.format = weight_format::q6_k;
            } else if (std::strcmp(argument, "q8_0") == 0) {
                value.format = weight_format::q8_0;
            } else {
                return false;
            }
        } else if (name == "--input-columns") {
            if (!parse_u32(argument, value.input_columns)) {
                return false;
            }
        } else if (name == "--output-columns") {
            if (!parse_u32(argument, value.output_columns)) {
                return false;
            }
        } else if (name == "--rows") {
            if (!parse_u32(argument, value.rows)) {
                return false;
            }
        } else if (name == "--cpu-threads") {
            if (!parse_u32(argument, value.cpu_threads)) {
                return false;
            }
        } else if (name == "--warmups") {
            if (!parse_u32(argument, value.warmups)) {
                return false;
            }
        } else if (name == "--samples") {
            if (!parse_u32(argument, value.samples)) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (value.weights_path.empty() || value.activation_path.empty() || value.output_path.empty() ||
        value.input_columns == 0 ||
        value.input_columns % (
            value.format == weight_format::q4_0 || value.format == weight_format::q8_0
                ? q4_0_block_elements : q4_k_block_elements) != 0 ||
        value.output_columns == 0 || (value.rows != 1 && value.rows != 4) ||
        value.cpu_threads == 0 || value.samples == 0) {
        return false;
    }
    result = value;
    return true;
}

std::vector<uint8_t> read_exact_file(const std::string & path, size_t expected_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<uint64_t>(input.tellg()) != expected_size) {
        std::fprintf(stderr, "%s does not contain exactly %zu bytes\n", path.c_str(), expected_size);
        return {};
    }
    std::vector<uint8_t> result(expected_size);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(expected_size))) {
        std::fprintf(stderr, "read %s failed\n", path.c_str());
        return {};
    }
    return result;
}

bool write_exact_file(const std::string & path, const void * data, size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!output) {
        std::fprintf(stderr, "write %s failed\n", path.c_str());
        return false;
    }
    return true;
}

void print_u64_array(const std::vector<uint64_t> & values) {
    std::putchar('[');
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::putchar(',');
        }
        std::printf("%llu", static_cast<unsigned long long>(values[index]));
    }
    std::putchar(']');
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    options config;
    if (!parse_options(argc, argv, config)) {
        usage(argv[0]);
        return 2;
    }
    const uint32_t block_elements = config.format == weight_format::q4_0
        ? q4_0_block_elements
        : config.format == weight_format::q4_k
            ? q4_k_block_elements
            : config.format == weight_format::q6_k ? q6_k_block_elements : q8_0_block_elements;
    const uint32_t block_bytes = config.format == weight_format::q4_0
        ? q4_0_block_bytes
        : config.format == weight_format::q4_k
            ? q4_k_block_bytes
            : config.format == weight_format::q6_k ? q6_k_block_bytes : q8_0_block_bytes;
    const ggml_type tensor_type = config.format == weight_format::q4_0
        ? GGML_TYPE_Q4_0
        : config.format == weight_format::q4_k
            ? GGML_TYPE_Q4_K
            : config.format == weight_format::q6_k ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
    const char * format_name = config.format == weight_format::q4_0
        ? "q4_0"
        : config.format == weight_format::q4_k
            ? "q4_k" : config.format == weight_format::q6_k ? "q6_k" : "q8_0";
    const size_t weight_bytes = static_cast<size_t>(config.output_columns) *
        (config.input_columns / block_elements) * block_bytes;
    const size_t activation_bytes = static_cast<size_t>(config.rows) *
        config.input_columns * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(config.rows) *
        config.output_columns * sizeof(float);
    const std::vector<uint8_t> weights = read_exact_file(config.weights_path, weight_bytes);
    const std::vector<uint8_t> activation = read_exact_file(config.activation_path, activation_bytes);
    if (weights.empty() || activation.empty()) {
        return 1;
    }

    ggml_init_params weight_params{};
    weight_params.mem_size = 2 * ggml_tensor_overhead();
    weight_params.no_alloc = true;
    ggml_context * weight_context = ggml_init(weight_params);
    if (weight_context == nullptr) {
        std::fprintf(stderr, "weight metadata context allocation failed\n");
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(weight_context, tensor_type,
        config.input_columns, config.output_columns);
    ggml_set_name(weight, "fixture.weight");
    const uint64_t prepare_start = monotonic_ns();
    ggml_backend_buffer_t weight_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(weight_context, ggml_backend_cpu_repack_buffer_type());
    if (weight_buffer == nullptr) {
        std::fprintf(stderr, "CPU_REPACK weight buffer allocation failed\n");
        ggml_free(weight_context);
        return 1;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    const uint64_t prepare_ns = monotonic_ns() - prepare_start;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "CPU backend initialization failed\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, static_cast<int>(config.cpu_threads));
    ggml_threadpool_params threadpool_params =
        ggml_threadpool_params_default(static_cast<int>(config.cpu_threads));
    ggml_threadpool_t threadpool = ggml_threadpool_new(&threadpool_params);
    if (threadpool == nullptr) {
        std::fprintf(stderr, "CPU threadpool initialization failed\n");
        return 1;
    }
    ggml_backend_cpu_set_threadpool(backend, threadpool);

    constexpr size_t graph_nodes = 8;
    ggml_init_params graph_params{};
    graph_params.mem_size = 8 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    graph_params.no_alloc = true;
    ggml_context * graph_context = ggml_init(graph_params);
    if (graph_context == nullptr) {
        std::fprintf(stderr, "graph metadata context allocation failed\n");
        return 1;
    }
    ggml_tensor * input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32,
        config.input_columns, config.rows);
    ggml_set_name(input, "fixture.activation");
    ggml_set_input(input);
    ggml_tensor * output = ggml_mul_mat(graph_context, weight, input);
    ggml_set_name(output, "fixture.output");
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors(graph_context, backend);
    if (graph_buffer == nullptr || !ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "CPU backend does not support the exact CPU_REPACK graph\n");
        return 1;
    }
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size());

    std::vector<uint64_t> complete_samples(config.samples);
    const uint32_t iterations = config.warmups + config.samples;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const uint64_t start = monotonic_ns();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const uint64_t end = monotonic_ns();
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "CPU_REPACK graph compute failed: %s\n", ggml_status_to_string(status));
            return 1;
        }
        if (iteration >= config.warmups) {
            complete_samples[iteration - config.warmups] = end - start;
        }
    }

    std::vector<uint8_t> result(output_bytes);
    ggml_backend_tensor_get(output, result.data(), 0, result.size());
    const bool output_written = write_exact_file(config.output_path, result.data(), result.size());
    std::printf("{\"schema_version\":1,\"kind\":\"llama-cpu-repack-native-node-samples\",")
        "\"mode\":\"cpu-repack\",\"weight_type\":\"%s\",\"input_columns\":%u,"
        "\"output_columns\":%u,\"rows\":%u,"
        "\"cpu_threads\":%u,\"warmups\":%u,\"retained_samples\":%u,\"prepare_ns\":%llu,"
        "\"weight_buffer_type\":\"%s\",\"weight_buffer_bytes\":%zu,"
        "\"output_bin_written\":%s,\"complete_ns\":",
        format_name, config.input_columns, config.output_columns, config.rows, config.cpu_threads,
        config.warmups,
        config.samples, static_cast<unsigned long long>(prepare_ns),
        ggml_backend_buffer_name(weight_buffer), ggml_backend_buffer_get_size(weight_buffer),
        output_written ? "true" : "false");
    print_u64_array(complete_samples);
    std::printf("}\n");

    ggml_backend_buffer_free(graph_buffer);
    ggml_free(graph_context);
    ggml_backend_free(backend);
    ggml_threadpool_free(threadpool);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_context);
    return output_written ? 0 : 1;
}
