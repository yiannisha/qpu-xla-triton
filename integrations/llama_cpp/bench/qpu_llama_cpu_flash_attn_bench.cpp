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

namespace {

struct options {
    std::string query_path;
    std::string key_path;
    std::string value_path;
    std::string mask_path;
    std::string output_path;
    uint32_t heads = 8;
    uint32_t kv_rows = 0;
    uint32_t head_dim = 256;
    uint32_t cpu_threads = 3;
    uint32_t warmups = 5;
    uint32_t samples = 31;
    float scale = 0.0625f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
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

bool parse_float(const char * text, float & result) {
    char * end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    result = value;
    return true;
}

void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s --query-f32 FILE --key-f16 FILE --value-f16 FILE --mask-f16 FILE "
        "--output-bin FILE --heads N --kv-rows N --head-dim N --cpu-threads N "
        "--warmups N --samples N --scale F --max-bias F --logit-softcap F\n",
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
        if (name == "--query-f32") {
            value.query_path = argument;
        } else if (name == "--key-f16") {
            value.key_path = argument;
        } else if (name == "--value-f16") {
            value.value_path = argument;
        } else if (name == "--mask-f16") {
            value.mask_path = argument;
        } else if (name == "--output-bin") {
            value.output_path = argument;
        } else if (name == "--heads") {
            if (!parse_u32(argument, value.heads)) {
                return false;
            }
        } else if (name == "--kv-rows") {
            if (!parse_u32(argument, value.kv_rows)) {
                return false;
            }
        } else if (name == "--head-dim") {
            if (!parse_u32(argument, value.head_dim)) {
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
        } else if (name == "--scale") {
            if (!parse_float(argument, value.scale)) {
                return false;
            }
        } else if (name == "--max-bias") {
            if (!parse_float(argument, value.max_bias)) {
                return false;
            }
        } else if (name == "--logit-softcap") {
            if (!parse_float(argument, value.logit_softcap)) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (value.query_path.empty() || value.key_path.empty() || value.value_path.empty() ||
        value.mask_path.empty() || value.output_path.empty() || value.heads == 0 ||
        value.kv_rows == 0 || value.head_dim == 0 || value.cpu_threads == 0 ||
        value.samples == 0) {
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
    return static_cast<bool>(output);
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
    const size_t query_bytes = static_cast<size_t>(config.heads) * config.head_dim * sizeof(float);
    const size_t kv_bytes = static_cast<size_t>(config.kv_rows) * config.head_dim * sizeof(ggml_fp16_t);
    const size_t mask_bytes = static_cast<size_t>(config.kv_rows) * sizeof(ggml_fp16_t);
    const size_t output_bytes = query_bytes;
    const std::vector<uint8_t> query_data = read_exact_file(config.query_path, query_bytes);
    const std::vector<uint8_t> key_data = read_exact_file(config.key_path, kv_bytes);
    const std::vector<uint8_t> value_data = read_exact_file(config.value_path, kv_bytes);
    const std::vector<uint8_t> mask_data = read_exact_file(config.mask_path, mask_bytes);
    if (query_data.empty() || key_data.empty() || value_data.empty() || mask_data.empty()) {
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "CPU backend initialization failed\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, static_cast<int>(config.cpu_threads));
    ggml_threadpool_params pool_params =
        ggml_threadpool_params_default(static_cast<int>(config.cpu_threads));
    ggml_threadpool_t threadpool = ggml_threadpool_new(&pool_params);
    if (threadpool == nullptr) {
        std::fprintf(stderr, "CPU threadpool initialization failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_cpu_set_threadpool(backend, threadpool);

    constexpr size_t graph_nodes = 8;
    ggml_init_params params{};
    params.mem_size = 16 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        std::fprintf(stderr, "graph metadata context allocation failed\n");
        return 1;
    }
    ggml_tensor * query = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, config.head_dim, 1, config.heads, 1);
    ggml_tensor * key = ggml_new_tensor_4d(
        context, GGML_TYPE_F16, config.head_dim, config.kv_rows, 1, 1);
    ggml_tensor * value = ggml_new_tensor_4d(
        context, GGML_TYPE_F16, config.head_dim, config.kv_rows, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(
        context, GGML_TYPE_F16, config.kv_rows, 1, 1, 1);
    ggml_set_input(query);
    ggml_set_input(key);
    ggml_set_input(value);
    ggml_set_input(mask);
    ggml_tensor * output = ggml_flash_attn_ext(
        context, query, key, value, mask,
        config.scale, config.max_bias, config.logit_softcap);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr || !ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "CPU backend does not support the exact FLASH_ATTN_EXT graph\n");
        return 1;
    }
    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size());
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size());
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size());
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size());

    std::vector<uint64_t> complete_samples(config.samples);
    const uint32_t iterations = config.warmups + config.samples;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const uint64_t start = monotonic_ns();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const uint64_t end = monotonic_ns();
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FLASH_ATTN_EXT graph compute failed: %s\n",
                ggml_status_to_string(status));
            return 1;
        }
        if (iteration >= config.warmups) {
            complete_samples[iteration - config.warmups] = end - start;
        }
    }
    std::vector<uint8_t> output_data(output_bytes);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size());
    if (!write_exact_file(config.output_path, output_data.data(), output_data.size())) {
        std::fprintf(stderr, "write %s failed\n", config.output_path.c_str());
        return 1;
    }
    std::printf(
        "{\"kind\":\"llama-cpu-flash-attention-node-samples\","
        "\"heads\":%u,\"kv_rows\":%u,\"head_dim\":%u,\"cpu_threads\":%u,"
        "\"warmups\":%u,\"samples\":%u,\"complete_ns\":",
        config.heads, config.kv_rows, config.head_dim, config.cpu_threads,
        config.warmups, config.samples);
    print_u64_array(complete_samples);
    std::printf("}\n");

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_cpu_set_threadpool(backend, nullptr);
    ggml_threadpool_free(threadpool);
    ggml_backend_free(backend);
    return 0;
}
