#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <vector>

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

struct options {
    int64_t rows = 16;
    int64_t columns = 6144;
    int64_t qpu_rows = 0;
    int cpu_threads = 1;
    int warmups = 1;
    int samples = 1;
};

struct resources {
    ggml_backend_t cpu = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * context = nullptr;

    ~resources() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(cpu);
    }
};

struct m1_resources {
    ggml_backend_t cpu = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_context * weight_context = nullptr;
    ggml_context * graph_context = nullptr;

    ~m1_resources() {
        ggml_backend_buffer_free(graph_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_free(cpu);
        ggml_free(graph_context);
        ggml_free(weight_context);
    }
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool positive_integer(const char * text, int64_t & result) {
    char * end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0) {
        return false;
    }
    result = value;
    return true;
}

void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--columns N] [--cpu-threads N] "
        "[--qpu-rows N] [--warmups N] [--samples N]\n",
        program);
}

bool parse_options(int argc, char ** argv, options & result) {
    options parsed;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        int64_t value = 0;
        if (!positive_integer(argv[index + 1], value)) {
            return false;
        }
        if (std::strcmp(argv[index], "--rows") == 0) {
            parsed.rows = value;
        } else if (std::strcmp(argv[index], "--columns") == 0) {
            parsed.columns = value;
        } else if (std::strcmp(argv[index], "--qpu-rows") == 0) {
            parsed.qpu_rows = value;
        } else if (std::strcmp(argv[index], "--cpu-threads") == 0 &&
            value <= std::numeric_limits<int>::max()) {
            parsed.cpu_threads = static_cast<int>(value);
        } else if (std::strcmp(argv[index], "--warmups") == 0 &&
            value <= std::numeric_limits<int>::max()) {
            parsed.warmups = static_cast<int>(value);
        } else if (std::strcmp(argv[index], "--samples") == 0 &&
            value <= std::numeric_limits<int>::max()) {
            parsed.samples = static_cast<int>(value);
        } else {
            return false;
        }
    }
    if (parsed.rows > INT64_MAX / parsed.columns ||
        parsed.rows * parsed.columns % 768 != 0 ||
        (parsed.qpu_rows != 0 &&
         (parsed.qpu_rows >= parsed.rows || parsed.cpu_threads < 2 ||
          parsed.qpu_rows > INT64_MAX / parsed.columns ||
          parsed.qpu_rows * parsed.columns % 768 != 0))) {
        return false;
    }
    result = parsed;
    return true;
}

void print_samples(const std::vector<uint64_t> & values) {
    std::putchar('[');
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::putchar(',');
        }
        std::printf("%llu", static_cast<unsigned long long>(values[index]));
    }
    std::putchar(']');
}

bool run_m1_inline_smoke() {
    using dispatch_count_fn = uint64_t (*)();
    auto dispatch_count = reinterpret_cast<dispatch_count_fn>(
        dlsym(RTLD_DEFAULT, "ggml_qpu_m1_dispatch_count"));
    if (dispatch_count == nullptr) {
        return false;
    }
    constexpr int64_t input_columns = 256;
    constexpr int64_t output_columns = 2048;
    constexpr size_t block_elements = 32;
    constexpr size_t block_bytes = 18;
    const size_t weight_bytes = static_cast<size_t>(output_columns) *
        (input_columns / block_elements) * block_bytes;
    std::vector<uint8_t> weight_values(weight_bytes);
    for (size_t block = 0; block < weight_bytes / block_bytes; ++block) {
        uint8_t * destination = weight_values.data() + block * block_bytes;
        const ggml_fp16_t scale = ggml_fp32_to_fp16(
            0.002F + 0.0001F * static_cast<float>(block % 17));
        std::memcpy(destination, &scale, sizeof(scale));
        for (size_t byte = sizeof(scale); byte < block_bytes; ++byte) {
            const uint8_t low = static_cast<uint8_t>((block + byte) % 16);
            const uint8_t high = static_cast<uint8_t>((3 * block + byte + 5) % 16);
            destination[byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    std::vector<float> activation(static_cast<size_t>(input_columns));
    for (size_t index = 0; index < activation.size(); ++index) {
        activation[index] = 0.75F * std::sin(static_cast<float>(index) * 0.031F) -
            0.2F * std::cos(static_cast<float>(index) * 0.007F);
    }

    setenv("GGML_QPU_M1_INLINE", "1", 1);
    setenv("GGML_QPU_M1_TENSORS", "fixture.m1.weight", 1);
    m1_resources state;
    ggml_init_params weight_params = {};
    weight_params.mem_size = 2 * ggml_tensor_overhead();
    weight_params.no_alloc = true;
    state.weight_context = ggml_init(weight_params);
    if (state.weight_context == nullptr) {
        return false;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(
        state.weight_context, GGML_TYPE_Q4_0, input_columns, output_columns);
    ggml_set_name(weight, "fixture.m1.weight");
    state.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        state.weight_context, ggml_backend_cpu_repack_buffer_type());
    if (state.weight_buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(
        state.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(weight, weight_values.data(), 0, weight_values.size());

    state.cpu = ggml_backend_cpu_init();
    if (state.cpu == nullptr) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(state.cpu, 1);
    constexpr size_t graph_nodes = 4;
    ggml_init_params graph_params = {};
    graph_params.mem_size = 4 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    graph_params.no_alloc = true;
    state.graph_context = ggml_init(graph_params);
    if (state.graph_context == nullptr) {
        return false;
    }
    ggml_tensor * input = ggml_new_tensor_2d(
        state.graph_context, GGML_TYPE_F32, input_columns, 1);
    ggml_tensor * output = ggml_mul_mat(state.graph_context, weight, input);
    ggml_set_input(input);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(
        state.graph_context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    state.graph_buffer = ggml_backend_alloc_ctx_tensors(state.graph_context, state.cpu);
    if (state.graph_buffer == nullptr) {
        return false;
    }
    ggml_backend_tensor_set(
        input, activation.data(), 0, activation.size() * sizeof(float));

    setenv("GGML_QPU_M1_INLINE", "0", 1);
    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    std::vector<float> expected(static_cast<size_t>(output_columns));
    ggml_backend_tensor_get(
        output, expected.data(), 0, expected.size() * sizeof(float));
    const uint64_t before = dispatch_count();
    setenv("GGML_QPU_M1_INLINE", "1", 1);
    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    std::vector<float> actual(static_cast<size_t>(output_columns));
    ggml_backend_tensor_get(
        output, actual.data(), 0, actual.size() * sizeof(float));
    if (dispatch_count() != before + 1) {
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        const float tolerance = 2.0e-4F + 2.0e-5F * std::fabs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

} // namespace

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
    using dispatch_count_fn = uint64_t (*)();
    auto dispatch_count = reinterpret_cast<dispatch_count_fn>(
        dlsym(RTLD_DEFAULT, "ggml_qpu_geglu_dispatch_count"));
    if (dispatch_count == nullptr) {
        std::fprintf(stderr, "inline QPU preload library is not loaded\n");
        return 1;
    }

    resources state;
    state.cpu = ggml_backend_cpu_init();
    if (state.cpu == nullptr) {
        return 1;
    }
    ggml_backend_cpu_set_n_threads(state.cpu, config.cpu_threads);
    constexpr size_t graph_nodes = 8;
    ggml_init_params params = {};
    params.mem_size = 8 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    params.no_alloc = true;
    state.context = ggml_init(params);
    if (state.context == nullptr) {
        return 1;
    }
    ggml_tensor * gate = ggml_new_tensor_2d(
        state.context, GGML_TYPE_F32, config.columns, config.rows);
    ggml_tensor * up = ggml_new_tensor_2d(
        state.context, GGML_TYPE_F32, config.columns, config.rows);
    ggml_tensor * output = ggml_geglu_split(state.context, gate, up);
    ggml_set_input(gate);
    ggml_set_input(up);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(state.context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    state.buffer = ggml_backend_alloc_ctx_tensors(state.context, state.cpu);
    if (state.buffer == nullptr) {
        return 1;
    }

    const size_t elements = static_cast<size_t>(config.rows * config.columns);
    const size_t bytes = elements * sizeof(float);
    std::vector<float> gate_values(elements);
    std::vector<float> up_values(elements);
    for (size_t index = 0; index < elements; ++index) {
        gate_values[index] = 2.5F * std::sin(static_cast<float>(index) * 0.017F) -
            0.5F * std::cos(static_cast<float>(index) * 0.003F);
        up_values[index] = 1.3F * std::cos(static_cast<float>(index) * 0.011F) +
            0.2F * std::sin(static_cast<float>(index) * 0.023F);
    }
    ggml_backend_tensor_set(gate, gate_values.data(), 0, bytes);
    ggml_backend_tensor_set(up, up_values.data(), 0, bytes);

    if (config.qpu_rows > 0) {
        const std::string partition = std::to_string(config.rows) + ":" +
            std::to_string(config.qpu_rows);
        setenv("GGML_QPU_CPU_INLINE_HYBRID", "1", 1);
        setenv("GGML_QPU_CPU_INLINE_PARTITIONS", partition.c_str(), 1);
    } else {
        setenv("GGML_QPU_CPU_INLINE_HYBRID", "0", 1);
        unsetenv("GGML_QPU_CPU_INLINE_PARTITIONS");
    }

    setenv("GGML_QPU_CPU_INLINE_GEGLU", "0", 1);
    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        return 1;
    }
    std::vector<float> expected(elements);
    ggml_backend_tensor_get(output, expected.data(), 0, bytes);
    const uint64_t before = dispatch_count();

    setenv("GGML_QPU_CPU_INLINE_GEGLU", "1", 1);
    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        return 1;
    }
    std::vector<float> actual(elements);
    ggml_backend_tensor_get(output, actual.data(), 0, bytes);
    const uint64_t after = dispatch_count();
    const bool exact = std::memcmp(expected.data(), actual.data(), bytes) == 0;
    const bool dispatched = after == before + 1;
    double maximum_absolute_error = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        maximum_absolute_error = std::max(maximum_absolute_error,
            std::fabs(static_cast<double>(actual[index]) - expected[index]));
    }

    std::vector<uint64_t> cpu_samples(static_cast<size_t>(config.samples));
    std::vector<uint64_t> qpu_samples(static_cast<size_t>(config.samples));
    for (int iteration = 0; iteration < config.warmups + config.samples; ++iteration) {
        const auto measure = [&](bool qpu) {
            setenv("GGML_QPU_CPU_INLINE_GEGLU", qpu ? "1" : "0", 1);
            const uint64_t start = monotonic_ns();
            const ggml_status status = ggml_backend_graph_compute(state.cpu, graph);
            const uint64_t end = monotonic_ns();
            if (status != GGML_STATUS_SUCCESS) {
                return UINT64_MAX;
            }
            return end - start;
        };
        uint64_t cpu_duration = 0;
        uint64_t qpu_duration = 0;
        if (iteration % 2 == 0) {
            cpu_duration = measure(false);
            qpu_duration = measure(true);
        } else {
            qpu_duration = measure(true);
            cpu_duration = measure(false);
        }
        if (cpu_duration == UINT64_MAX || qpu_duration == UINT64_MAX) {
            return 1;
        }
        if (iteration >= config.warmups) {
            const size_t sample = static_cast<size_t>(iteration - config.warmups);
            cpu_samples[sample] = cpu_duration;
            qpu_samples[sample] = qpu_duration;
        }
    }
    const uint64_t final_dispatches = dispatch_count();
    const uint64_t expected_dispatches = after +
        static_cast<uint64_t>(config.warmups + config.samples);
    const bool dispatch_count_valid = final_dispatches == expected_dispatches;
    const bool m1_passed = run_m1_inline_smoke();
    std::printf(
        "{\"schema_version\":1,\"kind\":\"qpu-ggml-inline-samples\","
        "\"m\":%lld,\"n\":%lld,\"qpu_rows\":%lld,\"cpu_threads\":%d,"
        "\"warmups\":%d,\"retained_samples\":%d,"
        "\"bitwise_exact\":%s,\"max_absolute_error\":%.9g,"
        "\"dispatches_before\":%llu,\"dispatches_after\":%llu,"
        "\"dispatches_final\":%llu,\"m1_passed\":%s,\"cpu_complete_ns\":" ,
        static_cast<long long>(config.rows),
        static_cast<long long>(config.columns),
        static_cast<long long>(config.qpu_rows),
        config.cpu_threads,
        config.warmups,
        config.samples,
        exact ? "true" : "false",
        maximum_absolute_error,
        static_cast<unsigned long long>(before),
        static_cast<unsigned long long>(after),
        static_cast<unsigned long long>(final_dispatches),
        m1_passed ? "true" : "false");
    print_samples(cpu_samples);
    std::printf(",\"qpu_inline_complete_ns\":");
    print_samples(qpu_samples);
    std::printf(",\"passed\":%s}\n",
        exact && dispatched && dispatch_count_valid && m1_passed ? "true" : "false");
    return exact && dispatched && dispatch_count_valid && m1_passed ? 0 : 1;
}
