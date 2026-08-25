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
#include <limits>
#include <vector>

namespace {

struct options {
    const char * plugin = nullptr;
    int64_t rows = 16;
    int64_t columns = 6144;
    int cpu_threads = 4;
    int warmups = 5;
    int samples = 31;
};

struct graph_owner {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    ~graph_owner() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool parse_positive(const char * text, int64_t & result) {
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
        "usage: %s --plugin FILE [--rows N] [--columns N] [--cpu-threads N] "
        "[--warmups N] [--samples N]\n",
        program);
}

bool parse_options(int argc, char ** argv, options & result) {
    options value;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        int64_t parsed = 0;
        const char * name = argv[index];
        if (std::strcmp(name, "--plugin") == 0) {
            value.plugin = argv[index + 1];
        } else if (!parse_positive(argv[index + 1], parsed)) {
            return false;
        } else if (std::strcmp(name, "--rows") == 0) {
            value.rows = parsed;
        } else if (std::strcmp(name, "--columns") == 0) {
            value.columns = parsed;
        } else if (std::strcmp(name, "--cpu-threads") == 0 &&
            parsed <= std::numeric_limits<int>::max()) {
            value.cpu_threads = static_cast<int>(parsed);
        } else if (std::strcmp(name, "--warmups") == 0 &&
            parsed <= std::numeric_limits<int>::max()) {
            value.warmups = static_cast<int>(parsed);
        } else if (std::strcmp(name, "--samples") == 0 &&
            parsed <= std::numeric_limits<int>::max()) {
            value.samples = static_cast<int>(parsed);
        } else {
            return false;
        }
    }
    if (value.plugin == nullptr ||
        static_cast<uint64_t>(value.rows) * static_cast<uint64_t>(value.columns) % 768 != 0) {
        return false;
    }
    result = value;
    return true;
}

bool build_graph(graph_owner & owner, ggml_backend_t backend, int64_t rows, int64_t columns) {
    constexpr size_t graph_nodes = 8;
    ggml_init_params params = {};
    params.mem_size = 8 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    params.no_alloc = true;
    owner.context = ggml_init(params);
    if (owner.context == nullptr) {
        return false;
    }
    owner.gate = ggml_new_tensor_2d(owner.context, GGML_TYPE_F32, columns, rows);
    owner.up = ggml_new_tensor_2d(owner.context, GGML_TYPE_F32, columns, rows);
    ggml_set_name(owner.gate, "bench.ffn_gate");
    ggml_set_name(owner.up, "bench.ffn_up");
    ggml_set_input(owner.gate);
    ggml_set_input(owner.up);
    owner.output = ggml_geglu_split(owner.context, owner.gate, owner.up);
    ggml_set_name(owner.output, "bench.ffn_geglu");
    ggml_set_output(owner.output);
    owner.graph = ggml_new_graph_custom(owner.context, graph_nodes, false);
    ggml_build_forward_expand(owner.graph, owner.output);
    owner.buffer = ggml_backend_alloc_ctx_tensors(owner.context, backend);
    return owner.buffer != nullptr && ggml_backend_supports_op(backend, owner.output);
}

void print_samples(const std::vector<uint64_t> & samples) {
    std::putchar('[');
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            std::putchar(',');
        }
        std::printf("%llu", static_cast<unsigned long long>(samples[index]));
    }
    std::putchar(']');
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

    setenv("GGML_QPU_ENABLE_GEGLU", "1", 1);
    ggml_backend_reg_t registry = ggml_backend_load(config.plugin);
    if (registry == nullptr || ggml_backend_reg_dev_count(registry) != 1) {
        std::fprintf(stderr, "QPU backend plugin load failed\n");
        return 1;
    }
    ggml_backend_t qpu = ggml_backend_dev_init(
        ggml_backend_reg_dev_get(registry, 0), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (qpu == nullptr || cpu == nullptr) {
        std::fprintf(stderr, "backend initialization failed\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, config.cpu_threads);
    ggml_threadpool_params threadpool_params =
        ggml_threadpool_params_default(config.cpu_threads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&threadpool_params);
    ggml_backend_cpu_set_threadpool(cpu, threadpool);

    graph_owner cpu_graph;
    graph_owner qpu_graph;
    if (!build_graph(cpu_graph, cpu, config.rows, config.columns) ||
        !build_graph(qpu_graph, qpu, config.rows, config.columns)) {
        std::fprintf(stderr, "GEGLU graph construction or allocation failed\n");
        return 1;
    }
    const size_t elements = static_cast<size_t>(config.rows * config.columns);
    std::vector<float> gate(elements);
    std::vector<float> up(elements);
    for (size_t index = 0; index < elements; ++index) {
        gate[index] = 2.5F * std::sin(static_cast<float>(index) * 0.017F) -
            0.5F * std::cos(static_cast<float>(index) * 0.003F);
        up[index] = 1.3F * std::cos(static_cast<float>(index) * 0.011F) +
            0.2F * std::sin(static_cast<float>(index) * 0.023F);
    }
    const size_t bytes = elements * sizeof(float);
    ggml_backend_tensor_set(cpu_graph.gate, gate.data(), 0, bytes);
    ggml_backend_tensor_set(cpu_graph.up, up.data(), 0, bytes);
    ggml_backend_tensor_set(qpu_graph.gate, gate.data(), 0, bytes);
    ggml_backend_tensor_set(qpu_graph.up, up.data(), 0, bytes);

    if (ggml_backend_graph_compute(cpu, cpu_graph.graph) != GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(qpu, qpu_graph.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "initial GEGLU graph compute failed\n");
        return 1;
    }
    std::vector<float> expected(elements);
    std::vector<float> actual(elements);
    ggml_backend_tensor_get(cpu_graph.output, expected.data(), 0, bytes);
    ggml_backend_tensor_get(qpu_graph.output, actual.data(), 0, bytes);
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    size_t tolerance_violations = 0;
    for (size_t index = 0; index < elements; ++index) {
        const double error = std::fabs(
            static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
        maximum_absolute_error = std::max(maximum_absolute_error, error);
        mean_absolute_error += error;
        if (error > 2.0e-4 + 2.0e-5 * std::fabs(static_cast<double>(expected[index]))) {
            ++tolerance_violations;
        }
    }
    mean_absolute_error /= static_cast<double>(elements);

    std::vector<uint64_t> cpu_samples(config.samples);
    std::vector<uint64_t> cpu_quantize_samples(config.samples);
    std::vector<uint64_t> cpu_geglu_quantize_samples(config.samples);
    std::vector<uint64_t> qpu_compute_samples(config.samples);
    std::vector<uint64_t> qpu_complete_samples(config.samples);
    const int iterations = config.warmups + config.samples;
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, config.columns);
    std::vector<uint8_t> q8_output(static_cast<size_t>(config.rows) * q8_row_bytes);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        uint64_t start = monotonic_ns();
        const ggml_status cpu_status = ggml_backend_graph_compute(cpu, cpu_graph.graph);
        uint64_t end = monotonic_ns();
        if (cpu_status != GGML_STATUS_SUCCESS) {
            return 1;
        }
        if (iteration >= config.warmups) {
            cpu_samples[iteration - config.warmups] = end - start;
        }

        start = monotonic_ns();
        const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q8_0,
            static_cast<const float *>(cpu_graph.output->data),
            q8_output.data(), 0, config.rows, config.columns, nullptr);
        end = monotonic_ns();
        if (quantized != q8_output.size()) {
            std::fprintf(stderr, "Q8_0 quantization returned an unexpected byte count\n");
            return 1;
        }
        if (iteration >= config.warmups) {
            cpu_quantize_samples[iteration - config.warmups] = end - start;
            cpu_geglu_quantize_samples[iteration - config.warmups] =
                cpu_samples[iteration - config.warmups] + end - start;
        }

        start = monotonic_ns();
        const ggml_status qpu_status = ggml_backend_graph_compute(qpu, qpu_graph.graph);
        end = monotonic_ns();
        if (qpu_status != GGML_STATUS_SUCCESS) {
            return 1;
        }
        if (iteration >= config.warmups) {
            qpu_compute_samples[iteration - config.warmups] = end - start;
        }

        start = monotonic_ns();
        ggml_backend_tensor_set(qpu_graph.gate, gate.data(), 0, bytes);
        ggml_backend_tensor_set(qpu_graph.up, up.data(), 0, bytes);
        const ggml_status complete_status = ggml_backend_graph_compute(qpu, qpu_graph.graph);
        end = monotonic_ns();
        if (complete_status != GGML_STATUS_SUCCESS) {
            return 1;
        }
        if (iteration >= config.warmups) {
            qpu_complete_samples[iteration - config.warmups] = end - start;
        }
    }

    std::printf(
        "{\"schema_version\":1,\"kind\":\"qpu-ggml-geglu-samples\","
        "\"m\":%lld,\"n\":%lld,\"elements\":%zu,\"cpu_threads\":%d,"
        "\"warmups\":%d,\"retained_samples\":%d,"
        "\"max_absolute_error\":%.9g,\"mean_absolute_error\":%.9g,"
        "\"tolerance_violation_count\":%zu,\"cpu_complete_ns\":",
        static_cast<long long>(config.rows),
        static_cast<long long>(config.columns),
        elements, config.cpu_threads, config.warmups, config.samples,
        maximum_absolute_error, mean_absolute_error, tolerance_violations);
    print_samples(cpu_samples);
    std::printf(",\"cpu_q8_0_quantize_ns\":");
    print_samples(cpu_quantize_samples);
    std::printf(",\"cpu_geglu_q8_0_ns\":");
    print_samples(cpu_geglu_quantize_samples);
    std::printf(",\"qpu_compute_ns\":");
    print_samples(qpu_compute_samples);
    std::printf(",\"qpu_complete_with_two_input_copies_ns\":");
    print_samples(qpu_complete_samples);
    std::printf("}\n");

    ggml_threadpool_free(threadpool);
    ggml_backend_free(cpu);
    ggml_backend_free(qpu);
    // The graph buffers still own buffer-interface callbacks from the plugin.
    // Keep the process-local registry loaded until normal process teardown.
    return tolerance_violations == 0 ? 0 : 1;
}
