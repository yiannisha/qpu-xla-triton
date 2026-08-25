#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Internal pinned-llama.cpp entry point used by its production model loader.
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

constexpr int64_t INPUT_COLUMNS = 64;
constexpr int64_t OUTPUT_COLUMNS = 16;
constexpr int64_t ROWS = 5;

struct resources {
    ggml_backend_reg_t registry = nullptr;
    ggml_backend_t qpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context * graph_context = nullptr;
    ggml_context * weight_context = nullptr;

    ~resources() {
        ggml_backend_buffer_free(graph_buffer);
        ggml_free(graph_context);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weight_context);
        ggml_backend_free(cpu);
        ggml_backend_free(qpu);
        if (registry != nullptr) {
            ggml_backend_unload(registry);
        }
    }
};

bool close_enough(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float & maximum_absolute_error,
    float & maximum_relative_error) {
    maximum_absolute_error = 0.0F;
    maximum_relative_error = 0.0F;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float absolute_error = std::fabs(actual[index] - expected[index]);
        const float relative_error = absolute_error /
            std::max(std::fabs(expected[index]), 1.0e-6F);
        maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
        maximum_relative_error = std::max(maximum_relative_error, relative_error);
        if (absolute_error > 2.0e-4F + 2.0e-5F * std::fabs(expected[index])) {
            std::fprintf(stderr,
                "output mismatch at %zu: qpu=%g cpu=%g absolute_error=%g\n",
                index, actual[index], expected[index], absolute_error);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/libggml-qpu.so\n", argv[0]);
        return 2;
    }

    setenv("GGML_QPU_ENABLE_Q4_0", "1", 1);
    setenv("GGML_QPU_MIN_M", "1", 1);
    resources state;
    state.registry = ggml_backend_load(argv[1]);
    if (state.registry == nullptr || ggml_backend_reg_dev_count(state.registry) != 1) {
        std::fprintf(stderr, "QPU backend plugin load failed\n");
        return 1;
    }
    ggml_backend_dev_t qpu_device = ggml_backend_reg_dev_get(state.registry, 0);
    state.qpu = ggml_backend_dev_init(qpu_device, nullptr);
    state.cpu = ggml_backend_cpu_init();
    if (state.qpu == nullptr || state.cpu == nullptr) {
        std::fprintf(stderr, "backend initialization failed\n");
        return 1;
    }

    ggml_init_params weight_params = {};
    weight_params.mem_size = 2 * ggml_tensor_overhead();
    weight_params.no_alloc = true;
    state.weight_context = ggml_init(weight_params);
    if (state.weight_context == nullptr) {
        std::fprintf(stderr, "weight metadata context allocation failed\n");
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(state.weight_context,
        GGML_TYPE_Q4_0, INPUT_COLUMNS, OUTPUT_COLUMNS);
    ggml_set_name(weight, "smoke.ffn_up.weight");
    state.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        state.weight_context, ggml_backend_cpu_repack_buffer_type());
    if (state.weight_buffer == nullptr) {
        std::fprintf(stderr, "CPU_REPACK weight buffer allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(state.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> unquantized_weight(
        static_cast<size_t>(INPUT_COLUMNS * OUTPUT_COLUMNS));
    for (size_t index = 0; index < unquantized_weight.size(); ++index) {
        unquantized_weight[index] =
            0.75F * std::sin(static_cast<float>(index) * 0.071F) -
            0.20F * std::cos(static_cast<float>(index) * 0.037F);
    }
    std::vector<uint8_t> canonical_weight(ggml_nbytes(weight));
    for (int64_t row = 0; row < OUTPUT_COLUMNS; ++row) {
        quantize_row_q4_0(
            unquantized_weight.data() + row * INPUT_COLUMNS,
            canonical_weight.data() + row * (INPUT_COLUMNS / 32) * 18,
            INPUT_COLUMNS);
    }
    ggml_backend_tensor_set(weight, canonical_weight.data(), 0, canonical_weight.size());

    constexpr size_t graph_nodes = 8;
    ggml_init_params graph_params = {};
    graph_params.mem_size = 8 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    graph_params.no_alloc = true;
    state.graph_context = ggml_init(graph_params);
    if (state.graph_context == nullptr) {
        std::fprintf(stderr, "graph metadata context allocation failed\n");
        return 1;
    }
    ggml_tensor * input = ggml_new_tensor_2d(state.graph_context,
        GGML_TYPE_F32, INPUT_COLUMNS, ROWS);
    ggml_set_name(input, "smoke.activation");
    ggml_set_input(input);
    ggml_tensor * output = ggml_mul_mat(state.graph_context, weight, input);
    ggml_set_name(output, "smoke.output");
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(
        state.graph_context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);

    state.graph_buffer = ggml_backend_alloc_ctx_tensors(state.graph_context, state.qpu);
    if (state.graph_buffer == nullptr ||
        !ggml_backend_dev_supports_op(qpu_device, output) ||
        !ggml_backend_supports_op(state.cpu, output)) {
        std::fprintf(stderr, "test graph is not supported by both backends\n");
        return 1;
    }

    std::vector<float> input_values(static_cast<size_t>(INPUT_COLUMNS * ROWS));
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] =
            0.90F * std::sin(static_cast<float>(index) * 0.113F) +
            0.15F * std::cos(static_cast<float>(index) * 0.019F);
    }
    ggml_backend_tensor_set(input, input_values.data(), 0,
        input_values.size() * sizeof(float));

    if (ggml_backend_graph_compute(state.qpu, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "QPU graph compute failed\n");
        return 1;
    }
    std::vector<float> qpu_output(static_cast<size_t>(OUTPUT_COLUMNS * ROWS));
    ggml_backend_tensor_get(output, qpu_output.data(), 0,
        qpu_output.size() * sizeof(float));

    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CPU graph compute failed\n");
        return 1;
    }
    std::vector<float> cpu_output(qpu_output.size());
    ggml_backend_tensor_get(output, cpu_output.data(), 0,
        cpu_output.size() * sizeof(float));

    float maximum_absolute_error = 0.0F;
    float maximum_relative_error = 0.0F;
    const bool matches = close_enough(qpu_output, cpu_output,
        maximum_absolute_error, maximum_relative_error);
    std::printf(
        "{\"schema_version\":1,\"kind\":\"qpu-ggml-backend-smoke\","
        "\"weight_buffer_type\":\"%s\",\"m\":%lld,\"k\":%lld,\"n\":%lld,"
        "\"max_absolute_error\":%.9g,\"max_relative_error\":%.9g,"
        "\"passed\":%s}\n",
        ggml_backend_buffer_name(state.weight_buffer),
        static_cast<long long>(ROWS),
        static_cast<long long>(INPUT_COLUMNS),
        static_cast<long long>(OUTPUT_COLUMNS),
        maximum_absolute_error, maximum_relative_error,
        matches ? "true" : "false");
    return matches ? 0 : 1;
}
