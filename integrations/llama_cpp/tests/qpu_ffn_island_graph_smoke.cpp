#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "qpu_llama_ffn_island.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

constexpr int64_t input_columns = 64;
constexpr int64_t intermediate_columns = 64;
constexpr int64_t hidden_columns = 64;

struct resources {
    ggml_backend_t cpu = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context * weight_context = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;

    ~resources() {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weight_context);
        ggml_backend_free(cpu);
    }
};

struct graph_resources {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * context = nullptr;

    ~graph_resources() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }
};

struct error_metrics {
    double maximum = 0.0;
    double mean = 0.0;
    bool finite = true;
};

std::vector<uint8_t> quantized_weight(
        int64_t inputs, int64_t outputs, float phase, float scale) {
    std::vector<float> source(static_cast<size_t>(inputs * outputs));
    for (int64_t output = 0; output < outputs; ++output) {
        for (int64_t input = 0; input < inputs; ++input) {
            const float index = static_cast<float>(output * inputs + input);
            source[static_cast<size_t>(output * inputs + input)] = scale * (
                0.8F * std::sin(index * 0.071F + phase) -
                0.2F * std::cos(index * 0.037F - phase));
        }
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, inputs);
    std::vector<uint8_t> result(static_cast<size_t>(outputs) * row_bytes);
    for (int64_t output = 0; output < outputs; ++output) {
        quantize_row_q4_0(source.data() + output * inputs,
            result.data() + static_cast<size_t>(output) * row_bytes, inputs);
    }
    return result;
}

bool initialize_weights(resources & state) {
    state.cpu = ggml_backend_cpu_init();
    if (state.cpu == nullptr) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(state.cpu, 4);

    ggml_init_params params = {};
    params.mem_size = 4 * ggml_tensor_overhead();
    params.no_alloc = true;
    state.weight_context = ggml_init(params);
    if (state.weight_context == nullptr) {
        return false;
    }
    state.gate = ggml_new_tensor_2d(state.weight_context,
        GGML_TYPE_Q4_0, input_columns, intermediate_columns);
    state.up = ggml_new_tensor_2d(state.weight_context,
        GGML_TYPE_Q4_0, input_columns, intermediate_columns);
    state.down = ggml_new_tensor_2d(state.weight_context,
        GGML_TYPE_Q4_0, intermediate_columns, hidden_columns);
    ggml_set_name(state.gate, "blk.0.ffn_gate.weight");
    ggml_set_name(state.up, "blk.0.ffn_up.weight");
    ggml_set_name(state.down, "blk.0.ffn_down.weight");

    state.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        state.weight_context, ggml_backend_cpu_repack_buffer_type());
    if (state.weight_buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(
        state.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const std::vector<uint8_t> gate = quantized_weight(
        input_columns, intermediate_columns, 0.2F, 0.08F);
    const std::vector<uint8_t> up = quantized_weight(
        input_columns, intermediate_columns, 0.7F, 0.07F);
    const std::vector<uint8_t> down = quantized_weight(
        intermediate_columns, hidden_columns, 1.1F, 0.06F);
    ggml_backend_tensor_set(state.gate, gate.data(), 0, gate.size());
    ggml_backend_tensor_set(state.up, up.data(), 0, up.size());
    ggml_backend_tensor_set(state.down, down.data(), 0, down.size());
    return true;
}

error_metrics compare(
        const std::vector<float> & actual, const std::vector<float> & expected) {
    error_metrics result;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            result.finite = false;
        }
        const double error = std::fabs(
            static_cast<double>(actual[index]) - expected[index]);
        result.maximum = std::max(result.maximum, error);
        result.mean += error;
    }
    result.mean /= static_cast<double>(actual.size());
    return result;
}

bool execute_graph(
        resources & state, int64_t rows, const char * mode,
        std::vector<float> & result) {
    graph_resources graph_state;
    constexpr size_t graph_nodes = 16;
    ggml_init_params params = {};
    params.mem_size = 12 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    params.no_alloc = true;
    graph_state.context = ggml_init(params);
    if (graph_state.context == nullptr) {
        return false;
    }
    ggml_tensor * input = ggml_new_tensor_2d(
        graph_state.context, GGML_TYPE_F32, input_columns, rows);
    ggml_tensor * gate_output = ggml_mul_mat(
        graph_state.context, state.gate, input);
    ggml_tensor * up_output = ggml_mul_mat(
        graph_state.context, state.up, input);
    ggml_tensor * activated = ggml_geglu_split(
        graph_state.context, gate_output, up_output);
    ggml_tensor * output = ggml_mul_mat(
        graph_state.context, state.down, activated);
    ggml_set_input(input);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(
        graph_state.context, graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    graph_state.buffer = ggml_backend_alloc_ctx_tensors(
        graph_state.context, state.cpu);
    if (graph_state.buffer == nullptr) {
        return false;
    }

    std::vector<float> input_values(static_cast<size_t>(rows * input_columns));
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = 0.9F * std::sin(static_cast<float>(index) * 0.113F) +
            0.15F * std::cos(static_cast<float>(index) * 0.019F);
    }
    ggml_backend_tensor_set(
        input, input_values.data(), 0, input_values.size() * sizeof(float));

    setenv("GGML_QPU_FFN_ISLAND", std::strcmp(mode, "cpu") == 0 ? "0" : "1", 1);
    if (std::strcmp(mode, "fallback") == 0) {
        setenv("GGML_QPU_FFN_ISLAND_FAIL_STAGE", "gate", 1);
        setenv("GGML_QPU_FFN_ISLAND_FAIL_LAYER", "0", 1);
    } else {
        unsetenv("GGML_QPU_FFN_ISLAND_FAIL_STAGE");
        unsetenv("GGML_QPU_FFN_ISLAND_FAIL_LAYER");
    }
    if (ggml_backend_graph_compute(state.cpu, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    result.resize(static_cast<size_t>(rows * hidden_columns));
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(float));
    return true;
}

bool check_rows(resources & state, int64_t rows) {
    std::vector<float> expected;
    std::vector<float> candidate;
    std::vector<float> fallback;
    if (!execute_graph(state, rows, "cpu", expected) ||
        !execute_graph(state, rows, "candidate", candidate) ||
        !execute_graph(state, rows, "fallback", fallback)) {
        std::fprintf(stderr, "FFN graph execution failed for M=%lld\n",
            static_cast<long long>(rows));
        return false;
    }
    const error_metrics candidate_error = compare(candidate, expected);
    const error_metrics fallback_error = compare(fallback, expected);
    const error_metrics qpu_fallback_error = compare(candidate, fallback);
    const bool passed = candidate_error.finite && fallback_error.finite &&
        candidate_error.maximum <= 0.002 && candidate_error.mean <= 2.0e-4 &&
        fallback_error.maximum <= 0.002 && fallback_error.mean <= 2.0e-4 &&
        qpu_fallback_error.maximum <= 0.002 && qpu_fallback_error.mean <= 2.0e-5;
    std::printf(
        "{\"kind\":\"qpu-ffn-island-graph-shape\",\"m\":%lld,"
        "\"candidate_max_absolute_error\":%.9g,"
        "\"candidate_mean_absolute_error\":%.9g,"
        "\"fallback_max_absolute_error\":%.9g,"
        "\"fallback_mean_absolute_error\":%.9g,"
        "\"qpu_fallback_max_absolute_error\":%.9g,"
        "\"qpu_fallback_mean_absolute_error\":%.9g,\"passed\":%s}\n",
        static_cast<long long>(rows),
        candidate_error.maximum, candidate_error.mean,
        fallback_error.maximum, fallback_error.mean,
        qpu_fallback_error.maximum, qpu_fallback_error.mean,
        passed ? "true" : "false");
    return passed;
}

} // namespace

int main() {
    setenv("GGML_QPU_FFN_ISLAND", "1", 1);
    setenv("GGML_QPU_FFN_ISLAND_FRACTION", "0.5", 1);
    setenv("GGML_QPU_FFN_ISLAND_MIN_ROWS", "1", 1);
    setenv("GGML_QPU_FFN_ISLAND_MAX_ROWS", "257", 1);
    setenv("GGML_QPU_FFN_ISLAND_WGS", "24", 1);
    setenv("GGML_QPU_TELEMETRY", "0", 1);

    resources state;
    if (!initialize_weights(state)) {
        std::fprintf(stderr, "FFN graph weight initialization failed\n");
        return 1;
    }
    const uint64_t initial_dispatches = ggml_qpu_ffn_island_dispatch_count();
    const uint64_t initial_islands = ggml_qpu_ffn_island_count();
    bool passed = true;
    for (const int64_t rows : std::array<int64_t, 3>{17, 129, 257}) {
        passed = check_rows(state, rows) && passed;
    }
    const uint64_t dispatches =
        ggml_qpu_ffn_island_dispatch_count() - initial_dispatches;
    const uint64_t islands = ggml_qpu_ffn_island_count() - initial_islands;
    // M=17 is intentionally below the production graph's M >= 64 island guard.
    passed = passed && dispatches == 8 && islands == 4;
    std::printf(
        "{\"kind\":\"qpu-ffn-island-graph-smoke\","
        "\"dispatch_count\":%llu,\"island_count\":%llu,\"passed\":%s}\n",
        static_cast<unsigned long long>(dispatches),
        static_cast<unsigned long long>(islands), passed ? "true" : "false");
    return passed ? 0 : 1;
}
