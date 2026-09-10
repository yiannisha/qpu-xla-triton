#include "qpu_llama_ffn_island.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t rows = 16;
constexpr uint32_t input_columns = 64;
constexpr uint32_t intermediate_columns = 64;
constexpr uint32_t hidden_columns = 64;
constexpr uint32_t block_elements = 32;
constexpr uint32_t q4_block_bytes = 18;
constexpr uint32_t q8_block_bytes = 34;

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
    const uint32_t absolute = bits & UINT32_C(0x7fffffff);
    if (absolute >= UINT32_C(0x7f800000)) {
        const uint16_t payload = absolute > UINT32_C(0x7f800000)
            ? static_cast<uint16_t>((absolute >> 13U) & UINT32_C(0x03ff)) : 0;
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00) | payload);
    }
    const int32_t exponent = static_cast<int32_t>((absolute >> 23U) & UINT32_C(0xff)) - 127;
    if (exponent > 15) {
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
    }
    if (exponent < -24) {
        return static_cast<uint16_t>(sign);
    }
    uint32_t mantissa = absolute & UINT32_C(0x007fffff);
    if (exponent < -14) {
        mantissa |= UINT32_C(0x00800000);
        const uint32_t shift = static_cast<uint32_t>(-exponent - 1);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
        const uint32_t halfway = UINT32_C(1) << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13U;
    const uint32_t remainder = mantissa & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (rounded & 1U) != 0)) {
        ++rounded;
        if (rounded == UINT32_C(0x400)) {
            return static_cast<uint16_t>(sign |
                (static_cast<uint32_t>(exponent + 16) << 10U));
        }
    }
    return static_cast<uint16_t>(sign |
        (static_cast<uint32_t>(exponent + 15) << 10U) | rounded);
}

std::vector<uint8_t> q4_weights(uint32_t inputs, uint32_t outputs, uint32_t salt) {
    const uint32_t blocks = inputs / block_elements;
    std::vector<uint8_t> result(
        static_cast<size_t>(outputs) * blocks * q4_block_bytes, 0);
    for (uint32_t output = 0; output < outputs; ++output) {
        for (uint32_t block = 0; block < blocks; ++block) {
            uint8_t * destination = result.data() +
                (static_cast<size_t>(output) * blocks + block) * q4_block_bytes;
            const uint16_t scale = float_to_half(0.0025f * static_cast<float>(1U + (output + salt) % 7U));
            std::memcpy(destination, &scale, sizeof(scale));
            for (uint32_t index = 0; index < 16; ++index) {
                const uint8_t low = static_cast<uint8_t>((output * 3U + block * 5U + index + salt) % 16U);
                const uint8_t high = static_cast<uint8_t>((output * 7U + block + index * 3U + salt) % 16U);
                destination[2U + index] = static_cast<uint8_t>(low | (high << 4U));
            }
        }
    }
    return result;
}

std::vector<uint8_t> q8_activation() {
    const uint32_t blocks = input_columns / block_elements;
    std::vector<uint8_t> result(static_cast<size_t>(rows) * blocks * q8_block_bytes, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            uint8_t * destination = result.data() +
                (static_cast<size_t>(row) * blocks + block) * q8_block_bytes;
            const uint16_t scale = float_to_half(0.01f * static_cast<float>(1U + row % 3U));
            std::memcpy(destination, &scale, sizeof(scale));
            for (uint32_t index = 0; index < block_elements; ++index) {
                const int value = static_cast<int>((row * 11U + block * 17U + index * 5U) % 255U) - 127;
                destination[2U + index] = static_cast<uint8_t>(static_cast<int8_t>(value));
            }
        }
    }
    return result;
}

bool execute_island(const std::vector<uint8_t> & activation, std::vector<float> & output) {
    if (ggml_qpu_ffn_island_begin("blk.0.ffn_gate.weight", activation.data(),
            activation.size(), rows, input_columns, intermediate_columns, 0) != 1 ||
        ggml_qpu_ffn_island_cpu_output_columns_for("blk.0.ffn_gate.weight") != 32 ||
        ggml_qpu_ffn_island_cpu_output_columns_for("blk.0.ffn_up.weight") != 32 ||
        ggml_qpu_ffn_island_cpu_intermediate_columns() != 32 ||
        ggml_qpu_ffn_island_down_reduction_columns_for("blk.0.ffn_down.weight") != 32 ||
        ggml_qpu_ffn_island_wait("blk.0.ffn_down.weight") != 1) {
        return false;
    }
    output.assign(static_cast<size_t>(rows) * hidden_columns, 0.0f);
    return ggml_qpu_ffn_island_join("blk.0.ffn_down.weight", output.data(),
               output.size() * sizeof(float), rows, hidden_columns, 0, 1) == 1 &&
        ggml_qpu_ffn_island_complete("blk.0.ffn_down.weight") == 1;
}

} // namespace

int main() {
    setenv("GGML_QPU_FFN_ISLAND", "1", 1);
    setenv("GGML_QPU_FFN_ISLAND_FRACTION", "0.5", 1);
    setenv("GGML_QPU_FFN_ISLAND_MIN_ROWS", "16", 1);
    setenv("GGML_QPU_FFN_ISLAND_MAX_ROWS", "16", 1);
    setenv("GGML_QPU_FFN_ISLAND_WGS", "24", 1);
    setenv("GGML_QPU_TELEMETRY", "0", 1);

    const std::vector<uint8_t> gate = q4_weights(input_columns, intermediate_columns, 1);
    const std::vector<uint8_t> up = q4_weights(input_columns, intermediate_columns, 2);
    const std::vector<uint8_t> down = q4_weights(intermediate_columns, hidden_columns, 3);
    const std::vector<uint8_t> activation = q8_activation();
    if (ggml_qpu_ffn_island_register_q4_0("blk.0.ffn_gate.weight", gate.data(),
            gate.size(), input_columns, intermediate_columns) != 1 ||
        ggml_qpu_ffn_island_register_q4_0("blk.0.ffn_up.weight", up.data(),
            up.size(), input_columns, intermediate_columns) != 1 ||
        ggml_qpu_ffn_island_register_q4_0("blk.0.ffn_down.weight", down.data(),
            down.size(), intermediate_columns, hidden_columns) != 1) {
        std::fprintf(stderr, "FFN-island weight registration failed\n");
        return 1;
    }

    std::vector<float> qpu_output;
    if (!execute_island(activation, qpu_output)) {
        std::fprintf(stderr, "QPU FFN-island execution failed\n");
        return 1;
    }
    setenv("GGML_QPU_FFN_ISLAND_FAIL_STAGE", "gate", 1);
    setenv("GGML_QPU_FFN_ISLAND_FAIL_LAYER", "0", 1);
    std::vector<float> fallback_output;
    if (!execute_island(activation, fallback_output)) {
        std::fprintf(stderr, "FFN-island fallback execution failed\n");
        return 1;
    }
    double maximum = 0.0;
    double mean = 0.0;
    for (size_t index = 0; index < qpu_output.size(); ++index) {
        const double error = std::fabs(
            static_cast<double>(qpu_output[index]) - fallback_output[index]);
        maximum = std::max(maximum, error);
        mean += error;
    }
    mean /= static_cast<double>(qpu_output.size());
    const bool passed = maximum <= 0.002 && mean <= 2.0e-5 &&
        ggml_qpu_ffn_island_dispatch_count() == 4 &&
        ggml_qpu_ffn_island_count() == 2;
    std::printf(
        "{\"kind\":\"qpu-ffn-island-smoke\",\"passed\":%s,"
        "\"max_absolute_error\":%.9g,\"mean_absolute_error\":%.9g,"
        "\"dispatch_count\":%llu,\"island_count\":%llu}\n",
        passed ? "true" : "false", maximum, mean,
        static_cast<unsigned long long>(ggml_qpu_ffn_island_dispatch_count()),
        static_cast<unsigned long long>(ggml_qpu_ffn_island_count()));
    return passed ? 0 : 1;
}
