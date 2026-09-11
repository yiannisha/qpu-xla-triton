#include "ggml.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void ggml_quantize_mat_q8_0_4x4(
    const float * x, void * output, int64_t columns, int64_t row_stride);
extern "C" void ggml_quantize_mat_q8_0_4x8(
    const float * x, void * output, int64_t columns, int64_t row_stride);

namespace {

using quantize_four_rows_fn = void (*)(const float *, void *, int64_t, int64_t);

bool check_quantizer(
        const char * name, quantize_four_rows_fn quantize,
        int64_t columns, int64_t row_stride) {
    constexpr int64_t rows = 8;

    std::vector<float> strided(static_cast<size_t>(rows * row_stride));
    std::vector<float> compact(static_cast<size_t>(rows * columns));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < row_stride; ++column) {
            const float value = column < columns
                ? 0.75F * std::sin(static_cast<float>(row * 101 + column) * 0.071F) +
                    static_cast<float>(row) * 0.19F
                : 1000.0F + static_cast<float>(row * row_stride + column);
            strided[static_cast<size_t>(row * row_stride + column)] = value;
            if (column < columns) {
                compact[static_cast<size_t>(row * columns + column)] = value;
            }
        }
    }

    const size_t group_bytes = 4 * ggml_row_size(GGML_TYPE_Q8_0, columns);
    std::vector<uint8_t> expected(2 * group_bytes);
    std::vector<uint8_t> actual(2 * group_bytes);
    std::vector<uint8_t> old_bug(2 * group_bytes);
    for (int64_t group = 0; group < 2; ++group) {
        quantize(compact.data() + group * 4 * columns,
            expected.data() + static_cast<size_t>(group) * group_bytes,
            columns, columns);
        quantize(strided.data() + group * 4 * row_stride,
            actual.data() + static_cast<size_t>(group) * group_bytes,
            columns, row_stride);
        quantize(strided.data() + group * 4 * row_stride,
            old_bug.data() + static_cast<size_t>(group) * group_bytes,
            columns, columns);
    }

    const bool exact = actual == expected;
    const bool detects_old_bug = old_bug != expected;
    std::printf("%s columns=%lld row_stride=%lld exact=%s detects_old_bug=%s\n",
        name, static_cast<long long>(columns), static_cast<long long>(row_stride),
        exact ? "true" : "false", detects_old_bug ? "true" : "false");
    return exact && detects_old_bug;
}

} // namespace

int main() {
    bool passed = true;
    for (const auto shape : std::array<std::array<int64_t, 2>, 2>{
             std::array<int64_t, 2>{64, 96},
             std::array<int64_t, 2>{5376, 6144},
         }) {
        passed = check_quantizer(
            "q8_0_4x4", ggml_quantize_mat_q8_0_4x4,
            shape[0], shape[1]) && passed;
        passed = check_quantizer(
            "q8_0_4x8", ggml_quantize_mat_q8_0_4x8,
            shape[0], shape[1]) && passed;
    }
    std::printf("{\"kind\":\"qpu-repack-stride-smoke\",\"passed\":%s}\n",
        passed ? "true" : "false");
    return passed ? 0 : 1;
}
