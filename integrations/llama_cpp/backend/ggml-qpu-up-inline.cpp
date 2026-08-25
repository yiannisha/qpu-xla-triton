#include "qpu_llama_q4_0.h"
#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-mx.h"
#include "ggml-column-w8-q8-0-mx.h"
#include "tiled-w8a8-gemm-dequantize.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct up_weight {
    qpu_llama_q4_0_linear * linear = nullptr;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t resident_column_start = 0;
    uint32_t resident_column_count = 0;
    qpu_llama_q4_0_weight_mode weight_mode = QPU_LLAMA_Q4_0_WEIGHT_EXACT;

    ~up_weight() {
        qpu_llama_q4_0_linear_destroy(linear);
    }
};

struct up_context {
    std::mutex mutex;
    qpu_llama_context * runtime = nullptr;
    bool initialization_attempted = false;
    std::unordered_map<std::string, std::unique_ptr<up_weight>> weights;
    qpu_llama_q4_0_submission * submission = nullptr;
    std::string pending_name;
    uint32_t pending_rows = 0;
    uint32_t pending_input_columns = 0;
    uint32_t pending_output_columns = 0;
    uint32_t pending_qpu_columns = 0;
    double pending_fraction = 0.0;
    qpu_llama_q4_0_weight_mode pending_weight_mode = QPU_LLAMA_Q4_0_WEIGHT_EXACT;
    uint64_t pending_submit_end = 0;
    std::atomic<bool> pending = false;
    std::atomic<uint32_t> cpu_columns = 0;
    std::atomic<uint64_t> dispatch_count = 0;

    ~up_context() {
        if (submission != nullptr) {
            qpu_llama_q4_0_submission_destroy(submission);
        }
        weights.clear();
        qpu_llama_context_destroy(runtime);
    }
};

struct m1_weight {
    qpu_llama_q4_0_linear * linear = nullptr;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;

    ~m1_weight() {
        qpu_llama_q4_0_linear_destroy(linear);
    }
};

struct m1_context {
    std::mutex mutex;
    qpu_llama_context * runtime = nullptr;
    bool initialization_attempted = false;
    std::unordered_map<std::string, std::unique_ptr<m1_weight>> weights;
    std::string last_name;
    std::atomic<uint32_t> cpu_columns = 0;
    std::atomic<uint64_t> dispatch_count = 0;

    ~m1_context() {
        weights.clear();
        qpu_llama_context_destroy(runtime);
    }
};

up_context & get_up_context() {
    static up_context context;
    return context;
}

m1_context & get_m1_context() {
    static m1_context context;
    return context;
}

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr &&
        (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "on") == 0);
}

bool m1_tensor_selected(const char * name) {
    const char * patterns = std::getenv("GGML_QPU_M1_TENSORS");
    if (name == nullptr || patterns == nullptr || patterns[0] == '\0') {
        return false;
    }
    while (*patterns != '\0') {
        const char * end = std::strchr(patterns, ',');
        const size_t size = end != nullptr
            ? static_cast<size_t>(end - patterns) : std::strlen(patterns);
        if (size > 0 && std::string(name).find(std::string(patterns, size)) !=
                std::string::npos) {
            return true;
        }
        if (end == nullptr) {
            break;
        }
        patterns = end + 1;
    }
    return false;
}

uint32_t maximum_rows() {
    const char * value = std::getenv("GGML_QPU_UP_MAX_ROWS");
    if (value == nullptr || value[0] == '\0') {
        return 512U;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 16UL || parsed > 65520UL) {
        return 0;
    }
    return static_cast<uint32_t>(parsed);
}

double maximum_fraction() {
    const char * value = std::getenv("GGML_QPU_UP_MAX_FRACTION");
    if (value == nullptr || value[0] == '\0') {
        return 0.5;
    }
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && *end == '\0' && parsed >= 0.0625 && parsed <= 1.0
        ? parsed : 0.0;
}

qpu_llama_q4_0_weight_mode selected_weight_mode() {
    const char * value = std::getenv("GGML_QPU_UP_WEIGHT_MODE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "exact") == 0) {
        return QPU_LLAMA_Q4_0_WEIGHT_EXACT;
    }
    if (std::strcmp(value, "column-w8") == 0) {
        return QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8;
    }
    if (std::strcmp(value, "rowcol-w8a8") == 0) {
        return QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8;
    }
    return QPU_LLAMA_Q4_0_WEIGHT_EXACT;
}

uint32_t selected_wgs_per_supergroup() {
    const char * value = std::getenv("GGML_QPU_UP_WGS");
    if (value == nullptr || value[0] == '\0') {
        return 24U;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0UL && parsed <= 255UL
        ? static_cast<uint32_t>(parsed) : 24U;
}

const char * weight_mode_name(qpu_llama_q4_0_weight_mode mode) {
    if (mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8) {
        return "column-w8";
    }
    if (mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) {
        return "rowcol-w8a8";
    }
    return "exact";
}

double selected_fraction(uint32_t rows) {
    const char * value = std::getenv("GGML_QPU_UP_FRACTION");
    if (value != nullptr && value[0] != '\0') {
        char * end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end != value && *end == '\0' && parsed > 0.0 && parsed <= 1.0) {
            return parsed;
        }
        return 0.0;
    }
    if (rows >= 256U) {
        return 1.0 / 3.0;
    }
    if (rows >= 128U) {
        return 5.0 / 16.0;
    }
    if (rows >= 64U) {
        return 1.0 / 4.0;
    }
    return 0.0;
}

bool has_up_suffix(const char * name) {
    if (name == nullptr) {
        return false;
    }
    const char suffix[] = ".ffn_up.weight";
    const size_t name_size = std::strlen(name);
    const size_t suffix_size = sizeof(suffix) - 1U;
    return name_size >= suffix_size &&
        std::memcmp(name + name_size - suffix_size, suffix, suffix_size) == 0;
}

qpu_llama_status initialize(up_context & context) {
    if (context.runtime != nullptr) {
        return QPU_LLAMA_OK;
    }
    if (context.initialization_attempted) {
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    context.initialization_attempted = true;
    return qpu_llama_context_create(nullptr, &context.runtime);
}

qpu_llama_status initialize(m1_context & context) {
    if (context.runtime != nullptr) {
        return QPU_LLAMA_OK;
    }
    if (context.initialization_attempted) {
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    context.initialization_attempted = true;
    return qpu_llama_context_create(nullptr, &context.runtime);
}

uint32_t aligned_columns(uint32_t columns, double fraction) {
    const double requested = std::floor(static_cast<double>(columns) * fraction);
    if (requested < 16.0) {
        return 0;
    }
    const uint32_t result = static_cast<uint32_t>(requested) & ~UINT32_C(15);
    return result < columns ? result : columns;
}

bool telemetry_enabled() {
    const char * value = std::getenv("GGML_QPU_TELEMETRY");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

} // namespace

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_m1_register_q4_0(
    const char * name,
    const void * weights,
    size_t weight_size,
    uint64_t input_columns,
    uint64_t output_columns) {
    if (!flag_enabled("GGML_QPU_M1_INLINE") || !m1_tensor_selected(name) ||
        weights == nullptr || input_columns == 0 || input_columns > UINT32_MAX ||
        input_columns % 32 != 0 || output_columns == 0 ||
        output_columns > UINT32_MAX || output_columns % 16 != 0) {
        return 0;
    }
    m1_context & context = get_m1_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.weights.find(name) != context.weights.end()) {
        return 1;
    }
    qpu_llama_status status = initialize(context);
    auto record = std::make_unique<m1_weight>();
    record->input_columns = static_cast<uint32_t>(input_columns);
    record->output_columns = static_cast<uint32_t>(output_columns);
    const qpu_llama_q4_0_linear_desc description = {
        /* .weights                = */ weights,
        /* .weight_size            = */ weight_size,
        /* .input_columns          = */ record->input_columns,
        /* .output_columns         = */ record->output_columns,
        /* .rows                   = */ 1,
        /* .resident_column_start  = */ 0,
        /* .resident_column_count  = */ record->output_columns,
        /* .weight_mode            = */ QPU_LLAMA_Q4_0_WEIGHT_EXACT,
        /* .workgroups_per_supergroup = */ selected_wgs_per_supergroup(),
        /* .expected_source_hash   = */ qpu_ggml_q4_0_q8_0_m1_source_hash,
    };
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_q4_0_linear_prepare(
            context.runtime, &description, &record->linear);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: could not persist M=1 tensor %s: %s (%s)\n",
            name, qpu_llama_status_string(status),
            context.runtime != nullptr
                ? qpu_llama_context_last_error(context.runtime) : "no context");
        return 0;
    }
    if (telemetry_enabled()) {
        std::fprintf(stderr,
            "qpu_llama_weight_json:{\"schema_version\":1,"
            "\"operation\":\"mul_mat_q4_0_q8_0_m1\",\"tensor\":\"%s\","
            "\"k\":%u,\"n\":%u,\"resident_bytes\":%zu}\n",
            name, record->input_columns, record->output_columns,
            qpu_llama_q4_0_linear_resident_bytes(record->linear));
    }
    context.weights.emplace(name, std::move(record));
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_m1_weight_registered(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    m1_context & context = get_m1_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    return context.weights.find(name) != context.weights.end() ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_m1_execute(
    const char * name,
    const void * activation_q8_0,
    size_t activation_size,
    float * destination,
    uint64_t rows,
    uint64_t input_columns,
    uint64_t output_columns,
    uint64_t activation_interleave) {
    if (name == nullptr || activation_q8_0 == nullptr || destination == nullptr || rows != 1 ||
        input_columns > UINT32_MAX || output_columns > UINT32_MAX ||
        activation_interleave > UINT32_MAX) {
        return 0;
    }
    m1_context & context = get_m1_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    const auto found = context.weights.find(name);
    if (found == context.weights.end()) {
        return 0;
    }
    m1_weight & weight = *found->second;
    if (weight.input_columns != input_columns ||
        weight.output_columns != output_columns) {
        return 0;
    }
    context.last_name = name;
    context.cpu_columns.store(weight.output_columns, std::memory_order_release);
    if (!flag_enabled("GGML_QPU_M1_INLINE")) {
        return 0;
    }
    const qpu_llama_q4_0_execution execution = {
        /* .activation         = */ activation_q8_0,
        /* .activation_size    = */ activation_size,
        /* .activation_offset  = */ 0,
        /* .destination        = */ destination,
        /* .destination_size   = */ static_cast<size_t>(output_columns) * sizeof(float),
        /* .destination_offset = */ 0,
        /* .column_start       = */ 0,
        /* .column_count       = */ weight.output_columns,
        /* .rows               = */ 1,
        /* .activation_interleave = */ static_cast<uint32_t>(activation_interleave),
    };
    qpu_llama_q4_0_submission * submission = nullptr;
    qpu_llama_status status = qpu_llama_q4_0_linear_submit(
        weight.linear, &execution, &submission);
    qpu_llama_q4_0_timing timing = {};
    bool used_fallback = false;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_q4_0_submission_wait(submission, &timing);
    }
    if (status != QPU_LLAMA_OK && submission != nullptr) {
        const qpu_llama_status fallback =
            qpu_llama_q4_0_submission_cpu_fallback(submission);
        used_fallback = fallback == QPU_LLAMA_OK;
        status = fallback;
    }
    qpu_llama_q4_0_submission_destroy(submission);
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: inline M=1 failed for %s: %s (%s)\n",
            name, qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        return 0;
    }
    context.cpu_columns.store(0, std::memory_order_release);
    ++context.dispatch_count;
    if (telemetry_enabled()) {
        std::fprintf(stderr,
            "qpu_llama_candidate_json:{\"schema_version\":1,"
            "\"backend\":\"QPU0\",\"operation\":\"mul_mat_q4_0_q8_0\","
            "\"program\":\"ggml-q4-0-q8-0-m1\",\"source_hash\":\"%s\","
            "\"binary_sha256\":\"%s\",\"exact_shape\":{"
            "\"operation\":\"q4_0-by-q8_0\",\"k_multiple\":32,"
            "\"n_multiple\":16},\"placement\":\"hybrid\","
            "\"partition\":{\"axis\":\"operators\","
            "\"qpu\":[\"mul_mat_q4_0_q8_0\"],\"cpu\":\"all_other_ops\"},"
            "\"integration_boundary\":\"cpu-repack-inline-m1\","
            "\"tensor\":\"%s\",\"m\":1,\"k\":%u,\"n\":%u,"
            "\"dispatch_count\":1,\"input_access_ns\":%llu,"
            "\"input_pack_ns\":%llu,\"input_copy_ns\":%llu,"
            "\"submit_wait_ns\":%llu,\"output_copy_ns\":%llu,"
            "\"complete_ns\":%llu,\"fallback\":%s}\n",
            qpu_ggml_q4_0_q8_0_m1_source_hash,
            qpu_ggml_q4_0_q8_0_m1_binary_hash,
            name, weight.input_columns, weight.output_columns,
            static_cast<unsigned long long>(timing.input_access_ns),
            static_cast<unsigned long long>(timing.input_pack_ns),
            static_cast<unsigned long long>(timing.input_copy_ns),
            static_cast<unsigned long long>(timing.submit_wait_ns),
            static_cast<unsigned long long>(timing.output_copy_ns),
            static_cast<unsigned long long>(timing.complete_ns),
            used_fallback ? "true" : "false");
    }
    return 1;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_m1_cpu_columns_for(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    m1_context & context = get_m1_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    return context.last_name == name
        ? context.cpu_columns.load(std::memory_order_acquire) : 0;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_m1_dispatch_count() {
    return get_m1_context().dispatch_count.load();
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_up_register_q4_0(
    const char * name,
    const void * weights,
    size_t weight_size,
    uint64_t input_columns,
    uint64_t output_columns) {
    if (!flag_enabled("GGML_QPU_UP_HYBRID") || !has_up_suffix(name) ||
        weights == nullptr || input_columns == 0 || input_columns > UINT32_MAX ||
        output_columns == 0 || output_columns > UINT32_MAX) {
        return 0;
    }
    const uint32_t rows = maximum_rows();
    const double max_fraction = maximum_fraction();
    if (rows == 0 || max_fraction == 0.0) {
        return 0;
    }
    const uint32_t output_count = static_cast<uint32_t>(output_columns);
    const uint32_t resident_count = aligned_columns(output_count, max_fraction);
    if (resident_count == 0) {
        return 0;
    }

    up_context & context = get_up_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.weights.find(name) != context.weights.end()) {
        return 1;
    }
    qpu_llama_status status = initialize(context);
    auto record = std::make_unique<up_weight>();
    record->input_columns = static_cast<uint32_t>(input_columns);
    record->output_columns = output_count;
    record->resident_column_count = resident_count;
    record->resident_column_start = output_count - resident_count;
    record->weight_mode = selected_weight_mode();
    const char * source_hash = record->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
        ? qpu_tiled_w8a8_gemm_dequantize_source_hash
        : record->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
            ? qpu_ggml_column_w8_q8_0_mx_source_hash
            : qpu_ggml_q4_0_q8_0_mx_source_hash;
    const qpu_llama_q4_0_linear_desc description = {
        /* .weights                = */ weights,
        /* .weight_size            = */ weight_size,
        /* .input_columns          = */ record->input_columns,
        /* .output_columns         = */ output_count,
        /* .rows                   = */ rows,
        /* .resident_column_start  = */ record->resident_column_start,
        /* .resident_column_count  = */ resident_count,
        /* .weight_mode            = */ record->weight_mode,
        /* .workgroups_per_supergroup = */ selected_wgs_per_supergroup(),
        /* .expected_source_hash   = */ source_hash,
    };
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_q4_0_linear_prepare(
            context.runtime, &description, &record->linear);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: could not persist %s: %s (%s)\n", name,
            qpu_llama_status_string(status),
            context.runtime != nullptr ? qpu_llama_context_last_error(context.runtime) : "no context");
        return 0;
    }
    if (telemetry_enabled()) {
        std::fprintf(stderr,
            "qpu_llama_weight_json:{\"schema_version\":1,\"operation\":\"ffn_up\","
            "\"tensor\":\"%s\",\"k\":%u,\"n\":%u,\"max_m\":%u,"
            "\"resident_column_start\":%u,\"resident_columns\":%u,"
            "\"resident_bytes\":%zu,\"weight_mode\":\"%s\","
            "\"wgs_per_supergroup\":%u}\n",
            name, record->input_columns, record->output_columns, rows,
            record->resident_column_start, record->resident_column_count,
            qpu_llama_q4_0_linear_resident_bytes(record->linear),
            weight_mode_name(record->weight_mode), selected_wgs_per_supergroup());
    }
    context.weights.emplace(name, std::move(record));
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_up_begin(
    const char * name,
    const void * activation_q8_0,
    size_t activation_size,
    float * destination,
    uint64_t rows,
    uint64_t input_columns,
    uint64_t output_columns,
    uint64_t activation_interleave) {
    if (!flag_enabled("GGML_QPU_UP_HYBRID") || name == nullptr ||
        activation_q8_0 == nullptr || destination == nullptr || rows == 0 ||
        rows > UINT32_MAX || input_columns > UINT32_MAX ||
        output_columns > UINT32_MAX || activation_interleave > UINT32_MAX) {
        return 0;
    }
    up_context & context = get_up_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.pending.load(std::memory_order_acquire)) {
        return 0;
    }
    const auto found = context.weights.find(name);
    if (found == context.weights.end()) {
        return 0;
    }
    up_weight & weight = *found->second;
    if (weight.input_columns != input_columns ||
        weight.output_columns != output_columns || rows > maximum_rows()) {
        return 0;
    }
    const double fraction = selected_fraction(static_cast<uint32_t>(rows));
    uint32_t qpu_columns = aligned_columns(weight.output_columns, fraction);
    qpu_columns = std::min(qpu_columns, weight.resident_column_count);
    if (qpu_columns == 0 || qpu_columns >= weight.output_columns) {
        return 0;
    }
    const uint32_t column_start = weight.output_columns - qpu_columns;
    const qpu_llama_q4_0_execution execution = {
        /* .activation        = */ activation_q8_0,
        /* .activation_size   = */ activation_size,
        /* .activation_offset = */ 0,
        /* .destination       = */ destination,
        /* .destination_size  = */ static_cast<size_t>(rows) * output_columns * sizeof(float),
        /* .destination_offset= */ 0,
        /* .column_start      = */ column_start,
        /* .column_count      = */ qpu_columns,
        /* .rows              = */ static_cast<uint32_t>(rows),
        /* .activation_interleave = */ static_cast<uint32_t>(activation_interleave),
    };
    const qpu_llama_status status = qpu_llama_q4_0_linear_submit(
        weight.linear, &execution, &context.submission);
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: ffn_up submission failed for %s: %s (%s)\n",
            name, qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        return 0;
    }
    context.pending_name = name;
    context.pending_rows = static_cast<uint32_t>(rows);
    context.pending_input_columns = static_cast<uint32_t>(input_columns);
    context.pending_output_columns = static_cast<uint32_t>(output_columns);
    context.pending_qpu_columns = qpu_columns;
    context.pending_fraction = static_cast<double>(qpu_columns) /
        static_cast<double>(weight.output_columns);
    context.pending_weight_mode = weight.weight_mode;
    context.pending_submit_end = monotonic_ns();
    context.cpu_columns.store(
        weight.output_columns - qpu_columns, std::memory_order_release);
    context.pending.store(true, std::memory_order_release);
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_up_pending() {
    return get_up_context().pending.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_up_weight_registered(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    up_context & context = get_up_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    return context.weights.find(name) != context.weights.end() ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_up_cpu_columns() {
    return get_up_context().cpu_columns.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_up_cpu_columns_for(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    up_context & context = get_up_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    return context.pending.load(std::memory_order_acquire) &&
        context.pending_name == name
        ? context.cpu_columns.load(std::memory_order_acquire) : 0;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_up_wait() {
    up_context & context = get_up_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (!context.pending.load(std::memory_order_acquire) ||
        context.submission == nullptr) {
        return 0;
    }
    const uint64_t wait_start = monotonic_ns();
    qpu_llama_q4_0_timing timing = {};
    qpu_llama_status status = qpu_llama_q4_0_submission_wait(
        context.submission, &timing);
    bool used_fallback = false;
    if (status != QPU_LLAMA_OK) {
        const qpu_llama_status fallback =
            qpu_llama_q4_0_submission_cpu_fallback(context.submission);
        used_fallback = fallback == QPU_LLAMA_OK;
        std::fprintf(stderr,
            "QPU: ffn_up completion failed for %s: %s; CPU fallback %s\n",
            context.pending_name.c_str(), qpu_llama_status_string(status),
            qpu_llama_status_string(fallback));
        status = fallback;
    }
    qpu_llama_q4_0_submission_destroy(context.submission);
    context.submission = nullptr;
    if (status == QPU_LLAMA_OK) {
        ++context.dispatch_count;
        if (telemetry_enabled()) {
            const char * program = context.pending_weight_mode ==
                    QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? "tiled-w8a8-gemm-dequantize"
                : context.pending_weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                    ? "ggml-column-w8-q8-0-mx" : "ggml-q4-0-q8-0-mx";
            const char * source_hash = context.pending_weight_mode ==
                    QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? qpu_tiled_w8a8_gemm_dequantize_source_hash
                : context.pending_weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                    ? qpu_ggml_column_w8_q8_0_mx_source_hash
                    : qpu_ggml_q4_0_q8_0_mx_source_hash;
            const char * binary_hash = context.pending_weight_mode ==
                    QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? qpu_tiled_w8a8_gemm_dequantize_binary_hash
                : context.pending_weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                    ? qpu_ggml_column_w8_q8_0_mx_binary_hash
                    : qpu_ggml_q4_0_q8_0_mx_binary_hash;
            const char * weight_contract = context.pending_weight_mode ==
                    QPU_LLAMA_Q4_0_WEIGHT_EXACT ? "Q4_0" : "APPROX_COLUMN_W8";
            const char * activation_contract = context.pending_weight_mode ==
                    QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? "APPROX_ROW_W8" : "CPU_REPACK_Q8_0x4";
            std::fprintf(stderr,
                "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
                "\"operation\":\"mul_mat_q4_0_q8_0\","
                "\"program\":\"%s\",\"source_hash\":\"%s\","
                "\"binary_sha256\":\"%s\","
                "\"exact_shape\":{\"weight\":\"%s\","
                "\"activation\":\"%s\",\"output\":\"F32\","
                "\"tile\":[16,16]},\"tensor\":\"%s\",\"placement\":\"hybrid\","
                "\"partition\":{\"axis\":\"output_columns\",\"fraction\":%.8g},"
                "\"strategy\":\"ffn_up_suffix_overlapped_with_cpu_gate\","
                "\"integration_boundary\":\"cpu-repack-up-to-geglu-join\","
                "\"m\":%u,\"k\":%u,\"n\":%u,"
                "\"qpu_columns\":%u,\"cpu_columns\":%u,\"elements\":%llu,"
                "\"input_access_ns\":%llu,\"input_pack_ns\":%llu,"
                "\"input_sync_ns\":%llu,\"input_copy_ns\":%llu,"
                "\"submit_ns\":%llu,\"wait_ns\":%llu,"
                "\"submit_wait_ns\":%llu,\"qpu_submit_wait_ns\":%llu,"
                "\"overlap_before_wait_ns\":%llu,"
                "\"output_sync_ns\":%llu,\"output_copy_ns\":%llu,"
                "\"complete_ns\":%llu,"
                "\"dispatch_count\":1,\"fallback\":%s}\n",
                program, source_hash, binary_hash, weight_contract, activation_contract,
                context.pending_name.c_str(), context.pending_fraction,
                context.pending_rows,
                context.pending_input_columns, context.pending_output_columns,
                context.pending_qpu_columns,
                context.pending_output_columns - context.pending_qpu_columns,
                static_cast<unsigned long long>(context.pending_rows) *
                    context.pending_qpu_columns,
                static_cast<unsigned long long>(timing.input_access_ns),
                static_cast<unsigned long long>(timing.input_pack_ns),
                static_cast<unsigned long long>(timing.input_sync_ns),
                static_cast<unsigned long long>(timing.input_copy_ns),
                static_cast<unsigned long long>(timing.submit_ns),
                static_cast<unsigned long long>(timing.wait_ns),
                static_cast<unsigned long long>(timing.submit_wait_ns),
                static_cast<unsigned long long>(timing.submit_wait_ns),
                static_cast<unsigned long long>(wait_start - context.pending_submit_end),
                static_cast<unsigned long long>(timing.output_sync_ns),
                static_cast<unsigned long long>(timing.output_copy_ns),
                static_cast<unsigned long long>(timing.complete_ns),
                used_fallback ? "true" : "false");
        }
    }
    context.pending.store(false, std::memory_order_release);
    context.cpu_columns.store(0, std::memory_order_release);
    return status == QPU_LLAMA_OK ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_up_dispatch_count() {
    return get_up_context().dispatch_count.load();
}
