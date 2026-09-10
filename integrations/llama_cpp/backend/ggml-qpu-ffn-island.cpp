#include "qpu_llama_runtime.h"

#include "ggml-geglu-q8-0-split.h"
#include "ggml-q4-0-q8-wordscale-mx.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t q4_block_elements = 32;
constexpr uint32_t q4_block_bytes = 18;
constexpr uint32_t q8_block_bytes = 34;
constexpr uint32_t output_tile = 16;
constexpr uint32_t row_tile = 16;

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

uint32_t unsigned_setting(const char * name, uint32_t fallback, uint32_t minimum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed >= minimum && parsed <= UINT32_MAX
        ? static_cast<uint32_t>(parsed) : 0;
}

double configured_fraction() {
    const char * value = std::getenv("GGML_QPU_FFN_ISLAND_FRACTION");
    if (value == nullptr || value[0] == '\0') {
        return 0.0;
    }
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && *end == '\0' && parsed >= 0.03125 && parsed <= 0.5
        ? parsed : 0.0;
}

uint32_t configured_columns(uint32_t intermediate_columns) {
    const double fraction = configured_fraction();
    if (fraction == 0.0) {
        return 0;
    }
    const uint32_t columns = static_cast<uint32_t>(
        std::floor(static_cast<double>(intermediate_columns) * fraction));
    const uint32_t aligned = columns & ~(q4_block_elements - 1U);
    return aligned > 0 && aligned < intermediate_columns ? aligned : 0;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & UINT16_C(0x8000)) << 16U;
    uint32_t exponent = (value >> 10U) & UINT16_C(0x001f);
    uint32_t mantissa = value & UINT16_C(0x03ff);
    uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            exponent = 113U;
            while ((mantissa & UINT32_C(0x0400)) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= UINT32_C(0x03ff);
            bits = sign | (exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
    const uint32_t absolute = bits & UINT32_C(0x7fffffff);
    if (absolute >= UINT32_C(0x7f800000)) {
        const uint16_t payload = absolute > UINT32_C(0x7f800000)
            ? static_cast<uint16_t>((absolute >> 13U) & UINT32_C(0x03ff)) : 0;
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00) | (payload != 0 ? payload : 0));
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
            rounded = 0;
            const uint32_t half_exponent = static_cast<uint32_t>(exponent + 16);
            return static_cast<uint16_t>(sign | (half_exponent << 10U));
        }
    }
    return static_cast<uint16_t>(sign |
        (static_cast<uint32_t>(exponent + 15) << 10U) | rounded);
}

struct buffer_owner {
    qpu_llama_buffer * value = nullptr;

    buffer_owner() = default;
    buffer_owner(const buffer_owner &) = delete;
    buffer_owner & operator=(const buffer_owner &) = delete;
    buffer_owner(buffer_owner && other) noexcept : value(other.value) {
        other.value = nullptr;
    }
    buffer_owner & operator=(buffer_owner && other) noexcept {
        if (this != &other) {
            qpu_llama_buffer_destroy(value);
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }
    ~buffer_owner() {
        qpu_llama_buffer_destroy(value);
    }
};

struct packed_matrix {
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t blocks = 0;
    std::vector<uint16_t> scales;
    std::vector<uint32_t> quants;
    buffer_owner scale_buffer;
    buffer_owner quant_buffer;

    size_t resident_bytes() const {
        return scales.size() * sizeof(uint16_t) + quants.size() * sizeof(uint32_t);
    }

    bool ready() const {
        return scale_buffer.value != nullptr && quant_buffer.value != nullptr;
    }
};

struct layer_record {
    int layer = -1;
    std::string prefix;
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t hidden_columns = 0;
    uint32_t cpu_columns = 0;
    uint32_t qpu_columns = 0;
    packed_matrix gate;
    packed_matrix up;
    packed_matrix down;

    bool ready() const {
        return layer >= 0 && input_columns > 0 && hidden_columns > 0 &&
            intermediate_columns > 0 && qpu_columns > 0 &&
            gate.ready() && up.ready() && down.ready();
    }

    size_t resident_bytes() const {
        return gate.resident_bytes() + up.resident_bytes() + down.resident_bytes();
    }
};

struct scratch_state {
    uint32_t rows = 0;
    uint32_t input_columns = 0;
    uint32_t qpu_columns = 0;
    uint32_t hidden_columns = 0;
    buffer_owner input_scales;
    buffer_owner input_quants;
    buffer_owner gate_output;
    buffer_owner up_output;
    buffer_owner intermediate_scales;
    buffer_owner intermediate_quants;
    buffer_owner output;
    std::vector<uint32_t> host_input_scales;
    std::vector<uint8_t> host_input_quants;
    std::vector<float> host_gate;
    std::vector<float> host_up;
    std::vector<uint32_t> host_intermediate_scales;
    std::vector<uint8_t> host_intermediate_quants;
    std::vector<float> host_output;
    std::vector<float> host_qpu_output;
};

enum role_bits : uint32_t {
    role_up_seen = 1U << 0,
    role_gate_seen = 1U << 1,
    role_geglu_seen = 1U << 2,
    role_down_seen = 1U << 3,
};

struct island_timing {
    uint64_t begin_start = 0;
    uint64_t input_access_ns = 0;
    uint64_t input_pack_ns = 0;
    uint64_t input_sync_ns = 0;
    uint64_t begin_end = 0;
    uint64_t gate_ns = 0;
    uint64_t up_ns = 0;
    uint64_t geglu_ns = 0;
    uint64_t down_ns = 0;
    uint64_t output_copy_ns = 0;
    uint64_t qpu_complete_ns = 0;
    uint64_t verification_ns = 0;
    uint64_t overlap_before_wait_ns = 0;
    uint64_t exposed_wait_ns = 0;
    uint64_t fallback_ns = 0;
    uint64_t join_start = 0;
    uint64_t join_ns = 0;
    uint32_t completed_dispatches = 0;
    bool verified = false;
    double verification_max_abs = 0.0;
    double verification_mean_abs = 0.0;
};

struct island_context {
    std::mutex mutex;
    std::condition_variable work_condition;
    std::condition_variable done_condition;
    qpu_llama_context * runtime = nullptr;
    qpu_llama_program * linear_program = nullptr;
    qpu_llama_program * geglu_program = nullptr;
    buffer_owner gelu_table_buffer;
    std::vector<uint16_t> gelu_table;
    std::unordered_map<std::string, std::unique_ptr<layer_record>> layers;
    scratch_state scratch;
    std::thread worker;
    bool worker_started = false;
    bool stop = false;
    bool work_ready = false;
    bool done = false;
    bool success = false;
    bool fallback = false;
    std::string failure_stage;
    std::string failure_detail;
    layer_record * active_layer = nullptr;
    uint32_t active_rows = 0;
    uint32_t padded_rows = 0;
    uint32_t activation_interleave = 0;
    island_timing timing;
    std::atomic<bool> pending = false;
    std::atomic<int> pending_layer{-1};
    std::atomic<uint32_t> active_cpu_columns{0};
    std::atomic<uint32_t> seen_roles{0};
    std::atomic<uint64_t> island_count{0};
    std::atomic<uint64_t> dispatch_count{0};

    ~island_context();
};

island_context & get_context() {
    static island_context context;
    return context;
}

bool parse_tensor_name(const char * name, int & layer, std::string & prefix, std::string & role) {
    if (name == nullptr) {
        return false;
    }
    const std::string text(name);
    const size_t marker = text.find(".ffn_");
    if (marker == std::string::npos || text.compare(text.size() >= 7 ? text.size() - 7 : 0,
            7, ".weight") != 0 || text.compare(0, 4, "blk.") != 0) {
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(text.c_str() + 4, &end, 10);
    if (end != text.c_str() + marker || parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    role = text.substr(marker + 5, text.size() - marker - 12);
    if (role != "gate" && role != "up" && role != "down") {
        return false;
    }
    layer = static_cast<int>(parsed);
    prefix = text.substr(0, marker);
    return true;
}

bool matches_active(const island_context & context, const char * name, const char * role) {
    int layer = -1;
    std::string prefix;
    std::string parsed_role;
    return context.pending.load(std::memory_order_acquire) &&
        parse_tensor_name(name, layer, prefix, parsed_role) && parsed_role == role &&
        layer == context.pending_layer.load(std::memory_order_acquire) &&
        context.active_layer != nullptr && prefix == context.active_layer->prefix;
}

qpu_llama_status create_buffer(
    qpu_llama_context * runtime,
    size_t size,
    buffer_owner & result,
    bool cached = false) {
    qpu_llama_buffer * value = nullptr;
    const qpu_llama_status status = cached
        ? qpu_llama_buffer_create_cached(runtime, size, &value)
        : qpu_llama_buffer_create(runtime, size, &value);
    if (status == QPU_LLAMA_OK) {
        result = buffer_owner();
        result.value = value;
    }
    return status;
}

qpu_llama_status upload_buffer(
    qpu_llama_context * runtime,
    const void * source,
    size_t size,
    buffer_owner & result) {
    qpu_llama_status status = create_buffer(runtime, size, result);
    if (status == QPU_LLAMA_OK) {
        std::memcpy(qpu_llama_buffer_data(result.value), source, size);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    return status;
}

qpu_llama_status pack_matrix(
    island_context & context,
    const void * weights,
    size_t weight_size,
    uint32_t total_input_columns,
    uint32_t total_output_columns,
    uint32_t input_block_start,
    uint32_t selected_blocks,
    uint32_t output_start,
    uint32_t selected_outputs,
    packed_matrix & result) {
    const uint32_t total_blocks = total_input_columns / q4_block_elements;
    const size_t expected = static_cast<size_t>(total_output_columns) * total_blocks * q4_block_bytes;
    if (weights == nullptr || total_input_columns % q4_block_elements != 0 ||
        selected_blocks == 0 || selected_outputs == 0 || selected_outputs % output_tile != 0 ||
        input_block_start + selected_blocks > total_blocks ||
        output_start + selected_outputs > total_output_columns || weight_size < expected) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    result.input_columns = selected_blocks * q4_block_elements;
    result.output_columns = selected_outputs;
    result.blocks = selected_blocks;
    result.scales.assign(static_cast<size_t>(selected_blocks) * selected_outputs, 0);
    result.quants.assign(static_cast<size_t>(selected_blocks) * 8U * selected_outputs, 0);
    const auto * source = static_cast<const uint8_t *>(weights);
    auto * quant_bytes = reinterpret_cast<uint8_t *>(result.quants.data());
    for (uint32_t output = 0; output < selected_outputs; ++output) {
        const uint32_t source_output = output_start + output;
        for (uint32_t block = 0; block < selected_blocks; ++block) {
            const uint32_t source_block = input_block_start + block;
            const uint8_t * native = source +
                (static_cast<size_t>(source_output) * total_blocks + source_block) * q4_block_bytes;
            std::memcpy(&result.scales[static_cast<size_t>(block) * selected_outputs + output],
                native, sizeof(uint16_t));
            for (uint32_t index = 0; index < q4_block_elements; ++index) {
                const uint8_t packed = native[2U + index % 16U];
                const uint8_t code = index < 16U ? packed & 0x0fU : packed >> 4U;
                const int8_t value = static_cast<int8_t>(static_cast<int>(code) - 8);
                const size_t byte_offset =
                    ((static_cast<size_t>(block) * 8U + index / 4U) * selected_outputs + output) *
                        sizeof(uint32_t) + index % 4U;
                quant_bytes[byte_offset] = static_cast<uint8_t>(value);
            }
        }
    }
    qpu_llama_status status = upload_buffer(context.runtime, result.scales.data(),
        result.scales.size() * sizeof(uint16_t), result.scale_buffer);
    if (status == QPU_LLAMA_OK) {
        status = upload_buffer(context.runtime, result.quants.data(),
            result.quants.size() * sizeof(uint32_t), result.quant_buffer);
    }
    return status;
}

qpu_llama_status initialize(island_context & context);
void worker_loop(island_context * context);

qpu_llama_status ensure_scratch(island_context & context) {
    uint32_t maximum_input = 0;
    uint32_t maximum_qpu = 0;
    uint32_t maximum_hidden = 0;
    for (const auto & item : context.layers) {
        const layer_record & layer = *item.second;
        maximum_input = std::max(maximum_input, layer.input_columns);
        maximum_qpu = std::max(maximum_qpu, layer.qpu_columns);
        maximum_hidden = std::max(maximum_hidden, layer.hidden_columns);
    }
    const uint32_t maximum_rows = unsigned_setting(
        "GGML_QPU_FFN_ISLAND_MAX_ROWS", 528U, row_tile);
    if (maximum_rows == 0 || maximum_input == 0 || maximum_qpu == 0 || maximum_hidden == 0) {
        return QPU_LLAMA_OK;
    }
    const uint32_t padded = (maximum_rows + row_tile - 1U) & ~(row_tile - 1U);
    if (context.scratch.rows >= padded && context.scratch.input_columns >= maximum_input &&
        context.scratch.qpu_columns >= maximum_qpu &&
        context.scratch.hidden_columns >= maximum_hidden) {
        return QPU_LLAMA_OK;
    }
    scratch_state replacement;
    replacement.rows = padded;
    replacement.input_columns = maximum_input;
    replacement.qpu_columns = maximum_qpu;
    replacement.hidden_columns = maximum_hidden;
    const uint32_t input_blocks = maximum_input / q4_block_elements;
    const uint32_t qpu_blocks = maximum_qpu / q4_block_elements;
    qpu_llama_status status = create_buffer(context.runtime,
        static_cast<size_t>(padded) * input_blocks * sizeof(uint32_t),
        replacement.input_scales, true);
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * maximum_input,
            replacement.input_quants, true);
    }
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * maximum_qpu * sizeof(float),
            replacement.gate_output, true);
    }
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * maximum_qpu * sizeof(float),
            replacement.up_output, true);
    }
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * qpu_blocks * sizeof(uint32_t),
            replacement.intermediate_scales, true);
    }
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * maximum_qpu,
            replacement.intermediate_quants, true);
    }
    if (status == QPU_LLAMA_OK) {
        status = create_buffer(context.runtime,
            static_cast<size_t>(padded) * maximum_hidden * sizeof(float),
            replacement.output, true);
    }
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    replacement.host_input_scales.resize(static_cast<size_t>(padded) * input_blocks);
    replacement.host_input_quants.resize(static_cast<size_t>(padded) * maximum_input);
    replacement.host_gate.resize(static_cast<size_t>(padded) * maximum_qpu);
    replacement.host_up.resize(static_cast<size_t>(padded) * maximum_qpu);
    replacement.host_intermediate_scales.resize(static_cast<size_t>(padded) * qpu_blocks);
    replacement.host_intermediate_quants.resize(static_cast<size_t>(padded) * maximum_qpu);
    replacement.host_output.resize(static_cast<size_t>(padded) * maximum_hidden);
    replacement.host_qpu_output.resize(static_cast<size_t>(padded) * maximum_hidden);
    context.scratch = std::move(replacement);
    return QPU_LLAMA_OK;
}

void build_gelu_table(island_context & context) {
    context.gelu_table.resize(UINT32_C(1) << 16U);
    for (uint32_t bits = 0; bits < (UINT32_C(1) << 16U); ++bits) {
        const float input = half_to_float(static_cast<uint16_t>(bits));
        const float output = 0.5f * input *
            (1.0f + std::erf(input / std::sqrt(2.0f)));
        context.gelu_table[bits] = float_to_half(output);
    }
}

qpu_llama_status initialize(island_context & context) {
    if (context.runtime != nullptr && context.linear_program != nullptr &&
        context.geglu_program != nullptr && context.gelu_table_buffer.value != nullptr &&
        context.worker_started) {
        return QPU_LLAMA_OK;
    }
    qpu_llama_status status = qpu_llama_context_create(nullptr, &context.runtime);
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    const qpu_llama_program_desc linear_description = {
        /* .code = */ qpu_ggml_q4_0_q8_wordscale_mx,
        /* .code_size = */ sizeof(qpu_ggml_q4_0_q8_wordscale_mx),
        /* .compiled_source_hash = */ qpu_ggml_q4_0_q8_wordscale_mx_source_hash,
        /* .expected_source_hash = */ qpu_ggml_q4_0_q8_wordscale_mx_source_hash,
        /* .binary_sha256 = */ qpu_ggml_q4_0_q8_wordscale_mx_binary_hash,
        /* .uniform_word_count = */ 11,
    };
    status = qpu_llama_program_create(context.runtime, &linear_description,
        &context.linear_program);
    const qpu_llama_program_desc geglu_description = {
        /* .code = */ qpu_ggml_geglu_q8_0_split,
        /* .code_size = */ sizeof(qpu_ggml_geglu_q8_0_split),
        /* .compiled_source_hash = */ qpu_ggml_geglu_q8_0_split_source_hash,
        /* .expected_source_hash = */ qpu_ggml_geglu_q8_0_split_source_hash,
        /* .binary_sha256 = */ qpu_ggml_geglu_q8_0_split_binary_hash,
        /* .uniform_word_count = */ 8,
    };
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_program_create(context.runtime, &geglu_description,
            &context.geglu_program);
    }
    if (status == QPU_LLAMA_OK) {
        build_gelu_table(context);
        status = upload_buffer(context.runtime, context.gelu_table.data(),
            context.gelu_table.size() * sizeof(uint16_t), context.gelu_table_buffer);
    }
    if (status == QPU_LLAMA_OK) {
        context.worker = std::thread(worker_loop, &context);
        context.worker_started = true;
    } else {
        context.gelu_table_buffer = buffer_owner();
        qpu_llama_program_destroy(context.geglu_program);
        qpu_llama_program_destroy(context.linear_program);
        qpu_llama_context_destroy(context.runtime);
        context.geglu_program = nullptr;
        context.linear_program = nullptr;
        context.runtime = nullptr;
    }
    return status;
}

island_context::~island_context() {
    if (worker_started) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            work_condition.notify_one();
        }
        worker.join();
    }
    layers.clear();
    scratch = scratch_state();
    gelu_table_buffer = buffer_owner();
    qpu_llama_program_destroy(geglu_program);
    qpu_llama_program_destroy(linear_program);
    qpu_llama_context_destroy(runtime);
}

const uint8_t * q8_scale_pointer(
    const uint8_t * source,
    size_t row_bytes,
    uint32_t row,
    uint32_t rows,
    uint32_t block,
    uint32_t interleave) {
    const uint32_t interleaved_rows = interleave == 0 ? 0 : rows - rows % 4U;
    if (row < interleaved_rows) {
        return source + static_cast<size_t>(row / 4U) * 4U * row_bytes +
            static_cast<size_t>(block) * 4U * q8_block_bytes +
            (row % 4U) * sizeof(uint16_t);
    }
    return source + static_cast<size_t>(row) * row_bytes +
        static_cast<size_t>(block) * q8_block_bytes;
}

int8_t q8_value(
    const uint8_t * source,
    size_t row_bytes,
    uint32_t row,
    uint32_t rows,
    uint32_t block,
    uint32_t index,
    uint32_t interleave) {
    const uint32_t interleaved_rows = interleave == 0 ? 0 : rows - rows % 4U;
    if (row < interleaved_rows) {
        const uint8_t * values = source + static_cast<size_t>(row / 4U) * 4U * row_bytes +
            static_cast<size_t>(block) * 4U * q8_block_bytes + 4U * sizeof(uint16_t);
        const uint32_t source_index = (index / interleave) * 4U * interleave +
            (row % 4U) * interleave + index % interleave;
        return static_cast<int8_t>(values[source_index]);
    }
    return static_cast<int8_t>(source[static_cast<size_t>(row) * row_bytes +
        static_cast<size_t>(block) * q8_block_bytes + sizeof(uint16_t) + index]);
}

qpu_llama_status pack_activation(
    island_context & context,
    const void * activation,
    size_t activation_size,
    uint32_t rows,
    uint32_t input_columns,
    uint32_t interleave) {
    const uint32_t blocks = input_columns / q4_block_elements;
    const size_t row_bytes = static_cast<size_t>(blocks) * q8_block_bytes;
    if (activation == nullptr || activation_size < static_cast<size_t>(rows) * row_bytes ||
        (interleave != 0 && interleave != 4 && interleave != 8)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    scratch_state & scratch = context.scratch;
    const auto * source = static_cast<const uint8_t *>(activation);
    std::fill_n(scratch.host_input_scales.data(),
        static_cast<size_t>(context.padded_rows) * blocks, 0U);
    std::fill_n(scratch.host_input_quants.data(),
        static_cast<size_t>(context.padded_rows) * input_columns, UINT8_C(0));
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            uint16_t scale = 0;
            std::memcpy(&scale,
                q8_scale_pointer(source, row_bytes, row, rows, block, interleave),
                sizeof(scale));
            scratch.host_input_scales[static_cast<size_t>(row) * blocks + block] = scale;
            uint8_t * destination = scratch.host_input_quants.data() +
                static_cast<size_t>(row) * input_columns + block * q4_block_elements;
            for (uint32_t index = 0; index < q4_block_elements; ++index) {
                destination[index] = static_cast<uint8_t>(q8_value(
                    source, row_bytes, row, rows, block, index, interleave));
            }
        }
    }
    const uint64_t access_start = monotonic_ns();
    qpu_llama_status status = qpu_llama_buffer_cpu_access_begin(
        scratch.input_scales.value, QPU_LLAMA_CPU_ACCESS_WRITE);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_begin(
            scratch.input_quants.value, QPU_LLAMA_CPU_ACCESS_WRITE);
    }
    const uint64_t access_end = monotonic_ns();
    if (status == QPU_LLAMA_OK) {
        std::memcpy(qpu_llama_buffer_data(scratch.input_scales.value),
            scratch.host_input_scales.data(),
            static_cast<size_t>(context.padded_rows) * blocks * sizeof(uint32_t));
        std::memcpy(qpu_llama_buffer_data(scratch.input_quants.value),
            scratch.host_input_quants.data(),
            static_cast<size_t>(context.padded_rows) * input_columns);
    }
    const uint64_t copy_end = monotonic_ns();
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_end(
            scratch.input_quants.value, QPU_LLAMA_CPU_ACCESS_WRITE);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_end(
            scratch.input_scales.value, QPU_LLAMA_CPU_ACCESS_WRITE);
    }
    const uint64_t sync_end = monotonic_ns();
    context.timing.input_access_ns = access_end - access_start;
    context.timing.input_pack_ns = copy_end - context.timing.begin_start;
    context.timing.input_sync_ns = sync_end - copy_end;
    return status;
}

qpu_llama_status buffer_address(qpu_llama_buffer * buffer, uint32_t & address) {
    return qpu_llama_buffer_gpu_address(buffer, 0, &address);
}

qpu_llama_status execute_linear(
    island_context & context,
    const packed_matrix & weight,
    qpu_llama_buffer * activation_quants,
    qpu_llama_buffer * activation_scales,
    qpu_llama_buffer * destination,
    uint32_t rows) {
    uint32_t activation_q_address = 0;
    uint32_t activation_scale_address = 0;
    uint32_t weight_q_address = 0;
    uint32_t weight_scale_address = 0;
    uint32_t destination_address = 0;
    qpu_llama_status status = buffer_address(activation_quants, activation_q_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(activation_scales, activation_scale_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(weight.quant_buffer.value, weight_q_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(weight.scale_buffer.value, weight_scale_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(destination, destination_address);
    const uint32_t uniforms[] = {
        weight.input_columns,
        activation_q_address,
        weight.output_columns * static_cast<uint32_t>(sizeof(uint32_t)),
        weight_q_address,
        weight.output_columns * static_cast<uint32_t>(sizeof(float)),
        destination_address,
        weight.blocks,
        weight.blocks * static_cast<uint32_t>(sizeof(uint32_t)),
        activation_scale_address,
        weight.output_columns * static_cast<uint32_t>(sizeof(uint16_t)),
        weight_scale_address,
    };
    qpu_llama_buffer * buffers[] = {
        activation_quants, activation_scales, weight.quant_buffer.value,
        weight.scale_buffer.value, destination,
    };
    const uint32_t workgroup_x = weight.output_columns / output_tile;
    const uint32_t workgroup_y = rows / row_tile;
    const qpu_llama_dispatch_desc dispatch = {
        /* .uniforms = */ uniforms,
        /* .uniform_word_count = */ 11,
        /* .buffers = */ buffers,
        /* .buffer_count = */ 5,
        /* .local_invocation = */ {16, 1, 1},
        /* .workgroup = */ {workgroup_x, workgroup_y, 1},
        /* .wgs_per_sg = */ unsigned_setting("GGML_QPU_FFN_ISLAND_WGS", 24U, 1U),
        /* .thread_count = */ workgroup_x * workgroup_y,
        /* .propagate_nan = */ 0,
        /* .single_segment = */ 0,
        /* .threading = */ 0,
    };
    return status == QPU_LLAMA_OK
        ? qpu_llama_program_execute(context.linear_program, &dispatch) : status;
}

qpu_llama_status execute_geglu(island_context & context, uint32_t rows, uint32_t columns) {
    uint32_t gate_address = 0;
    uint32_t up_address = 0;
    uint32_t scale_address = 0;
    uint32_t quant_address = 0;
    uint32_t table_address = 0;
    qpu_llama_status status = buffer_address(context.scratch.gate_output.value, gate_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(context.scratch.up_output.value, up_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(context.scratch.intermediate_scales.value, scale_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(context.scratch.intermediate_quants.value, quant_address);
    if (status == QPU_LLAMA_OK) status = buffer_address(context.gelu_table_buffer.value, table_address);
    const uint32_t blocks = columns / q4_block_elements;
    const uint32_t uniforms[] = {
        blocks, gate_address, up_address, scale_address, quant_address, table_address,
        float_bits(1.0f / 127.0f), float_bits(std::ldexp(1.0f, -24)),
    };
    qpu_llama_buffer * buffers[] = {
        context.scratch.gate_output.value, context.scratch.up_output.value,
        context.scratch.intermediate_scales.value,
        context.scratch.intermediate_quants.value, context.gelu_table_buffer.value,
    };
    const qpu_llama_dispatch_desc dispatch = {
        /* .uniforms = */ uniforms,
        /* .uniform_word_count = */ 8,
        /* .buffers = */ buffers,
        /* .buffer_count = */ 5,
        /* .local_invocation = */ {16, 1, 1},
        /* .workgroup = */ {blocks, rows, 1},
        /* .wgs_per_sg = */ unsigned_setting("GGML_QPU_FFN_ISLAND_WGS", 24U, 1U),
        /* .thread_count = */ blocks * rows,
        /* .propagate_nan = */ 0,
        /* .single_segment = */ 0,
        /* .threading = */ 0,
    };
    return status == QPU_LLAMA_OK
        ? qpu_llama_program_execute(context.geglu_program, &dispatch) : status;
}

bool injected(const island_context & context, const char * stage) {
    const char * requested = std::getenv("GGML_QPU_FFN_ISLAND_FAIL_STAGE");
    if (requested == nullptr || std::strcmp(requested, stage) != 0) {
        return false;
    }
    const char * layer_text = std::getenv("GGML_QPU_FFN_ISLAND_FAIL_LAYER");
    if (layer_text == nullptr || layer_text[0] == '\0') {
        return true;
    }
    char * end = nullptr;
    const long layer = std::strtol(layer_text, &end, 10);
    return end != layer_text && *end == '\0' && context.active_layer != nullptr &&
        layer == context.active_layer->layer;
}

qpu_llama_status execute_stage(
    island_context & context,
    const char * stage,
    uint64_t & duration,
    const std::function<qpu_llama_status()> & operation) {
    const uint64_t start = monotonic_ns();
    if (injected(context, stage)) {
        duration = monotonic_ns() - start;
        context.failure_stage = stage;
        context.failure_detail = "injected failure";
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    const qpu_llama_status status = operation();
    duration = monotonic_ns() - start;
    if (status == QPU_LLAMA_OK) {
        ++context.timing.completed_dispatches;
    } else {
        context.failure_stage = stage;
        context.failure_detail = qpu_llama_status_string(status);
        const char * detail = qpu_llama_context_last_error(context.runtime);
        if (detail != nullptr && detail[0] != '\0') {
            context.failure_detail += ": ";
            context.failure_detail += detail;
        }
    }
    return status;
}

int8_t packed_weight_value(const packed_matrix & weight, uint32_t block, uint32_t output,
        uint32_t index) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(weight.quants.data());
    const size_t offset =
        ((static_cast<size_t>(block) * 8U + index / 4U) * weight.output_columns + output) *
            sizeof(uint32_t) + index % 4U;
    return static_cast<int8_t>(bytes[offset]);
}

void cpu_linear(
    const std::vector<uint8_t> & activation_quants,
    const std::vector<uint32_t> & activation_scales,
    uint32_t rows,
    const packed_matrix & weight,
    std::vector<float> & output) {
    output.resize(static_cast<size_t>(rows) * weight.output_columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < weight.output_columns; ++column) {
            float sum = 0.0f;
            for (uint32_t block = 0; block < weight.blocks; ++block) {
                int32_t dot = 0;
                const auto * values = reinterpret_cast<const int8_t *>(activation_quants.data()) +
                    static_cast<size_t>(row) * weight.input_columns + block * q4_block_elements;
                for (uint32_t index = 0; index < q4_block_elements; ++index) {
                    dot += static_cast<int32_t>(values[index]) *
                        static_cast<int32_t>(packed_weight_value(weight, block, column, index));
                }
                const uint16_t activation_scale = static_cast<uint16_t>(
                    activation_scales[static_cast<size_t>(row) * weight.blocks + block]);
                const uint16_t weight_scale =
                    weight.scales[static_cast<size_t>(block) * weight.output_columns + column];
                sum += static_cast<float>(dot) * half_to_float(activation_scale) *
                    half_to_float(weight_scale);
            }
            output[static_cast<size_t>(row) * weight.output_columns + column] = sum;
        }
    }
}

void cpu_geglu_quantize(island_context & context, uint32_t rows, uint32_t columns) {
    scratch_state & scratch = context.scratch;
    const uint32_t blocks = columns / q4_block_elements;
    std::fill_n(scratch.host_intermediate_scales.data(),
        static_cast<size_t>(rows) * blocks, 0U);
    std::fill_n(scratch.host_intermediate_quants.data(),
        static_cast<size_t>(rows) * columns, UINT8_C(0));
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            float values[q4_block_elements];
            float maximum = 0.0f;
            for (uint32_t index = 0; index < q4_block_elements; ++index) {
                const size_t offset = static_cast<size_t>(row) * columns +
                    block * q4_block_elements + index;
                const uint16_t table_index = float_to_half(scratch.host_gate[offset]);
                values[index] = half_to_float(context.gelu_table[table_index]) *
                    scratch.host_up[offset];
                maximum = std::max(maximum, std::fabs(values[index]));
            }
            const float scale = maximum / 127.0f;
            scratch.host_intermediate_scales[static_cast<size_t>(row) * blocks + block] =
                float_to_half(scale);
            for (uint32_t index = 0; index < q4_block_elements; ++index) {
                const float scaled = scale != 0.0f ? values[index] / scale : 0.0f;
                long quantized = std::lrint(scaled);
                quantized = std::max(-127L, std::min(127L, quantized));
                scratch.host_intermediate_quants[static_cast<size_t>(row) * columns +
                    block * q4_block_elements + index] =
                        static_cast<uint8_t>(static_cast<int8_t>(quantized));
            }
        }
    }
}

void cpu_fallback(island_context & context, const layer_record & layer, uint32_t rows) {
    scratch_state & scratch = context.scratch;
    cpu_linear(scratch.host_input_quants, scratch.host_input_scales,
        rows, layer.gate, scratch.host_gate);
    cpu_linear(scratch.host_input_quants, scratch.host_input_scales,
        rows, layer.up, scratch.host_up);
    cpu_geglu_quantize(context, rows, layer.qpu_columns);
    cpu_linear(scratch.host_intermediate_quants, scratch.host_intermediate_scales,
        rows, layer.down, scratch.host_output);
}

qpu_llama_status copy_output(island_context & context, const layer_record & layer) {
    const uint64_t start = monotonic_ns();
    qpu_llama_status status = qpu_llama_buffer_cpu_access_begin(
        context.scratch.output.value, QPU_LLAMA_CPU_ACCESS_READ);
    if (status == QPU_LLAMA_OK) {
        std::memcpy(context.scratch.host_output.data(),
            qpu_llama_buffer_data(context.scratch.output.value),
            static_cast<size_t>(context.active_rows) * layer.hidden_columns * sizeof(float));
        status = qpu_llama_buffer_cpu_access_end(
            context.scratch.output.value, QPU_LLAMA_CPU_ACCESS_READ);
    }
    context.timing.output_copy_ns = monotonic_ns() - start;
    return status;
}

void worker_loop(island_context * context) {
    std::unique_lock<std::mutex> lock(context->mutex);
    for (;;) {
        context->work_condition.wait(lock, [&] { return context->stop || context->work_ready; });
        if (context->stop) {
            return;
        }
        layer_record * layer = context->active_layer;
        const uint32_t rows = context->padded_rows;
        context->work_ready = false;
        lock.unlock();

        qpu_llama_status status = execute_stage(*context, "gate", context->timing.gate_ns, [&] {
            return execute_linear(*context, layer->gate,
                context->scratch.input_quants.value, context->scratch.input_scales.value,
                context->scratch.gate_output.value, rows);
        });
        if (status == QPU_LLAMA_OK) {
            status = execute_stage(*context, "up", context->timing.up_ns, [&] {
                return execute_linear(*context, layer->up,
                    context->scratch.input_quants.value, context->scratch.input_scales.value,
                    context->scratch.up_output.value, rows);
            });
        }
        if (status == QPU_LLAMA_OK) {
            status = execute_stage(*context, "geglu", context->timing.geglu_ns, [&] {
                return execute_geglu(*context, rows, layer->qpu_columns);
            });
        }
        if (status == QPU_LLAMA_OK) {
            status = execute_stage(*context, "down", context->timing.down_ns, [&] {
                return execute_linear(*context, layer->down,
                    context->scratch.intermediate_quants.value,
                    context->scratch.intermediate_scales.value,
                    context->scratch.output.value, rows);
            });
        }
        if (status == QPU_LLAMA_OK && injected(*context, "output")) {
            status = QPU_LLAMA_INTERNAL_ERROR;
            context->failure_stage = "output";
            context->failure_detail = "injected failure";
        }
        if (status == QPU_LLAMA_OK) {
            status = copy_output(*context, *layer);
            if (status != QPU_LLAMA_OK) {
                context->failure_stage = "output";
                context->failure_detail = qpu_llama_status_string(status);
            }
        }
        bool fallback = status != QPU_LLAMA_OK;
        if (fallback) {
            const uint64_t start = monotonic_ns();
            cpu_fallback(*context, *layer, context->active_rows);
            context->timing.fallback_ns = monotonic_ns() - start;
            status = QPU_LLAMA_OK;
        } else {
            context->timing.qpu_complete_ns = monotonic_ns() - context->timing.begin_start;
            if (flag_enabled("GGML_QPU_FFN_ISLAND_VERIFY")) {
                const uint64_t start = monotonic_ns();
                const size_t elements = static_cast<size_t>(context->active_rows) *
                    layer->hidden_columns;
                std::copy_n(context->scratch.host_output.data(), elements,
                    context->scratch.host_qpu_output.data());
                cpu_fallback(*context, *layer, context->active_rows);
                double absolute_sum = 0.0;
                double maximum = 0.0;
                for (size_t index = 0; index < elements; ++index) {
                    const double error = std::fabs(
                        static_cast<double>(context->scratch.host_output[index]) -
                        static_cast<double>(context->scratch.host_qpu_output[index]));
                    absolute_sum += error;
                    maximum = std::max(maximum, error);
                }
                context->timing.verified = true;
                context->timing.verification_max_abs = maximum;
                context->timing.verification_mean_abs = elements != 0
                    ? absolute_sum / static_cast<double>(elements) : 0.0;
                std::copy_n(context->scratch.host_qpu_output.data(), elements,
                    context->scratch.host_output.data());
                context->timing.verification_ns = monotonic_ns() - start;
            }
        }
        if (context->timing.qpu_complete_ns == 0) {
            context->timing.qpu_complete_ns = monotonic_ns() - context->timing.begin_start;
        }

        lock.lock();
        context->success = status == QPU_LLAMA_OK;
        context->fallback = fallback;
        context->done = true;
        context->done_condition.notify_all();
    }
}

bool telemetry_enabled() {
    const char * value = std::getenv("GGML_QPU_TELEMETRY");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

void emit_weight(const layer_record & layer, const char * role, const packed_matrix & weight) {
    if (!telemetry_enabled()) {
        return;
    }
    std::fprintf(stderr,
        "qpu_llama_weight_json:{\"schema_version\":1,"
        "\"operation\":\"ffn_island\",\"layer\":%d,\"role\":\"%s\","
        "\"k\":%u,\"n\":%u,\"cpu_intermediate_columns\":%u,"
        "\"qpu_intermediate_columns\":%u,\"resident_bytes\":%zu}\n",
        layer.layer, role, weight.input_columns, weight.output_columns,
        layer.cpu_columns, layer.qpu_columns, weight.resident_bytes());
}

} // namespace

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_ffn_island_register_q4_0(
    const char * name,
    const void * weights,
    size_t weight_size,
    uint64_t input_columns,
    uint64_t output_columns) {
    if (!flag_enabled("GGML_QPU_FFN_ISLAND") || input_columns == 0 ||
        input_columns > UINT32_MAX || output_columns == 0 || output_columns > UINT32_MAX ||
        input_columns % q4_block_elements != 0) {
        return 0;
    }
    int layer_index = -1;
    std::string prefix;
    std::string role;
    if (!parse_tensor_name(name, layer_index, prefix, role)) {
        return 0;
    }
    island_context & context = get_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    qpu_llama_status status = initialize(context);
    auto & item = context.layers[prefix];
    if (item == nullptr) {
        item = std::make_unique<layer_record>();
        item->layer = layer_index;
        item->prefix = prefix;
    }
    layer_record & layer = *item;
    const uint32_t input = static_cast<uint32_t>(input_columns);
    const uint32_t output = static_cast<uint32_t>(output_columns);
    if (role == "gate" || role == "up") {
        const uint32_t qpu_columns = configured_columns(output);
        if (qpu_columns == 0) {
            return 0;
        }
        if ((layer.intermediate_columns != 0 && layer.intermediate_columns != output) ||
            (layer.input_columns != 0 && layer.input_columns != input) ||
            (layer.qpu_columns != 0 && layer.qpu_columns != qpu_columns)) {
            return 0;
        }
        layer.input_columns = input;
        layer.intermediate_columns = output;
        layer.qpu_columns = qpu_columns;
        layer.cpu_columns = output - qpu_columns;
        packed_matrix & target = role == "gate" ? layer.gate : layer.up;
        if (target.ready()) {
            return 1;
        }
        if (status == QPU_LLAMA_OK) {
            status = pack_matrix(context, weights, weight_size, input, output,
                0, input / q4_block_elements, layer.cpu_columns, qpu_columns, target);
        }
        if (status == QPU_LLAMA_OK) {
            emit_weight(layer, role.c_str(), target);
        }
    } else {
        const uint32_t qpu_columns = configured_columns(input);
        if (qpu_columns == 0) {
            return 0;
        }
        if ((layer.intermediate_columns != 0 && layer.intermediate_columns != input) ||
            (layer.qpu_columns != 0 && layer.qpu_columns != qpu_columns) ||
            (layer.hidden_columns != 0 && layer.hidden_columns != output)) {
            return 0;
        }
        layer.intermediate_columns = input;
        layer.qpu_columns = qpu_columns;
        layer.cpu_columns = input - qpu_columns;
        layer.hidden_columns = output;
        if (layer.down.ready()) {
            return 1;
        }
        if (status == QPU_LLAMA_OK) {
            status = pack_matrix(context, weights, weight_size, input, output,
                layer.cpu_columns / q4_block_elements,
                qpu_columns / q4_block_elements, 0, output, layer.down);
        }
        if (status == QPU_LLAMA_OK) {
            emit_weight(layer, role.c_str(), layer.down);
        }
    }
    if (status == QPU_LLAMA_OK) {
        status = ensure_scratch(context);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: FFN-island registration failed for %s: %s (%s)\n",
            name, qpu_llama_status_string(status),
            context.runtime != nullptr ? qpu_llama_context_last_error(context.runtime) : "no context");
        return 0;
    }
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_ffn_island_begin(
    const char * projection_name,
    const void * activation_q8_0,
    size_t activation_size,
    uint64_t rows,
    uint64_t input_columns,
    uint64_t intermediate_columns,
    uint64_t activation_interleave) {
    if (!flag_enabled("GGML_QPU_FFN_ISLAND") || projection_name == nullptr ||
        activation_q8_0 == nullptr || rows == 0 || rows > UINT32_MAX ||
        input_columns > UINT32_MAX || intermediate_columns > UINT32_MAX ||
        activation_interleave > UINT32_MAX) {
        return 0;
    }
    const uint32_t minimum_rows = unsigned_setting(
        "GGML_QPU_FFN_ISLAND_MIN_ROWS", 64U, 1U);
    const uint32_t maximum_rows = unsigned_setting(
        "GGML_QPU_FFN_ISLAND_MAX_ROWS", 528U, row_tile);
    if (minimum_rows == 0 || maximum_rows == 0 || rows < minimum_rows || rows > maximum_rows) {
        return 0;
    }
    int layer_index = -1;
    std::string prefix;
    std::string role;
    if (!parse_tensor_name(projection_name, layer_index, prefix, role) ||
        (role != "gate" && role != "up")) {
        return 0;
    }
    island_context & context = get_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.pending.load(std::memory_order_acquire)) {
        return 0;
    }
    const auto found = context.layers.find(prefix);
    if (found == context.layers.end() || !found->second->ready()) {
        return 0;
    }
    layer_record & layer = *found->second;
    if (layer.input_columns != input_columns ||
        layer.intermediate_columns != intermediate_columns) {
        return 0;
    }
    context.active_layer = &layer;
    context.active_rows = static_cast<uint32_t>(rows);
    context.padded_rows = (context.active_rows + row_tile - 1U) & ~(row_tile - 1U);
    context.activation_interleave = static_cast<uint32_t>(activation_interleave);
    context.timing = {};
    context.timing.begin_start = monotonic_ns();
    context.done = false;
    context.success = false;
    context.fallback = false;
    context.failure_stage.clear();
    context.failure_detail.clear();
    context.seen_roles.store(role == "gate" ? role_gate_seen : role_up_seen,
        std::memory_order_release);
    const qpu_llama_status status = pack_activation(context, activation_q8_0,
        activation_size, context.active_rows, layer.input_columns,
        context.activation_interleave);
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: FFN-island input preparation failed for %s: %s (%s)\n",
            projection_name, qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        context.active_layer = nullptr;
        return 0;
    }
    context.timing.begin_end = monotonic_ns();
    context.active_cpu_columns.store(layer.cpu_columns, std::memory_order_release);
    context.pending_layer.store(layer.layer, std::memory_order_release);
    context.pending.store(true, std::memory_order_release);
    context.work_ready = true;
    context.work_condition.notify_one();
    return 1;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_ffn_island_cpu_output_columns_for(const char * name) {
    island_context & context = get_context();
    int layer = -1;
    std::string prefix;
    std::string role;
    if (!context.pending.load(std::memory_order_acquire) ||
        !parse_tensor_name(name, layer, prefix, role) ||
        layer != context.pending_layer.load(std::memory_order_acquire) ||
        context.active_layer == nullptr || prefix != context.active_layer->prefix ||
        (role != "gate" && role != "up")) {
        return 0;
    }
    context.seen_roles.fetch_or(role == "gate" ? role_gate_seen : role_up_seen,
        std::memory_order_relaxed);
    return context.active_cpu_columns.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_ffn_island_cpu_intermediate_columns() {
    island_context & context = get_context();
    if (!context.pending.load(std::memory_order_acquire)) {
        return 0;
    }
    context.seen_roles.fetch_or(role_geglu_seen, std::memory_order_relaxed);
    return context.active_cpu_columns.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_ffn_island_down_reduction_columns_for(const char * name) {
    island_context & context = get_context();
    if (!matches_active(context, name, "down")) {
        return 0;
    }
    context.seen_roles.fetch_or(role_down_seen, std::memory_order_relaxed);
    return context.active_cpu_columns.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_ffn_island_wait(const char * down_name) {
    island_context & context = get_context();
    if (!matches_active(context, down_name, "down")) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(context.mutex);
    const uint64_t start = monotonic_ns();
    context.timing.overlap_before_wait_ns = start - context.timing.begin_end;
    context.done_condition.wait(lock, [&] { return context.done; });
    context.timing.exposed_wait_ns = monotonic_ns() - start;
    context.timing.join_start = monotonic_ns();
    return context.success ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_ffn_island_join(
    const char * down_name,
    float * destination,
    size_t destination_size,
    uint64_t rows,
    uint64_t hidden_columns,
    uint64_t thread_index,
    uint64_t thread_count) {
    island_context & context = get_context();
    if (!matches_active(context, down_name, "down") || destination == nullptr ||
        thread_count == 0 || thread_index >= thread_count || context.active_layer == nullptr ||
        rows != context.active_rows || hidden_columns != context.active_layer->hidden_columns ||
        destination_size < rows * hidden_columns * sizeof(float)) {
        return 0;
    }
    const uint64_t row_start = rows * thread_index / thread_count;
    const uint64_t row_end = rows * (thread_index + 1U) / thread_count;
    const float * partial = context.scratch.host_output.data();
    for (uint64_t row = row_start; row < row_end; ++row) {
        const size_t base = static_cast<size_t>(row) * hidden_columns;
        for (uint64_t column = 0; column < hidden_columns; ++column) {
            destination[base + column] += partial[base + column];
        }
    }
    return 1;
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_ffn_island_complete(const char * down_name) {
    island_context & context = get_context();
    if (!matches_active(context, down_name, "down")) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(context.mutex);
    context.timing.join_ns = monotonic_ns() - context.timing.join_start;
    const layer_record & layer = *context.active_layer;
    context.dispatch_count.fetch_add(context.timing.completed_dispatches,
        std::memory_order_relaxed);
    context.island_count.fetch_add(1, std::memory_order_relaxed);
    if (telemetry_enabled()) {
        const uint32_t roles = context.seen_roles.load(std::memory_order_acquire);
        std::fprintf(stderr,
            "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
            "\"operation\":\"ffn_island\",\"strategy\":\"complete_channel_partition\","
            "\"integration_boundary\":\"cpu-repack-first-projection-through-down-join\","
            "\"layer\":%d,\"m\":%u,\"padded_m\":%u,\"k\":%u,"
            "\"intermediate_columns\":%u,\"hidden_columns\":%u,"
            "\"partition\":{\"axis\":\"intermediate_columns\",\"fraction\":%.8g,"
            "\"cpu_columns\":%u,\"qpu_columns\":%u},"
            "\"programs\":[{\"name\":\"ggml-q4-0-q8-wordscale-mx\","
            "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\"},"
            "{\"name\":\"ggml-geglu-q8-0-split\",\"source_hash\":\"%s\","
            "\"binary_sha256\":\"%s\"}],"
            "\"cpu_path\":{\"up_restricted\":%s,\"gate_restricted\":%s,"
            "\"geglu_restricted\":%s,\"down_restricted\":%s},"
            "\"timing_ns\":{\"input_access\":%llu,\"input_pack_and_copy\":%llu,"
            "\"input_sync\":%llu,\"gate\":%llu,\"up\":%llu,\"geglu\":%llu,"
            "\"down\":%llu,\"output_copy\":%llu,\"qpu_complete\":%llu,"
            "\"verification\":%llu,"
            "\"cpu_overlap_before_wait\":%llu,\"exposed_wait\":%llu,"
            "\"fallback\":%llu,\"join\":%llu},"
            "\"verification\":{\"enabled\":%s,\"max_absolute_error\":%.9g,"
            "\"mean_absolute_error\":%.9g},"
            "\"resident_weight_bytes\":%zu,\"dispatch_count\":%u,"
            "\"fallback\":%s,\"failure_stage\":\"%s\",\"failure_detail\":\"%s\"}\n",
            layer.layer, context.active_rows, context.padded_rows, layer.input_columns,
            layer.intermediate_columns, layer.hidden_columns,
            static_cast<double>(layer.qpu_columns) / layer.intermediate_columns,
            layer.cpu_columns, layer.qpu_columns,
            qpu_ggml_q4_0_q8_wordscale_mx_source_hash,
            qpu_ggml_q4_0_q8_wordscale_mx_binary_hash,
            qpu_ggml_geglu_q8_0_split_source_hash,
            qpu_ggml_geglu_q8_0_split_binary_hash,
            (roles & role_up_seen) != 0 ? "true" : "false",
            (roles & role_gate_seen) != 0 ? "true" : "false",
            (roles & role_geglu_seen) != 0 ? "true" : "false",
            (roles & role_down_seen) != 0 ? "true" : "false",
            static_cast<unsigned long long>(context.timing.input_access_ns),
            static_cast<unsigned long long>(context.timing.input_pack_ns),
            static_cast<unsigned long long>(context.timing.input_sync_ns),
            static_cast<unsigned long long>(context.timing.gate_ns),
            static_cast<unsigned long long>(context.timing.up_ns),
            static_cast<unsigned long long>(context.timing.geglu_ns),
            static_cast<unsigned long long>(context.timing.down_ns),
            static_cast<unsigned long long>(context.timing.output_copy_ns),
            static_cast<unsigned long long>(context.timing.qpu_complete_ns),
            static_cast<unsigned long long>(context.timing.verification_ns),
            static_cast<unsigned long long>(context.timing.overlap_before_wait_ns),
            static_cast<unsigned long long>(context.timing.exposed_wait_ns),
            static_cast<unsigned long long>(context.timing.fallback_ns),
            static_cast<unsigned long long>(context.timing.join_ns),
            context.timing.verified ? "true" : "false",
            context.timing.verification_max_abs,
            context.timing.verification_mean_abs,
            layer.resident_bytes(), context.timing.completed_dispatches,
            context.fallback ? "true" : "false", context.failure_stage.c_str(),
            context.failure_detail.c_str());
    }
    context.pending.store(false, std::memory_order_release);
    context.pending_layer.store(-1, std::memory_order_release);
    context.active_cpu_columns.store(0, std::memory_order_release);
    context.active_layer = nullptr;
    return 1;
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_ffn_island_dispatch_count() {
    return get_context().dispatch_count.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_ffn_island_count() {
    return get_context().island_count.load(std::memory_order_acquire);
}
