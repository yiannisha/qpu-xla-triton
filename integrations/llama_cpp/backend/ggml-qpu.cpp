#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "qpu_llama_runtime.h"
#include "qpu_llama_q4_0.h"

#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-m4.h"
#include "ggml-q4-0-q8-0-mx.h"
#include "ggml-geglu-split-fp32.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/sysinfo.h>

namespace {

constexpr size_t QPU_BUFFER_ALIGNMENT = 64;

struct qpu_device_context;

struct qpu_buffer_context {
    qpu_llama_buffer * buffer = nullptr;
};

struct qpu_linear_owner {
    qpu_llama_q4_0_linear * linear = nullptr;
    uint32_t row_capacity = 0;

    qpu_linear_owner() = default;
    qpu_linear_owner(qpu_llama_q4_0_linear * value, uint32_t capacity) :
        linear(value), row_capacity(capacity) {}
    qpu_linear_owner(const qpu_linear_owner &) = delete;
    qpu_linear_owner & operator=(const qpu_linear_owner &) = delete;
    qpu_linear_owner(qpu_linear_owner && other) noexcept :
        linear(other.linear), row_capacity(other.row_capacity) {
        other.linear = nullptr;
        other.row_capacity = 0;
    }
    qpu_linear_owner & operator=(qpu_linear_owner && other) noexcept {
        if (this != &other) {
            qpu_llama_q4_0_linear_destroy(linear);
            linear = other.linear;
            row_capacity = other.row_capacity;
            other.linear = nullptr;
            other.row_capacity = 0;
        }
        return *this;
    }
    ~qpu_linear_owner() {
        qpu_llama_q4_0_linear_destroy(linear);
    }
};

struct qpu_backend_context {
    qpu_device_context * device = nullptr;
    std::mutex mutex;
    std::unordered_map<const ggml_tensor *, qpu_linear_owner> linears;
    qpu_llama_program * geglu_program = nullptr;
    qpu_llama_buffer * geglu_table = nullptr;
    uint64_t dispatch_count = 0;
    uint64_t weight_cache_hits = 0;
    uint64_t weight_cache_misses = 0;
};

struct qpu_device_context {
    std::string name = "QPU0";
    std::string description = "VideoCore VII QPU via DRM/V3D";
    qpu_llama_context * runtime = nullptr;
    ggml_backend_buffer_type buffer_type = {};
    uint32_t minimum_rows = 16;
    uint64_t minimum_geglu_elements = 768;
    bool enable_q4_0 = false;
    bool enable_geglu = false;

    ~qpu_device_context() {
        qpu_llama_context_destroy(runtime);
    }
};

struct qpu_registry_context {
    std::unique_ptr<qpu_device_context> device_context;
    std::unique_ptr<ggml_backend_device> device;
};

static ggml_guid_t qpu_backend_guid() {
    static ggml_guid guid = {
        0x71, 0x70, 0x75, 0x2d, 0x76, 0x63, 0x37, 0x2d,
        0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x30, 0x30, 0x31,
    };
    return &guid;
}

static const char * qpu_buffer_type_name(ggml_backend_buffer_type_t) {
    return "QPU0_Host";
}

static void qpu_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<qpu_buffer_context *>(buffer->context);
    qpu_llama_buffer_destroy(context->buffer);
    delete context;
}

static void * qpu_buffer_base(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<qpu_buffer_context *>(buffer->context);
    return qpu_llama_buffer_data(context->buffer);
}

static enum ggml_status qpu_buffer_init_tensor(
    ggml_backend_buffer_t,
    ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void qpu_buffer_memset_tensor(
    ggml_backend_buffer_t,
    ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size) {
    std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
}

static void qpu_buffer_set_tensor(
    ggml_backend_buffer_t,
    ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
}

static void qpu_buffer_get_tensor(
    ggml_backend_buffer_t,
    const ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

static bool qpu_buffer_copy_tensor(
    ggml_backend_buffer_t,
    const ggml_tensor * source,
    ggml_tensor * destination) {
    if (!ggml_backend_buffer_is_host(source->buffer)) {
        return false;
    }
    std::memcpy(destination->data, source->data, ggml_nbytes(source));
    return true;
}

static void qpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<qpu_buffer_context *>(buffer->context);
    std::memset(qpu_llama_buffer_data(context->buffer), value,
        qpu_llama_buffer_size(context->buffer));
}

static const ggml_backend_buffer_i qpu_buffer_interface = {
    /* .free_buffer   = */ qpu_buffer_free,
    /* .get_base      = */ qpu_buffer_base,
    /* .init_tensor   = */ qpu_buffer_init_tensor,
    /* .memset_tensor = */ qpu_buffer_memset_tensor,
    /* .set_tensor    = */ qpu_buffer_set_tensor,
    /* .get_tensor    = */ qpu_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ qpu_buffer_copy_tensor,
    /* .clear         = */ qpu_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t qpu_buffer_alloc(
    ggml_backend_buffer_type_t buffer_type,
    size_t size) {
    auto * device = static_cast<qpu_device_context *>(buffer_type->context);
    auto context = std::make_unique<qpu_buffer_context>();
    const size_t physical_size = std::max<size_t>(size, 1);
    const qpu_llama_status status = qpu_llama_buffer_create(
        device->runtime, physical_size, &context->buffer);
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: buffer allocation of %zu bytes failed: %s (%s)\n",
            size, qpu_llama_status_string(status),
            qpu_llama_context_last_error(device->runtime));
        return nullptr;
    }
    return ggml_backend_buffer_init(
        buffer_type, qpu_buffer_interface, context.release(), size);
}

static size_t qpu_buffer_alignment(ggml_backend_buffer_type_t) {
    return QPU_BUFFER_ALIGNMENT;
}

static size_t qpu_buffer_max_size(ggml_backend_buffer_type_t) {
    return UINT32_MAX;
}

static bool qpu_buffer_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const ggml_backend_buffer_type_i qpu_buffer_type_interface = {
    /* .get_name       = */ qpu_buffer_type_name,
    /* .alloc_buffer   = */ qpu_buffer_alloc,
    /* .get_alignment  = */ qpu_buffer_alignment,
    /* .get_max_size   = */ qpu_buffer_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ qpu_buffer_is_host,
};

static const char * qpu_backend_name(ggml_backend_t) {
    return "QPU";
}

static void qpu_backend_free(ggml_backend_t backend) {
    auto * context = static_cast<qpu_backend_context *>(backend->context);
    qpu_llama_program_destroy(context->geglu_program);
    qpu_llama_buffer_destroy(context->geglu_table);
    delete context;
    delete backend;
}

static uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool is_cpu_repack(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->buffer != nullptr &&
        std::strcmp(ggml_backend_buffer_name(tensor->buffer), "CPU_REPACK") == 0;
}

static bool unpack_q4_0_4x4(
    const ggml_tensor * tensor,
    std::vector<uint8_t> & destination) {
    constexpr size_t block_bytes = 18;
    constexpr size_t packed_group_bytes = 4 * block_bytes;
    const int64_t input_columns = tensor->ne[0];
    const int64_t output_columns = tensor->ne[1];
    if (input_columns <= 0 || input_columns % 32 != 0 ||
        output_columns <= 0 || output_columns % 4 != 0 || tensor->data == nullptr) {
        return false;
    }
    const size_t blocks = static_cast<size_t>(input_columns / 32);
    destination.resize(static_cast<size_t>(output_columns) * blocks * block_bytes);
    const auto * source = static_cast<const uint8_t *>(tensor->data);
    for (size_t group = 0; group < static_cast<size_t>(output_columns) / 4; ++group) {
        for (size_t block = 0; block < blocks; ++block) {
            const uint8_t * packed = source + (group * blocks + block) * packed_group_bytes;
            const uint8_t * packed_quants = packed + 8;
            for (size_t row = 0; row < 4; ++row) {
                uint8_t * output = destination.data() +
                    ((group * 4 + row) * blocks + block) * block_bytes;
                std::memcpy(output, packed + row * 2, 2);
                for (size_t chunk = 0; chunk < 4; ++chunk) {
                    const uint8_t * packed_chunk = packed_quants +
                        (chunk * 4 + row) * 4;
                    for (size_t byte = 0; byte < 4; ++byte) {
                        output[2 + chunk * 4 + byte] =
                            packed_chunk[byte] ^ UINT8_C(0x88);
                    }
                }
            }
        }
    }
    return true;
}

static bool canonical_q4_0(
    const ggml_tensor * tensor,
    std::vector<uint8_t> & destination) {
    if (is_cpu_repack(tensor)) {
        return unpack_q4_0_4x4(tensor, destination);
    }
    if (tensor == nullptr || tensor->data == nullptr || tensor->type != GGML_TYPE_Q4_0) {
        return false;
    }
    const size_t size = ggml_nbytes(tensor);
    destination.resize(size);
    std::memcpy(destination.data(), tensor->data, size);
    return true;
}

static const char * source_hash_for_rows(uint32_t rows) {
    return rows == 1 ? qpu_ggml_q4_0_q8_0_m1_source_hash :
        rows == 4 ? qpu_ggml_q4_0_q8_0_m4_source_hash :
        qpu_ggml_q4_0_q8_0_mx_source_hash;
}

static const char * binary_hash_for_rows(uint32_t rows) {
    return rows == 1 ? qpu_ggml_q4_0_q8_0_m1_binary_hash :
        rows == 4 ? qpu_ggml_q4_0_q8_0_m4_binary_hash :
        qpu_ggml_q4_0_q8_0_mx_binary_hash;
}

static const char * program_for_rows(uint32_t rows) {
    return rows == 1 ? "ggml-q4-0-q8-0-m1" :
        rows == 4 ? "ggml-q4-0-q8-0-m4" :
        "ggml-q4-0-q8-0-mx";
}

static qpu_llama_status get_or_prepare_linear(
    qpu_backend_context * context,
    const ggml_tensor * weight,
    uint32_t rows,
    qpu_llama_q4_0_linear ** result) {
    auto existing = context->linears.find(weight);
    if (existing != context->linears.end() && existing->second.row_capacity >= rows) {
        ++context->weight_cache_hits;
        *result = existing->second.linear;
        return QPU_LLAMA_OK;
    }
    if (existing != context->linears.end()) {
        context->linears.erase(existing);
    }

    std::vector<uint8_t> canonical;
    if (!canonical_q4_0(weight, canonical)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_q4_0_linear * prepared = nullptr;
    const qpu_llama_q4_0_linear_desc description = {
        /* .weights               = */ canonical.data(),
        /* .weight_size           = */ canonical.size(),
        /* .input_columns         = */ static_cast<uint32_t>(weight->ne[0]),
        /* .output_columns        = */ static_cast<uint32_t>(weight->ne[1]),
        /* .rows                  = */ rows,
        /* .resident_column_start = */ 0,
        /* .resident_column_count = */ static_cast<uint32_t>(weight->ne[1]),
        /* .weight_mode            = */ QPU_LLAMA_Q4_0_WEIGHT_EXACT,
        /* .workgroups_per_supergroup = */ 24U,
        /* .expected_source_hash  = */ source_hash_for_rows(rows),
    };
    const qpu_llama_status status = qpu_llama_q4_0_linear_prepare(
        context->device->runtime, &description, &prepared);
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    ++context->weight_cache_misses;
    auto inserted = context->linears.emplace(weight, qpu_linear_owner(prepared, rows));
    *result = inserted.first->second.linear;
    return QPU_LLAMA_OK;
}

static bool supports_q4_0_mul_mat(const ggml_tensor * operation) {
    if (operation == nullptr || operation->op != GGML_OP_MUL_MAT ||
        operation->src[0] == nullptr || operation->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * weight = operation->src[0];
    const ggml_tensor * activation = operation->src[1];
    return weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32 &&
        weight->ne[0] > 0 && weight->ne[0] % 32 == 0 &&
        weight->ne[1] > 0 && weight->ne[1] % 16 == 0 &&
        activation->ne[0] == weight->ne[0] && activation->ne[1] > 0 &&
        activation->ne[1] <= 4 * UINT16_MAX &&
        weight->ne[2] == 1 && weight->ne[3] == 1 &&
        activation->ne[2] == 1 && activation->ne[3] == 1 &&
        ggml_is_contiguous(activation) && ggml_is_contiguous(operation);
}

static bool supports_geglu_split(const ggml_tensor * operation) {
    if (operation == nullptr || operation->op != GGML_OP_GLU ||
        ggml_get_glu_op(operation) != GGML_GLU_OP_GEGLU ||
        operation->src[0] == nullptr || operation->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * gate = operation->src[0];
    const ggml_tensor * up = operation->src[1];
    const int64_t elements = ggml_nelements(operation);
    return gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
        operation->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, operation) &&
        elements > 0 && elements % 768 == 0 &&
        ggml_is_contiguous(gate) && ggml_is_contiguous(up) &&
        ggml_is_contiguous(operation);
}

static bool qpu_tensor_address(
    const ggml_tensor * tensor,
    qpu_llama_buffer ** buffer,
    uint32_t * address) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        std::strcmp(ggml_backend_buffer_name(tensor->buffer), "QPU0_Host") != 0) {
        return false;
    }
    auto * buffer_context = static_cast<qpu_buffer_context *>(tensor->buffer->context);
    const auto * base = static_cast<const uint8_t *>(
        qpu_llama_buffer_data(buffer_context->buffer));
    const auto * data = static_cast<const uint8_t *>(tensor->data);
    if (data < base) {
        return false;
    }
    const size_t offset = static_cast<size_t>(data - base);
    if (offset > qpu_llama_buffer_size(buffer_context->buffer) ||
        ggml_nbytes(tensor) > qpu_llama_buffer_size(buffer_context->buffer) - offset ||
        qpu_llama_buffer_gpu_address(buffer_context->buffer, offset, address) != QPU_LLAMA_OK) {
        return false;
    }
    *buffer = buffer_context->buffer;
    return true;
}

static qpu_llama_status get_geglu_program(qpu_backend_context * context) {
    if (context->geglu_program != nullptr) {
        return QPU_LLAMA_OK;
    }
    qpu_llama_status status = qpu_llama_buffer_create(context->device->runtime,
        UINT32_C(1) << 17U, &context->geglu_table);
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    auto * table = static_cast<ggml_fp16_t *>(
        qpu_llama_buffer_data(context->geglu_table));
    for (uint32_t index = 0; index < (UINT32_C(1) << 16U); ++index) {
        const float input = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(index));
        const float gelu = 0.5F * input * (1.0F + std::tanh(
            0.7978845608028654F * input *
            (1.0F + 0.044715F * input * input)));
        table[index] = ggml_fp32_to_fp16(gelu);
    }
    const qpu_llama_program_desc description = {
        /* .code                 = */ qpu_ggml_geglu_split_fp32,
        /* .code_size            = */ sizeof(qpu_ggml_geglu_split_fp32),
        /* .compiled_source_hash = */ qpu_ggml_geglu_split_fp32_source_hash,
        /* .expected_source_hash = */ qpu_ggml_geglu_split_fp32_source_hash,
        /* .binary_sha256        = */ qpu_ggml_geglu_split_fp32_binary_hash,
        /* .uniform_word_count   = */ 5,
    };
    status = qpu_llama_program_create(context->device->runtime,
        &description, &context->geglu_program);
    if (status != QPU_LLAMA_OK) {
        qpu_llama_buffer_destroy(context->geglu_table);
        context->geglu_table = nullptr;
    }
    return status;
}

static enum ggml_status compute_geglu_split(
    qpu_backend_context * context,
    ggml_tensor * operation) {
    qpu_llama_buffer * gate_buffer = nullptr;
    qpu_llama_buffer * up_buffer = nullptr;
    qpu_llama_buffer * destination_buffer = nullptr;
    uint32_t gate_address = 0;
    uint32_t up_address = 0;
    uint32_t destination_address = 0;
    if (!qpu_tensor_address(operation->src[0], &gate_buffer, &gate_address) ||
        !qpu_tensor_address(operation->src[1], &up_buffer, &up_address) ||
        !qpu_tensor_address(operation, &destination_buffer, &destination_address)) {
        std::fprintf(stderr, "QPU: GEGLU tensors are not in QPU host buffers\n");
        return GGML_STATUS_FAILED;
    }
    qpu_llama_status status = get_geglu_program(context);
    const uint64_t start = monotonic_ns();
    if (status == QPU_LLAMA_OK) {
        const uint64_t elements = static_cast<uint64_t>(ggml_nelements(operation));
        uint32_t table_address = 0;
        status = qpu_llama_buffer_gpu_address(
            context->geglu_table, 0, &table_address);
        const uint32_t uniforms[] = {
            static_cast<uint32_t>(elements / (12U * 64U)),
            gate_address,
            up_address,
            destination_address,
            table_address,
        };
        qpu_llama_buffer * buffers[] = {
            gate_buffer, up_buffer, destination_buffer, context->geglu_table,
        };
        const qpu_llama_dispatch_desc dispatch = {
            /* .uniforms          = */ uniforms,
            /* .uniform_word_count= */ 5,
            /* .buffers           = */ buffers,
            /* .buffer_count      = */ 4,
            /* .local_invocation  = */ {16, 1, 1},
            /* .workgroup         = */ {12, 1, 1},
            /* .wgs_per_sg        = */ 24,
            /* .thread_count      = */ 12,
            /* .propagate_nan     = */ 0,
            /* .single_segment    = */ 0,
            /* .threading         = */ 0,
        };
        if (status == QPU_LLAMA_OK) {
            status = qpu_llama_program_execute(context->geglu_program, &dispatch);
        }
    }
    const uint64_t end = monotonic_ns();
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: GEGLU M=%lld N=%lld failed: %s (%s)\n",
            static_cast<long long>(operation->ne[1]),
            static_cast<long long>(operation->ne[0]),
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context->device->runtime));
        return GGML_STATUS_FAILED;
    }
    ++context->dispatch_count;
    std::fprintf(stderr,
        "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
        "\"operation\":\"geglu_split_fp32\","
        "\"program\":\"ggml-geglu-split-fp32\","
        "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\","
        "\"exact_shape\":{\"layout\":\"contiguous-f32-split-geglu\","
        "\"element_multiple\":768},"
        "\"placement\":\"hybrid\","
        "\"partition\":{\"axis\":\"operators\","
        "\"qpu\":[\"geglu_split_fp32\"],\"cpu\":\"all_other_ops\"},"
        "\"tensor\":\"%s\","
        "\"m\":%lld,\"n\":%lld,\"elements\":%lld,\"dispatch_count\":1,"
        "\"submit_wait_ns\":%llu,\"complete_ns\":%llu}\n",
        qpu_ggml_geglu_split_fp32_source_hash,
        qpu_ggml_geglu_split_fp32_binary_hash,
        operation->name,
        static_cast<long long>(operation->ne[1]),
        static_cast<long long>(operation->ne[0]),
        static_cast<long long>(ggml_nelements(operation)),
        static_cast<unsigned long long>(end - start),
        static_cast<unsigned long long>(end - start));
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status compute_q4_0_mul_mat(
    qpu_backend_context * context,
    ggml_tensor * operation) {
    const ggml_tensor * weight = operation->src[0];
    const ggml_tensor * activation = operation->src[1];
    const uint32_t rows = static_cast<uint32_t>(activation->ne[1]);
    const uint32_t input_columns = static_cast<uint32_t>(weight->ne[0]);
    const size_t activation_row_bytes =
        static_cast<size_t>(input_columns / 32) * 34;
    std::vector<uint8_t> quantized(static_cast<size_t>(rows) * activation_row_bytes);

    const uint64_t complete_start = monotonic_ns();
    const uint64_t quantize_start = complete_start;
    for (uint32_t row = 0; row < rows; ++row) {
        const auto * input = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(activation->data) +
            static_cast<size_t>(row) * activation->nb[1]);
        quantize_row_q8_0(input,
            quantized.data() + static_cast<size_t>(row) * activation_row_bytes,
            input_columns);
    }
    const uint64_t quantize_end = monotonic_ns();

    qpu_llama_q4_0_linear * linear = nullptr;
    qpu_llama_status status = get_or_prepare_linear(context, weight, rows, &linear);
    qpu_llama_q4_0_timing timing = {};
    if (status == QPU_LLAMA_OK) {
        const qpu_llama_q4_0_execution execution = {
            /* .activation         = */ quantized.data(),
            /* .activation_size    = */ quantized.size(),
            /* .activation_offset  = */ 0,
            /* .destination        = */ operation->data,
            /* .destination_size   = */ ggml_nbytes(operation),
            /* .destination_offset = */ 0,
            /* .column_start       = */ 0,
            /* .column_count       = */ static_cast<uint32_t>(weight->ne[1]),
            /* .rows               = */ rows,
            /* .activation_interleave = */ 0,
        };
        status = qpu_llama_q4_0_linear_execute(linear, &execution, &timing);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: %s M=%u K=%u N=%lld failed: %s (%s)\n",
            weight->name, rows, input_columns, static_cast<long long>(weight->ne[1]),
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context->device->runtime));
        return GGML_STATUS_FAILED;
    }
    ++context->dispatch_count;
    const uint64_t complete_end = monotonic_ns();
    std::fprintf(stderr,
        "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
        "\"operation\":\"mul_mat_q4_0_q8_0\",\"program\":\"%s\","
        "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\","
        "\"exact_shape\":{\"operation\":\"q4_0-by-q8_0\","
        "\"k_multiple\":32,\"n_multiple\":16},"
        "\"placement\":\"hybrid\","
        "\"partition\":{\"axis\":\"operators\","
        "\"qpu\":[\"mul_mat_q4_0_q8_0\"],\"cpu\":\"all_other_ops\"},"
        "\"tensor\":\"%s\","
        "\"m\":%u,\"k\":%u,\"n\":%lld,\"dispatch_count\":1,"
        "\"quantize_ns\":%llu,\"input_copy_ns\":%llu,"
        "\"submit_wait_ns\":%llu,\"output_copy_ns\":%llu,"
        "\"complete_ns\":%llu,\"backend_complete_ns\":%llu,"
        "\"weight_cache_hits\":%llu,\"weight_cache_misses\":%llu}\n",
        program_for_rows(rows),
        source_hash_for_rows(rows),
        binary_hash_for_rows(rows),
        weight->name,
        rows,
        input_columns,
        static_cast<long long>(weight->ne[1]),
        static_cast<unsigned long long>(quantize_end - quantize_start),
        static_cast<unsigned long long>(timing.input_copy_ns),
        static_cast<unsigned long long>(timing.submit_wait_ns),
        static_cast<unsigned long long>(timing.output_copy_ns),
        static_cast<unsigned long long>(timing.complete_ns),
        static_cast<unsigned long long>(complete_end - complete_start),
        static_cast<unsigned long long>(context->weight_cache_hits),
        static_cast<unsigned long long>(context->weight_cache_misses));
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status qpu_backend_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * graph) {
    auto * context = static_cast<qpu_backend_context *>(backend->context);
    std::lock_guard<std::mutex> lock(context->mutex);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        ggml_tensor * operation = ggml_graph_node(graph, index);
        if (operation->view_src != nullptr || operation->op == GGML_OP_NONE) {
            continue;
        }
        const bool q4_0_enabled = context->device->enable_q4_0 &&
            supports_q4_0_mul_mat(operation) &&
            operation->src[1]->ne[1] >= context->device->minimum_rows;
        const bool geglu_enabled = context->device->enable_geglu &&
            supports_geglu_split(operation) &&
            static_cast<uint64_t>(ggml_nelements(operation)) >=
                context->device->minimum_geglu_elements;
        if (!q4_0_enabled && !geglu_enabled) {
            std::fprintf(stderr, "QPU: unsupported operation reached graph compute: %s\n",
                ggml_op_desc(operation));
            return GGML_STATUS_FAILED;
        }
        const enum ggml_status status = geglu_enabled
            ? compute_geglu_split(context, operation)
            : compute_q4_0_mul_mat(context, operation);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i qpu_backend_interface = {
    /* .get_name           = */ qpu_backend_name,
    /* .free               = */ qpu_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ qpu_backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * qpu_device_name(ggml_backend_dev_t device) {
    return static_cast<qpu_device_context *>(device->context)->name.c_str();
}

static const char * qpu_device_description(ggml_backend_dev_t device) {
    return static_cast<qpu_device_context *>(device->context)->description.c_str();
}

static void qpu_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    struct sysinfo info = {};
    if (sysinfo(&info) != 0) {
        *free = 0;
        *total = 0;
        return;
    }
    *free = static_cast<size_t>(info.freeram) * info.mem_unit;
    *total = static_cast<size_t>(info.totalram) * info.mem_unit;
}

static enum ggml_backend_dev_type qpu_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static void qpu_device_properties(
    ggml_backend_dev_t device,
    ggml_backend_dev_props * properties) {
    properties->name = qpu_device_name(device);
    properties->description = qpu_device_description(device);
    qpu_device_memory(device, &properties->memory_free, &properties->memory_total);
    properties->type = qpu_device_type(device);
    properties->device_id = nullptr;
    properties->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t qpu_device_init_backend(ggml_backend_dev_t device, const char *) {
    auto context = std::make_unique<qpu_backend_context>();
    context->device = static_cast<qpu_device_context *>(device->context);
    return new ggml_backend {
        /* .guid    = */ qpu_backend_guid(),
        /* .iface   = */ qpu_backend_interface,
        /* .device  = */ device,
        /* .context = */ context.release(),
    };
}

static ggml_backend_buffer_type_t qpu_device_buffer_type(ggml_backend_dev_t device) {
    return &static_cast<qpu_device_context *>(device->context)->buffer_type;
}

static bool qpu_device_supports_op(
    ggml_backend_dev_t device,
    const ggml_tensor * operation) {
    auto * context = static_cast<qpu_device_context *>(device->context);
    if (supports_q4_0_mul_mat(operation)) {
        return context->enable_q4_0 &&
            operation->src[1]->ne[1] >= context->minimum_rows;
    }
    return context->enable_geglu && supports_geglu_split(operation) &&
        static_cast<uint64_t>(ggml_nelements(operation)) >=
            context->minimum_geglu_elements;
}

static bool qpu_device_supports_buffer_type(
    ggml_backend_dev_t device,
    ggml_backend_buffer_type_t buffer_type) {
    return buffer_type == qpu_device_buffer_type(device) ||
        ggml_backend_buft_is_host(buffer_type);
}

static bool qpu_device_offload_op(
    ggml_backend_dev_t device,
    const ggml_tensor * operation) {
    return qpu_device_supports_op(device, operation);
}

static const ggml_backend_device_i qpu_device_interface = {
    /* .get_name             = */ qpu_device_name,
    /* .get_description      = */ qpu_device_description,
    /* .get_memory           = */ qpu_device_memory,
    /* .get_type             = */ qpu_device_type,
    /* .get_props            = */ qpu_device_properties,
    /* .init_backend         = */ qpu_device_init_backend,
    /* .get_buffer_type      = */ qpu_device_buffer_type,
    /* .get_host_buffer_type = */ qpu_device_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ qpu_device_supports_op,
    /* .supports_buft        = */ qpu_device_supports_buffer_type,
    /* .offload_op           = */ qpu_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * qpu_registry_name(ggml_backend_reg_t) {
    return "QPU";
}

static size_t qpu_registry_device_count(ggml_backend_reg_t registry) {
    auto * context = static_cast<qpu_registry_context *>(registry->context);
    return context->device != nullptr ? 1 : 0;
}

static ggml_backend_dev_t qpu_registry_device(
    ggml_backend_reg_t registry,
    size_t index) {
    auto * context = static_cast<qpu_registry_context *>(registry->context);
    GGML_ASSERT(index == 0 && context->device != nullptr);
    return context->device.get();
}

static const ggml_backend_reg_i qpu_registry_interface = {
    /* .get_name         = */ qpu_registry_name,
    /* .get_device_count = */ qpu_registry_device_count,
    /* .get_device       = */ qpu_registry_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg_t qpu_backend_registry() {
    static ggml_backend_reg registry = {};
    static qpu_registry_context context;
    static std::once_flag once;
    std::call_once(once, [] {
        auto device_context = std::make_unique<qpu_device_context>();
        const auto enabled = [](const char * name) {
            const char * value = std::getenv(name);
            return value != nullptr &&
                (std::strcmp(value, "1") == 0 ||
                 std::strcmp(value, "true") == 0 ||
                 std::strcmp(value, "on") == 0);
        };
        device_context->enable_q4_0 = enabled("GGML_QPU_ENABLE_Q4_0");
        device_context->enable_geglu = enabled("GGML_QPU_ENABLE_GEGLU");
        if (const char * value = std::getenv("GGML_QPU_MIN_M")) {
            char * end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0' && parsed > 0 && parsed <= 4U * UINT16_MAX) {
                device_context->minimum_rows = static_cast<uint32_t>(parsed);
            }
        }
        if (const char * value = std::getenv("GGML_QPU_MIN_GEGLU_ELEMENTS")) {
            char * end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end != value && *end == '\0' && parsed > 0) {
                device_context->minimum_geglu_elements = parsed;
            }
        }
        const qpu_llama_status status = qpu_llama_context_create(
            nullptr, &device_context->runtime);
        if (status != QPU_LLAMA_OK) {
            std::fprintf(stderr, "QPU: VideoCore VII device unavailable: %s\n",
                qpu_llama_status_string(status));
        } else {
            device_context->buffer_type = ggml_backend_buffer_type {
                /* .iface   = */ qpu_buffer_type_interface,
                /* .device  = */ nullptr,
                /* .context = */ device_context.get(),
            };
            context.device_context = std::move(device_context);
            context.device = std::make_unique<ggml_backend_device>(ggml_backend_device {
                /* .iface   = */ qpu_device_interface,
                /* .reg     = */ &registry,
                /* .context = */ context.device_context.get(),
            });
            context.device_context->buffer_type.device = context.device.get();
            std::fprintf(stderr,
                "QPU: registered VideoCore VII backend (minimum matmul M=%u, "
                "minimum GEGLU elements=%llu, Q4_0=%s, GEGLU=%s)\n",
                context.device_context->minimum_rows,
                static_cast<unsigned long long>(
                    context.device_context->minimum_geglu_elements),
                context.device_context->enable_q4_0 ? "enabled" : "disabled",
                context.device_context->enable_geglu ? "enabled" : "disabled");
        }
        registry = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ qpu_registry_interface,
            /* .context     = */ &context,
        };
    });
    return &registry;
}

struct qpu_inline_geglu_context {
    std::mutex mutex;
    qpu_backend_context backend;
    qpu_llama_buffer * scratch = nullptr;
    size_t scratch_capacity = 0;

    ~qpu_inline_geglu_context() {
        qpu_llama_buffer_destroy(scratch);
        qpu_llama_program_destroy(backend.geglu_program);
        qpu_llama_buffer_destroy(backend.geglu_table);
    }
};

static void cpu_fallback_geglu(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t elements) {
    for (uint64_t index = 0; index < elements; ++index) {
        const float input = ggml_fp16_to_fp32(ggml_fp32_to_fp16(gate[index]));
        const float gelu = 0.5F * input * (1.0F + std::tanh(
            0.7978845608028654F * input *
            (1.0F + 0.044715F * input * input)));
        destination[index] =
            ggml_fp16_to_fp32(ggml_fp32_to_fp16(gelu)) * up[index];
    }
}

static int qpu_inline_geglu(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns) {
    if (gate == nullptr || up == nullptr || destination == nullptr || rows == 0 ||
        columns == 0 || rows > UINT64_MAX / columns) {
        return 0;
    }
    const uint64_t elements = rows * columns;
    if (elements % 768 != 0 || elements > SIZE_MAX / (3 * sizeof(float))) {
        return 0;
    }

    static qpu_inline_geglu_context hook;
    std::lock_guard<std::mutex> lock(hook.mutex);
    auto * registry_context = static_cast<qpu_registry_context *>(
        qpu_backend_registry()->context);
    if (registry_context->device_context == nullptr) {
        cpu_fallback_geglu(gate, up, destination, elements);
        return 1;
    }
    hook.backend.device = registry_context->device_context.get();

    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    const size_t required = 3 * bytes;
    qpu_llama_status status = QPU_LLAMA_OK;
    if (hook.scratch_capacity < required) {
        qpu_llama_buffer_destroy(hook.scratch);
        hook.scratch = nullptr;
        hook.scratch_capacity = 0;
        status = qpu_llama_buffer_create(
            hook.backend.device->runtime, required, &hook.scratch);
        if (status == QPU_LLAMA_OK) {
            hook.scratch_capacity = required;
        }
    }

    const uint64_t complete_start = monotonic_ns();
    uint64_t input_copy_end = complete_start;
    uint64_t submit_end = complete_start;
    uint32_t gate_address = 0;
    uint32_t up_address = 0;
    uint32_t destination_address = 0;
    uint32_t table_address = 0;
    if (status == QPU_LLAMA_OK) {
        auto * scratch = static_cast<uint8_t *>(qpu_llama_buffer_data(hook.scratch));
        std::memcpy(scratch, gate, bytes);
        std::memcpy(scratch + bytes, up, bytes);
        input_copy_end = monotonic_ns();
        status = get_geglu_program(&hook.backend);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            hook.scratch, 0, &gate_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            hook.scratch, bytes, &up_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            hook.scratch, 2 * bytes, &destination_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            hook.backend.geglu_table, 0, &table_address);
    }
    if (status == QPU_LLAMA_OK) {
        const uint32_t uniforms[] = {
            static_cast<uint32_t>(elements / (12U * 64U)),
            gate_address,
            up_address,
            destination_address,
            table_address,
        };
        qpu_llama_buffer * buffers[] = {
            hook.scratch, hook.backend.geglu_table,
        };
        const qpu_llama_dispatch_desc dispatch = {
            /* .uniforms          = */ uniforms,
            /* .uniform_word_count= */ 5,
            /* .buffers           = */ buffers,
            /* .buffer_count      = */ 2,
            /* .local_invocation  = */ {16, 1, 1},
            /* .workgroup         = */ {12, 1, 1},
            /* .wgs_per_sg        = */ 24,
            /* .thread_count      = */ 12,
            /* .propagate_nan     = */ 0,
            /* .single_segment    = */ 0,
            /* .threading         = */ 0,
        };
        status = qpu_llama_program_execute(hook.backend.geglu_program, &dispatch);
        submit_end = monotonic_ns();
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr, "QPU: inline GEGLU failed, using exact CPU fallback: %s (%s)\n",
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(hook.backend.device->runtime));
        cpu_fallback_geglu(gate, up, destination, elements);
        return 1;
    }

    const uint64_t output_copy_start = monotonic_ns();
    const auto * scratch = static_cast<const uint8_t *>(
        qpu_llama_buffer_data(hook.scratch));
    std::memcpy(destination, scratch + 2 * bytes, bytes);
    const uint64_t complete_end = monotonic_ns();
    ++hook.backend.dispatch_count;
    std::fprintf(stderr,
        "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
        "\"operation\":\"geglu_split_fp32\","
        "\"program\":\"ggml-geglu-split-fp32\","
        "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\","
        "\"exact_shape\":{\"layout\":\"contiguous-f32-split-geglu\","
        "\"element_multiple\":768},\"placement\":\"hybrid\","
        "\"partition\":{\"axis\":\"operators\","
        "\"qpu\":[\"geglu_split_fp32\"],\"cpu\":\"all_other_ops\"},"
        "\"integration_boundary\":\"ggml-cpu-inline-hook\","
        "\"m\":%llu,\"n\":%llu,\"elements\":%llu,\"dispatch_count\":1,"
        "\"input_copy_ns\":%llu,\"submit_wait_ns\":%llu,"
        "\"output_copy_ns\":%llu,\"complete_ns\":%llu}\n",
        qpu_ggml_geglu_split_fp32_source_hash,
        qpu_ggml_geglu_split_fp32_binary_hash,
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(columns),
        static_cast<unsigned long long>(elements),
        static_cast<unsigned long long>(input_copy_end - complete_start),
        static_cast<unsigned long long>(submit_end - input_copy_end),
        static_cast<unsigned long long>(complete_end - output_copy_start),
        static_cast<unsigned long long>(complete_end - complete_start));
    return 1;
}

static int qpu_backend_score() {
    return qpu_registry_device_count(qpu_backend_registry()) != 0 ? 1 : 0;
}

} // namespace

GGML_BACKEND_DL_IMPL(qpu_backend_registry)
GGML_BACKEND_DL_SCORE_IMPL(qpu_backend_score)

extern "C" __attribute__((visibility("default"))) int ggml_qpu_geglu_f32(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns) {
    return qpu_inline_geglu(gate, up, destination, rows, columns);
}
