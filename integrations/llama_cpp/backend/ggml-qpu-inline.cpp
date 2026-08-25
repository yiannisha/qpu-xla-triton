#include "ggml.h"

#include "qpu_llama_runtime.h"
#include "ggml-geglu-split-fp32.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

struct inline_geglu_context {
    std::mutex mutex;
    qpu_llama_context * runtime = nullptr;
    qpu_llama_program * program = nullptr;
    qpu_llama_buffer * table = nullptr;
    qpu_llama_buffer * scratch = nullptr;
    qpu_llama_submission * submission = nullptr;
    size_t scratch_capacity = 0;
    bool initialization_attempted = false;
    bool hybrid_pending = false;
    const float * pending_gate = nullptr;
    const float * pending_up = nullptr;
    float * pending_destination = nullptr;
    uint64_t pending_qpu_rows = 0;
    uint64_t pending_total_rows = 0;
    uint64_t pending_columns = 0;
    uint64_t pending_cpu_threads = 0;
    size_t pending_bytes = 0;
    uint64_t pending_complete_start = 0;
    uint64_t pending_input_copy_end = 0;
    std::atomic<uint64_t> dispatch_count = 0;

    ~inline_geglu_context() {
        qpu_llama_submission_destroy(submission);
        qpu_llama_buffer_destroy(scratch);
        qpu_llama_buffer_destroy(table);
        qpu_llama_program_destroy(program);
        qpu_llama_context_destroy(runtime);
    }
};

inline_geglu_context & get_context() {
    static inline_geglu_context context;
    return context;
}

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void cpu_fallback_geglu(
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

qpu_llama_status initialize(inline_geglu_context & context) {
    if (context.program != nullptr) {
        return QPU_LLAMA_OK;
    }
    if (context.initialization_attempted) {
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    context.initialization_attempted = true;
    qpu_llama_status status = qpu_llama_context_create(nullptr, &context.runtime);
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    status = qpu_llama_buffer_create(
        context.runtime, UINT32_C(1) << 17U, &context.table);
    if (status != QPU_LLAMA_OK) {
        return status;
    }
    auto * table = static_cast<ggml_fp16_t *>(qpu_llama_buffer_data(context.table));
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
    return qpu_llama_program_create(
        context.runtime, &description, &context.program);
}

qpu_llama_status ensure_scratch(inline_geglu_context & context, size_t required) {
    if (context.scratch_capacity >= required) {
        return QPU_LLAMA_OK;
    }
    qpu_llama_buffer_destroy(context.scratch);
    context.scratch = nullptr;
    context.scratch_capacity = 0;
    const qpu_llama_status status = qpu_llama_buffer_create_cached(
        context.runtime, required, &context.scratch);
    if (status == QPU_LLAMA_OK) {
        context.scratch_capacity = required;
    }
    return status;
}

int execute_geglu(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns,
    uint64_t total_rows,
    uint64_t cpu_threads) {
    if (gate == nullptr || up == nullptr || destination == nullptr || rows == 0 ||
        columns == 0 || rows > UINT64_MAX / columns || total_rows < rows ||
        total_rows > UINT64_MAX / columns) {
        return 0;
    }
    const uint64_t elements = rows * columns;
    if (elements % 768 != 0 || elements > SIZE_MAX / (3 * sizeof(float))) {
        return 0;
    }

    inline_geglu_context & context = get_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    qpu_llama_status status = initialize(context);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    if (status == QPU_LLAMA_OK) {
        status = ensure_scratch(context, 3 * bytes);
    }

    const uint64_t complete_start = monotonic_ns();
    uint64_t input_copy_end = complete_start;
    uint64_t submit_end = complete_start;
    uint32_t gate_address = 0;
    uint32_t up_address = 0;
    uint32_t destination_address = 0;
    uint32_t table_address = 0;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_begin(
            context.scratch, QPU_LLAMA_CPU_ACCESS_WRITE);
    }
    if (status == QPU_LLAMA_OK) {
        auto * scratch = static_cast<uint8_t *>(qpu_llama_buffer_data(context.scratch));
        std::memcpy(scratch, gate, bytes);
        std::memcpy(scratch + bytes, up, bytes);
        status = qpu_llama_buffer_cpu_access_end(
            context.scratch, QPU_LLAMA_CPU_ACCESS_WRITE);
        input_copy_end = monotonic_ns();
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.scratch, 0, &gate_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.scratch, bytes, &up_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            context.scratch, 2 * bytes, &destination_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.table, 0, &table_address);
    }
    if (status == QPU_LLAMA_OK) {
        const uint32_t uniforms[] = {
            static_cast<uint32_t>(elements / (12U * 64U)),
            gate_address,
            up_address,
            destination_address,
            table_address,
        };
        qpu_llama_buffer * buffers[] = {context.scratch, context.table};
        const qpu_llama_dispatch_desc dispatch = {
            /* .uniforms           = */ uniforms,
            /* .uniform_word_count = */ 5,
            /* .buffers            = */ buffers,
            /* .buffer_count       = */ 2,
            /* .local_invocation   = */ {16, 1, 1},
            /* .workgroup          = */ {12, 1, 1},
            /* .wgs_per_sg         = */ 24,
            /* .thread_count       = */ 12,
            /* .propagate_nan      = */ 0,
            /* .single_segment     = */ 0,
            /* .threading          = */ 0,
        };
        status = qpu_llama_program_execute(context.program, &dispatch);
        submit_end = monotonic_ns();
    }
    if (status != QPU_LLAMA_OK) {
        if (context.runtime != nullptr) {
            std::fprintf(stderr,
                "QPU: inline GEGLU unavailable, using exact CPU fallback: %s (%s)\n",
                qpu_llama_status_string(status),
                qpu_llama_context_last_error(context.runtime));
        } else {
            std::fprintf(stderr,
                "QPU: inline GEGLU unavailable, using exact CPU fallback: %s\n",
                qpu_llama_status_string(status));
        }
        cpu_fallback_geglu(gate, up, destination, elements);
        return 1;
    }

    const uint64_t output_copy_start = monotonic_ns();
    status = qpu_llama_buffer_cpu_access_begin(
        context.scratch, QPU_LLAMA_CPU_ACCESS_READ);
    if (status == QPU_LLAMA_OK) {
        const auto * scratch = static_cast<const uint8_t *>(
            qpu_llama_buffer_data(context.scratch));
        std::memcpy(destination, scratch + 2 * bytes, bytes);
        status = qpu_llama_buffer_cpu_access_end(
            context.scratch, QPU_LLAMA_CPU_ACCESS_READ);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr,
            "QPU: inline GEGLU cached output access failed, using exact CPU fallback: %s (%s)\n",
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        cpu_fallback_geglu(gate, up, destination, elements);
        return 1;
    }
    const uint64_t complete_end = monotonic_ns();
    ++context.dispatch_count;
    const char * telemetry = std::getenv("GGML_QPU_TELEMETRY");
    if (telemetry == nullptr || std::strcmp(telemetry, "0") != 0) {
        if (total_rows == rows) {
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
        } else {
            std::fprintf(stderr,
                "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
                "\"operation\":\"geglu_split_fp32\","
                "\"program\":\"ggml-geglu-split-fp32\","
                "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\","
                "\"exact_shape\":{\"layout\":\"contiguous-f32-split-geglu\","
                "\"element_multiple\":768},\"placement\":\"hybrid\","
                "\"partition\":{\"axis\":\"rows\",\"qpu_rows\":%llu,"
                "\"cpu_rows\":%llu,\"cpu_threads\":%llu},"
                "\"integration_boundary\":\"ggml-cpu-inline-hook\","
                "\"m\":%llu,\"n\":%llu,\"elements\":%llu,"
                "\"qpu_elements\":%llu,\"dispatch_count\":1,"
                "\"input_copy_ns\":%llu,\"submit_wait_ns\":%llu,"
                "\"output_copy_ns\":%llu,\"complete_ns\":%llu}\n",
                qpu_ggml_geglu_split_fp32_source_hash,
                qpu_ggml_geglu_split_fp32_binary_hash,
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(total_rows - rows),
                static_cast<unsigned long long>(cpu_threads),
                static_cast<unsigned long long>(total_rows),
                static_cast<unsigned long long>(columns),
                static_cast<unsigned long long>(total_rows * columns),
                static_cast<unsigned long long>(elements),
                static_cast<unsigned long long>(input_copy_end - complete_start),
                static_cast<unsigned long long>(submit_end - input_copy_end),
                static_cast<unsigned long long>(complete_end - output_copy_start),
                static_cast<unsigned long long>(complete_end - complete_start));
        }
    }
    return 1;
}

int begin_hybrid_geglu(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns,
    uint64_t qpu_rows,
    uint64_t cpu_threads) {
    if (gate == nullptr || up == nullptr || destination == nullptr ||
        qpu_rows == 0 || qpu_rows >= rows || columns == 0 || cpu_threads == 0 ||
        qpu_rows > UINT64_MAX / columns || rows > UINT64_MAX / columns) {
        return 0;
    }
    const uint64_t elements = qpu_rows * columns;
    if (elements % 768 != 0 || elements > SIZE_MAX / (3 * sizeof(float))) {
        return 0;
    }
    inline_geglu_context & context = get_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.hybrid_pending) {
        return 0;
    }
    qpu_llama_status status = initialize(context);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    if (status == QPU_LLAMA_OK) {
        status = ensure_scratch(context, 3 * bytes);
    }

    context.pending_gate = gate;
    context.pending_up = up;
    context.pending_destination = destination;
    context.pending_qpu_rows = qpu_rows;
    context.pending_total_rows = rows;
    context.pending_columns = columns;
    context.pending_cpu_threads = cpu_threads;
    context.pending_bytes = bytes;
    context.pending_complete_start = monotonic_ns();
    context.pending_input_copy_end = context.pending_complete_start;
    context.hybrid_pending = true;

    uint32_t gate_address = 0;
    uint32_t up_address = 0;
    uint32_t destination_address = 0;
    uint32_t table_address = 0;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_begin(
            context.scratch, QPU_LLAMA_CPU_ACCESS_WRITE);
    }
    if (status == QPU_LLAMA_OK) {
        auto * scratch = static_cast<uint8_t *>(qpu_llama_buffer_data(context.scratch));
        std::memcpy(scratch, gate, bytes);
        std::memcpy(scratch + bytes, up, bytes);
        status = qpu_llama_buffer_cpu_access_end(
            context.scratch, QPU_LLAMA_CPU_ACCESS_WRITE);
        context.pending_input_copy_end = monotonic_ns();
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.scratch, 0, &gate_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.scratch, bytes, &up_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            context.scratch, 2 * bytes, &destination_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(context.table, 0, &table_address);
    }
    if (status == QPU_LLAMA_OK) {
        const uint32_t uniforms[] = {
            static_cast<uint32_t>(elements / (12U * 64U)),
            gate_address,
            up_address,
            destination_address,
            table_address,
        };
        qpu_llama_buffer * buffers[] = {context.scratch, context.table};
        const qpu_llama_dispatch_desc dispatch = {
            /* .uniforms           = */ uniforms,
            /* .uniform_word_count = */ 5,
            /* .buffers            = */ buffers,
            /* .buffer_count       = */ 2,
            /* .local_invocation   = */ {16, 1, 1},
            /* .workgroup          = */ {12, 1, 1},
            /* .wgs_per_sg         = */ 24,
            /* .thread_count       = */ 12,
            /* .propagate_nan      = */ 0,
            /* .single_segment     = */ 0,
            /* .threading          = */ 0,
        };
        status = qpu_llama_program_submit(
            context.program, &dispatch, &context.submission);
    }
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr,
            "QPU: hybrid GEGLU submission unavailable, using exact CPU fallback: %s (%s)\n",
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        cpu_fallback_geglu(gate, up, destination, elements);
    }
    return 1;
}

int wait_hybrid_geglu() {
    inline_geglu_context & context = get_context();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (!context.hybrid_pending) {
        return 0;
    }
    qpu_llama_status status = QPU_LLAMA_OK;
    uint64_t qpu_complete = context.pending_input_copy_end;
    uint64_t output_copy_start = qpu_complete;
    if (context.submission != nullptr) {
        status = qpu_llama_submission_wait(context.submission);
        qpu_complete = monotonic_ns();
        output_copy_start = qpu_complete;
        if (status == QPU_LLAMA_OK) {
            status = qpu_llama_buffer_cpu_access_begin(
                context.scratch, QPU_LLAMA_CPU_ACCESS_READ);
        }
        if (status == QPU_LLAMA_OK) {
            const auto * scratch = static_cast<const uint8_t *>(
                qpu_llama_buffer_data(context.scratch));
            std::memcpy(
                context.pending_destination,
                scratch + 2 * context.pending_bytes,
                context.pending_bytes);
            status = qpu_llama_buffer_cpu_access_end(
                context.scratch, QPU_LLAMA_CPU_ACCESS_READ);
        }
        qpu_llama_submission_destroy(context.submission);
        context.submission = nullptr;
    }
    const uint64_t complete_end = monotonic_ns();
    if (status != QPU_LLAMA_OK) {
        std::fprintf(stderr,
            "QPU: hybrid GEGLU completion failed, using exact CPU fallback: %s (%s)\n",
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context.runtime));
        cpu_fallback_geglu(
            context.pending_gate,
            context.pending_up,
            context.pending_destination,
            context.pending_qpu_rows * context.pending_columns);
    } else if (qpu_complete != context.pending_input_copy_end) {
        ++context.dispatch_count;
        const char * telemetry = std::getenv("GGML_QPU_TELEMETRY");
        if (telemetry == nullptr || std::strcmp(telemetry, "0") != 0) {
            std::fprintf(stderr,
                "qpu_llama_candidate_json:{\"schema_version\":1,\"backend\":\"QPU0\","
                "\"operation\":\"geglu_split_fp32\","
                "\"program\":\"ggml-geglu-split-fp32\","
                "\"source_hash\":\"%s\",\"binary_sha256\":\"%s\","
                "\"exact_shape\":{\"layout\":\"contiguous-f32-split-geglu\","
                "\"element_multiple\":768},\"placement\":\"hybrid\","
                "\"partition\":{\"axis\":\"rows\",\"qpu_rows\":%llu,"
                "\"cpu_rows\":%llu,\"cpu_threads\":%llu},"
                "\"integration_boundary\":\"ggml-cpu-inline-hook\","
                "\"m\":%llu,\"n\":%llu,\"elements\":%llu,"
                "\"qpu_elements\":%llu,\"dispatch_count\":1,"
                "\"input_copy_ns\":%llu,\"submit_overlap_wait_ns\":%llu,"
                "\"output_copy_ns\":%llu,\"complete_ns\":%llu}\n",
                qpu_ggml_geglu_split_fp32_source_hash,
                qpu_ggml_geglu_split_fp32_binary_hash,
                static_cast<unsigned long long>(context.pending_qpu_rows),
                static_cast<unsigned long long>(
                    context.pending_total_rows - context.pending_qpu_rows),
                static_cast<unsigned long long>(context.pending_cpu_threads),
                static_cast<unsigned long long>(context.pending_total_rows),
                static_cast<unsigned long long>(context.pending_columns),
                static_cast<unsigned long long>(
                    context.pending_total_rows * context.pending_columns),
                static_cast<unsigned long long>(
                    context.pending_qpu_rows * context.pending_columns),
                static_cast<unsigned long long>(
                    context.pending_input_copy_end - context.pending_complete_start),
                static_cast<unsigned long long>(
                    qpu_complete - context.pending_input_copy_end),
                static_cast<unsigned long long>(complete_end - output_copy_start),
                static_cast<unsigned long long>(
                    complete_end - context.pending_complete_start));
        }
    }
    context.hybrid_pending = false;
    context.pending_gate = nullptr;
    context.pending_up = nullptr;
    context.pending_destination = nullptr;
    return 1;
}

} // namespace

extern "C" __attribute__((visibility("default"))) int ggml_qpu_geglu_f32(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns) {
    return execute_geglu(gate, up, destination, rows, columns, rows, 0);
}

extern "C" __attribute__((visibility("default"))) int ggml_qpu_geglu_f32_hybrid(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns,
    uint64_t qpu_rows,
    uint64_t cpu_threads) {
    if (qpu_rows == 0 || qpu_rows >= rows || cpu_threads == 0) {
        return 0;
    }
    return execute_geglu(
        gate, up, destination, qpu_rows, columns, rows, cpu_threads);
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_geglu_f32_hybrid_begin(
    const float * gate,
    const float * up,
    float * destination,
    uint64_t rows,
    uint64_t columns,
    uint64_t qpu_rows,
    uint64_t cpu_threads) {
    return begin_hybrid_geglu(
        gate, up, destination, rows, columns, qpu_rows, cpu_threads);
}

extern "C" __attribute__((visibility("default"))) int
ggml_qpu_geglu_f32_hybrid_wait() {
    return wait_hybrid_geglu();
}

extern "C" __attribute__((visibility("default"))) uint64_t
ggml_qpu_geglu_dispatch_count() {
    return get_context().dispatch_count.load();
}
