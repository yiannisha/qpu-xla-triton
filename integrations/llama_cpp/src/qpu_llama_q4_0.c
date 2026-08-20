#include "qpu_llama_q4_0.h"

#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-m4.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q4_0_BLOCK_ELEMENTS 32U
#define Q4_0_BLOCK_BYTES 18U
#define Q8_0_BLOCK_BYTES 34U
#define Q4_0_OUTPUT_TILE 16U

struct qpu_llama_q4_0_linear {
    qpu_llama_context *context;
    qpu_llama_program *program;
    qpu_llama_buffer *weight;
    qpu_llama_buffer *activation_staging;
    qpu_llama_buffer *staging;
    pthread_mutex_t mutex;
    uint32_t blocks;
    uint32_t output_columns;
    uint32_t rows;
    uint32_t kernel_rows;
    uint32_t resident_column_start;
    uint32_t resident_column_count;
    size_t weight_row_bytes;
    size_t resident_bytes;
};

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) + (uint64_t) value.tv_nsec;
}

static bool multiply_size(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static bool valid_range(size_t total, size_t offset, size_t size) {
    return offset <= total && size <= total - offset;
}

qpu_llama_status qpu_llama_q4_0_linear_prepare(
    qpu_llama_context *context,
    const qpu_llama_q4_0_linear_desc *desc,
    qpu_llama_q4_0_linear **result) {
    if (context == NULL || desc == NULL || result == NULL || desc->weights == NULL ||
        desc->expected_source_hash == NULL || (desc->rows != 1 && desc->rows != 4) ||
        desc->input_columns == 0 || desc->input_columns % Q4_0_BLOCK_ELEMENTS != 0 ||
        desc->output_columns == 0 || desc->resident_column_count == 0 ||
        desc->resident_column_start % Q4_0_OUTPUT_TILE != 0 ||
        desc->resident_column_count % Q4_0_OUTPUT_TILE != 0 ||
        desc->resident_column_start > desc->output_columns ||
        desc->resident_column_count > desc->output_columns - desc->resident_column_start) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    const size_t blocks = desc->input_columns / Q4_0_BLOCK_ELEMENTS;
    size_t weight_row_bytes = 0;
    size_t required_source_bytes = 0;
    size_t resident_bytes = 0;
    size_t staging_elements = 0;
    size_t staging_bytes = 0;
    size_t activation_staging_bytes = 0;
    const uint32_t kernel_rows = 4;
    if (!multiply_size(blocks, Q4_0_BLOCK_BYTES, &weight_row_bytes) ||
        !multiply_size(desc->output_columns, weight_row_bytes, &required_source_bytes) ||
        !multiply_size(desc->resident_column_count, weight_row_bytes, &resident_bytes) ||
        !multiply_size(kernel_rows, desc->resident_column_count, &staging_elements) ||
        !multiply_size(staging_elements, sizeof(float), &staging_bytes) ||
        !multiply_size(kernel_rows, blocks * Q8_0_BLOCK_BYTES, &activation_staging_bytes) ||
        desc->weight_size < required_source_bytes) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_q4_0_linear *linear = calloc(1, sizeof(*linear));
    if (linear == NULL) {
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    linear->context = context;
    linear->blocks = (uint32_t) blocks;
    linear->output_columns = desc->output_columns;
    linear->rows = desc->rows;
    linear->kernel_rows = kernel_rows;
    linear->resident_column_start = desc->resident_column_start;
    linear->resident_column_count = desc->resident_column_count;
    linear->weight_row_bytes = weight_row_bytes;
    linear->resident_bytes = resident_bytes + staging_bytes + activation_staging_bytes;
    if (pthread_mutex_init(&linear->mutex, NULL) != 0) {
        free(linear);
        return QPU_LLAMA_INTERNAL_ERROR;
    }

    qpu_llama_status status = qpu_llama_buffer_create(context, resident_bytes, &linear->weight);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create(context, staging_bytes, &linear->staging);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create(context, activation_staging_bytes, &linear->activation_staging);
    }
    if (status == QPU_LLAMA_OK) {
        const uint8_t *source = desc->weights;
        source += (size_t) desc->resident_column_start * weight_row_bytes;
        memcpy(qpu_llama_buffer_data(linear->weight), source, resident_bytes);
        atomic_thread_fence(memory_order_seq_cst);
        const void *code = desc->rows == 1
            ? (const void *) qpu_ggml_q4_0_q8_0_m1
            : (const void *) qpu_ggml_q4_0_q8_0_m4;
        const size_t code_size = desc->rows == 1
            ? sizeof(qpu_ggml_q4_0_q8_0_m1)
            : sizeof(qpu_ggml_q4_0_q8_0_m4);
        const char *source_hash = desc->rows == 1
            ? qpu_ggml_q4_0_q8_0_m1_source_hash
            : qpu_ggml_q4_0_q8_0_m4_source_hash;
        const char *binary_hash = desc->rows == 1
            ? qpu_ggml_q4_0_q8_0_m1_binary_hash
            : qpu_ggml_q4_0_q8_0_m4_binary_hash;
        const qpu_llama_program_desc program_desc = {
            .code = code,
            .code_size = code_size,
            .compiled_source_hash = source_hash,
            .expected_source_hash = desc->expected_source_hash,
            .binary_sha256 = binary_hash,
            .uniform_word_count = 13,
        };
        status = qpu_llama_program_create(context, &program_desc, &linear->program);
    }
    if (status != QPU_LLAMA_OK) {
        qpu_llama_q4_0_linear_destroy(linear);
        return status;
    }
    *result = linear;
    return QPU_LLAMA_OK;
}

void qpu_llama_q4_0_linear_destroy(qpu_llama_q4_0_linear *linear) {
    if (linear == NULL) {
        return;
    }
    qpu_llama_program_destroy(linear->program);
    qpu_llama_buffer_destroy(linear->staging);
    qpu_llama_buffer_destroy(linear->activation_staging);
    qpu_llama_buffer_destroy(linear->weight);
    pthread_mutex_destroy(&linear->mutex);
    free(linear);
}

qpu_llama_status qpu_llama_q4_0_linear_execute(
    qpu_llama_q4_0_linear *linear,
    const qpu_llama_q4_0_execution *execution,
    qpu_llama_q4_0_timing *timing) {
    if (linear == NULL || execution == NULL || execution->activation == NULL ||
        execution->destination == NULL || execution->column_count == 0 ||
        execution->column_start % Q4_0_OUTPUT_TILE != 0 ||
        execution->column_count % Q4_0_OUTPUT_TILE != 0 ||
        execution->column_start < linear->resident_column_start ||
        execution->column_count > linear->resident_column_count ||
        execution->column_start - linear->resident_column_start >
            linear->resident_column_count - execution->column_count) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    size_t activation_row_bytes = 0;
    size_t activation_bytes = 0;
    size_t destination_elements = 0;
    size_t destination_bytes = 0;
    if (!multiply_size(linear->blocks, Q8_0_BLOCK_BYTES, &activation_row_bytes) ||
        !multiply_size(linear->rows, activation_row_bytes, &activation_bytes) ||
        !multiply_size(linear->rows, linear->output_columns, &destination_elements) ||
        !multiply_size(destination_elements, sizeof(float), &destination_bytes) ||
        !valid_range(execution->activation_size, execution->activation_offset, activation_bytes) ||
        !valid_range(execution->destination_size, execution->destination_offset, destination_bytes)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&linear->mutex);
    const uint64_t complete_start = monotonic_ns();
    const uint8_t *source = execution->activation;
    source += execution->activation_offset;
    uint8_t *expanded = qpu_llama_buffer_data(linear->activation_staging);
    if (linear->rows == 1) {
        for (uint32_t row = 0; row < linear->kernel_rows; ++row) {
            memcpy(expanded + (size_t) row * activation_row_bytes, source, activation_row_bytes);
        }
    } else {
        memcpy(expanded, source, activation_bytes);
    }
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t input_copy_end = monotonic_ns();
    uint32_t activation_address = 0;
    uint32_t weight_address = 0;
    uint32_t staging_address = 0;
    qpu_llama_status status = qpu_llama_buffer_gpu_address(
        linear->activation_staging, 0, &activation_address);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(linear->weight, 0, &weight_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(linear->staging, 0, &staging_address);
    }
    const uint32_t local_column_start = execution->column_start - linear->resident_column_start;
    const uint32_t uniforms[] = {
        linear->blocks,
        (uint32_t) activation_row_bytes,
        activation_address,
        (uint32_t) linear->weight_row_bytes,
        weight_address,
        linear->resident_column_count * (uint32_t) sizeof(float),
        staging_address,
        local_column_start,
        local_column_start,
        UINT32_C(0x0f0f0f0f),
        UINT32_C(0x08080808),
        Q4_0_BLOCK_BYTES,
        Q8_0_BLOCK_BYTES,
    };
    qpu_llama_buffer *buffers[] = {linear->activation_staging, linear->weight, linear->staging};
    const qpu_llama_dispatch_desc dispatch = {
        .uniforms = uniforms,
        .uniform_word_count = sizeof(uniforms) / sizeof(uniforms[0]),
        .buffers = buffers,
        .buffer_count = sizeof(buffers) / sizeof(buffers[0]),
        .local_invocation = {16, 1, 1},
        .workgroup = {execution->column_count / Q4_0_OUTPUT_TILE, 1, 1},
        .wgs_per_sg = 24,
        .thread_count = execution->column_count / Q4_0_OUTPUT_TILE,
        .propagate_nan = 0,
        .single_segment = 0,
        .threading = 0,
    };
    const uint64_t submit_start = monotonic_ns();
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_program_execute(linear->program, &dispatch);
    }
    const uint64_t submit_end = monotonic_ns();
    uint64_t copy_end = submit_end;
    if (status == QPU_LLAMA_OK) {
        uint8_t *destination = execution->destination;
        destination += execution->destination_offset;
        const uint8_t *staging = qpu_llama_buffer_data(linear->staging);
        for (uint32_t row = 0; row < linear->rows; ++row) {
            const size_t destination_offset =
                ((size_t) row * linear->output_columns + execution->column_start) * sizeof(float);
            const size_t staging_offset =
                ((size_t) row * linear->resident_column_count + local_column_start) * sizeof(float);
            memcpy(destination + destination_offset, staging + staging_offset,
                (size_t) execution->column_count * sizeof(float));
        }
        atomic_thread_fence(memory_order_seq_cst);
        copy_end = monotonic_ns();
    }
    if (timing != NULL) {
        timing->input_copy_ns = input_copy_end - complete_start;
        timing->submit_wait_ns = submit_end - submit_start;
        timing->output_copy_ns = copy_end - submit_end;
        timing->complete_ns = copy_end - complete_start;
    }
    pthread_mutex_unlock(&linear->mutex);
    return status;
}

size_t qpu_llama_q4_0_linear_resident_bytes(const qpu_llama_q4_0_linear *linear) {
    return linear != NULL ? linear->resident_bytes : 0;
}
