#include "qpu_llama_q4_0.h"

#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-m4.h"
#include "ggml-q4-0-q8-0-mx.h"
#include "ggml-column-w8-q8-0-mx.h"
#include "tiled-w8a8-gemm-dequantize.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define Q4_0_BLOCK_ELEMENTS 32U
#define Q4_0_BLOCK_BYTES 18U
#define Q8_0_BLOCK_BYTES 34U
#define Q4_0_OUTPUT_TILE 16U
#define QPU_BUFFER_ALIGNMENT 64U

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
    uint32_t workgroups_per_supergroup;
    size_t weight_row_bytes;
    size_t device_weight_row_bytes;
    size_t device_activation_row_bytes;
    size_t weight_q_offset;
    size_t activation_q_offset;
    size_t activation_scale_row_bytes;
    size_t resident_bytes;
    bool packed_mx;
    qpu_llama_q4_0_weight_mode weight_mode;
    bool busy;
};

struct qpu_llama_q4_0_submission {
    qpu_llama_q4_0_linear *linear;
    qpu_llama_submission *runtime_submission;
    uint8_t *destination;
    size_t destination_offset;
    uint32_t rows;
    uint32_t column_start;
    uint32_t column_count;
    uint32_t local_column_start;
    uint64_t complete_start;
    uint64_t input_access_end;
    uint64_t input_pack_end;
    uint64_t input_copy_end;
    uint64_t submit_start;
    uint64_t submit_end;
    bool active;
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

static bool align_size(size_t value, size_t alignment, size_t *result) {
    if (value > SIZE_MAX - (alignment - 1U)) {
        return false;
    }
    *result = (value + alignment - 1U) & ~(alignment - 1U);
    return true;
}

static float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = (uint32_t) (value & UINT16_C(0x8000)) << 16U;
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
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void pack_interleaved_q8_0x4_block(
    const uint8_t *source,
    uint8_t *scales[4],
    uint8_t *values[4],
    uint32_t interleave) {
    for (uint32_t row = 0; row < 4U; ++row) {
        memcpy(scales[row], source + row * sizeof(uint16_t), sizeof(uint16_t));
    }
    source += 4U * sizeof(uint16_t);
#if defined(__aarch64__)
    if (interleave == 4U) {
        const uint32x4x4_t low = vld4q_u32((const uint32_t *) source);
        const uint32x4x4_t high = vld4q_u32((const uint32_t *) (source + 64U));
        for (uint32_t row = 0; row < 4U; ++row) {
            vst1q_u32((uint32_t *) values[row], low.val[row]);
            vst1q_u32((uint32_t *) (values[row] + 16U), high.val[row]);
        }
        return;
    }
    if (interleave == 8U) {
        const uint64x2x4_t low = vld4q_u64((const uint64_t *) source);
        const uint64x2x4_t high = vld4q_u64((const uint64_t *) (source + 64U));
        for (uint32_t row = 0; row < 4U; ++row) {
            vst1q_u64((uint64_t *) values[row], low.val[row]);
            vst1q_u64((uint64_t *) (values[row] + 16U), high.val[row]);
        }
        return;
    }
#endif
    for (uint32_t row = 0; row < 4U; ++row) {
        for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
            const uint32_t source_index =
                (index / interleave) * 4U * interleave +
                row * interleave + index % interleave;
            values[row][index] = source[source_index];
        }
    }
}

static void q8_0_source_block(
    const uint8_t *source,
    size_t activation_row_bytes,
    uint32_t row,
    uint32_t total_rows,
    uint32_t block,
    uint32_t interleave,
    const uint8_t **scale,
    const uint8_t **values) {
    const uint32_t interleaved_rows = interleave == 0U
        ? 0U : total_rows - total_rows % 4U;
    if (row < interleaved_rows) {
        const uint8_t *group = source +
            (size_t) (row / 4U) * 4U * activation_row_bytes +
            (size_t) block * 4U * Q8_0_BLOCK_BYTES;
        *scale = group + (row % 4U) * sizeof(uint16_t);
        *values = group + 4U * sizeof(uint16_t);
        return;
    }
    const uint8_t *native = source + (size_t) row * activation_row_bytes +
        (size_t) block * Q8_0_BLOCK_BYTES;
    *scale = native;
    *values = native + sizeof(uint16_t);
}

static int8_t q8_0_source_value(
    const uint8_t *values,
    uint32_t row,
    uint32_t index,
    uint32_t interleave) {
    if (interleave == 0U) {
        return (int8_t) values[index];
    }
    const uint32_t source_index =
        (index / interleave) * 4U * interleave +
        (row % 4U) * interleave + index % interleave;
    return (int8_t) values[source_index];
}

qpu_llama_status qpu_llama_q4_0_linear_prepare(
    qpu_llama_context *context,
    const qpu_llama_q4_0_linear_desc *desc,
    qpu_llama_q4_0_linear **result) {
    if (context == NULL || desc == NULL || result == NULL || desc->weights == NULL ||
        desc->expected_source_hash == NULL || desc->rows == 0 ||
        desc->rows > 4U * UINT16_MAX ||
        desc->input_columns == 0 || desc->input_columns % Q4_0_BLOCK_ELEMENTS != 0 ||
        desc->output_columns == 0 || desc->resident_column_count == 0 ||
        desc->resident_column_start % Q4_0_OUTPUT_TILE != 0 ||
        desc->resident_column_count % Q4_0_OUTPUT_TILE != 0 ||
        (desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT &&
            desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8 &&
            desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) ||
        (desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT && desc->rows <= 4U) ||
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
    size_t weight_scale_bytes = 0;
    size_t weight_q_bytes = 0;
    size_t weight_q_offset = 0;
    size_t activation_scale_bytes = 0;
    size_t activation_q_bytes = 0;
    size_t activation_q_offset = 0;
    const uint32_t row_tile = desc->rows <= 4U ? 4U : 16U;
    const uint32_t kernel_rows = (desc->rows + row_tile - 1U) & ~(row_tile - 1U);
    const bool packed_mx = desc->rows > 4U;
    size_t device_weight_row_bytes = 0;
    size_t device_activation_row_bytes = 0;
    size_t activation_scale_row_bytes = 0;
    if (!multiply_size(blocks, Q4_0_BLOCK_BYTES, &weight_row_bytes) ||
        !multiply_size(desc->output_columns, weight_row_bytes, &required_source_bytes) ||
        !multiply_size(kernel_rows, desc->resident_column_count, &staging_elements) ||
        !multiply_size(staging_elements, sizeof(float), &staging_bytes) ||
        desc->weight_size < required_source_bytes) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    if (packed_mx) {
        device_weight_row_bytes = (size_t) desc->resident_column_count * sizeof(uint32_t);
        device_activation_row_bytes = blocks * Q4_0_BLOCK_ELEMENTS;
        activation_scale_row_bytes =
            desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? sizeof(float) : blocks * sizeof(uint16_t);
        const size_t scale_element_bytes =
            desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT
                ? sizeof(float) : sizeof(uint16_t);
        const size_t scale_count =
            desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT ? 1U : blocks;
        if (!multiply_size(scale_count * scale_element_bytes,
                desc->resident_column_count,
                &weight_scale_bytes) ||
            !align_size(weight_scale_bytes, QPU_BUFFER_ALIGNMENT, &weight_q_offset) ||
            !multiply_size(blocks * 8U + 4U, device_weight_row_bytes,
                &weight_q_bytes) ||
            weight_q_offset > SIZE_MAX - weight_q_bytes ||
            !multiply_size(kernel_rows, activation_scale_row_bytes,
                &activation_scale_bytes) ||
            !align_size(activation_scale_bytes, QPU_BUFFER_ALIGNMENT,
                &activation_q_offset) ||
            !multiply_size(kernel_rows, device_activation_row_bytes,
                &activation_q_bytes) ||
            activation_q_bytes > SIZE_MAX - 16U ||
            activation_q_offset > SIZE_MAX - activation_q_bytes - 16U) {
            return QPU_LLAMA_INVALID_ARGUMENT;
        }
        resident_bytes = weight_q_offset + weight_q_bytes;
        activation_staging_bytes = activation_q_offset + activation_q_bytes + 16U;
    } else {
        device_weight_row_bytes = weight_row_bytes;
        device_activation_row_bytes = blocks * Q8_0_BLOCK_BYTES;
        if (!multiply_size(desc->resident_column_count, device_weight_row_bytes,
                &resident_bytes) ||
            !multiply_size(kernel_rows, device_activation_row_bytes,
                &activation_staging_bytes)) {
            return QPU_LLAMA_INVALID_ARGUMENT;
        }
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
    linear->workgroups_per_supergroup = desc->workgroups_per_supergroup != 0U
        ? desc->workgroups_per_supergroup : 24U;
    linear->weight_row_bytes = weight_row_bytes;
    linear->device_weight_row_bytes = device_weight_row_bytes;
    linear->device_activation_row_bytes = device_activation_row_bytes;
    linear->weight_q_offset = weight_q_offset;
    linear->activation_q_offset = activation_q_offset;
    linear->activation_scale_row_bytes = activation_scale_row_bytes;
    linear->resident_bytes = resident_bytes + staging_bytes + activation_staging_bytes;
    linear->packed_mx = packed_mx;
    linear->weight_mode = desc->weight_mode;
    if (pthread_mutex_init(&linear->mutex, NULL) != 0) {
        free(linear);
        return QPU_LLAMA_INTERNAL_ERROR;
    }

    qpu_llama_status status = qpu_llama_buffer_create(context, resident_bytes, &linear->weight);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create_cached(context, staging_bytes, &linear->staging);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create_cached(
            context, activation_staging_bytes, &linear->activation_staging);
    }
    if (status == QPU_LLAMA_OK) {
        const uint8_t *source = desc->weights;
        source += (size_t) desc->resident_column_start * weight_row_bytes;
        uint8_t *destination = qpu_llama_buffer_data(linear->weight);
        if (packed_mx) {
            memset(destination, 0, resident_bytes);
            uint8_t *weight_q = destination + weight_q_offset;
            for (uint32_t column = 0; column < desc->resident_column_count; ++column) {
                float column_scale = 0.0f;
                if (desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT) {
                    float maximum = 0.0f;
                    for (uint32_t block = 0; block < blocks; ++block) {
                        const uint8_t *source_block = source +
                            (size_t) column * weight_row_bytes +
                            (size_t) block * Q4_0_BLOCK_BYTES;
                        uint16_t scale_bits = 0;
                        memcpy(&scale_bits, source_block, sizeof(scale_bits));
                        const float block_scale = fp16_to_fp32(scale_bits);
                        for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
                            const uint8_t packed = source_block[2U + index % 16U];
                            const uint8_t code = index < 16U
                                ? packed & 0x0fU : packed >> 4U;
                            const float value = fabsf((float) ((int) code - 8) * block_scale);
                            maximum = value > maximum ? value : maximum;
                        }
                    }
                    column_scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
                    memcpy(destination + (size_t) column * sizeof(float),
                        &column_scale, sizeof(column_scale));
                }
                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint8_t *source_block = source +
                        (size_t) column * weight_row_bytes +
                        (size_t) block * Q4_0_BLOCK_BYTES;
                    uint16_t block_scale_bits = 0;
                    memcpy(&block_scale_bits, source_block, sizeof(block_scale_bits));
                    const float block_scale = fp16_to_fp32(block_scale_bits);
                    if (desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_EXACT) {
                        memcpy(destination +
                            ((size_t) block * desc->resident_column_count + column) *
                                sizeof(uint16_t),
                            source_block, sizeof(uint16_t));
                    }
                    for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
                        const uint8_t packed = source_block[2U + index % 16U];
                        const uint8_t code = index < 16U ? packed & 0x0fU : packed >> 4U;
                        const int q4_value = (int) code - 8;
                        int8_t value = (int8_t) q4_value;
                        if (desc->weight_mode != QPU_LLAMA_Q4_0_WEIGHT_EXACT) {
                            const float scaled = column_scale > 0.0f
                                ? (float) q4_value * block_scale / column_scale : 0.0f;
                            long rounded = (long) (scaled >= 0.0f
                                ? scaled + 0.5f : scaled - 0.5f);
                            rounded = rounded < -127L ? -127L : rounded > 127L ? 127L : rounded;
                            value = (int8_t) rounded;
                        }
                        const size_t qword = (size_t) block * 8U + index / 4U;
                        weight_q[qword * device_weight_row_bytes +
                            (size_t) column * sizeof(uint32_t) + index % 4U] =
                                (uint8_t) value;
                    }
                }
            }
        } else {
            memcpy(destination, source, resident_bytes);
        }
        atomic_thread_fence(memory_order_seq_cst);
        const void *code = desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
            ? (const void *) qpu_tiled_w8a8_gemm_dequantize
            : desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                ? (const void *) qpu_ggml_column_w8_q8_0_mx
            : desc->rows == 1
            ? (const void *) qpu_ggml_q4_0_q8_0_m1
            : desc->rows == 4
                ? (const void *) qpu_ggml_q4_0_q8_0_m4
                : (const void *) qpu_ggml_q4_0_q8_0_mx;
        const size_t code_size = desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
            ? sizeof(qpu_tiled_w8a8_gemm_dequantize)
            : desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                ? sizeof(qpu_ggml_column_w8_q8_0_mx)
            : desc->rows == 1
            ? sizeof(qpu_ggml_q4_0_q8_0_m1)
            : desc->rows == 4
                ? sizeof(qpu_ggml_q4_0_q8_0_m4)
                : sizeof(qpu_ggml_q4_0_q8_0_mx);
        const char *source_hash = desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
            ? qpu_tiled_w8a8_gemm_dequantize_source_hash
            : desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                ? qpu_ggml_column_w8_q8_0_mx_source_hash
            : desc->rows == 1
            ? qpu_ggml_q4_0_q8_0_m1_source_hash
            : desc->rows == 4
                ? qpu_ggml_q4_0_q8_0_m4_source_hash
                : qpu_ggml_q4_0_q8_0_mx_source_hash;
        const char *binary_hash = desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
            ? qpu_tiled_w8a8_gemm_dequantize_binary_hash
            : desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                ? qpu_ggml_column_w8_q8_0_mx_binary_hash
            : desc->rows == 1
            ? qpu_ggml_q4_0_q8_0_m1_binary_hash
            : desc->rows == 4
                ? qpu_ggml_q4_0_q8_0_m4_binary_hash
                : qpu_ggml_q4_0_q8_0_mx_binary_hash;
        const qpu_llama_program_desc program_desc = {
            .code = code,
            .code_size = code_size,
            .compiled_source_hash = source_hash,
            .expected_source_hash = desc->expected_source_hash,
            .binary_sha256 = binary_hash,
            .uniform_word_count = desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8
                ? 9U : desc->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8
                    ? 10U : packed_mx ? 11U : 13U,
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

qpu_llama_status qpu_llama_q4_0_linear_submit(
    qpu_llama_q4_0_linear *linear,
    const qpu_llama_q4_0_execution *execution,
    qpu_llama_q4_0_submission **result) {
    const uint32_t rows = execution != NULL && execution->rows != 0
        ? execution->rows : linear != NULL ? linear->rows : 0;
    if (linear == NULL || execution == NULL || result == NULL ||
        execution->activation == NULL ||
        execution->destination == NULL || execution->column_count == 0 ||
        rows == 0 || rows > linear->rows ||
        execution->column_start % Q4_0_OUTPUT_TILE != 0 ||
        execution->column_count % Q4_0_OUTPUT_TILE != 0 ||
        execution->column_start < linear->resident_column_start ||
        execution->column_count > linear->resident_column_count ||
        execution->column_start - linear->resident_column_start >
            linear->resident_column_count - execution->column_count) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    if (execution->activation_interleave != 0U &&
        execution->activation_interleave != 4U &&
        execution->activation_interleave != 8U) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    size_t activation_row_bytes = 0;
    size_t activation_bytes = 0;
    size_t destination_elements = 0;
    size_t destination_bytes = 0;
    const uint32_t row_tile = linear->rows <= 4U ? 4U : 16U;
    const uint32_t kernel_rows = (rows + row_tile - 1U) & ~(row_tile - 1U);
    if (!multiply_size(linear->blocks, Q8_0_BLOCK_BYTES, &activation_row_bytes) ||
        !multiply_size(rows, activation_row_bytes, &activation_bytes) ||
        !multiply_size(rows, linear->output_columns, &destination_elements) ||
        !multiply_size(destination_elements, sizeof(float), &destination_bytes) ||
        !valid_range(execution->activation_size, execution->activation_offset, activation_bytes) ||
        !valid_range(execution->destination_size, execution->destination_offset, destination_bytes)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }

    qpu_llama_q4_0_submission *submission = calloc(1, sizeof(*submission));
    if (submission == NULL) {
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    pthread_mutex_lock(&linear->mutex);
    if (linear->busy) {
        pthread_mutex_unlock(&linear->mutex);
        free(submission);
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    linear->busy = true;
    pthread_mutex_unlock(&linear->mutex);
    const uint64_t complete_start = monotonic_ns();
    const uint8_t *source = execution->activation;
    source += execution->activation_offset;
    uint8_t *expanded = qpu_llama_buffer_data(linear->activation_staging);
    qpu_llama_status status = qpu_llama_buffer_cpu_access_begin(
        linear->activation_staging, QPU_LLAMA_CPU_ACCESS_WRITE);
    if (status != QPU_LLAMA_OK) {
        pthread_mutex_lock(&linear->mutex);
        linear->busy = false;
        pthread_mutex_unlock(&linear->mutex);
        free(submission);
        return status;
    }
    const uint64_t input_access_end = monotonic_ns();
    if (linear->packed_mx &&
        linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) {
        memset(expanded, 0, linear->activation_q_offset +
            (size_t) linear->kernel_rows * linear->device_activation_row_bytes + 16U);
        uint8_t *activation_q = expanded + linear->activation_q_offset;
        const uint32_t interleave = execution->activation_interleave;
        for (uint32_t row = 0; row < rows; ++row) {
            float maximum = 0.0f;
            for (uint32_t block = 0; block < linear->blocks; ++block) {
                const uint8_t *scale_source = NULL;
                const uint8_t *value_source = NULL;
                q8_0_source_block(source, activation_row_bytes, row, rows, block,
                    interleave, &scale_source, &value_source);
                uint16_t scale_bits = 0;
                memcpy(&scale_bits, scale_source, sizeof(scale_bits));
                const float block_scale = fp16_to_fp32(scale_bits);
                for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
                    const float value = fabsf((float) q8_0_source_value(
                        value_source, row, index, interleave) * block_scale);
                    maximum = value > maximum ? value : maximum;
                }
            }
            const float row_scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
            memcpy(expanded + (size_t) row * sizeof(float),
                &row_scale, sizeof(row_scale));
            for (uint32_t block = 0; block < linear->blocks; ++block) {
                const uint8_t *scale_source = NULL;
                const uint8_t *value_source = NULL;
                q8_0_source_block(source, activation_row_bytes, row, rows, block,
                    interleave, &scale_source, &value_source);
                uint16_t scale_bits = 0;
                memcpy(&scale_bits, scale_source, sizeof(scale_bits));
                const float block_scale = fp16_to_fp32(scale_bits);
                for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
                    const float scaled = row_scale > 0.0f
                        ? (float) q8_0_source_value(value_source, row, index, interleave) *
                            block_scale / row_scale
                        : 0.0f;
                    long rounded = (long) (scaled >= 0.0f
                        ? scaled + 0.5f : scaled - 0.5f);
                    rounded = rounded < -127L ? -127L : rounded > 127L ? 127L : rounded;
                    activation_q[(size_t) row * linear->device_activation_row_bytes +
                        (size_t) block * Q4_0_BLOCK_ELEMENTS + index] = (uint8_t) (int8_t) rounded;
                }
            }
        }
    } else if (linear->packed_mx) {
        memset(expanded, 0, linear->activation_q_offset +
            (size_t) linear->kernel_rows * linear->device_activation_row_bytes + 16U);
        uint8_t *activation_q = expanded + linear->activation_q_offset;
        const uint32_t interleave = execution->activation_interleave;
        const uint32_t interleaved_rows = interleave == 0U ? 0U : rows - rows % 4U;
        for (uint32_t row = 0; row < interleaved_rows; row += 4U) {
            for (uint32_t block = 0; block < linear->blocks; ++block) {
                const uint8_t *source_block = source +
                    (size_t) row * activation_row_bytes +
                    (size_t) block * 4U * Q8_0_BLOCK_BYTES;
                uint8_t *scale_destinations[4];
                uint8_t *value_destinations[4];
                for (uint32_t group_row = 0; group_row < 4U; ++group_row) {
                    scale_destinations[group_row] = expanded +
                        (size_t) (row + group_row) * linear->activation_scale_row_bytes +
                        (size_t) block * sizeof(uint16_t);
                    value_destinations[group_row] = activation_q +
                        (size_t) (row + group_row) * linear->device_activation_row_bytes +
                        (size_t) block * Q4_0_BLOCK_ELEMENTS;
                }
                pack_interleaved_q8_0x4_block(source_block, scale_destinations,
                    value_destinations, interleave);
            }
        }
        for (uint32_t row = interleaved_rows; row < rows; ++row) {
            for (uint32_t block = 0; block < linear->blocks; ++block) {
                const uint8_t *source_block = source + (size_t) row * activation_row_bytes +
                    (size_t) block * Q8_0_BLOCK_BYTES;
                memcpy(expanded + (size_t) row * linear->activation_scale_row_bytes +
                    (size_t) block * sizeof(uint16_t), source_block, sizeof(uint16_t));
                memcpy(activation_q + (size_t) row * linear->device_activation_row_bytes +
                    (size_t) block * Q4_0_BLOCK_ELEMENTS,
                    source_block + sizeof(uint16_t), Q4_0_BLOCK_ELEMENTS);
            }
        }
    } else if (linear->rows == 1U) {
        for (uint32_t row = 0; row < kernel_rows; ++row) {
            memcpy(expanded + (size_t) row * activation_row_bytes, source, activation_row_bytes);
        }
    } else {
        memcpy(expanded, source, activation_bytes);
        if (kernel_rows > rows) {
            memset(expanded + activation_bytes, 0,
                (size_t) (kernel_rows - rows) * activation_row_bytes);
        }
    }
    const uint64_t input_pack_end = monotonic_ns();
    status = qpu_llama_buffer_cpu_access_end(
        linear->activation_staging, QPU_LLAMA_CPU_ACCESS_WRITE);
    const uint64_t input_copy_end = monotonic_ns();
    uint32_t activation_address = 0;
    uint32_t activation_q_address = 0;
    uint32_t weight_address = 0;
    uint32_t weight_q_address = 0;
    uint32_t staging_address = 0;
    const uint32_t local_column_start =
        execution->column_start - linear->resident_column_start;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(
            linear->activation_staging, 0, &activation_address);
    }
    if (status == QPU_LLAMA_OK && linear->packed_mx) {
        status = qpu_llama_buffer_gpu_address(linear->activation_staging,
            linear->activation_q_offset, &activation_q_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(linear->weight, 0, &weight_address);
    }
    if (status == QPU_LLAMA_OK && linear->packed_mx) {
        status = qpu_llama_buffer_gpu_address(linear->weight,
            linear->weight_q_offset +
                (size_t) local_column_start * sizeof(uint32_t),
            &weight_q_address);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_gpu_address(linear->staging,
            linear->packed_mx ? (size_t) local_column_start * sizeof(float) : 0,
            &staging_address);
    }
    uint32_t uniforms[13] = {0};
    uint32_t uniform_word_count = 0;
    if (linear->packed_mx &&
        linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) {
        const uint32_t packed_uniforms[] = {
            (uint32_t) linear->device_activation_row_bytes,
            activation_q_address,
            (uint32_t) linear->device_weight_row_bytes,
            weight_q_address,
            linear->resident_column_count * (uint32_t) sizeof(float),
            staging_address,
            linear->blocks * 8U,
            activation_address,
            weight_address + local_column_start * (uint32_t) sizeof(float),
        };
        memcpy(uniforms, packed_uniforms, sizeof(packed_uniforms));
        uniform_word_count = sizeof(packed_uniforms) / sizeof(packed_uniforms[0]);
    } else if (linear->packed_mx &&
        linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8) {
        const uint32_t packed_uniforms[] = {
            (uint32_t) linear->device_activation_row_bytes,
            activation_q_address,
            (uint32_t) linear->device_weight_row_bytes,
            weight_q_address,
            linear->resident_column_count * (uint32_t) sizeof(float),
            staging_address,
            linear->blocks,
            (uint32_t) linear->activation_scale_row_bytes,
            activation_address,
            weight_address + local_column_start * (uint32_t) sizeof(float),
        };
        memcpy(uniforms, packed_uniforms, sizeof(packed_uniforms));
        uniform_word_count = sizeof(packed_uniforms) / sizeof(packed_uniforms[0]);
    } else if (linear->packed_mx) {
        const uint32_t packed_uniforms[] = {
            (uint32_t) linear->device_activation_row_bytes,
            activation_q_address,
            (uint32_t) linear->device_weight_row_bytes,
            weight_q_address,
            linear->resident_column_count * (uint32_t) sizeof(float),
            staging_address,
            linear->blocks,
            (uint32_t) linear->activation_scale_row_bytes,
            activation_address,
            linear->resident_column_count * (uint32_t) sizeof(uint16_t),
            weight_address + local_column_start * (uint32_t) sizeof(uint16_t),
        };
        memcpy(uniforms, packed_uniforms, sizeof(packed_uniforms));
        uniform_word_count = sizeof(packed_uniforms) / sizeof(packed_uniforms[0]);
    } else {
        const uint32_t native_uniforms[] = {
            linear->blocks,
            (uint32_t) linear->device_activation_row_bytes,
            activation_address,
            (uint32_t) linear->device_weight_row_bytes,
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
        memcpy(uniforms, native_uniforms, sizeof(native_uniforms));
        uniform_word_count = sizeof(native_uniforms) / sizeof(native_uniforms[0]);
    }
    qpu_llama_buffer *buffers[] = {linear->activation_staging, linear->weight, linear->staging};
    const uint32_t workgroup_x = execution->column_count / Q4_0_OUTPUT_TILE;
    const uint32_t workgroup_y = kernel_rows / row_tile;
    const uint64_t thread_count = (uint64_t) workgroup_x * workgroup_y;
    if (thread_count == 0 || thread_count > UINT32_MAX) {
        pthread_mutex_lock(&linear->mutex);
        linear->busy = false;
        pthread_mutex_unlock(&linear->mutex);
        free(submission);
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    const qpu_llama_dispatch_desc dispatch = {
        .uniforms = uniforms,
        .uniform_word_count = uniform_word_count,
        .buffers = buffers,
        .buffer_count = sizeof(buffers) / sizeof(buffers[0]),
        .local_invocation = {16, 1, 1},
        .workgroup = {workgroup_x, workgroup_y, 1},
        .wgs_per_sg = linear->workgroups_per_supergroup,
        .thread_count = (uint32_t) thread_count,
        .propagate_nan = 0,
        .single_segment = 0,
        .threading = 0,
    };
    const uint64_t submit_start = monotonic_ns();
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_program_submit(
            linear->program, &dispatch, &submission->runtime_submission);
    }
    if (status != QPU_LLAMA_OK) {
        pthread_mutex_lock(&linear->mutex);
        linear->busy = false;
        pthread_mutex_unlock(&linear->mutex);
        free(submission);
        return status;
    }
    const uint64_t submit_end = monotonic_ns();
    submission->linear = linear;
    submission->destination = execution->destination;
    submission->destination_offset = execution->destination_offset;
    submission->rows = rows;
    submission->column_start = execution->column_start;
    submission->column_count = execution->column_count;
    submission->local_column_start = local_column_start;
    submission->complete_start = complete_start;
    submission->input_access_end = input_access_end;
    submission->input_pack_end = input_pack_end;
    submission->input_copy_end = input_copy_end;
    submission->submit_start = submit_start;
    submission->submit_end = submit_end;
    submission->active = true;
    *result = submission;
    return QPU_LLAMA_OK;
}

qpu_llama_status qpu_llama_q4_0_submission_wait(
    qpu_llama_q4_0_submission *submission,
    qpu_llama_q4_0_timing *timing) {
    if (submission == NULL || !submission->active || submission->linear == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_q4_0_linear *linear = submission->linear;
    const uint64_t wait_start = monotonic_ns();
    qpu_llama_status status = qpu_llama_submission_wait(
        submission->runtime_submission);
    const uint64_t submit_end = monotonic_ns();
    uint64_t output_access_end = submit_end;
    uint64_t output_copy_end = submit_end;
    uint64_t copy_end = submit_end;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_begin(
            linear->staging, QPU_LLAMA_CPU_ACCESS_READ);
        output_access_end = monotonic_ns();
    }
    if (status == QPU_LLAMA_OK) {
        uint8_t *destination = submission->destination + submission->destination_offset;
        const uint8_t *staging = qpu_llama_buffer_data(linear->staging);
        for (uint32_t row = 0; row < submission->rows; ++row) {
            const size_t destination_offset =
                ((size_t) row * linear->output_columns + submission->column_start) *
                    sizeof(float);
            const size_t staging_offset =
                ((size_t) row * linear->resident_column_count +
                    submission->local_column_start) * sizeof(float);
            memcpy(destination + destination_offset, staging + staging_offset,
                (size_t) submission->column_count * sizeof(float));
        }
        output_copy_end = monotonic_ns();
        status = qpu_llama_buffer_cpu_access_end(
            linear->staging, QPU_LLAMA_CPU_ACCESS_READ);
        copy_end = monotonic_ns();
    }
    if (timing != NULL) {
        timing->input_access_ns = submission->input_access_end - submission->complete_start;
        timing->input_pack_ns = submission->input_pack_end - submission->input_access_end;
        timing->input_sync_ns = submission->input_copy_end - submission->input_pack_end;
        timing->input_copy_ns = submission->input_copy_end - submission->complete_start;
        timing->submit_ns = submission->submit_end - submission->submit_start;
        timing->wait_ns = submit_end - wait_start;
        timing->submit_wait_ns = submit_end - submission->submit_start;
        timing->output_sync_ns = (output_access_end - submit_end) +
            (copy_end - output_copy_end);
        timing->output_copy_ns = output_copy_end - output_access_end;
        timing->complete_ns = copy_end - submission->complete_start;
    }
    qpu_llama_submission_destroy(submission->runtime_submission);
    submission->runtime_submission = NULL;
    submission->active = false;
    pthread_mutex_lock(&linear->mutex);
    linear->busy = false;
    pthread_mutex_unlock(&linear->mutex);
    return status;
}

qpu_llama_status qpu_llama_q4_0_submission_cpu_fallback(
    qpu_llama_q4_0_submission *submission) {
    if (submission == NULL || submission->linear == NULL ||
        !submission->linear->packed_mx) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_q4_0_linear *linear = submission->linear;
    qpu_llama_status status = qpu_llama_buffer_cpu_access_begin(
        linear->activation_staging, QPU_LLAMA_CPU_ACCESS_READ);
    const bool activation_access = status == QPU_LLAMA_OK;
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_cpu_access_begin(
            linear->weight, QPU_LLAMA_CPU_ACCESS_READ);
    }
    if (status != QPU_LLAMA_OK) {
        if (activation_access) {
            (void) qpu_llama_buffer_cpu_access_end(
                linear->activation_staging, QPU_LLAMA_CPU_ACCESS_READ);
        }
        return status;
    }
    const uint8_t *activation = qpu_llama_buffer_data(linear->activation_staging);
    const uint8_t *activation_q = activation + linear->activation_q_offset;
    const uint8_t *weight = qpu_llama_buffer_data(linear->weight);
    const uint8_t *weight_q = weight + linear->weight_q_offset;
    float *destination = (float *) (submission->destination +
        submission->destination_offset);
    for (uint32_t row = 0; row < submission->rows; ++row) {
        for (uint32_t column = 0; column < submission->column_count; ++column) {
            const uint32_t local_column = submission->local_column_start + column;
            float value = 0.0f;
            int32_t rowcol_dot = 0;
            for (uint32_t block = 0; block < linear->blocks; ++block) {
                int32_t dot = 0;
                const int8_t *activation_block = (const int8_t *) activation_q +
                    (size_t) row * linear->device_activation_row_bytes +
                    (size_t) block * Q4_0_BLOCK_ELEMENTS;
                for (uint32_t index = 0; index < Q4_0_BLOCK_ELEMENTS; ++index) {
                    const size_t qword = (size_t) block * 8U + index / 4U;
                    const int8_t weight_value = (int8_t) weight_q[
                        qword * linear->device_weight_row_bytes +
                        (size_t) local_column * sizeof(uint32_t) + index % 4U];
                    dot += (int32_t) activation_block[index] * (int32_t) weight_value;
                }
                if (linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) {
                    rowcol_dot += dot;
                } else {
                    uint16_t activation_scale_bits = 0;
                    memcpy(&activation_scale_bits,
                        activation + (size_t) row * linear->activation_scale_row_bytes +
                            (size_t) block * sizeof(uint16_t),
                        sizeof(activation_scale_bits));
                    if (linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8) {
                        value += (float) dot * fp16_to_fp32(activation_scale_bits);
                    } else {
                        uint16_t weight_scale_bits = 0;
                        memcpy(&weight_scale_bits,
                            weight + ((size_t) block * linear->resident_column_count +
                                local_column) * sizeof(uint16_t),
                            sizeof(weight_scale_bits));
                        value += (float) dot * fp16_to_fp32(activation_scale_bits) *
                            fp16_to_fp32(weight_scale_bits);
                    }
                }
            }
            if (linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8) {
                float row_scale = 0.0f;
                float column_scale = 0.0f;
                memcpy(&row_scale, activation + (size_t) row * sizeof(float),
                    sizeof(row_scale));
                memcpy(&column_scale, weight + (size_t) local_column * sizeof(float),
                    sizeof(column_scale));
                value = (float) rowcol_dot * row_scale * column_scale;
            } else if (linear->weight_mode == QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8) {
                float column_scale = 0.0f;
                memcpy(&column_scale, weight + (size_t) local_column * sizeof(float),
                    sizeof(column_scale));
                value *= column_scale;
            }
            destination[(size_t) row * linear->output_columns +
                submission->column_start + column] = value;
        }
    }
    const qpu_llama_status weight_end = qpu_llama_buffer_cpu_access_end(
        linear->weight, QPU_LLAMA_CPU_ACCESS_READ);
    const qpu_llama_status activation_end = qpu_llama_buffer_cpu_access_end(
        linear->activation_staging, QPU_LLAMA_CPU_ACCESS_READ);
    return weight_end != QPU_LLAMA_OK ? weight_end : activation_end;
}

void qpu_llama_q4_0_submission_destroy(qpu_llama_q4_0_submission *submission) {
    if (submission == NULL) {
        return;
    }
    if (submission->active) {
        (void) qpu_llama_q4_0_submission_wait(submission, NULL);
    }
    free(submission);
}

qpu_llama_status qpu_llama_q4_0_linear_execute(
    qpu_llama_q4_0_linear *linear,
    const qpu_llama_q4_0_execution *execution,
    qpu_llama_q4_0_timing *timing) {
    qpu_llama_q4_0_submission *submission = NULL;
    qpu_llama_status status = qpu_llama_q4_0_linear_submit(
        linear, execution, &submission);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_q4_0_submission_wait(submission, timing);
    }
    qpu_llama_q4_0_submission_destroy(submission);
    return status;
}

size_t qpu_llama_q4_0_linear_resident_bytes(const qpu_llama_q4_0_linear *linear) {
    return linear != NULL ? linear->resident_bytes : 0;
}
