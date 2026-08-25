#include "qpu_llama_q4_0.h"
#include "ggml-q4-0-q8-0-mx.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    BLOCKS = 2,
    INPUT_COLUMNS = 32 * BLOCKS,
    OUTPUT_COLUMNS = 32,
    ROWS = 5,
    RESIDENT_START = 16,
    RESIDENT_COLUMNS = 16,
    Q4_BLOCK_BYTES = 18,
    Q8_BLOCK_BYTES = 34,
    WEIGHT_ROW_BYTES = Q4_BLOCK_BYTES * BLOCKS,
    ACTIVATION_ROW_BYTES = Q8_BLOCK_BYTES * BLOCKS,
};

static void pack_q8_0x4(
    uint8_t destination[ROWS * ACTIVATION_ROW_BYTES],
    const uint8_t source[ROWS][ACTIVATION_ROW_BYTES],
    unsigned int interleave) {
    memset(destination, 0, ROWS * ACTIVATION_ROW_BYTES);
    const unsigned int interleaved_rows = ROWS - ROWS % 4U;
    for (unsigned int row_base = 0; row_base < interleaved_rows; row_base += 4U) {
        uint8_t *group = destination + row_base * ACTIVATION_ROW_BYTES;
        for (unsigned int block = 0; block < BLOCKS; ++block) {
            uint8_t *packed = group + block * 4U * Q8_BLOCK_BYTES;
            for (unsigned int lane = 0; lane < 4U; ++lane) {
                const uint8_t *native = source[row_base + lane] + block * Q8_BLOCK_BYTES;
                memcpy(packed + lane * sizeof(uint16_t), native, sizeof(uint16_t));
                for (unsigned int index = 0; index < 32U; ++index) {
                    const unsigned int packed_index =
                        (index / interleave) * 4U * interleave +
                        lane * interleave + index % interleave;
                    packed[4U * sizeof(uint16_t) + packed_index] = native[2U + index];
                }
            }
        }
    }
    for (unsigned int row = interleaved_rows; row < ROWS; ++row) {
        memcpy(destination + row * ACTIVATION_ROW_BYTES,
            source[row], ACTIVATION_ROW_BYTES);
    }
}

int main(void) {
    uint8_t weights[OUTPUT_COLUMNS][WEIGHT_ROW_BYTES] = {{0}};
    uint8_t native_activation[ROWS][ACTIVATION_ROW_BYTES] = {{0}};
    uint8_t activation[ROWS * ACTIVATION_ROW_BYTES] = {0};
    float expected[ROWS][OUTPUT_COLUMNS] = {{0}};
    const uint16_t one_fp16 = UINT16_C(0x3c00);
    const uint16_t half_fp16 = UINT16_C(0x3800);
    for (unsigned int row = 0; row < ROWS; ++row) {
        for (unsigned int block = 0; block < BLOCKS; ++block) {
            const uint16_t scale = (row + block) % 2U == 0U ? one_fp16 : half_fp16;
            uint8_t *activation_block =
                native_activation[row] + block * Q8_BLOCK_BYTES;
            memcpy(activation_block, &scale, sizeof(scale));
            for (unsigned int index = 0; index < 32U; ++index) {
                activation_block[2U + index] = (uint8_t) (int8_t)
                    ((int) index - 12 - (int) row + (int) block * 3);
            }
        }
    }
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        for (unsigned int block = 0; block < BLOCKS; ++block) {
            const uint16_t scale = (column + block) % 2U == 0U ? one_fp16 : half_fp16;
            uint8_t *weight_block = weights[column] + block * Q4_BLOCK_BYTES;
            memcpy(weight_block, &scale, sizeof(scale));
            for (unsigned int index = 0; index < 16U; ++index) {
                const uint8_t low = (uint8_t) ((index + column + block) % 16U);
                const uint8_t high =
                    (uint8_t) ((index * 3U + column + block * 2U) % 16U);
                weight_block[2U + index] = low | (uint8_t) (high << 4U);
                for (unsigned int row = 0; row < ROWS; ++row) {
                    const uint8_t *activation_block =
                        native_activation[row] + block * Q8_BLOCK_BYTES;
                    const float combined_scale =
                        ((row + block) % 2U == 0U ? 1.0f : 0.5f) *
                        ((column + block) % 2U == 0U ? 1.0f : 0.5f);
                    expected[row][column] += combined_scale * (float) (
                        ((int32_t) low - 8) *
                            (int32_t) (int8_t) activation_block[2U + index] +
                        ((int32_t) high - 8) *
                            (int32_t) (int8_t) activation_block[18U + index]);
                }
            }
        }
    }

    qpu_llama_context *context = NULL;
    qpu_llama_status status = qpu_llama_context_create(NULL, &context);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "context: %s\n", qpu_llama_status_string(status));
        return 1;
    }
    const qpu_llama_q4_0_linear_desc desc = {
        .weights = weights,
        .weight_size = sizeof(weights),
        .input_columns = INPUT_COLUMNS,
        .output_columns = OUTPUT_COLUMNS,
        .rows = ROWS,
        .resident_column_start = RESIDENT_START,
        .resident_column_count = RESIDENT_COLUMNS,
        .expected_source_hash = qpu_ggml_q4_0_q8_0_mx_source_hash,
    };
    qpu_llama_q4_0_linear *linear = NULL;
    status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "prepare: %s (%s)\n", qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        qpu_llama_context_destroy(context);
        return 1;
    }
    int result = 0;
    for (unsigned int interleave = 4U; interleave <= 8U; interleave += 4U) {
        pack_q8_0x4(activation, native_activation, interleave);
        float destination[ROWS][OUTPUT_COLUMNS];
        for (unsigned int row = 0; row < ROWS; ++row) {
            for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
                destination[row][column] = 123456.0f;
            }
        }
        const qpu_llama_q4_0_execution execution = {
            .activation = activation,
            .activation_size = sizeof(activation),
            .activation_offset = 0,
            .destination = destination,
            .destination_size = sizeof(destination),
            .destination_offset = 0,
            .column_start = RESIDENT_START,
            .column_count = RESIDENT_COLUMNS,
            .rows = ROWS,
            .activation_interleave = interleave,
        };
        qpu_llama_q4_0_submission *submission = NULL;
        qpu_llama_q4_0_timing timing = {0};
        status = qpu_llama_q4_0_linear_submit(linear, &execution, &submission);
        if (status == QPU_LLAMA_OK) {
            status = qpu_llama_q4_0_submission_wait(submission, &timing);
        }
        if (status != QPU_LLAMA_OK) {
            fprintf(stderr, "interleave %u: %s (%s)\n", interleave,
                qpu_llama_status_string(status), qpu_llama_context_last_error(context));
            result = 1;
        }
        for (unsigned int row = 0; row < ROWS; ++row) {
            for (unsigned int column = 0; column < RESIDENT_START; ++column) {
                if (destination[row][column] != 123456.0f) {
                    fprintf(stderr,
                        "interleave %u: CPU interval changed at row %u column %u\n",
                        interleave, row, column);
                    result = 1;
                }
            }
            for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
                if (destination[row][column] != expected[row][column]) {
                    fprintf(stderr,
                        "interleave %u row %u column %u: expected %.2f, got %.2f\n",
                        interleave, row, column,
                        expected[row][column], destination[row][column]);
                    result = 1;
                }
                destination[row][column] = -98765.0f;
            }
        }
        status = qpu_llama_q4_0_submission_cpu_fallback(submission);
        if (status != QPU_LLAMA_OK) {
            fprintf(stderr, "fallback interleave %u: %s\n", interleave,
                qpu_llama_status_string(status));
            result = 1;
        }
        for (unsigned int row = 0; row < ROWS; ++row) {
            for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
                if (destination[row][column] != expected[row][column]) {
                    fprintf(stderr,
                        "fallback interleave %u row %u column %u: expected %.2f, got %.2f\n",
                        interleave, row, column,
                        expected[row][column], destination[row][column]);
                    result = 1;
                }
            }
        }
        printf("rows=%u blocks=%u interleave=%u input_copy_ns=%llu "
            "submit_wait_ns=%llu output_copy_ns=%llu complete_ns=%llu\n",
            ROWS, BLOCKS, interleave,
            (unsigned long long) timing.input_copy_ns,
            (unsigned long long) timing.submit_wait_ns,
            (unsigned long long) timing.output_copy_ns,
            (unsigned long long) timing.complete_ns);
        qpu_llama_q4_0_submission_destroy(submission);
        if (result != 0) {
            break;
        }
    }
    qpu_llama_q4_0_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    return result;
}
