#include "qpu_llama_q4_k.h"
#include "ggml-q4-k-q8-k-m4.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    INPUT_COLUMNS = 256,
    OUTPUT_COLUMNS = 32,
    ROWS = 4,
    RESIDENT_START = 16,
    RESIDENT_COLUMNS = 16,
    Q4_K_BYTES = 144,
    Q8_K_BYTES = 292,
};

int main(void) {
    uint8_t weights[OUTPUT_COLUMNS][Q4_K_BYTES] = {{0}};
    uint8_t activation[ROWS][Q8_K_BYTES] = {{0}};
    const uint16_t one_fp16 = UINT16_C(0x3c00);
    const float one_fp32 = 1.0f;
    for (unsigned int row = 0; row < ROWS; ++row) {
        const int8_t value = (int8_t) (row + 1U);
        memcpy(activation[row], &one_fp32, sizeof(one_fp32));
        memset(activation[row] + 4, (uint8_t) value, INPUT_COLUMNS);
        const int16_t sum = (int16_t) (16 * value);
        for (unsigned int group = 0; group < 16; ++group) {
            memcpy(activation[row] + 260 + 2 * group, &sum, sizeof(sum));
        }
    }
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        memcpy(weights[column], &one_fp16, sizeof(one_fp16));
        weights[column][4] = 1;
        weights[column][5] = 1;
        weights[column][6] = 1;
        weights[column][7] = 1;
        weights[column][12] = 1;
        weights[column][13] = 1;
        weights[column][14] = 1;
        weights[column][15] = 1;
        const uint8_t code = (uint8_t) (column % 16U);
        memset(weights[column] + 16, code | (uint8_t) (code << 4U), 128);
    }

    qpu_llama_context *context = NULL;
    qpu_llama_status status = qpu_llama_context_create(NULL, &context);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "context: %s\n", qpu_llama_status_string(status));
        return 1;
    }
    const qpu_llama_q4_k_linear_desc desc = {
        .weights = weights,
        .weight_size = sizeof(weights),
        .input_columns = INPUT_COLUMNS,
        .output_columns = OUTPUT_COLUMNS,
        .resident_column_start = RESIDENT_START,
        .resident_column_count = RESIDENT_COLUMNS,
        .expected_source_hash = qpu_ggml_q4_k_q8_k_m4_source_hash,
    };
    qpu_llama_q4_k_linear *linear = NULL;
    status = qpu_llama_q4_k_linear_prepare(context, &desc, &linear);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "prepare: %s (%s)\n", qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        qpu_llama_context_destroy(context);
        return 1;
    }
    float destination[ROWS][OUTPUT_COLUMNS];
    for (unsigned int row = 0; row < ROWS; ++row) {
        for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
            destination[row][column] = 123456.0f;
        }
    }
    const qpu_llama_q4_k_execution execution = {
        .activation = activation,
        .activation_size = sizeof(activation),
        .activation_offset = 0,
        .destination = destination,
        .destination_size = sizeof(destination),
        .destination_offset = 0,
        .column_start = RESIDENT_START,
        .column_count = RESIDENT_COLUMNS,
    };
    qpu_llama_q4_k_timing timing;
    status = qpu_llama_q4_k_linear_execute(linear, &execution, &timing);
    int result = 0;
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "execute: %s (%s)\n", qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        result = 1;
    }
    for (unsigned int row = 0; row < ROWS; ++row) {
        for (unsigned int column = 0; column < RESIDENT_START; ++column) {
            if (destination[row][column] != 123456.0f) {
                fprintf(stderr, "CPU interval changed at row %u column %u\n", row, column);
                result = 1;
            }
        }
        for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
            const float expected = (float) (256U * (row + 1U) * (column % 16U));
            if (destination[row][column] != expected) {
                fprintf(stderr, "row %u column %u: expected %.0f, got %.0f\n",
                    row, column, expected, destination[row][column]);
                result = 1;
            }
        }
    }
    printf("input_copy_ns=%llu submit_wait_ns=%llu output_copy_ns=%llu complete_ns=%llu\n",
        (unsigned long long) timing.input_copy_ns,
        (unsigned long long) timing.submit_wait_ns,
        (unsigned long long) timing.output_copy_ns,
        (unsigned long long) timing.complete_ns);
    qpu_llama_q4_k_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    return result;
}
