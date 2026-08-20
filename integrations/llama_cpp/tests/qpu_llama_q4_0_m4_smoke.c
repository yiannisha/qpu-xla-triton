#include "qpu_llama_q4_0.h"
#include "ggml-q4-0-q8-0-m4.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    INPUT_COLUMNS = 32,
    OUTPUT_COLUMNS = 32,
    ROWS = 4,
    RESIDENT_START = 16,
    RESIDENT_COLUMNS = 16,
};

int main(void) {
    uint8_t weights[OUTPUT_COLUMNS][18] = {{0}};
    uint8_t activation[ROWS][34] = {{0}};
    int32_t expected[ROWS][OUTPUT_COLUMNS] = {{0}};
    const uint16_t one_fp16 = UINT16_C(0x3c00);
    for (unsigned int row = 0; row < ROWS; ++row) {
        memcpy(activation[row], &one_fp16, sizeof(one_fp16));
        for (unsigned int index = 0; index < INPUT_COLUMNS; ++index) {
            activation[row][2 + index] = (uint8_t) (int8_t) ((int) index - 12 - (int) row);
        }
    }
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        memcpy(weights[column], &one_fp16, sizeof(one_fp16));
        for (unsigned int index = 0; index < INPUT_COLUMNS / 2; ++index) {
            const uint8_t low = (uint8_t) ((index + column) % 16U);
            const uint8_t high = (uint8_t) ((index * 3U + column) % 16U);
            weights[column][2 + index] = low | (uint8_t) (high << 4U);
            for (unsigned int row = 0; row < ROWS; ++row) {
                expected[row][column] +=
                    ((int32_t) low - 8) * (int32_t) (int8_t) activation[row][2 + index];
                expected[row][column] +=
                    ((int32_t) high - 8) * (int32_t) (int8_t) activation[row][18 + index];
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
        .expected_source_hash = qpu_ggml_q4_0_q8_0_m4_source_hash,
    };
    qpu_llama_q4_0_linear *linear = NULL;
    status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
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
    const qpu_llama_q4_0_execution execution = {
        .activation = activation,
        .activation_size = sizeof(activation),
        .activation_offset = 0,
        .destination = destination,
        .destination_size = sizeof(destination),
        .destination_offset = 0,
        .column_start = RESIDENT_START,
        .column_count = RESIDENT_COLUMNS,
    };
    qpu_llama_q4_0_timing timing;
    status = qpu_llama_q4_0_linear_execute(linear, &execution, &timing);
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
            if (destination[row][column] != (float) expected[row][column]) {
                fprintf(stderr, "row %u column %u: expected %d, got %.0f\n",
                    row, column, expected[row][column], destination[row][column]);
                result = 1;
            }
        }
    }
    printf("input_copy_ns=%llu submit_wait_ns=%llu output_copy_ns=%llu complete_ns=%llu\n",
        (unsigned long long) timing.input_copy_ns,
        (unsigned long long) timing.submit_wait_ns,
        (unsigned long long) timing.output_copy_ns,
        (unsigned long long) timing.complete_ns);
    qpu_llama_q4_0_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    return result;
}
