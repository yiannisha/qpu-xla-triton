#include "qpu_llama_q8_0.h"
#include "ggml-q8-0-q8-0-m4.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    INPUT_COLUMNS = 32,
    OUTPUT_COLUMNS = 32,
    ROWS = 4,
    RESIDENT_START = 16,
    RESIDENT_COLUMNS = 16,
    Q8_0_BYTES = 34,
};

int main(void) {
    uint8_t weights[OUTPUT_COLUMNS][Q8_0_BYTES] = {{0}};
    uint8_t activation[ROWS][Q8_0_BYTES] = {{0}};
    const uint16_t one_fp16 = UINT16_C(0x3c00);
    for (unsigned int row = 0; row < ROWS; ++row) {
        memcpy(activation[row], &one_fp16, sizeof(one_fp16));
        memset(activation[row] + 2, (uint8_t) (int8_t) (row + 1U), INPUT_COLUMNS);
    }
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        memcpy(weights[column], &one_fp16, sizeof(one_fp16));
        memset(weights[column] + 2, (uint8_t) (int8_t) ((int) column - 16), INPUT_COLUMNS);
    }
    qpu_llama_context *context = NULL;
    qpu_llama_status status = qpu_llama_context_create(NULL, &context);
    if (status != QPU_LLAMA_OK) {
        return 1;
    }
    const qpu_llama_q8_0_linear_desc desc = {
        .weights = weights,
        .weight_size = sizeof(weights),
        .input_columns = INPUT_COLUMNS,
        .output_columns = OUTPUT_COLUMNS,
        .resident_column_start = RESIDENT_START,
        .resident_column_count = RESIDENT_COLUMNS,
        .expected_source_hash = qpu_ggml_q8_0_q8_0_m4_source_hash,
    };
    qpu_llama_q8_0_linear *linear = NULL;
    status = qpu_llama_q8_0_linear_prepare(context, &desc, &linear);
    if (status != QPU_LLAMA_OK) {
        qpu_llama_context_destroy(context);
        return 1;
    }
    float destination[ROWS][OUTPUT_COLUMNS];
    for (unsigned int row = 0; row < ROWS; ++row) {
        for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
            destination[row][column] = 123456.0f;
        }
    }
    const qpu_llama_q8_0_execution execution = {
        .activation = activation,
        .activation_size = sizeof(activation),
        .destination = destination,
        .destination_size = sizeof(destination),
        .column_start = RESIDENT_START,
        .column_count = RESIDENT_COLUMNS,
    };
    status = qpu_llama_q8_0_linear_execute(linear, &execution, NULL);
    int result = status == QPU_LLAMA_OK ? 0 : 1;
    for (unsigned int row = 0; row < ROWS; ++row) {
        for (unsigned int column = 0; column < RESIDENT_START; ++column) {
            if (destination[row][column] != 123456.0f) {
                result = 1;
            }
        }
        for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
            const float expected = (float) (32 * (int) (row + 1U) * ((int) column - 16));
            if (destination[row][column] != expected) {
                fprintf(stderr, "row %u column %u: expected %.0f, got %.0f\n",
                    row, column, expected, destination[row][column]);
                result = 1;
            }
        }
    }
    qpu_llama_q8_0_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    return result;
}
