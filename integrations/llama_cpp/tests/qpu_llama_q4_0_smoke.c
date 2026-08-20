#include "qpu_llama_q4_0.h"
#include "ggml-q4-0-q8-0-m1.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    INPUT_COLUMNS = 32,
    OUTPUT_COLUMNS = 32,
    RESIDENT_START = 16,
    RESIDENT_COLUMNS = 16,
};

int main(void) {
    uint8_t weights[OUTPUT_COLUMNS][18];
    uint8_t activation[34];
    int32_t expected[OUTPUT_COLUMNS] = {0};
    memset(weights, 0, sizeof(weights));
    memset(activation, 0, sizeof(activation));
    const uint16_t activation_scale = UINT16_C(0x3c00);
    memcpy(activation, &activation_scale, sizeof(activation_scale));
    for (unsigned int index = 0; index < INPUT_COLUMNS; ++index) {
        activation[2 + index] = (uint8_t) (int8_t) ((int) index - 16);
    }
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        weights[column][0] = 0x00;
        weights[column][1] = 0x3c;
        for (unsigned int index = 0; index < INPUT_COLUMNS / 2; ++index) {
            const uint8_t low = (uint8_t) ((index + column) % 16U);
            const uint8_t high = (uint8_t) ((index * 3U + column) % 16U);
            weights[column][2 + index] = low | (uint8_t) (high << 4U);
            expected[column] += ((int32_t) low - 8) * (int32_t) (int8_t) activation[2 + index];
            expected[column] += ((int32_t) high - 8) * (int32_t) (int8_t) activation[18 + index];
        }
    }

    qpu_llama_context *context = NULL;
    qpu_llama_status status = qpu_llama_context_create(NULL, &context);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "context: %s\n", qpu_llama_status_string(status));
        return 1;
    }
    float destination[OUTPUT_COLUMNS];
    for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
        destination[column] = 123456.0f;
    }
    const qpu_llama_q4_0_linear_desc desc = {
        .weights = weights,
        .weight_size = sizeof(weights),
        .input_columns = INPUT_COLUMNS,
        .output_columns = OUTPUT_COLUMNS,
        .rows = 1,
        .resident_column_start = RESIDENT_START,
        .resident_column_count = RESIDENT_COLUMNS,
        .expected_source_hash = qpu_ggml_q4_0_q8_0_m1_source_hash,
    };
    qpu_llama_q4_0_linear *linear = NULL;
    qpu_llama_q4_0_linear_desc invalid_desc = desc;
    invalid_desc.expected_source_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    status = qpu_llama_q4_0_linear_prepare(context, &invalid_desc, &linear);
    if (status != QPU_LLAMA_HASH_MISMATCH || linear != NULL) {
        fprintf(stderr, "invalid source hash did not fail safely\n");
        return 1;
    }
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_ALLOCATION);
    status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_NONE);
    if (status != QPU_LLAMA_ALLOCATION_FAILED || linear != NULL) {
        fprintf(stderr, "injected allocation failure did not clean up\n");
        return 1;
    }
    status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "prepare: %s (%s)\n", qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        return 1;
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
    for (unsigned int column = 0; column < RESIDENT_START; ++column) {
        if (destination[column] != 123456.0f) {
            fprintf(stderr, "CPU partition column %u was modified\n", column);
            result = 1;
        }
    }
    for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
        if (destination[column] != (float) expected[column]) {
            fprintf(stderr, "column %u: expected %d, got %.0f\n", column, expected[column], destination[column]);
            result = 1;
        }
    }
    for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
        destination[column] = -777.0f;
    }
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_SUBMISSION);
    status = qpu_llama_q4_0_linear_execute(linear, &execution, NULL);
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_NONE);
    if (status != QPU_LLAMA_SUBMISSION_FAILED) {
        fprintf(stderr, "injected submission failure returned %s\n", qpu_llama_status_string(status));
        result = 1;
    }
    for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
        if (destination[column] != -777.0f) {
            fprintf(stderr, "submission failure exposed partial output at column %u\n", column);
            result = 1;
        }
    }
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_WAIT);
    status = qpu_llama_q4_0_linear_execute(linear, &execution, NULL);
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_NONE);
    if (status != QPU_LLAMA_TIMEOUT) {
        fprintf(stderr, "injected wait failure returned %s\n", qpu_llama_status_string(status));
        result = 1;
    }
    for (unsigned int column = RESIDENT_START; column < OUTPUT_COLUMNS; ++column) {
        if (destination[column] != -777.0f) {
            fprintf(stderr, "wait failure exposed partial output at column %u\n", column);
            result = 1;
        }
    }
    printf("submit_wait_ns=%llu output_copy_ns=%llu complete_ns=%llu resident_bytes=%zu\n",
        (unsigned long long) timing.submit_wait_ns,
        (unsigned long long) timing.output_copy_ns,
        (unsigned long long) timing.complete_ns,
        qpu_llama_q4_0_linear_resident_bytes(linear));
    qpu_llama_q4_0_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    return result;
}
