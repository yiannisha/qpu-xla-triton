#include "qpu_llama_runtime.h"
#include "w8a8-gemv.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    LOGICAL_REDUCTION = 16,
    REDUCTION_WORDS = LOGICAL_REDUCTION / 4,
    OUTPUT_COLUMNS = 16,
};

static uint32_t pack_four(const int8_t values[4]) {
    uint32_t packed = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        packed |= (uint32_t) (uint8_t) values[index] << (index * 8U);
    }
    return packed;
}

static int fail(qpu_llama_context *context, qpu_llama_status status, const char *operation) {
    fprintf(stderr, "%s: %s (%s)\n", operation, qpu_llama_status_string(status),
        qpu_llama_context_last_error(context));
    return 1;
}

int main(void) {
    qpu_llama_context *context = NULL;
    qpu_llama_buffer *source = NULL;
    qpu_llama_buffer *weight = NULL;
    qpu_llama_buffer *destination = NULL;
    qpu_llama_program *program = NULL;
    qpu_llama_status status = qpu_llama_context_create(NULL, &context);
    if (status != QPU_LLAMA_OK) {
        return fail(context, status, "context");
    }
    status = qpu_llama_buffer_create(context, REDUCTION_WORDS * sizeof(uint32_t), &source);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create(
            context,
            REDUCTION_WORDS * OUTPUT_COLUMNS * sizeof(uint32_t),
            &weight);
    }
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_buffer_create(context, OUTPUT_COLUMNS * sizeof(int32_t), &destination);
    }
    if (status != QPU_LLAMA_OK) {
        const int result = fail(context, status, "buffers");
        qpu_llama_buffer_destroy(destination);
        qpu_llama_buffer_destroy(weight);
        qpu_llama_buffer_destroy(source);
        qpu_llama_context_destroy(context);
        return result;
    }

    int8_t activations[LOGICAL_REDUCTION];
    int8_t weights[LOGICAL_REDUCTION][OUTPUT_COLUMNS];
    int32_t expected[OUTPUT_COLUMNS] = {0};
    for (unsigned int reduction = 0; reduction < LOGICAL_REDUCTION; ++reduction) {
        activations[reduction] = (int8_t) reduction - 8;
        for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
            weights[reduction][column] = (int8_t) ((reduction * 3U + column * 5U) % 15U) - 7;
            expected[column] += (int32_t) activations[reduction] * weights[reduction][column];
        }
    }
    uint32_t *packed_source = qpu_llama_buffer_data(source);
    uint32_t *packed_weight = qpu_llama_buffer_data(weight);
    int32_t *actual = qpu_llama_buffer_data(destination);
    for (unsigned int word = 0; word < REDUCTION_WORDS; ++word) {
        packed_source[word] = pack_four(&activations[word * 4U]);
        for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
            int8_t values[4];
            for (unsigned int lane = 0; lane < 4; ++lane) {
                values[lane] = weights[word * 4U + lane][column];
            }
            packed_weight[word * OUTPUT_COLUMNS + column] = pack_four(values);
        }
    }
    memset(actual, 0xa5, OUTPUT_COLUMNS * sizeof(*actual));

    const qpu_llama_program_desc program_desc = {
        .code = qpu_w8a8_gemv,
        .code_size = sizeof(qpu_w8a8_gemv),
        .compiled_source_hash = qpu_w8a8_gemv_source_hash,
        .expected_source_hash = qpu_w8a8_gemv_source_hash,
        .binary_sha256 = qpu_w8a8_gemv_binary_hash,
        .uniform_word_count = 5,
    };
    status = qpu_llama_program_create(context, &program_desc, &program);
    if (status != QPU_LLAMA_OK) {
        return fail(context, status, "program");
    }
    uint32_t source_address = 0;
    uint32_t weight_address = 0;
    uint32_t destination_address = 0;
    if (qpu_llama_buffer_gpu_address(source, 0, &source_address) != QPU_LLAMA_OK ||
        qpu_llama_buffer_gpu_address(weight, 0, &weight_address) != QPU_LLAMA_OK ||
        qpu_llama_buffer_gpu_address(destination, 0, &destination_address) != QPU_LLAMA_OK) {
        return 1;
    }
    const uint32_t uniforms[] = {
        REDUCTION_WORDS,
        source_address,
        OUTPUT_COLUMNS * sizeof(uint32_t),
        weight_address,
        destination_address,
    };
    qpu_llama_buffer *buffers[] = {source, weight, destination};
    const qpu_llama_dispatch_desc dispatch = {
        .uniforms = uniforms,
        .uniform_word_count = sizeof(uniforms) / sizeof(uniforms[0]),
        .buffers = buffers,
        .buffer_count = sizeof(buffers) / sizeof(buffers[0]),
        .local_invocation = {16, 1, 1},
        .workgroup = {1, 1, 1},
        .wgs_per_sg = 24,
        .thread_count = 1,
        .propagate_nan = 0,
        .single_segment = 0,
        .threading = 0,
    };
    status = qpu_llama_program_execute(program, &dispatch);
    int result = 0;
    if (status != QPU_LLAMA_OK) {
        result = fail(context, status, "execute");
    } else {
        for (unsigned int column = 0; column < OUTPUT_COLUMNS; ++column) {
            if (actual[column] != expected[column]) {
                fprintf(stderr, "column %u: expected %d, got %d\n", column, expected[column], actual[column]);
                result = 1;
            }
        }
    }
    qpu_llama_program_destroy(program);
    qpu_llama_buffer_destroy(destination);
    qpu_llama_buffer_destroy(weight);
    qpu_llama_buffer_destroy(source);
    qpu_llama_context_destroy(context);
    if (result == 0) {
        puts("native W8A8 GEMV hardware differential passed");
    }
    return result;
}
