#include "qpu_llama_q4_0.h"
#include "qpu_llama_q4_k.h"
#include "qpu_llama_q6_k.h"
#include "qpu_llama_q8_0.h"

#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-m4.h"
#include "ggml-q4-k-q8-k-m4.h"
#include "ggml-q6-k-q8-k-m4.h"
#include "ggml-q8-0-q8-0-m4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GUARD_BYTES = 64,
    OUTPUT_COLUMNS = 48,
    OUTPUT_TILE = 16,
    ROWS = 4,
    REPEAT_COUNT = 100,
};

static const float OUTPUT_SENTINEL = 123456.0f;
static const uint8_t GUARD_VALUE = UINT8_C(0xa5);

typedef qpu_llama_status (*prepare_fn)(
    qpu_llama_context *context,
    const void *weights,
    size_t weight_size,
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t resident_start,
    uint32_t resident_count,
    const char *source_hash,
    void **result);

typedef qpu_llama_status (*execute_fn)(
    void *linear,
    const void *activation,
    size_t activation_size,
    size_t activation_offset,
    void *destination,
    size_t destination_size,
    size_t destination_offset,
    uint32_t column_start,
    uint32_t column_count);

typedef void (*destroy_fn)(void *linear);

typedef struct format_ops {
    const char *name;
    uint32_t rows;
    uint32_t input_columns;
    size_t weight_block_bytes;
    size_t activation_block_bytes;
    const char *source_hash;
    prepare_fn prepare;
    execute_fn execute;
    destroy_fn destroy;
} format_ops;

static qpu_llama_status prepare_q4_0_rows(
    qpu_llama_context *context,
    const void *weights,
    size_t weight_size,
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t resident_start,
    uint32_t resident_count,
    const char *source_hash,
    uint32_t rows,
    void **result) {
    const qpu_llama_q4_0_linear_desc desc = {
        .weights = weights,
        .weight_size = weight_size,
        .input_columns = input_columns,
        .output_columns = output_columns,
        .rows = rows,
        .resident_column_start = resident_start,
        .resident_column_count = resident_count,
        .expected_source_hash = source_hash,
    };
    qpu_llama_q4_0_linear *linear = NULL;
    const qpu_llama_status status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
    *result = linear;
    return status;
}

static qpu_llama_status prepare_q4_0_m1(
    qpu_llama_context *context,
    const void *weights,
    size_t weight_size,
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t resident_start,
    uint32_t resident_count,
    const char *source_hash,
    void **result) {
    return prepare_q4_0_rows(
        context,
        weights,
        weight_size,
        input_columns,
        output_columns,
        resident_start,
        resident_count,
        source_hash,
        1,
        result);
}

static qpu_llama_status prepare_q4_0_m4(
    qpu_llama_context *context,
    const void *weights,
    size_t weight_size,
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t resident_start,
    uint32_t resident_count,
    const char *source_hash,
    void **result) {
    return prepare_q4_0_rows(
        context,
        weights,
        weight_size,
        input_columns,
        output_columns,
        resident_start,
        resident_count,
        source_hash,
        ROWS,
        result);
}

static qpu_llama_status execute_q4_0(
    void *linear,
    const void *activation,
    size_t activation_size,
    size_t activation_offset,
    void *destination,
    size_t destination_size,
    size_t destination_offset,
    uint32_t column_start,
    uint32_t column_count) {
    const qpu_llama_q4_0_execution execution = {
        .activation = activation,
        .activation_size = activation_size,
        .activation_offset = activation_offset,
        .destination = destination,
        .destination_size = destination_size,
        .destination_offset = destination_offset,
        .column_start = column_start,
        .column_count = column_count,
    };
    return qpu_llama_q4_0_linear_execute(linear, &execution, NULL);
}

static void destroy_q4_0(void *linear) {
    qpu_llama_q4_0_linear_destroy(linear);
}

#define DEFINE_M4_ADAPTERS(NAME, LINEAR_TYPE, DESC_TYPE, EXECUTION_TYPE, PREPARE, EXECUTE, DESTROY) \
    static qpu_llama_status prepare_##NAME(                                                   \
        qpu_llama_context *context,                                                           \
        const void *weights,                                                                  \
        size_t weight_size,                                                                   \
        uint32_t input_columns,                                                               \
        uint32_t output_columns,                                                              \
        uint32_t resident_start,                                                              \
        uint32_t resident_count,                                                              \
        const char *source_hash,                                                              \
        void **result) {                                                                      \
        const DESC_TYPE desc = {                                                              \
            .weights = weights,                                                               \
            .weight_size = weight_size,                                                       \
            .input_columns = input_columns,                                                   \
            .output_columns = output_columns,                                                 \
            .resident_column_start = resident_start,                                          \
            .resident_column_count = resident_count,                                          \
            .expected_source_hash = source_hash,                                              \
        };                                                                                    \
        LINEAR_TYPE *linear = NULL;                                                           \
        const qpu_llama_status status = PREPARE(context, &desc, &linear);                     \
        *result = linear;                                                                     \
        return status;                                                                        \
    }                                                                                         \
    static qpu_llama_status execute_##NAME(                                                   \
        void *linear,                                                                         \
        const void *activation,                                                               \
        size_t activation_size,                                                               \
        size_t activation_offset,                                                             \
        void *destination,                                                                    \
        size_t destination_size,                                                              \
        size_t destination_offset,                                                            \
        uint32_t column_start,                                                                \
        uint32_t column_count) {                                                              \
        const EXECUTION_TYPE execution = {                                                    \
            .activation = activation,                                                         \
            .activation_size = activation_size,                                               \
            .activation_offset = activation_offset,                                           \
            .destination = destination,                                                       \
            .destination_size = destination_size,                                             \
            .destination_offset = destination_offset,                                         \
            .column_start = column_start,                                                     \
            .column_count = column_count,                                                     \
        };                                                                                    \
        return EXECUTE(linear, &execution, NULL);                                             \
    }                                                                                         \
    static void destroy_##NAME(void *linear) {                                                \
        DESTROY(linear);                                                                      \
    }

DEFINE_M4_ADAPTERS(
    q4_k,
    qpu_llama_q4_k_linear,
    qpu_llama_q4_k_linear_desc,
    qpu_llama_q4_k_execution,
    qpu_llama_q4_k_linear_prepare,
    qpu_llama_q4_k_linear_execute,
    qpu_llama_q4_k_linear_destroy)

DEFINE_M4_ADAPTERS(
    q6_k,
    qpu_llama_q6_k_linear,
    qpu_llama_q6_k_linear_desc,
    qpu_llama_q6_k_execution,
    qpu_llama_q6_k_linear_prepare,
    qpu_llama_q6_k_linear_execute,
    qpu_llama_q6_k_linear_destroy)

DEFINE_M4_ADAPTERS(
    q8_0,
    qpu_llama_q8_0_linear,
    qpu_llama_q8_0_linear_desc,
    qpu_llama_q8_0_execution,
    qpu_llama_q8_0_linear_prepare,
    qpu_llama_q8_0_linear_execute,
    qpu_llama_q8_0_linear_destroy)

static int bytes_equal(const uint8_t *data, size_t size, uint8_t expected) {
    for (size_t index = 0; index < size; ++index) {
        if (data[index] != expected) {
            return 0;
        }
    }
    return 1;
}

static int bytes_zero(const uint8_t *data, size_t size) {
    return bytes_equal(data, size, 0);
}

static void reset_destination(uint8_t *storage, size_t destination_bytes) {
    memset(storage, GUARD_VALUE, GUARD_BYTES + destination_bytes + GUARD_BYTES);
    float *destination = (float *) (storage + GUARD_BYTES);
    for (size_t index = 0; index < destination_bytes / sizeof(float); ++index) {
        destination[index] = OUTPUT_SENTINEL;
    }
}

static int verify_guards_and_inputs(
    const format_ops *ops,
    const uint8_t *weights,
    size_t weight_bytes,
    const uint8_t *activation,
    size_t activation_bytes,
    const uint8_t *destination,
    size_t destination_bytes) {
    if (!bytes_equal(weights, GUARD_BYTES, GUARD_VALUE) ||
        !bytes_zero(weights + GUARD_BYTES, weight_bytes) ||
        !bytes_equal(weights + GUARD_BYTES + weight_bytes, GUARD_BYTES, GUARD_VALUE)) {
        fprintf(stderr, "%s modified guarded read-only weights\n", ops->name);
        return 0;
    }
    if (!bytes_equal(activation, GUARD_BYTES, GUARD_VALUE) ||
        !bytes_zero(activation + GUARD_BYTES, activation_bytes) ||
        !bytes_equal(activation + GUARD_BYTES + activation_bytes, GUARD_BYTES, GUARD_VALUE)) {
        fprintf(stderr, "%s modified guarded read-only activation\n", ops->name);
        return 0;
    }
    if (!bytes_equal(destination, GUARD_BYTES, GUARD_VALUE) ||
        !bytes_equal(
            destination + GUARD_BYTES + destination_bytes, GUARD_BYTES, GUARD_VALUE)) {
        fprintf(stderr, "%s modified a destination canary\n", ops->name);
        return 0;
    }
    return 1;
}

static int verify_destination(
    const format_ops *ops,
    const uint8_t *storage,
    uint32_t selected_start,
    uint32_t selected_count,
    int expect_output) {
    const float *destination = (const float *) (storage + GUARD_BYTES);
    for (uint32_t row = 0; row < ops->rows; ++row) {
        for (uint32_t column = 0; column < OUTPUT_COLUMNS; ++column) {
            const int selected = expect_output && column >= selected_start &&
                column < selected_start + selected_count;
            const float expected = selected ? 0.0f : OUTPUT_SENTINEL;
            const float actual = destination[(size_t) row * OUTPUT_COLUMNS + column];
            if (actual != expected) {
                fprintf(stderr,
                    "%s row %u column %u: expected %.0f, got %.0f\n",
                    ops->name,
                    row,
                    column,
                    expected,
                    actual);
                return 0;
            }
        }
    }
    return 1;
}

static int execute_and_check(
    qpu_llama_context *context,
    const format_ops *ops,
    void *linear,
    uint8_t *weights,
    size_t weight_bytes,
    uint8_t *activation,
    size_t activation_bytes,
    uint8_t *destination,
    size_t destination_bytes,
    size_t advertised_activation_size,
    size_t advertised_destination_size,
    uint32_t column_start,
    uint32_t column_count,
    qpu_llama_status expected_status) {
    reset_destination(destination, destination_bytes);
    const qpu_llama_status status = ops->execute(
        linear,
        activation,
        advertised_activation_size,
        GUARD_BYTES,
        destination,
        advertised_destination_size,
        GUARD_BYTES,
        column_start,
        column_count);
    if (status != expected_status) {
        fprintf(stderr,
            "%s execute expected %s, got %s (%s)\n",
            ops->name,
            qpu_llama_status_string(expected_status),
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        return 0;
    }
    if (!verify_guards_and_inputs(
            ops,
            weights,
            weight_bytes,
            activation,
            activation_bytes,
            destination,
            destination_bytes)) {
        return 0;
    }
    return verify_destination(
        ops,
        destination,
        column_start,
        column_count,
        status == QPU_LLAMA_OK);
}

static int expect_invalid_prepare(
    qpu_llama_context *context,
    const format_ops *ops,
    const uint8_t *weights,
    size_t weight_size,
    uint32_t input_columns,
    uint32_t resident_start,
    uint32_t resident_count) {
    void *linear = NULL;
    const qpu_llama_status status = ops->prepare(
        context,
        weights,
        weight_size,
        input_columns,
        OUTPUT_COLUMNS,
        resident_start,
        resident_count,
        ops->source_hash,
        &linear);
    if (status != QPU_LLAMA_INVALID_ARGUMENT || linear != NULL) {
        fprintf(stderr,
            "%s invalid prepare returned %s and %p\n",
            ops->name,
            qpu_llama_status_string(status),
            linear);
        ops->destroy(linear);
        return 0;
    }
    return 1;
}

static int run_format(qpu_llama_context *context, const format_ops *ops) {
    const size_t weight_bytes = (size_t) OUTPUT_COLUMNS * ops->weight_block_bytes;
    const size_t activation_bytes = (size_t) ops->rows * ops->activation_block_bytes;
    const size_t destination_bytes =
        (size_t) ops->rows * OUTPUT_COLUMNS * sizeof(float);
    uint8_t *weights = malloc(GUARD_BYTES + weight_bytes + GUARD_BYTES);
    uint8_t *activation = malloc(GUARD_BYTES + activation_bytes + GUARD_BYTES);
    uint8_t *destination = malloc(GUARD_BYTES + destination_bytes + GUARD_BYTES);
    if (weights == NULL || activation == NULL || destination == NULL) {
        fprintf(stderr, "%s host allocation failed\n", ops->name);
        free(destination);
        free(activation);
        free(weights);
        return 0;
    }
    memset(weights, GUARD_VALUE, GUARD_BYTES + weight_bytes + GUARD_BYTES);
    memset(weights + GUARD_BYTES, 0, weight_bytes);
    memset(activation, GUARD_VALUE, GUARD_BYTES + activation_bytes + GUARD_BYTES);
    memset(activation + GUARD_BYTES, 0, activation_bytes);
    reset_destination(destination, destination_bytes);

    int passed = 1;
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns - 1U,
        0,
        OUTPUT_COLUMNS);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        1,
        OUTPUT_TILE);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        OUTPUT_TILE - 1U,
        OUTPUT_TILE);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        0,
        OUTPUT_TILE - 1U);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        0,
        OUTPUT_TILE + 1U);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        0,
        0);
    passed &= expect_invalid_prepare(
        context,
        ops,
        weights + GUARD_BYTES,
        weight_bytes - 1U,
        ops->input_columns,
        0,
        OUTPUT_COLUMNS);

    void *linear = NULL;
    qpu_llama_status status = ops->prepare(
        context,
        weights + GUARD_BYTES,
        weight_bytes,
        ops->input_columns,
        OUTPUT_COLUMNS,
        0,
        OUTPUT_COLUMNS,
        ops->source_hash,
        &linear);
    if (status != QPU_LLAMA_OK || linear == NULL) {
        fprintf(stderr,
            "%s full prepare failed: %s (%s)\n",
            ops->name,
            qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        passed = 0;
        goto cleanup;
    }

    const uint32_t partitions[][2] = {
        {0, OUTPUT_COLUMNS},
        {0, OUTPUT_TILE},
        {0, OUTPUT_COLUMNS - OUTPUT_TILE},
        {OUTPUT_TILE, OUTPUT_TILE},
        {OUTPUT_TILE, OUTPUT_COLUMNS - OUTPUT_TILE},
        {OUTPUT_COLUMNS - OUTPUT_TILE, OUTPUT_TILE},
    };
    for (size_t index = 0; index < sizeof(partitions) / sizeof(partitions[0]); ++index) {
        passed &= execute_and_check(
            context,
            ops,
            linear,
            weights,
            weight_bytes,
            activation,
            activation_bytes,
            destination,
            destination_bytes,
            GUARD_BYTES + activation_bytes,
            GUARD_BYTES + destination_bytes,
            partitions[index][0],
            partitions[index][1],
            QPU_LLAMA_OK);
    }

    for (unsigned int iteration = 0; iteration < REPEAT_COUNT; ++iteration) {
        const size_t index = iteration % (sizeof(partitions) / sizeof(partitions[0]));
        passed &= execute_and_check(
            context,
            ops,
            linear,
            weights,
            weight_bytes,
            activation,
            activation_bytes,
            destination,
            destination_bytes,
            GUARD_BYTES + activation_bytes,
            GUARD_BYTES + destination_bytes,
            partitions[index][0],
            partitions[index][1],
            QPU_LLAMA_OK);
    }

    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        0,
        0,
        QPU_LLAMA_INVALID_ARGUMENT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        0,
        OUTPUT_TILE - 1U,
        QPU_LLAMA_INVALID_ARGUMENT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        0,
        OUTPUT_TILE + 1U,
        QPU_LLAMA_INVALID_ARGUMENT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        1,
        OUTPUT_TILE,
        QPU_LLAMA_INVALID_ARGUMENT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes - 1U,
        GUARD_BYTES + destination_bytes,
        0,
        OUTPUT_TILE,
        QPU_LLAMA_INVALID_ARGUMENT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes - 1U,
        0,
        OUTPUT_TILE,
        QPU_LLAMA_INVALID_ARGUMENT);

    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_SUBMISSION);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        0,
        OUTPUT_COLUMNS,
        QPU_LLAMA_SUBMISSION_FAILED);
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_WAIT);
    passed &= execute_and_check(
        context,
        ops,
        linear,
        weights,
        weight_bytes,
        activation,
        activation_bytes,
        destination,
        destination_bytes,
        GUARD_BYTES + activation_bytes,
        GUARD_BYTES + destination_bytes,
        0,
        OUTPUT_COLUMNS,
        QPU_LLAMA_TIMEOUT);
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_NONE);

cleanup:
    qpu_llama_context_set_failure_mask(context, QPU_LLAMA_FAIL_NONE);
    ops->destroy(linear);
    free(destination);
    free(activation);
    free(weights);
    if (passed) {
        printf("%s native safety matrix passed\n", ops->name);
    }
    return passed;
}

int main(void) {
    static const format_ops formats[] = {
        {
            .name = "Q4_0 x Q8_0 M1",
            .rows = 1,
            .input_columns = 32,
            .weight_block_bytes = 18,
            .activation_block_bytes = 34,
            .source_hash = qpu_ggml_q4_0_q8_0_m1_source_hash,
            .prepare = prepare_q4_0_m1,
            .execute = execute_q4_0,
            .destroy = destroy_q4_0,
        },
        {
            .name = "Q4_0 x Q8_0 M4",
            .rows = ROWS,
            .input_columns = 32,
            .weight_block_bytes = 18,
            .activation_block_bytes = 34,
            .source_hash = qpu_ggml_q4_0_q8_0_m4_source_hash,
            .prepare = prepare_q4_0_m4,
            .execute = execute_q4_0,
            .destroy = destroy_q4_0,
        },
        {
            .name = "Q4_K x Q8_K M4",
            .rows = ROWS,
            .input_columns = 256,
            .weight_block_bytes = 144,
            .activation_block_bytes = 292,
            .source_hash = qpu_ggml_q4_k_q8_k_m4_source_hash,
            .prepare = prepare_q4_k,
            .execute = execute_q4_k,
            .destroy = destroy_q4_k,
        },
        {
            .name = "Q6_K x Q8_K M4",
            .rows = ROWS,
            .input_columns = 256,
            .weight_block_bytes = 210,
            .activation_block_bytes = 292,
            .source_hash = qpu_ggml_q6_k_q8_k_m4_source_hash,
            .prepare = prepare_q6_k,
            .execute = execute_q6_k,
            .destroy = destroy_q6_k,
        },
        {
            .name = "Q8_0 x Q8_0 M4",
            .rows = ROWS,
            .input_columns = 32,
            .weight_block_bytes = 34,
            .activation_block_bytes = 34,
            .source_hash = qpu_ggml_q8_0_q8_0_m4_source_hash,
            .prepare = prepare_q8_0,
            .execute = execute_q8_0,
            .destroy = destroy_q8_0,
        },
    };

    int passed = 1;
    for (size_t index = 0; index < sizeof(formats) / sizeof(formats[0]); ++index) {
        qpu_llama_context *context = NULL;
        const qpu_llama_status status = qpu_llama_context_create(NULL, &context);
        if (status != QPU_LLAMA_OK) {
            fprintf(stderr,
                "%s context recovery failed: %s\n",
                formats[index].name,
                qpu_llama_status_string(status));
            return 1;
        }
        passed &= run_format(context, &formats[index]);
        qpu_llama_context_destroy(context);
    }
    return passed ? 0 : 1;
}
