#ifndef QPU_LLAMA_Q6_K_H
#define QPU_LLAMA_Q6_K_H

#include "qpu_llama_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qpu_llama_q6_k_linear qpu_llama_q6_k_linear;

typedef struct qpu_llama_q6_k_linear_desc {
    const void *weights;
    size_t weight_size;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t resident_column_start;
    uint32_t resident_column_count;
    uint32_t workgroups_per_supergroup;
    const char *expected_source_hash;
} qpu_llama_q6_k_linear_desc;

typedef struct qpu_llama_q6_k_execution {
    const void *activation;
    size_t activation_size;
    size_t activation_offset;
    void *destination;
    size_t destination_size;
    size_t destination_offset;
    uint32_t column_start;
    uint32_t column_count;
} qpu_llama_q6_k_execution;

typedef struct qpu_llama_q6_k_timing {
    uint64_t input_copy_ns;
    uint64_t submit_wait_ns;
    uint64_t output_copy_ns;
    uint64_t complete_ns;
} qpu_llama_q6_k_timing;

qpu_llama_status qpu_llama_q6_k_linear_prepare(
    qpu_llama_context *context,
    const qpu_llama_q6_k_linear_desc *desc,
    qpu_llama_q6_k_linear **result);
void qpu_llama_q6_k_linear_destroy(qpu_llama_q6_k_linear *linear);
qpu_llama_status qpu_llama_q6_k_linear_execute(
    qpu_llama_q6_k_linear *linear,
    const qpu_llama_q6_k_execution *execution,
    qpu_llama_q6_k_timing *timing);
size_t qpu_llama_q6_k_linear_resident_bytes(const qpu_llama_q6_k_linear *linear);

#ifdef __cplusplus
}
#endif

#endif
