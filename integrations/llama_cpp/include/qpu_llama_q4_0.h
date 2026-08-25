#ifndef QPU_LLAMA_Q4_0_H
#define QPU_LLAMA_Q4_0_H

#include "qpu_llama_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qpu_llama_q4_0_linear qpu_llama_q4_0_linear;
typedef struct qpu_llama_q4_0_submission qpu_llama_q4_0_submission;

typedef enum qpu_llama_q4_0_weight_mode {
    QPU_LLAMA_Q4_0_WEIGHT_EXACT = 0,
    QPU_LLAMA_Q4_0_WEIGHT_COLUMN_W8 = 1,
    QPU_LLAMA_Q4_0_WEIGHT_ROWCOL_W8A8 = 2,
} qpu_llama_q4_0_weight_mode;

typedef struct qpu_llama_q4_0_linear_desc {
    const void *weights;
    size_t weight_size;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t rows;
    uint32_t resident_column_start;
    uint32_t resident_column_count;
    qpu_llama_q4_0_weight_mode weight_mode;
    uint32_t workgroups_per_supergroup;
    const char *expected_source_hash;
} qpu_llama_q4_0_linear_desc;

typedef struct qpu_llama_q4_0_execution {
    const void *activation;
    size_t activation_size;
    size_t activation_offset;
    void *destination;
    size_t destination_size;
    size_t destination_offset;
    uint32_t column_start;
    uint32_t column_count;
    // Zero uses the prepared row capacity for backwards compatibility.
    uint32_t rows;
    // Zero is native row-major Q8_0; 4 or 8 is CPU_REPACK block_q8_0x4.
    uint32_t activation_interleave;
} qpu_llama_q4_0_execution;

typedef struct qpu_llama_q4_0_timing {
    uint64_t input_access_ns;
    uint64_t input_pack_ns;
    uint64_t input_sync_ns;
    uint64_t input_copy_ns;
    uint64_t submit_ns;
    uint64_t wait_ns;
    uint64_t submit_wait_ns;
    uint64_t output_sync_ns;
    uint64_t output_copy_ns;
    uint64_t complete_ns;
} qpu_llama_q4_0_timing;

qpu_llama_status qpu_llama_q4_0_linear_prepare(
    qpu_llama_context *context,
    const qpu_llama_q4_0_linear_desc *desc,
    qpu_llama_q4_0_linear **result);
void qpu_llama_q4_0_linear_destroy(qpu_llama_q4_0_linear *linear);
qpu_llama_status qpu_llama_q4_0_linear_execute(
    qpu_llama_q4_0_linear *linear,
    const qpu_llama_q4_0_execution *execution,
    qpu_llama_q4_0_timing *timing);
qpu_llama_status qpu_llama_q4_0_linear_submit(
    qpu_llama_q4_0_linear *linear,
    const qpu_llama_q4_0_execution *execution,
    qpu_llama_q4_0_submission **result);
qpu_llama_status qpu_llama_q4_0_submission_wait(
    qpu_llama_q4_0_submission *submission,
    qpu_llama_q4_0_timing *timing);
qpu_llama_status qpu_llama_q4_0_submission_cpu_fallback(
    qpu_llama_q4_0_submission *submission);
void qpu_llama_q4_0_submission_destroy(qpu_llama_q4_0_submission *submission);
size_t qpu_llama_q4_0_linear_resident_bytes(const qpu_llama_q4_0_linear *linear);

#ifdef __cplusplus
}
#endif

#endif
