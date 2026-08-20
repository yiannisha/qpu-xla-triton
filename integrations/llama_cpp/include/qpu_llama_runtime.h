#ifndef QPU_LLAMA_RUNTIME_H
#define QPU_LLAMA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qpu_llama_context qpu_llama_context;
typedef struct qpu_llama_buffer qpu_llama_buffer;
typedef struct qpu_llama_program qpu_llama_program;

typedef enum qpu_llama_status {
    QPU_LLAMA_OK = 0,
    QPU_LLAMA_INVALID_ARGUMENT = 1,
    QPU_LLAMA_DEVICE_UNAVAILABLE = 2,
    QPU_LLAMA_CAPABILITY_MISSING = 3,
    QPU_LLAMA_ALLOCATION_FAILED = 4,
    QPU_LLAMA_IO_FAILED = 5,
    QPU_LLAMA_SUBMISSION_FAILED = 6,
    QPU_LLAMA_TIMEOUT = 7,
    QPU_LLAMA_HASH_MISMATCH = 8,
    QPU_LLAMA_INTERNAL_ERROR = 9,
} qpu_llama_status;

typedef enum qpu_llama_failure_point {
    QPU_LLAMA_FAIL_NONE = 0,
    QPU_LLAMA_FAIL_DEVICE_OPEN = 1U << 0,
    QPU_LLAMA_FAIL_ALLOCATION = 1U << 1,
    QPU_LLAMA_FAIL_SUBMISSION = 1U << 2,
    QPU_LLAMA_FAIL_WAIT = 1U << 3,
    QPU_LLAMA_FAIL_SOURCE_HASH = 1U << 4,
} qpu_llama_failure_point;

typedef struct qpu_llama_context_config {
    const char *render_node;
    uint64_t timeout_ns;
    uint32_t failure_mask;
} qpu_llama_context_config;

typedef struct qpu_llama_capabilities {
    char render_node[128];
    uint64_t hub_ident1;
    uint64_t hub_ident2;
    uint64_t hub_ident3;
    uint64_t core0_ident0;
    uint64_t core0_ident1;
    uint64_t core0_ident2;
    uint8_t supports_csd;
    uint8_t supports_tfu;
} qpu_llama_capabilities;

typedef struct qpu_llama_program_desc {
    const void *code;
    size_t code_size;
    const char *compiled_source_hash;
    const char *expected_source_hash;
    const char *binary_sha256;
    uint32_t uniform_word_count;
} qpu_llama_program_desc;

typedef struct qpu_llama_dispatch_desc {
    const uint32_t *uniforms;
    uint32_t uniform_word_count;
    qpu_llama_buffer *const *buffers;
    uint32_t buffer_count;
    uint32_t local_invocation[3];
    uint32_t workgroup[3];
    uint32_t wgs_per_sg;
    uint32_t thread_count;
    uint8_t propagate_nan;
    uint8_t single_segment;
    uint8_t threading;
} qpu_llama_dispatch_desc;

qpu_llama_status qpu_llama_context_create(
    const qpu_llama_context_config *config,
    qpu_llama_context **result);
void qpu_llama_context_destroy(qpu_llama_context *context);
const char *qpu_llama_context_last_error(const qpu_llama_context *context);
qpu_llama_status qpu_llama_context_capabilities(
    const qpu_llama_context *context,
    qpu_llama_capabilities *result);
void qpu_llama_context_set_failure_mask(qpu_llama_context *context, uint32_t failure_mask);

qpu_llama_status qpu_llama_buffer_create(
    qpu_llama_context *context,
    size_t size,
    qpu_llama_buffer **result);
void qpu_llama_buffer_destroy(qpu_llama_buffer *buffer);
void *qpu_llama_buffer_data(qpu_llama_buffer *buffer);
size_t qpu_llama_buffer_size(const qpu_llama_buffer *buffer);
qpu_llama_status qpu_llama_buffer_gpu_address(
    const qpu_llama_buffer *buffer,
    size_t byte_offset,
    uint32_t *result);

qpu_llama_status qpu_llama_validate_program(const qpu_llama_program_desc *desc);
qpu_llama_status qpu_llama_program_create(
    qpu_llama_context *context,
    const qpu_llama_program_desc *desc,
    qpu_llama_program **result);
void qpu_llama_program_destroy(qpu_llama_program *program);
qpu_llama_status qpu_llama_program_execute(
    qpu_llama_program *program,
    const qpu_llama_dispatch_desc *dispatch);

const char *qpu_llama_status_string(qpu_llama_status status);

#ifdef __cplusplus
}
#endif

#endif
