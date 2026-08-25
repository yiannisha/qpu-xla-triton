#include "qpu_llama_runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_program_validation(void) {
    static const uint8_t code[] = {0x61, 0x62, 0x63, 0, 0, 0, 0, 0};
    static const char source_hash[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    qpu_llama_program_desc desc = {
        .code = code,
        .code_size = sizeof(code),
        .compiled_source_hash = source_hash,
        .expected_source_hash = source_hash,
        .binary_sha256 = "1d4e65b8a6b941ab0ad349370f1f9ad95e81f7eef104e102450e7c08126b0559",
        .uniform_word_count = 1,
    };
    assert(qpu_llama_validate_program(&desc) == QPU_LLAMA_OK);
    desc.expected_source_hash = "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    assert(qpu_llama_validate_program(&desc) == QPU_LLAMA_HASH_MISMATCH);
}

static void test_device_failure_injection(void) {
    qpu_llama_context *context = NULL;
    const qpu_llama_context_config config = {
        .render_node = NULL,
        .timeout_ns = 0,
        .failure_mask = QPU_LLAMA_FAIL_DEVICE_OPEN,
    };
    assert(qpu_llama_context_create(&config, &context) == QPU_LLAMA_DEVICE_UNAVAILABLE);
    assert(context == NULL);
}

static void test_cached_buffer_argument_validation(void) {
    qpu_llama_buffer *buffer = NULL;
    assert(qpu_llama_buffer_create_cached(NULL, 4096, &buffer) ==
        QPU_LLAMA_INVALID_ARGUMENT);
    assert(qpu_llama_buffer_cpu_access_begin(
        NULL, QPU_LLAMA_CPU_ACCESS_READ) == QPU_LLAMA_INVALID_ARGUMENT);
    assert(qpu_llama_buffer_cpu_access_end(
        NULL, QPU_LLAMA_CPU_ACCESS_WRITE) == QPU_LLAMA_INVALID_ARGUMENT);
}

static void test_async_submission_argument_validation(void) {
    qpu_llama_submission *submission = NULL;
    assert(qpu_llama_program_submit(NULL, NULL, &submission) ==
        QPU_LLAMA_INVALID_ARGUMENT);
    assert(qpu_llama_submission_wait(NULL) == QPU_LLAMA_INVALID_ARGUMENT);
    qpu_llama_submission_destroy(NULL);
}

int main(void) {
    test_program_validation();
    test_device_failure_injection();
    test_cached_buffer_argument_validation();
    test_async_submission_argument_validation();
    puts("qpu llama runtime tests passed");
    return 0;
}
