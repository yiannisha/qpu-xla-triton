#include "qpu_llama_runtime.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    qpu_llama_context *context = NULL;
    const qpu_llama_context_config config = {0};
    qpu_llama_status status = qpu_llama_context_create(&config, &context);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "context: %s\n", qpu_llama_status_string(status));
        return 1;
    }
    qpu_llama_capabilities capabilities;
    status = qpu_llama_context_capabilities(context, &capabilities);
    if (status != QPU_LLAMA_OK) {
        qpu_llama_context_destroy(context);
        return 1;
    }
    qpu_llama_buffer *buffer = NULL;
    status = qpu_llama_buffer_create(context, 4096, &buffer);
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "buffer: %s (%s)\n", qpu_llama_status_string(status),
            qpu_llama_context_last_error(context));
        qpu_llama_context_destroy(context);
        return 1;
    }
    uint32_t address = 0;
    status = qpu_llama_buffer_gpu_address(buffer, 0, &address);
    printf("render_node=%s csd=%u tfu=%u bo_gpu_address=0x%08x\n",
        capabilities.render_node, capabilities.supports_csd, capabilities.supports_tfu, address);
    qpu_llama_buffer_destroy(buffer);
    qpu_llama_context_destroy(context);
    return status == QPU_LLAMA_OK ? 0 : 1;
}
