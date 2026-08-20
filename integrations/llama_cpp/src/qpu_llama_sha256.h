#ifndef QPU_LLAMA_SHA256_H
#define QPU_LLAMA_SHA256_H

#include <stddef.h>
#include <stdint.h>

void qpu_llama_sha256(const void *data, size_t size, uint8_t digest[32]);
void qpu_llama_sha256_hex(const void *data, size_t size, char hex[65]);

#endif
