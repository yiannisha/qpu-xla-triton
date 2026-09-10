#ifndef QPU_LLAMA_FFN_ISLAND_H
#define QPU_LLAMA_FFN_ISLAND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ggml_qpu_ffn_island_register_q4_0(
    const char * name,
    const void * weights,
    size_t weight_size,
    uint64_t input_columns,
    uint64_t output_columns);

int ggml_qpu_ffn_island_begin(
    const char * projection_name,
    const void * activation_q8_0,
    size_t activation_size,
    uint64_t rows,
    uint64_t input_columns,
    uint64_t intermediate_columns,
    uint64_t activation_interleave);

uint64_t ggml_qpu_ffn_island_cpu_output_columns_for(const char * name);
uint64_t ggml_qpu_ffn_island_cpu_intermediate_columns(void);
uint64_t ggml_qpu_ffn_island_down_reduction_columns_for(const char * name);
int ggml_qpu_ffn_island_wait(const char * down_name);
int ggml_qpu_ffn_island_join(
    const char * down_name,
    float * destination,
    size_t destination_size,
    uint64_t rows,
    uint64_t hidden_columns,
    uint64_t thread_index,
    uint64_t thread_count);
int ggml_qpu_ffn_island_complete(const char * down_name);
uint64_t ggml_qpu_ffn_island_dispatch_count(void);
uint64_t ggml_qpu_ffn_island_count(void);

#ifdef __cplusplus
}
#endif

#endif
