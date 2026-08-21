#include "qpu_llama_q4_0.h"
#include "qpu_llama_q4_k.h"
#include "qpu_llama_q6_k.h"
#include "qpu_llama_q8_0.h"

#include "ggml-q4-0-q8-0-m1.h"
#include "ggml-q4-0-q8-0-m4.h"
#include "ggml-q4-k-q8-k-m4.h"
#include "ggml-q6-k-q8-k-m4.h"
#include "ggml-q8-0-q8-0-m4.h"
#include "quants.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q4_0_BLOCK_ELEMENTS 32U
#define Q4_0_BLOCK_BYTES 18U
#define Q8_0_BLOCK_BYTES 34U
#define Q4_K_BLOCK_ELEMENTS 256U
#define Q4_K_BLOCK_BYTES 144U
#define Q8_K_BLOCK_BYTES 292U
#define Q6_K_BLOCK_BYTES 210U

typedef enum weight_format {
    WEIGHT_Q4_0,
    WEIGHT_Q4_K,
    WEIGHT_Q6_K,
    WEIGHT_Q8_0,
} weight_format;

typedef enum bench_mode {
    BENCH_CPU,
    BENCH_QPU,
    BENCH_HYBRID,
} bench_mode;

typedef struct options {
    const char *weights_path;
    const char *activation_path;
    const char *output_path;
    bench_mode mode;
    weight_format format;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t rows;
    uint32_t qpu_column_start;
    uint32_t qpu_column_count;
    uint32_t qpu_wgs_per_sg;
    uint32_t cpu_threads;
    uint32_t warmups;
    uint32_t samples;
} options;

typedef struct cpu_job {
    const uint8_t *weights;
    const uint8_t *activation;
    float *destination;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t rows;
    uint32_t column_start;
    uint32_t column_count;
    size_t weight_row_bytes;
    size_t activation_row_bytes;
    weight_format format;
} cpu_job;

typedef struct cpu_pool cpu_pool;

typedef struct cpu_worker {
    cpu_pool *pool;
    pthread_t thread;
    uint32_t index;
    uint64_t generation;
} cpu_worker;

struct cpu_pool {
    pthread_mutex_t mutex;
    pthread_cond_t start_condition;
    pthread_cond_t done_condition;
    cpu_worker *workers;
    uint32_t thread_count;
    uint32_t pending;
    uint64_t generation;
    bool stopping;
    cpu_job job;
};

typedef struct error_metrics {
    double max_absolute;
    double max_relative;
    double mean_absolute;
    double p99_absolute;
    uint64_t nan_count;
    uint64_t inf_count;
    uint64_t tolerance_violation_count;
    uint32_t worst_row;
    uint32_t worst_column;
    float worst_reference;
    float worst_actual;
} error_metrics;

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) + (uint64_t) value.tv_nsec;
}

static bool parse_u32(const char *text, uint32_t *result) {
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    *result = (uint32_t) value;
    return true;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --weights FILE --activation-f32 FILE --mode cpu|qpu|hybrid "
        "[--weight-type q4_0|q4_k|q6_k|q8_0] "
        "--input-columns N --output-columns N --rows 1|4 --qpu-column-start N "
        "--qpu-column-count N [--qpu-wgs-per-sg N] --cpu-threads N --warmups N --samples N "
        "[--output-bin FILE]\n",
        program);
}

static bool parse_mode(const char *text, bench_mode *result) {
    if (strcmp(text, "cpu") == 0) {
        *result = BENCH_CPU;
        return true;
    }
    if (strcmp(text, "qpu") == 0) {
        *result = BENCH_QPU;
        return true;
    }
    if (strcmp(text, "hybrid") == 0) {
        *result = BENCH_HYBRID;
        return true;
    }
    return false;
}

static bool parse_weight_format(const char *text, weight_format *result) {
    if (strcmp(text, "q4_0") == 0) {
        *result = WEIGHT_Q4_0;
        return true;
    }
    if (strcmp(text, "q4_k") == 0) {
        *result = WEIGHT_Q4_K;
        return true;
    }
    if (strcmp(text, "q6_k") == 0) {
        *result = WEIGHT_Q6_K;
        return true;
    }
    if (strcmp(text, "q8_0") == 0) {
        *result = WEIGHT_Q8_0;
        return true;
    }
    return false;
}

static bool parse_options(int argc, char **argv, options *result) {
    options value = {
        .mode = BENCH_CPU,
        .cpu_threads = 1,
        .qpu_wgs_per_sg = 24,
        .warmups = 5,
        .samples = 31,
    };
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        const char *name = argv[index];
        const char *argument = argv[index + 1];
        if (strcmp(name, "--weights") == 0) {
            value.weights_path = argument;
        } else if (strcmp(name, "--activation-f32") == 0) {
            value.activation_path = argument;
        } else if (strcmp(name, "--output-bin") == 0) {
            value.output_path = argument;
        } else if (strcmp(name, "--mode") == 0) {
            if (!parse_mode(argument, &value.mode)) {
                return false;
            }
        } else if (strcmp(name, "--weight-type") == 0) {
            if (!parse_weight_format(argument, &value.format)) {
                return false;
            }
        } else if (strcmp(name, "--input-columns") == 0) {
            if (!parse_u32(argument, &value.input_columns)) {
                return false;
            }
        } else if (strcmp(name, "--output-columns") == 0) {
            if (!parse_u32(argument, &value.output_columns)) {
                return false;
            }
        } else if (strcmp(name, "--rows") == 0) {
            if (!parse_u32(argument, &value.rows)) {
                return false;
            }
        } else if (strcmp(name, "--qpu-column-start") == 0) {
            if (!parse_u32(argument, &value.qpu_column_start)) {
                return false;
            }
        } else if (strcmp(name, "--qpu-column-count") == 0) {
            if (!parse_u32(argument, &value.qpu_column_count)) {
                return false;
            }
        } else if (strcmp(name, "--qpu-wgs-per-sg") == 0) {
            if (!parse_u32(argument, &value.qpu_wgs_per_sg)) {
                return false;
            }
        } else if (strcmp(name, "--cpu-threads") == 0) {
            if (!parse_u32(argument, &value.cpu_threads)) {
                return false;
            }
        } else if (strcmp(name, "--warmups") == 0) {
            if (!parse_u32(argument, &value.warmups)) {
                return false;
            }
        } else if (strcmp(name, "--samples") == 0) {
            if (!parse_u32(argument, &value.samples)) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (value.weights_path == NULL || value.activation_path == NULL || value.input_columns == 0 ||
        value.input_columns % (value.format == WEIGHT_Q4_0
                || value.format == WEIGHT_Q8_0 ? Q4_0_BLOCK_ELEMENTS : Q4_K_BLOCK_ELEMENTS) != 0 ||
        value.output_columns == 0 ||
        (value.rows != 1 && value.rows != 4) || value.qpu_wgs_per_sg == 0 ||
        value.qpu_wgs_per_sg > UINT8_MAX || value.cpu_threads == 0 || value.samples == 0) {
        return false;
    }
    if (value.format != WEIGHT_Q4_0 && value.rows != 4) {
        return false;
    }
    if (value.format == WEIGHT_Q4_0 && value.qpu_wgs_per_sg != 24U) {
        return false;
    }
    if (value.mode == BENCH_QPU) {
        value.qpu_column_start = 0;
        value.qpu_column_count = value.output_columns;
    }
    if (value.mode == BENCH_CPU) {
        value.qpu_column_start = value.output_columns;
        value.qpu_column_count = 0;
    }
    if (value.qpu_column_start > value.output_columns ||
        value.qpu_column_count > value.output_columns - value.qpu_column_start ||
        (value.mode != BENCH_CPU &&
            (value.qpu_column_start % 16U != 0 || value.qpu_column_count == 0 ||
                value.qpu_column_count % 16U != 0)) ||
        (value.mode == BENCH_HYBRID &&
            (value.qpu_column_start == 0 ||
                value.qpu_column_start + value.qpu_column_count != value.output_columns))) {
        return false;
    }
    *result = value;
    return true;
}

static void *read_exact_file(const char *path, size_t expected_size) {
    FILE *input = fopen(path, "rb");
    if (input == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    void *data = malloc(expected_size);
    const bool ok = data != NULL && fread(data, 1, expected_size, input) == expected_size &&
        fgetc(input) == EOF && !ferror(input);
    if (fclose(input) != 0) {
        fprintf(stderr, "close %s: %s\n", path, strerror(errno));
        free(data);
        return NULL;
    }
    if (!ok) {
        fprintf(stderr, "%s does not contain exactly %zu bytes\n", path, expected_size);
        free(data);
        return NULL;
    }
    return data;
}

static bool write_exact_file(const char *path, const void *data, size_t size) {
    FILE *output = fopen(path, "wb");
    if (output == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return false;
    }
    const bool ok = fwrite(data, 1, size, output) == size && fflush(output) == 0;
    if (fclose(output) != 0) {
        fprintf(stderr, "close %s: %s\n", path, strerror(errno));
        return false;
    }
    if (!ok) {
        fprintf(stderr, "write %s: %s\n", path, strerror(errno));
    }
    return ok;
}

static void cpu_compute_range(const cpu_job *job, uint32_t worker_index, uint32_t worker_count) {
    const uint32_t begin = job->column_start +
        (uint32_t) ((uint64_t) job->column_count * worker_index / worker_count);
    const uint32_t end = job->column_start +
        (uint32_t) ((uint64_t) job->column_count * (worker_index + 1U) / worker_count);
    for (uint32_t row = 0; row < job->rows; ++row) {
        const uint8_t *activation = job->activation + (size_t) row * job->activation_row_bytes;
        for (uint32_t column = begin; column < end; ++column) {
            const uint8_t *weight = job->weights + (size_t) column * job->weight_row_bytes;
            float *destination = job->destination + (size_t) row * job->output_columns + column;
            if (job->format == WEIGHT_Q4_0) {
                ggml_vec_dot_q4_0_q8_0(
                    (int) job->input_columns, destination, 0, weight, 0, activation, 0, 1);
            } else if (job->format == WEIGHT_Q4_K) {
                ggml_vec_dot_q4_K_q8_K(
                    (int) job->input_columns, destination, 0, weight, 0, activation, 0, 1);
            } else if (job->format == WEIGHT_Q6_K) {
                ggml_vec_dot_q6_K_q8_K(
                    (int) job->input_columns, destination, 0, weight, 0, activation, 0, 1);
            } else {
                ggml_vec_dot_q8_0_q8_0(
                    (int) job->input_columns, destination, 0, weight, 0, activation, 0, 1);
            }
        }
    }
}

static void *cpu_worker_main(void *argument) {
    cpu_worker *worker = argument;
    cpu_pool *pool = worker->pool;
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stopping) {
        while (!pool->stopping && worker->generation == pool->generation) {
            pthread_cond_wait(&pool->start_condition, &pool->mutex);
        }
        if (pool->stopping) {
            break;
        }
        const cpu_job job = pool->job;
        worker->generation = pool->generation;
        pthread_mutex_unlock(&pool->mutex);
        cpu_compute_range(&job, worker->index, pool->thread_count);
        pthread_mutex_lock(&pool->mutex);
        if (--pool->pending == 0) {
            pthread_cond_signal(&pool->done_condition);
        }
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static bool cpu_pool_create(uint32_t thread_count, cpu_pool *pool) {
    memset(pool, 0, sizeof(*pool));
    pool->thread_count = thread_count;
    pool->workers = calloc(thread_count, sizeof(*pool->workers));
    if (pool->workers == NULL || pthread_mutex_init(&pool->mutex, NULL) != 0 ||
        pthread_cond_init(&pool->start_condition, NULL) != 0 ||
        pthread_cond_init(&pool->done_condition, NULL) != 0) {
        return false;
    }
    for (uint32_t index = 0; index < thread_count; ++index) {
        pool->workers[index].pool = pool;
        pool->workers[index].index = index;
        if (pthread_create(&pool->workers[index].thread, NULL, cpu_worker_main, &pool->workers[index]) != 0) {
            pthread_mutex_lock(&pool->mutex);
            pool->stopping = true;
            pthread_cond_broadcast(&pool->start_condition);
            pthread_mutex_unlock(&pool->mutex);
            for (uint32_t joined = 0; joined < index; ++joined) {
                pthread_join(pool->workers[joined].thread, NULL);
            }
            return false;
        }
    }
    return true;
}

static void cpu_pool_start(cpu_pool *pool, const cpu_job *job) {
    pthread_mutex_lock(&pool->mutex);
    pool->job = *job;
    pool->pending = pool->thread_count;
    ++pool->generation;
    pthread_cond_broadcast(&pool->start_condition);
    pthread_mutex_unlock(&pool->mutex);
}

static void cpu_pool_wait(cpu_pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->pending != 0) {
        pthread_cond_wait(&pool->done_condition, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

static void cpu_pool_destroy(cpu_pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->stopping = true;
    pthread_cond_broadcast(&pool->start_condition);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t index = 0; index < pool->thread_count; ++index) {
        pthread_join(pool->workers[index].thread, NULL);
    }
    pthread_cond_destroy(&pool->done_condition);
    pthread_cond_destroy(&pool->start_condition);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->workers);
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *) left;
    const double b = *(const double *) right;
    return (a > b) - (a < b);
}

static error_metrics calculate_error(const float *reference, const float *actual, size_t count,
    uint32_t output_columns) {
    error_metrics result = {0};
    double *absolute_errors = malloc(count * sizeof(*absolute_errors));
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const float expected = reference[index];
        const float observed = actual[index];
        if (isnan(observed)) {
            ++result.nan_count;
        }
        if (isinf(observed)) {
            ++result.inf_count;
        }
        const double absolute = fabs((double) observed - (double) expected);
        const double relative = absolute / fmax(fabs((double) expected), 1e-12);
        if (absolute_errors != NULL) {
            absolute_errors[index] = absolute;
        }
        sum += absolute;
        if (absolute > 2e-4 + 2e-5 * fabs((double) expected)) {
            ++result.tolerance_violation_count;
        }
        if (absolute > result.max_absolute) {
            result.max_absolute = absolute;
            result.worst_row = (uint32_t) (index / output_columns);
            result.worst_column = (uint32_t) (index % output_columns);
            result.worst_reference = expected;
            result.worst_actual = observed;
        }
        if (relative > result.max_relative) {
            result.max_relative = relative;
        }
    }
    result.mean_absolute = sum / (double) count;
    if (absolute_errors != NULL) {
        qsort(absolute_errors, count, sizeof(*absolute_errors), compare_double);
        const size_t p99_index = count == 0 ? 0 : (size_t) ceil(0.99 * (double) count) - 1U;
        result.p99_absolute = count == 0 ? 0.0 : absolute_errors[p99_index];
    } else {
        result.p99_absolute = NAN;
    }
    free(absolute_errors);
    return result;
}

static void print_u64_array(const uint64_t *values, uint32_t count) {
    putchar('[');
    for (uint32_t index = 0; index < count; ++index) {
        if (index != 0) {
            putchar(',');
        }
        printf("%llu", (unsigned long long) values[index]);
    }
    putchar(']');
}

static const char *mode_name(bench_mode mode) {
    return mode == BENCH_CPU ? "cpu" : mode == BENCH_QPU ? "qpu" : "hybrid";
}

static const char *weight_format_name(weight_format format) {
    return format == WEIGHT_Q4_0
        ? "q4_0" : format == WEIGHT_Q4_K ? "q4_k" : format == WEIGHT_Q6_K ? "q6_k" : "q8_0";
}

int main(int argc, char **argv) {
    options config;
    if (!parse_options(argc, argv, &config)) {
        usage(argv[0]);
        return 2;
    }
    const uint32_t block_elements = config.format == WEIGHT_Q4_0 || config.format == WEIGHT_Q8_0
        ? Q4_0_BLOCK_ELEMENTS : Q4_K_BLOCK_ELEMENTS;
    const uint32_t weight_block_bytes = config.format == WEIGHT_Q4_0
        ? Q4_0_BLOCK_BYTES
        : config.format == WEIGHT_Q4_K
            ? Q4_K_BLOCK_BYTES : config.format == WEIGHT_Q6_K ? Q6_K_BLOCK_BYTES : Q8_0_BLOCK_BYTES;
    const uint32_t activation_block_bytes =
        config.format == WEIGHT_Q4_0 || config.format == WEIGHT_Q8_0
            ? Q8_0_BLOCK_BYTES : Q8_K_BLOCK_BYTES;
    const size_t blocks = config.input_columns / block_elements;
    const size_t weight_row_bytes = blocks * weight_block_bytes;
    const size_t activation_row_bytes = blocks * activation_block_bytes;
    const size_t weight_bytes = (size_t) config.output_columns * weight_row_bytes;
    const size_t activation_f32_bytes =
        (size_t) config.rows * config.input_columns * sizeof(float);
    const size_t activation_q8_bytes = (size_t) config.rows * activation_row_bytes;
    const size_t output_elements = (size_t) config.rows * config.output_columns;
    const size_t output_bytes = output_elements * sizeof(float);
    uint8_t *weights = read_exact_file(config.weights_path, weight_bytes);
    float *activation_f32 = read_exact_file(config.activation_path, activation_f32_bytes);
    float *reference = malloc(output_bytes);
    if (weights == NULL || activation_f32 == NULL || reference == NULL) {
        free(reference);
        free(activation_f32);
        free(weights);
        return 1;
    }

    cpu_pool pool;
    if (!cpu_pool_create(config.cpu_threads, &pool)) {
        fprintf(stderr, "failed to create CPU worker pool\n");
        free(reference);
        free(activation_f32);
        free(weights);
        return 1;
    }

    qpu_llama_context *context = NULL;
    qpu_llama_q4_0_linear *linear = NULL;
    qpu_llama_q4_k_linear *linear_q4_k = NULL;
    qpu_llama_q6_k_linear *linear_q6_k = NULL;
    qpu_llama_q8_0_linear *linear_q8_0 = NULL;
    uint8_t *activation_q8 = malloc(activation_q8_bytes);
    float *destination = malloc(output_bytes);
    uint64_t prepare_ns = 0;
    qpu_llama_status status = activation_q8 == NULL || destination == NULL
        ? QPU_LLAMA_ALLOCATION_FAILED
        : QPU_LLAMA_OK;
    if (config.mode != BENCH_CPU && status == QPU_LLAMA_OK) {
        const uint64_t prepare_start = monotonic_ns();
        status = qpu_llama_context_create(NULL, &context);
        if (status == QPU_LLAMA_OK) {
            if (config.format == WEIGHT_Q4_0) {
                const qpu_llama_q4_0_linear_desc desc = {
                    .weights = weights,
                    .weight_size = weight_bytes,
                    .input_columns = config.input_columns,
                    .output_columns = config.output_columns,
                    .rows = config.rows,
                    .resident_column_start = config.qpu_column_start,
                    .resident_column_count = config.qpu_column_count,
                    .expected_source_hash = config.rows == 1
                        ? qpu_ggml_q4_0_q8_0_m1_source_hash
                        : qpu_ggml_q4_0_q8_0_m4_source_hash,
                };
                status = qpu_llama_q4_0_linear_prepare(context, &desc, &linear);
            } else if (config.format == WEIGHT_Q4_K) {
                const qpu_llama_q4_k_linear_desc desc = {
                    .weights = weights,
                    .weight_size = weight_bytes,
                    .input_columns = config.input_columns,
                    .output_columns = config.output_columns,
                    .resident_column_start = config.qpu_column_start,
                    .resident_column_count = config.qpu_column_count,
                    .workgroups_per_supergroup = config.qpu_wgs_per_sg,
                    .expected_source_hash = qpu_ggml_q4_k_q8_k_m4_source_hash,
                };
                status = qpu_llama_q4_k_linear_prepare(context, &desc, &linear_q4_k);
            } else if (config.format == WEIGHT_Q6_K) {
                const qpu_llama_q6_k_linear_desc desc = {
                    .weights = weights,
                    .weight_size = weight_bytes,
                    .input_columns = config.input_columns,
                    .output_columns = config.output_columns,
                    .resident_column_start = config.qpu_column_start,
                    .resident_column_count = config.qpu_column_count,
                    .workgroups_per_supergroup = config.qpu_wgs_per_sg,
                    .expected_source_hash = qpu_ggml_q6_k_q8_k_m4_source_hash,
                };
                status = qpu_llama_q6_k_linear_prepare(context, &desc, &linear_q6_k);
            } else {
                const qpu_llama_q8_0_linear_desc desc = {
                    .weights = weights,
                    .weight_size = weight_bytes,
                    .input_columns = config.input_columns,
                    .output_columns = config.output_columns,
                    .resident_column_start = config.qpu_column_start,
                    .resident_column_count = config.qpu_column_count,
                    .workgroups_per_supergroup = config.qpu_wgs_per_sg,
                    .expected_source_hash = qpu_ggml_q8_0_q8_0_m4_source_hash,
                };
                status = qpu_llama_q8_0_linear_prepare(context, &desc, &linear_q8_0);
            }
        }
        prepare_ns = monotonic_ns() - prepare_start;
    }
    if (status != QPU_LLAMA_OK) {
        fprintf(stderr, "setup: %s%s%s\n", qpu_llama_status_string(status),
            context == NULL ? "" : " (", context == NULL ? "" : qpu_llama_context_last_error(context));
        qpu_llama_q8_0_linear_destroy(linear_q8_0);
        qpu_llama_q6_k_linear_destroy(linear_q6_k);
        qpu_llama_q4_k_linear_destroy(linear_q4_k);
        qpu_llama_q4_0_linear_destroy(linear);
        qpu_llama_context_destroy(context);
        free(destination);
        free(activation_q8);
        cpu_pool_destroy(&pool);
        free(reference);
        free(activation_f32);
        free(weights);
        return 1;
    }

    for (uint32_t row = 0; row < config.rows; ++row) {
        if (config.format == WEIGHT_Q4_0 || config.format == WEIGHT_Q8_0) {
            quantize_row_q8_0(activation_f32 + (size_t) row * config.input_columns,
                activation_q8 + (size_t) row * activation_row_bytes, config.input_columns);
        } else {
            quantize_row_q8_K(activation_f32 + (size_t) row * config.input_columns,
                activation_q8 + (size_t) row * activation_row_bytes, config.input_columns);
        }
    }
    cpu_job full_job = {
        .weights = weights,
        .activation = activation_q8,
        .destination = reference,
        .input_columns = config.input_columns,
        .output_columns = config.output_columns,
        .rows = config.rows,
        .column_start = 0,
        .column_count = config.output_columns,
        .weight_row_bytes = weight_row_bytes,
        .activation_row_bytes = activation_row_bytes,
        .format = config.format,
    };
    cpu_pool_start(&pool, &full_job);
    cpu_pool_wait(&pool);

    uint64_t *complete_samples = calloc(config.samples, sizeof(uint64_t));
    uint64_t *quantize_samples = calloc(config.samples, sizeof(uint64_t));
    uint64_t *input_copy_samples = calloc(config.samples, sizeof(uint64_t));
    uint64_t *submit_wait_samples = calloc(config.samples, sizeof(uint64_t));
    uint64_t *output_copy_samples = calloc(config.samples, sizeof(uint64_t));
    if (complete_samples == NULL || quantize_samples == NULL || input_copy_samples == NULL ||
        submit_wait_samples == NULL || output_copy_samples == NULL) {
        fprintf(stderr, "sample allocation failed\n");
        return 1;
    }
    cpu_job candidate_job = full_job;
    candidate_job.destination = destination;
    candidate_job.column_count = config.qpu_column_start;
    qpu_llama_q4_0_execution execution = {
        .activation = activation_q8,
        .activation_size = activation_q8_bytes,
        .activation_offset = 0,
        .destination = destination,
        .destination_size = output_bytes,
        .destination_offset = 0,
        .column_start = config.qpu_column_start,
        .column_count = config.qpu_column_count,
    };
    qpu_llama_q4_k_execution execution_q4_k = {
        .activation = activation_q8,
        .activation_size = activation_q8_bytes,
        .activation_offset = 0,
        .destination = destination,
        .destination_size = output_bytes,
        .destination_offset = 0,
        .column_start = config.qpu_column_start,
        .column_count = config.qpu_column_count,
    };
    qpu_llama_q6_k_execution execution_q6_k = {
        .activation = activation_q8,
        .activation_size = activation_q8_bytes,
        .activation_offset = 0,
        .destination = destination,
        .destination_size = output_bytes,
        .destination_offset = 0,
        .column_start = config.qpu_column_start,
        .column_count = config.qpu_column_count,
    };
    qpu_llama_q8_0_execution execution_q8_0 = {
        .activation = activation_q8,
        .activation_size = activation_q8_bytes,
        .activation_offset = 0,
        .destination = destination,
        .destination_size = output_bytes,
        .destination_offset = 0,
        .column_start = config.qpu_column_start,
        .column_count = config.qpu_column_count,
    };
    const uint32_t iterations = config.warmups + config.samples;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const uint64_t complete_start = monotonic_ns();
        const uint64_t quantize_start = complete_start;
        for (uint32_t row = 0; row < config.rows; ++row) {
            if (config.format == WEIGHT_Q4_0 || config.format == WEIGHT_Q8_0) {
                quantize_row_q8_0(activation_f32 + (size_t) row * config.input_columns,
                    activation_q8 + (size_t) row * activation_row_bytes, config.input_columns);
            } else {
                quantize_row_q8_K(activation_f32 + (size_t) row * config.input_columns,
                    activation_q8 + (size_t) row * activation_row_bytes, config.input_columns);
            }
        }
        const uint64_t quantize_end = monotonic_ns();
        qpu_llama_q4_0_timing qpu_timing = {0};
        qpu_llama_q4_k_timing qpu_timing_q4_k = {0};
        qpu_llama_q6_k_timing qpu_timing_q6_k = {0};
        qpu_llama_q8_0_timing qpu_timing_q8_0 = {0};
        if (config.mode == BENCH_CPU) {
            candidate_job.column_count = config.output_columns;
            cpu_pool_start(&pool, &candidate_job);
            cpu_pool_wait(&pool);
        } else if (config.mode == BENCH_QPU) {
            if (config.format == WEIGHT_Q4_0) {
                status = qpu_llama_q4_0_linear_execute(linear, &execution, &qpu_timing);
            } else if (config.format == WEIGHT_Q4_K) {
                status = qpu_llama_q4_k_linear_execute(
                    linear_q4_k, &execution_q4_k, &qpu_timing_q4_k);
            } else if (config.format == WEIGHT_Q6_K) {
                status = qpu_llama_q6_k_linear_execute(
                    linear_q6_k, &execution_q6_k, &qpu_timing_q6_k);
            } else {
                status = qpu_llama_q8_0_linear_execute(
                    linear_q8_0, &execution_q8_0, &qpu_timing_q8_0);
            }
        } else {
            cpu_pool_start(&pool, &candidate_job);
            if (config.format == WEIGHT_Q4_0) {
                status = qpu_llama_q4_0_linear_execute(linear, &execution, &qpu_timing);
            } else if (config.format == WEIGHT_Q4_K) {
                status = qpu_llama_q4_k_linear_execute(
                    linear_q4_k, &execution_q4_k, &qpu_timing_q4_k);
            } else if (config.format == WEIGHT_Q6_K) {
                status = qpu_llama_q6_k_linear_execute(
                    linear_q6_k, &execution_q6_k, &qpu_timing_q6_k);
            } else {
                status = qpu_llama_q8_0_linear_execute(
                    linear_q8_0, &execution_q8_0, &qpu_timing_q8_0);
            }
            cpu_pool_wait(&pool);
        }
        const uint64_t complete_end = monotonic_ns();
        if (status != QPU_LLAMA_OK) {
            fprintf(stderr, "execute: %s (%s)\n", qpu_llama_status_string(status),
                qpu_llama_context_last_error(context));
            return 1;
        }
        if (iteration >= config.warmups) {
            const uint32_t sample = iteration - config.warmups;
            complete_samples[sample] = complete_end - complete_start;
            quantize_samples[sample] = quantize_end - quantize_start;
            input_copy_samples[sample] = config.format == WEIGHT_Q4_0
                ? qpu_timing.input_copy_ns
                : config.format == WEIGHT_Q4_K
                    ? qpu_timing_q4_k.input_copy_ns
                    : config.format == WEIGHT_Q6_K
                        ? qpu_timing_q6_k.input_copy_ns : qpu_timing_q8_0.input_copy_ns;
            submit_wait_samples[sample] = config.format == WEIGHT_Q4_0
                ? qpu_timing.submit_wait_ns
                : config.format == WEIGHT_Q4_K
                    ? qpu_timing_q4_k.submit_wait_ns
                    : config.format == WEIGHT_Q6_K
                        ? qpu_timing_q6_k.submit_wait_ns : qpu_timing_q8_0.submit_wait_ns;
            output_copy_samples[sample] = config.format == WEIGHT_Q4_0
                ? qpu_timing.output_copy_ns
                : config.format == WEIGHT_Q4_K
                    ? qpu_timing_q4_k.output_copy_ns
                    : config.format == WEIGHT_Q6_K
                        ? qpu_timing_q6_k.output_copy_ns : qpu_timing_q8_0.output_copy_ns;
        }
    }

    const bool output_written = config.output_path != NULL &&
        write_exact_file(config.output_path, destination, output_bytes);
    const error_metrics errors = calculate_error(reference, destination, output_elements, config.output_columns);
    printf("{\"schema_version\":1,\"kind\":\"llama-qpu-native-operator-samples\","
        "\"mode\":\"%s\",\"weight_type\":\"%s\",\"input_columns\":%u,"
        "\"output_columns\":%u,\"rows\":%u,"
        "\"cpu_threads\":%u,\"qpu_column_start\":%u,\"qpu_column_count\":%u,"
        "\"qpu_wgs_per_sg\":%u,"
        "\"warmups\":%u,\"retained_samples\":%u,\"prepare_ns\":%llu,"
        "\"resident_bytes\":%zu,\"output_bin_requested\":%s,\"output_bin_written\":%s,"
        "\"activation_contract\":\"native-%s-from-f32-each-sample\","
        "\"cpu_kernel\":\"%s\",\"complete_ns\":",
        mode_name(config.mode), weight_format_name(config.format), config.input_columns,
        config.output_columns, config.rows,
        config.cpu_threads, config.qpu_column_start, config.qpu_column_count,
        config.qpu_wgs_per_sg, config.warmups,
        config.samples, (unsigned long long) prepare_ns,
        config.format == WEIGHT_Q4_0 ? qpu_llama_q4_0_linear_resident_bytes(linear)
            : config.format == WEIGHT_Q4_K
                ? qpu_llama_q4_k_linear_resident_bytes(linear_q4_k)
                : config.format == WEIGHT_Q6_K
                    ? qpu_llama_q6_k_linear_resident_bytes(linear_q6_k)
                    : qpu_llama_q8_0_linear_resident_bytes(linear_q8_0),
        config.output_path == NULL ? "false" : "true", output_written ? "true" : "false",
        config.format == WEIGHT_Q4_0 || config.format == WEIGHT_Q8_0 ? "q8-0" : "q8-k",
        config.format == WEIGHT_Q4_0
            ? "ggml_vec_dot_q4_0_q8_0"
            : config.format == WEIGHT_Q4_K
                ? "ggml_vec_dot_q4_K_q8_K"
                : config.format == WEIGHT_Q6_K
                    ? "ggml_vec_dot_q6_K_q8_K" : "ggml_vec_dot_q8_0_q8_0");
    print_u64_array(complete_samples, config.samples);
    printf(",\"quantize_ns\":");
    print_u64_array(quantize_samples, config.samples);
    printf(",\"qpu_input_copy_ns\":");
    print_u64_array(input_copy_samples, config.samples);
    printf(",\"qpu_submit_wait_ns\":");
    print_u64_array(submit_wait_samples, config.samples);
    printf(",\"qpu_output_copy_ns\":");
    print_u64_array(output_copy_samples, config.samples);
    printf(",\"correctness\":{\"max_absolute\":%.17g,\"max_relative\":%.17g,"
        "\"mean_absolute\":%.17g,\"p99_absolute\":%.17g,\"atol\":0.0002,"
        "\"rtol\":0.00002,\"tolerance_violation_count\":%llu,\"nan_count\":%llu,"
        "\"inf_count\":%llu,\"worst_row\":%u,\"worst_column\":%u,"
        "\"worst_reference\":%.9g,\"worst_actual\":%.9g}}\n",
        errors.max_absolute, errors.max_relative, errors.mean_absolute, errors.p99_absolute,
        (unsigned long long) errors.tolerance_violation_count,
        (unsigned long long) errors.nan_count, (unsigned long long) errors.inf_count,
        errors.worst_row, errors.worst_column, errors.worst_reference, errors.worst_actual);

    free(output_copy_samples);
    free(submit_wait_samples);
    free(input_copy_samples);
    free(quantize_samples);
    free(complete_samples);
    qpu_llama_q8_0_linear_destroy(linear_q8_0);
    qpu_llama_q6_k_linear_destroy(linear_q6_k);
    qpu_llama_q4_k_linear_destroy(linear_q4_k);
    qpu_llama_q4_0_linear_destroy(linear);
    qpu_llama_context_destroy(context);
    free(destination);
    free(activation_q8);
    cpu_pool_destroy(&pool);
    free(reference);
    free(activation_f32);
    free(weights);
    return errors.nan_count == 0 && errors.inf_count == 0 &&
            (config.output_path == NULL || output_written)
        ? 0
        : 1;
}
