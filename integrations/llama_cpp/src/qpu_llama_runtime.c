#include "qpu_llama_runtime.h"

#include "qpu_llama_sha256.h"

#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define QPU_LLAMA_DEFAULT_TIMEOUT_NS UINT64_C(10000000000)
#define QPU_LLAMA_MAX_SUBMIT_BUFFERS 64U

struct qpu_llama_context {
    int fd;
    uint64_t timeout_ns;
    uint32_t failure_mask;
    bool poisoned;
    bool submission_active;
    pthread_mutex_t mutex;
    qpu_llama_capabilities capabilities;
    char last_error[256];
};

struct qpu_llama_buffer {
    qpu_llama_context *context;
    uint32_t handle;
    uint32_t gpu_address;
    size_t size;
    void *mapping;
    int dmabuf_fd;
};

struct qpu_llama_program {
    qpu_llama_context *context;
    qpu_llama_buffer *code;
    qpu_llama_buffer *uniforms;
    uint32_t uniform_word_count;
};

struct qpu_llama_submission {
    qpu_llama_context *context;
    uint32_t handles[QPU_LLAMA_MAX_SUBMIT_BUFFERS];
    uint32_t handle_count;
    bool complete;
};

static void set_error(qpu_llama_context *context, const char *operation) {
    if (context == NULL) {
        return;
    }
    const int saved_errno = errno;
    if (saved_errno != 0) {
        snprintf(context->last_error, sizeof(context->last_error), "%s: %s", operation, strerror(saved_errno));
    } else {
        snprintf(context->last_error, sizeof(context->last_error), "%s", operation);
    }
}

static qpu_llama_status query_parameter(qpu_llama_context *context, uint32_t parameter, uint64_t *value) {
    struct drm_v3d_get_param request = {.param = parameter, .pad = 0, .value = 0};
    if (ioctl(context->fd, DRM_IOCTL_V3D_GET_PARAM, &request) != 0) {
        set_error(context, "DRM_IOCTL_V3D_GET_PARAM");
        return QPU_LLAMA_IO_FAILED;
    }
    *value = request.value;
    return QPU_LLAMA_OK;
}

static bool is_v3d_fd(int fd) {
    struct drm_v3d_get_param request = {
        .param = DRM_V3D_PARAM_SUPPORTS_CSD,
        .pad = 0,
        .value = 0,
    };
    return ioctl(fd, DRM_IOCTL_V3D_GET_PARAM, &request) == 0 && request.value != 0;
}

static int open_render_node(const char *requested, char resolved[128]) {
    if (requested != NULL && requested[0] != '\0') {
        const int fd = open(requested, O_RDWR | O_CLOEXEC);
        if (fd >= 0 && is_v3d_fd(fd)) {
            snprintf(resolved, 128, "%s", requested);
            return fd;
        }
        if (fd >= 0) {
            close(fd);
        }
        errno = ENODEV;
        return -1;
    }
    for (unsigned int minor = 128; minor < 192; ++minor) {
        char path[128];
        snprintf(path, sizeof(path), "/dev/dri/renderD%u", minor);
        const int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        if (is_v3d_fd(fd)) {
            snprintf(resolved, 128, "%s", path);
            return fd;
        }
        close(fd);
    }
    errno = ENODEV;
    return -1;
}

static void destroy_buffer_unlocked(qpu_llama_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    if (buffer->mapping != NULL && buffer->mapping != MAP_FAILED) {
        munmap(buffer->mapping, buffer->size);
    }
    if (buffer->context != NULL && buffer->context->fd >= 0 && buffer->handle != 0) {
        struct drm_gem_close request = {.handle = buffer->handle, .pad = 0};
        ioctl(buffer->context->fd, DRM_IOCTL_GEM_CLOSE, &request);
    }
    if (buffer->dmabuf_fd >= 0) {
        close(buffer->dmabuf_fd);
    }
    free(buffer);
}

static qpu_llama_status create_buffer_unlocked(
    qpu_llama_context *context,
    size_t size,
    qpu_llama_buffer **result) {
    if (size == 0 || size > UINT32_MAX || result == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    if ((context->failure_mask & QPU_LLAMA_FAIL_ALLOCATION) != 0) {
        errno = ENOMEM;
        set_error(context, "injected BO allocation failure");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    qpu_llama_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        set_error(context, "allocate BO metadata");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    buffer->dmabuf_fd = -1;
    struct drm_v3d_create_bo create = {.size = (uint32_t) size, .flags = 0, .handle = 0, .offset = 0};
    if (ioctl(context->fd, DRM_IOCTL_V3D_CREATE_BO, &create) != 0) {
        set_error(context, "DRM_IOCTL_V3D_CREATE_BO");
        free(buffer);
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    buffer->context = context;
    buffer->handle = create.handle;
    buffer->gpu_address = create.offset;
    buffer->size = size;
    struct drm_v3d_mmap_bo map = {.handle = create.handle, .flags = 0, .offset = 0};
    if (ioctl(context->fd, DRM_IOCTL_V3D_MMAP_BO, &map) != 0) {
        set_error(context, "DRM_IOCTL_V3D_MMAP_BO");
        destroy_buffer_unlocked(buffer);
        return QPU_LLAMA_IO_FAILED;
    }
    buffer->mapping = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, context->fd, (off_t) map.offset);
    if (buffer->mapping == MAP_FAILED) {
        buffer->mapping = NULL;
        set_error(context, "mmap V3D BO");
        destroy_buffer_unlocked(buffer);
        return QPU_LLAMA_IO_FAILED;
    }
    *result = buffer;
    return QPU_LLAMA_OK;
}

static qpu_llama_status create_cached_buffer_unlocked(
    qpu_llama_context *context,
    size_t size,
    qpu_llama_buffer **result) {
    if (size == 0 || size > UINT32_MAX || result == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    if ((context->failure_mask & QPU_LLAMA_FAIL_ALLOCATION) != 0) {
        errno = ENOMEM;
        set_error(context, "injected cached allocation failure");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    const char *heap_path = getenv("QPU_LLAMA_DMA_HEAP");
    if (heap_path == NULL || heap_path[0] == '\0') {
        heap_path = "/dev/dma_heap/vidbuf_cached";
    }
    const int heap_fd = open(heap_path, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        set_error(context, "open cached DMA heap");
        return QPU_LLAMA_DEVICE_UNAVAILABLE;
    }
    struct dma_heap_allocation_data allocation = {
        .len = size,
        .fd = 0,
        .fd_flags = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation) != 0) {
        const int allocation_errno = errno;
        close(heap_fd);
        errno = allocation_errno;
        set_error(context, "DMA_HEAP_IOCTL_ALLOC");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    close(heap_fd);

    qpu_llama_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        close((int) allocation.fd);
        set_error(context, "allocate cached BO metadata");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    buffer->context = context;
    buffer->size = size;
    buffer->dmabuf_fd = (int) allocation.fd;
    struct drm_prime_handle prime = {
        .handle = 0,
        .flags = 0,
        .fd = buffer->dmabuf_fd,
    };
    if (ioctl(context->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime) != 0) {
        set_error(context, "DRM_IOCTL_PRIME_FD_TO_HANDLE");
        destroy_buffer_unlocked(buffer);
        return QPU_LLAMA_IO_FAILED;
    }
    buffer->handle = prime.handle;
    struct drm_v3d_get_bo_offset offset = {
        .handle = buffer->handle,
        .offset = 0,
    };
    if (ioctl(context->fd, DRM_IOCTL_V3D_GET_BO_OFFSET, &offset) != 0) {
        set_error(context, "DRM_IOCTL_V3D_GET_BO_OFFSET");
        destroy_buffer_unlocked(buffer);
        return QPU_LLAMA_IO_FAILED;
    }
    buffer->gpu_address = offset.offset;
    buffer->mapping = mmap(
        NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, buffer->dmabuf_fd, 0);
    if (buffer->mapping == MAP_FAILED) {
        buffer->mapping = NULL;
        set_error(context, "mmap cached DMA buffer");
        destroy_buffer_unlocked(buffer);
        return QPU_LLAMA_IO_FAILED;
    }
    *result = buffer;
    return QPU_LLAMA_OK;
}

qpu_llama_status qpu_llama_context_create(
    const qpu_llama_context_config *config,
    qpu_llama_context **result) {
    if (result == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    qpu_llama_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    context->fd = -1;
    context->timeout_ns = config != NULL && config->timeout_ns != 0
        ? config->timeout_ns
        : QPU_LLAMA_DEFAULT_TIMEOUT_NS;
    context->failure_mask = config != NULL ? config->failure_mask : 0;
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        free(context);
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    if ((context->failure_mask & QPU_LLAMA_FAIL_DEVICE_OPEN) != 0) {
        errno = ENODEV;
        set_error(context, "injected V3D device absence");
        qpu_llama_context_destroy(context);
        return QPU_LLAMA_DEVICE_UNAVAILABLE;
    }
    const char *render_node = config != NULL ? config->render_node : NULL;
    context->fd = open_render_node(render_node, context->capabilities.render_node);
    if (context->fd < 0) {
        set_error(context, "open V3D render node");
        qpu_llama_context_destroy(context);
        return QPU_LLAMA_DEVICE_UNAVAILABLE;
    }
    uint64_t supports_csd = 0;
    uint64_t supports_tfu = 0;
    qpu_llama_status status = query_parameter(context, DRM_V3D_PARAM_SUPPORTS_CSD, &supports_csd);
    if (status == QPU_LLAMA_OK) {
        status = query_parameter(context, DRM_V3D_PARAM_SUPPORTS_TFU, &supports_tfu);
    }
    if (status != QPU_LLAMA_OK || supports_csd == 0) {
        if (status == QPU_LLAMA_OK) {
            set_error(context, "V3D compute shaders are not supported");
        }
        qpu_llama_context_destroy(context);
        return status == QPU_LLAMA_OK ? QPU_LLAMA_CAPABILITY_MISSING : status;
    }
    context->capabilities.supports_csd = (uint8_t) supports_csd;
    context->capabilities.supports_tfu = (uint8_t) supports_tfu;
    const uint32_t parameters[] = {
        DRM_V3D_PARAM_V3D_HUB_IDENT1,
        DRM_V3D_PARAM_V3D_HUB_IDENT2,
        DRM_V3D_PARAM_V3D_HUB_IDENT3,
        DRM_V3D_PARAM_V3D_CORE0_IDENT0,
        DRM_V3D_PARAM_V3D_CORE0_IDENT1,
        DRM_V3D_PARAM_V3D_CORE0_IDENT2,
    };
    uint64_t *destinations[] = {
        &context->capabilities.hub_ident1,
        &context->capabilities.hub_ident2,
        &context->capabilities.hub_ident3,
        &context->capabilities.core0_ident0,
        &context->capabilities.core0_ident1,
        &context->capabilities.core0_ident2,
    };
    for (size_t index = 0; index < sizeof(parameters) / sizeof(parameters[0]); ++index) {
        status = query_parameter(context, parameters[index], destinations[index]);
        if (status != QPU_LLAMA_OK) {
            qpu_llama_context_destroy(context);
            return status;
        }
    }
    *result = context;
    return QPU_LLAMA_OK;
}

void qpu_llama_context_destroy(qpu_llama_context *context) {
    if (context == NULL) {
        return;
    }
    if (context->fd >= 0) {
        close(context->fd);
    }
    pthread_mutex_destroy(&context->mutex);
    free(context);
}

const char *qpu_llama_context_last_error(const qpu_llama_context *context) {
    return context != NULL ? context->last_error : "no context";
}

qpu_llama_status qpu_llama_context_capabilities(
    const qpu_llama_context *context,
    qpu_llama_capabilities *result) {
    if (context == NULL || result == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = context->capabilities;
    return QPU_LLAMA_OK;
}

void qpu_llama_context_set_failure_mask(qpu_llama_context *context, uint32_t failure_mask) {
    if (context == NULL) {
        return;
    }
    pthread_mutex_lock(&context->mutex);
    context->failure_mask = failure_mask;
    pthread_mutex_unlock(&context->mutex);
}

qpu_llama_status qpu_llama_buffer_create(
    qpu_llama_context *context,
    size_t size,
    qpu_llama_buffer **result) {
    if (context == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&context->mutex);
    const qpu_llama_status status = create_buffer_unlocked(context, size, result);
    pthread_mutex_unlock(&context->mutex);
    return status;
}

qpu_llama_status qpu_llama_buffer_create_cached(
    qpu_llama_context *context,
    size_t size,
    qpu_llama_buffer **result) {
    if (context == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&context->mutex);
    const qpu_llama_status status =
        create_cached_buffer_unlocked(context, size, result);
    pthread_mutex_unlock(&context->mutex);
    return status;
}

void qpu_llama_buffer_destroy(qpu_llama_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    qpu_llama_context *context = buffer->context;
    pthread_mutex_lock(&context->mutex);
    destroy_buffer_unlocked(buffer);
    pthread_mutex_unlock(&context->mutex);
}

void *qpu_llama_buffer_data(qpu_llama_buffer *buffer) {
    return buffer != NULL ? buffer->mapping : NULL;
}

size_t qpu_llama_buffer_size(const qpu_llama_buffer *buffer) {
    return buffer != NULL ? buffer->size : 0;
}

qpu_llama_status qpu_llama_buffer_gpu_address(
    const qpu_llama_buffer *buffer,
    size_t byte_offset,
    uint32_t *result) {
    if (buffer == NULL || result == NULL || byte_offset >= buffer->size || byte_offset > UINT32_MAX - buffer->gpu_address) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = buffer->gpu_address + (uint32_t) byte_offset;
    return QPU_LLAMA_OK;
}

static qpu_llama_status buffer_cpu_access(
    qpu_llama_buffer *buffer,
    qpu_llama_cpu_access access,
    bool begin) {
    if (buffer == NULL ||
        (access != QPU_LLAMA_CPU_ACCESS_READ &&
         access != QPU_LLAMA_CPU_ACCESS_WRITE &&
         access != QPU_LLAMA_CPU_ACCESS_READ_WRITE)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    atomic_thread_fence(memory_order_seq_cst);
    if (buffer->dmabuf_fd < 0) {
        return QPU_LLAMA_OK;
    }
    uint64_t direction = DMA_BUF_SYNC_RW;
    if (access == QPU_LLAMA_CPU_ACCESS_READ) {
        direction = DMA_BUF_SYNC_READ;
    } else if (access == QPU_LLAMA_CPU_ACCESS_WRITE) {
        direction = DMA_BUF_SYNC_WRITE;
    }
    struct dma_buf_sync sync = {
        .flags = direction | (begin ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END),
    };
    pthread_mutex_lock(&buffer->context->mutex);
    if (ioctl(buffer->dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        set_error(buffer->context, "DMA_BUF_IOCTL_SYNC");
        pthread_mutex_unlock(&buffer->context->mutex);
        return QPU_LLAMA_IO_FAILED;
    }
    pthread_mutex_unlock(&buffer->context->mutex);
    atomic_thread_fence(memory_order_seq_cst);
    return QPU_LLAMA_OK;
}

qpu_llama_status qpu_llama_buffer_cpu_access_begin(
    qpu_llama_buffer *buffer,
    qpu_llama_cpu_access access) {
    return buffer_cpu_access(buffer, access, true);
}

qpu_llama_status qpu_llama_buffer_cpu_access_end(
    qpu_llama_buffer *buffer,
    qpu_llama_cpu_access access) {
    return buffer_cpu_access(buffer, access, false);
}

static bool valid_hash(const char *value) {
    if (value == NULL || strlen(value) != 64) {
        return false;
    }
    return strspn(value, "0123456789abcdef") == 64;
}

qpu_llama_status qpu_llama_validate_program(const qpu_llama_program_desc *desc) {
    if (desc == NULL || desc->code == NULL || desc->code_size == 0 || desc->code_size % sizeof(uint64_t) != 0 ||
        desc->uniform_word_count == 0 || !valid_hash(desc->compiled_source_hash) ||
        !valid_hash(desc->expected_source_hash) || !valid_hash(desc->binary_sha256)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    if (strcmp(desc->compiled_source_hash, desc->expected_source_hash) != 0) {
        return QPU_LLAMA_HASH_MISMATCH;
    }
    char actual_hash[65];
    qpu_llama_sha256_hex(desc->code, desc->code_size, actual_hash);
    return strcmp(actual_hash, desc->binary_sha256) == 0 ? QPU_LLAMA_OK : QPU_LLAMA_HASH_MISMATCH;
}

qpu_llama_status qpu_llama_program_create(
    qpu_llama_context *context,
    const qpu_llama_program_desc *desc,
    qpu_llama_program **result) {
    if (context == NULL || result == NULL) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    if ((context->failure_mask & QPU_LLAMA_FAIL_SOURCE_HASH) != 0) {
        set_error(context, "injected source hash mismatch");
        return QPU_LLAMA_HASH_MISMATCH;
    }
    const qpu_llama_status validation = qpu_llama_validate_program(desc);
    if (validation != QPU_LLAMA_OK) {
        set_error(context, "QPU program validation failed");
        return validation;
    }
    qpu_llama_program *program = calloc(1, sizeof(*program));
    if (program == NULL) {
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    program->context = context;
    program->uniform_word_count = desc->uniform_word_count;
    pthread_mutex_lock(&context->mutex);
    qpu_llama_status status = create_buffer_unlocked(context, desc->code_size, &program->code);
    if (status == QPU_LLAMA_OK) {
        status = create_buffer_unlocked(
            context,
            (size_t) desc->uniform_word_count * sizeof(uint32_t),
            &program->uniforms);
    }
    if (status == QPU_LLAMA_OK) {
        memcpy(program->code->mapping, desc->code, desc->code_size);
        atomic_thread_fence(memory_order_seq_cst);
    }
    if (status != QPU_LLAMA_OK) {
        destroy_buffer_unlocked(program->uniforms);
        destroy_buffer_unlocked(program->code);
        free(program);
    } else {
        *result = program;
    }
    pthread_mutex_unlock(&context->mutex);
    return status;
}

void qpu_llama_program_destroy(qpu_llama_program *program) {
    if (program == NULL) {
        return;
    }
    qpu_llama_context *context = program->context;
    pthread_mutex_lock(&context->mutex);
    destroy_buffer_unlocked(program->uniforms);
    destroy_buffer_unlocked(program->code);
    free(program);
    pthread_mutex_unlock(&context->mutex);
}

static uint32_t divide_round_up(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1U) / denominator;
}

static void append_handle(uint32_t *handles, uint32_t *count, uint32_t handle) {
    for (uint32_t index = 0; index < *count; ++index) {
        if (handles[index] == handle) {
            return;
        }
    }
    handles[(*count)++] = handle;
}

qpu_llama_status qpu_llama_program_submit(
    qpu_llama_program *program,
    const qpu_llama_dispatch_desc *dispatch,
    qpu_llama_submission **result) {
    if (program == NULL || dispatch == NULL || dispatch->uniforms == NULL ||
        result == NULL ||
        dispatch->uniform_word_count != program->uniform_word_count ||
        dispatch->buffer_count > QPU_LLAMA_MAX_SUBMIT_BUFFERS - 2U ||
        (dispatch->buffer_count != 0 && dispatch->buffers == NULL)) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    *result = NULL;
    uint64_t local_size = 1;
    for (size_t index = 0; index < 3; ++index) {
        if (dispatch->local_invocation[index] == 0 || dispatch->workgroup[index] == 0 ||
            dispatch->workgroup[index] > UINT16_MAX) {
            return QPU_LLAMA_INVALID_ARGUMENT;
        }
        local_size *= dispatch->local_invocation[index];
    }
    if (local_size > UINT8_MAX || dispatch->wgs_per_sg == 0 || dispatch->wgs_per_sg > UINT8_MAX ||
        dispatch->thread_count == 0) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_context *context = program->context;
    qpu_llama_submission *submission = calloc(1, sizeof(*submission));
    if (submission == NULL) {
        set_error(context, "allocate submission metadata");
        return QPU_LLAMA_ALLOCATION_FAILED;
    }
    submission->context = context;
    pthread_mutex_lock(&context->mutex);
    if (context->poisoned || context->submission_active) {
        const bool poisoned = context->poisoned;
        set_error(context, context->poisoned
            ? "V3D context is poisoned after a wait failure"
            : "V3D context already has an active submission");
        pthread_mutex_unlock(&context->mutex);
        free(submission);
        return poisoned ? QPU_LLAMA_IO_FAILED : QPU_LLAMA_INTERNAL_ERROR;
    }
    if ((context->failure_mask & QPU_LLAMA_FAIL_SUBMISSION) != 0) {
        set_error(context, "injected CSD submission failure");
        pthread_mutex_unlock(&context->mutex);
        free(submission);
        return QPU_LLAMA_SUBMISSION_FAILED;
    }
    memcpy(program->uniforms->mapping, dispatch->uniforms, dispatch->uniform_word_count * sizeof(uint32_t));
    atomic_thread_fence(memory_order_seq_cst);

    append_handle(submission->handles, &submission->handle_count, program->code->handle);
    append_handle(submission->handles, &submission->handle_count, program->uniforms->handle);
    for (uint32_t index = 0; index < dispatch->buffer_count; ++index) {
        if (dispatch->buffers[index] == NULL || dispatch->buffers[index]->context != context) {
            pthread_mutex_unlock(&context->mutex);
            free(submission);
            return QPU_LLAMA_INVALID_ARGUMENT;
        }
        append_handle(
            submission->handles,
            &submission->handle_count,
            dispatch->buffers[index]->handle);
    }
    const uint32_t groups_per_batch = divide_round_up(dispatch->wgs_per_sg * (uint32_t) local_size, 16U);
    uint32_t uniform_address = 0;
    qpu_llama_status status = qpu_llama_buffer_gpu_address(program->uniforms, 0, &uniform_address);
    if (status != QPU_LLAMA_OK) {
        pthread_mutex_unlock(&context->mutex);
        free(submission);
        return status;
    }
    struct drm_v3d_submit_csd request = {
        .cfg = {
            dispatch->workgroup[0] << 16U,
            dispatch->workgroup[1] << 16U,
            dispatch->workgroup[2] << 16U,
            ((groups_per_batch - 1U) << 12U) | (dispatch->wgs_per_sg << 8U) | (uint32_t) local_size,
            dispatch->thread_count,
            program->code->gpu_address | ((uint32_t) dispatch->propagate_nan << 2U) |
                ((uint32_t) dispatch->single_segment << 1U) | (uint32_t) dispatch->threading,
            uniform_address,
        },
        .coef = {0, 0, 0, 0},
        .bo_handles = (uintptr_t) submission->handles,
        .bo_handle_count = submission->handle_count,
        .in_sync = 0,
        .out_sync = 0,
        .perfmon_id = 0,
        .extensions = 0,
        .flags = 0,
        .pad = 0,
    };
    if (ioctl(context->fd, DRM_IOCTL_V3D_SUBMIT_CSD, &request) != 0) {
        set_error(context, "DRM_IOCTL_V3D_SUBMIT_CSD");
        pthread_mutex_unlock(&context->mutex);
        free(submission);
        return QPU_LLAMA_SUBMISSION_FAILED;
    }
    context->submission_active = true;
    pthread_mutex_unlock(&context->mutex);
    *result = submission;
    return QPU_LLAMA_OK;
}

qpu_llama_status qpu_llama_submission_wait(qpu_llama_submission *submission) {
    if (submission == NULL || submission->context == NULL || submission->complete) {
        return QPU_LLAMA_INVALID_ARGUMENT;
    }
    qpu_llama_context *context = submission->context;
    pthread_mutex_lock(&context->mutex);
    if (!context->submission_active) {
        pthread_mutex_unlock(&context->mutex);
        return QPU_LLAMA_INTERNAL_ERROR;
    }
    const bool inject_wait_failure = (context->failure_mask & QPU_LLAMA_FAIL_WAIT) != 0;
    // Every BO referenced by one V3D submission carries the same completion fence.
    struct drm_v3d_wait_bo wait = {
        .handle = submission->handles[0],
        .pad = 0,
        .timeout_ns = context->timeout_ns,
    };
    if (ioctl(context->fd, DRM_IOCTL_V3D_WAIT_BO, &wait) != 0) {
        const int wait_errno = errno;
        set_error(context, "DRM_IOCTL_V3D_WAIT_BO");
        context->poisoned = true;
        context->submission_active = false;
        submission->complete = true;
        pthread_mutex_unlock(&context->mutex);
        return wait_errno == ETIME || wait_errno == ETIMEDOUT ? QPU_LLAMA_TIMEOUT : QPU_LLAMA_IO_FAILED;
    }
    atomic_thread_fence(memory_order_seq_cst);
    if (inject_wait_failure) {
        set_error(context, "injected BO wait timeout");
        context->poisoned = true;
        context->submission_active = false;
        submission->complete = true;
        pthread_mutex_unlock(&context->mutex);
        return QPU_LLAMA_TIMEOUT;
    }
    context->submission_active = false;
    submission->complete = true;
    pthread_mutex_unlock(&context->mutex);
    return QPU_LLAMA_OK;
}

void qpu_llama_submission_destroy(qpu_llama_submission *submission) {
    if (submission == NULL) {
        return;
    }
    if (!submission->complete) {
        (void) qpu_llama_submission_wait(submission);
    }
    free(submission);
}

qpu_llama_status qpu_llama_program_execute(
    qpu_llama_program *program,
    const qpu_llama_dispatch_desc *dispatch) {
    qpu_llama_submission *submission = NULL;
    qpu_llama_status status = qpu_llama_program_submit(program, dispatch, &submission);
    if (status == QPU_LLAMA_OK) {
        status = qpu_llama_submission_wait(submission);
    }
    qpu_llama_submission_destroy(submission);
    return status;
}

const char *qpu_llama_status_string(qpu_llama_status status) {
    switch (status) {
        case QPU_LLAMA_OK: return "ok";
        case QPU_LLAMA_INVALID_ARGUMENT: return "invalid argument";
        case QPU_LLAMA_DEVICE_UNAVAILABLE: return "device unavailable";
        case QPU_LLAMA_CAPABILITY_MISSING: return "capability missing";
        case QPU_LLAMA_ALLOCATION_FAILED: return "allocation failed";
        case QPU_LLAMA_IO_FAILED: return "I/O failed";
        case QPU_LLAMA_SUBMISSION_FAILED: return "submission failed";
        case QPU_LLAMA_TIMEOUT: return "timeout";
        case QPU_LLAMA_HASH_MISMATCH: return "hash mismatch";
        case QPU_LLAMA_INTERNAL_ERROR: return "internal error";
    }
    return "unknown status";
}
