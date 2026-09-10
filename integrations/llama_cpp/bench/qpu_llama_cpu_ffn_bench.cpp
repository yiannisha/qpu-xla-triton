#include "ggml-alloc.h"
#include "ggml-backend.h"
#if defined(QPU_LLAMA_HAVE_BLAS)
#include "ggml-blas.h"
#endif
#include "ggml-cpu.h"
#include "ggml.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

constexpr uint32_t q4_0_block_elements = 32;
constexpr uint32_t q4_0_block_bytes = 18;

struct options {
    std::string gate_path;
    std::string up_path;
    std::string down_path;
    std::string activation_path;
    std::string output_path;
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    uint32_t rows = 0;
    uint32_t cpu_threads = 4;
    uint32_t warmups = 5;
    uint32_t samples = 31;
    bool use_blas = false;
    bool serve = false;
};

struct graph_owner {
    ggml_context * weight_context = nullptr;
    ggml_context * graph_context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t blas_backend = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_tensor * gate_weight = nullptr;
    ggml_tensor * up_weight = nullptr;
    ggml_tensor * down_weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * activated = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    ~graph_owner() {
        ggml_backend_sched_free(scheduler);
        ggml_backend_buffer_free(graph_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_free(blas_backend);
        ggml_backend_free(backend);
        ggml_threadpool_free(threadpool);
        ggml_free(graph_context);
        ggml_free(weight_context);
    }
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool parse_u32(const char * text, uint32_t & result, bool allow_zero = false) {
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        (!allow_zero && value == 0) || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(value);
    return true;
}

void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s --gate FILE --up FILE --down FILE --activation-f32 FILE "
        "--input-columns N --intermediate-columns N --output-columns N --rows N "
        "[--backend cpu-repack|openblas] [--cpu-threads N] [--warmups N] "
        "[--samples N] [--output-bin FILE] [--serve]\n",
        program);
}

bool parse_options(int argc, char ** argv, options & result) {
    options value;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (name == "--serve") {
            value.serve = true;
            continue;
        }
        if (index + 1 >= argc) {
            return false;
        }
        const char * argument = argv[++index];
        if (name == "--gate") {
            value.gate_path = argument;
        } else if (name == "--up") {
            value.up_path = argument;
        } else if (name == "--down") {
            value.down_path = argument;
        } else if (name == "--activation-f32") {
            value.activation_path = argument;
        } else if (name == "--output-bin") {
            value.output_path = argument;
        } else if (name == "--input-columns") {
            if (!parse_u32(argument, value.input_columns)) {
                return false;
            }
        } else if (name == "--intermediate-columns") {
            if (!parse_u32(argument, value.intermediate_columns)) {
                return false;
            }
        } else if (name == "--output-columns") {
            if (!parse_u32(argument, value.output_columns)) {
                return false;
            }
        } else if (name == "--rows") {
            if (!parse_u32(argument, value.rows)) {
                return false;
            }
        } else if (name == "--backend") {
            if (std::strcmp(argument, "cpu-repack") == 0) {
                value.use_blas = false;
            } else if (std::strcmp(argument, "openblas") == 0) {
                value.use_blas = true;
            } else {
                return false;
            }
        } else if (name == "--cpu-threads") {
            if (!parse_u32(argument, value.cpu_threads)) {
                return false;
            }
        } else if (name == "--warmups") {
            if (!parse_u32(argument, value.warmups, true)) {
                return false;
            }
        } else if (name == "--samples") {
            if (!parse_u32(argument, value.samples)) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (value.gate_path.empty() || value.up_path.empty() || value.down_path.empty() ||
        value.activation_path.empty() || value.input_columns % q4_0_block_elements != 0 ||
        value.intermediate_columns % q4_0_block_elements != 0 ||
        value.output_columns == 0 || value.rows == 0) {
        return false;
    }
    result = value;
    return true;
}

std::vector<uint8_t> read_exact_file(const std::string & path, size_t expected_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<uint64_t>(input.tellg()) != expected_size) {
        std::fprintf(stderr, "%s does not contain exactly %zu bytes\n", path.c_str(), expected_size);
        return {};
    }
    std::vector<uint8_t> result(expected_size);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()),
            static_cast<std::streamsize>(expected_size))) {
        return {};
    }
    return result;
}

bool write_exact_file(const std::string & path, const void * data, size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(output);
}

void print_samples(const std::vector<uint64_t> & samples) {
    std::putchar('[');
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            std::putchar(',');
        }
        std::printf("%llu", static_cast<unsigned long long>(samples[index]));
    }
    std::putchar(']');
}

bool build_graph(graph_owner & owner, const options & config,
        const std::vector<uint8_t> & gate_data,
        const std::vector<uint8_t> & up_data,
        const std::vector<uint8_t> & down_data,
        const std::vector<uint8_t> & activation_data) {
    ggml_init_params weight_params{};
    weight_params.mem_size = 4 * ggml_tensor_overhead();
    weight_params.no_alloc = true;
    owner.weight_context = ggml_init(weight_params);
    if (owner.weight_context == nullptr) {
        return false;
    }
    owner.gate_weight = ggml_new_tensor_2d(owner.weight_context, GGML_TYPE_Q4_0,
        config.input_columns, config.intermediate_columns);
    owner.up_weight = ggml_new_tensor_2d(owner.weight_context, GGML_TYPE_Q4_0,
        config.input_columns, config.intermediate_columns);
    owner.down_weight = ggml_new_tensor_2d(owner.weight_context, GGML_TYPE_Q4_0,
        config.intermediate_columns, config.output_columns);
    ggml_set_name(owner.gate_weight, "fixture.ffn_gate.weight");
    ggml_set_name(owner.up_weight, "fixture.ffn_up.weight");
    ggml_set_name(owner.down_weight, "fixture.ffn_down.weight");
    ggml_backend_buffer_type_t weight_buffer_type = config.use_blas
        ? ggml_backend_cpu_buffer_type()
        : ggml_backend_cpu_repack_buffer_type();
    owner.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        owner.weight_context, weight_buffer_type);
    if (owner.weight_buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(owner.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(owner.gate_weight, gate_data.data(), 0, gate_data.size());
    ggml_backend_tensor_set(owner.up_weight, up_data.data(), 0, up_data.size());
    ggml_backend_tensor_set(owner.down_weight, down_data.data(), 0, down_data.size());

    owner.backend = ggml_backend_cpu_init();
    if (owner.backend == nullptr) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(owner.backend, static_cast<int>(config.cpu_threads));
    ggml_threadpool_params threadpool_params =
        ggml_threadpool_params_default(static_cast<int>(config.cpu_threads));
    owner.threadpool = ggml_threadpool_new(&threadpool_params);
    if (owner.threadpool == nullptr) {
        return false;
    }
    ggml_backend_cpu_set_threadpool(owner.backend, owner.threadpool);

    constexpr size_t graph_nodes = 16;
    ggml_init_params graph_params{};
    graph_params.mem_size = 16 * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_nodes, false);
    graph_params.no_alloc = true;
    owner.graph_context = ggml_init(graph_params);
    if (owner.graph_context == nullptr) {
        return false;
    }
    owner.input = ggml_new_tensor_2d(owner.graph_context, GGML_TYPE_F32,
        config.input_columns, config.rows);
    ggml_set_name(owner.input, "fixture.ffn_input");
    ggml_set_input(owner.input);
    owner.gate = ggml_mul_mat(owner.graph_context, owner.gate_weight, owner.input);
    owner.up = ggml_mul_mat(owner.graph_context, owner.up_weight, owner.input);
    ggml_set_name(owner.gate, "fixture.ffn_gate");
    ggml_set_name(owner.up, "fixture.ffn_up");
    owner.activated = ggml_geglu_split(owner.graph_context, owner.gate, owner.up);
    ggml_set_name(owner.activated, "fixture.ffn_geglu");
    owner.output = ggml_mul_mat(owner.graph_context, owner.down_weight, owner.activated);
    ggml_set_name(owner.output, "fixture.ffn_output");
    ggml_set_output(owner.output);
    owner.graph = ggml_new_graph_custom(owner.graph_context, graph_nodes, false);
    ggml_build_forward_expand(owner.graph, owner.output);
    if (!ggml_backend_supports_op(owner.backend, owner.gate) ||
        !ggml_backend_supports_op(owner.backend, owner.up) ||
        !ggml_backend_supports_op(owner.backend, owner.activated) ||
        !ggml_backend_supports_op(owner.backend, owner.output)) {
        return false;
    }
    if (config.use_blas) {
#if defined(QPU_LLAMA_HAVE_BLAS)
        owner.blas_backend = ggml_backend_blas_init();
        if (owner.blas_backend == nullptr) {
            return false;
        }
        ggml_backend_blas_set_n_threads(
            owner.blas_backend, static_cast<int>(config.cpu_threads));
        if (!ggml_backend_supports_op(owner.blas_backend, owner.gate) ||
            !ggml_backend_supports_op(owner.blas_backend, owner.up) ||
            !ggml_backend_supports_op(owner.blas_backend, owner.output) ||
            ggml_backend_supports_op(owner.blas_backend, owner.activated)) {
            std::fprintf(stderr, "OpenBLAS does not support the expected FFN placement\n");
            return false;
        }
        ggml_backend_t backends[] = {owner.blas_backend, owner.backend};
        owner.scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, graph_nodes, false, true);
        if (owner.scheduler == nullptr ||
            !ggml_backend_sched_alloc_graph(owner.scheduler, owner.graph)) {
            return false;
        }
        if (ggml_backend_sched_get_tensor_backend(owner.scheduler, owner.gate) !=
                owner.blas_backend ||
            ggml_backend_sched_get_tensor_backend(owner.scheduler, owner.up) !=
                owner.blas_backend ||
            ggml_backend_sched_get_tensor_backend(owner.scheduler, owner.activated) !=
                owner.backend ||
            ggml_backend_sched_get_tensor_backend(owner.scheduler, owner.output) !=
                owner.blas_backend) {
            std::fprintf(stderr,
                "scheduler placements: gate=%s up=%s geglu=%s down=%s\n",
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(
                    owner.scheduler, owner.gate)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(
                    owner.scheduler, owner.up)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(
                    owner.scheduler, owner.activated)),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(
                    owner.scheduler, owner.output)));
            return false;
        }
#else
        std::fprintf(stderr, "this benchmark was built without a GGML BLAS backend\n");
        return false;
#endif
    } else {
        owner.graph_buffer = ggml_backend_alloc_ctx_tensors(
            owner.graph_context, owner.backend);
        if (owner.graph_buffer == nullptr) {
            return false;
        }
    }
    ggml_backend_tensor_set(owner.input, activation_data.data(), 0, activation_data.size());
    return true;
}

bool compute(graph_owner & owner, uint64_t & elapsed_ns) {
    const uint64_t start = monotonic_ns();
    const ggml_status status = owner.scheduler != nullptr
        ? ggml_backend_sched_graph_compute(owner.scheduler, owner.graph)
        : ggml_backend_graph_compute(owner.backend, owner.graph);
    elapsed_ns = monotonic_ns() - start;
    return status == GGML_STATUS_SUCCESS;
}

bool dump_output(graph_owner & owner, const options & config, const std::string & path) {
    const size_t bytes = static_cast<size_t>(config.rows) * config.output_columns * sizeof(float);
    std::vector<uint8_t> output(bytes);
    ggml_backend_tensor_get(owner.output, output.data(), 0, output.size());
    return write_exact_file(path, output.data(), output.size());
}

int serve(graph_owner & owner, const options & config) {
    for (uint32_t iteration = 0; iteration < config.warmups; ++iteration) {
        uint64_t ignored = 0;
        if (!compute(owner, ignored)) {
            return 1;
        }
    }
    std::puts("{\"ready\":true}");
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "run") {
            uint64_t elapsed = 0;
            if (!compute(owner, elapsed)) {
                std::puts("{\"error\":\"compute failed\"}");
                std::fflush(stdout);
                return 1;
            }
            std::printf("{\"complete_ns\":%llu}\n",
                static_cast<unsigned long long>(elapsed));
        } else if (line.rfind("dump ", 0) == 0) {
            const std::string path = line.substr(5);
            std::printf("{\"dumped\":%s}\n",
                dump_output(owner, config, path) ? "true" : "false");
        } else if (line == "quit") {
            std::puts("{\"stopped\":true}");
            std::fflush(stdout);
            return 0;
        } else {
            std::puts("{\"error\":\"unknown command\"}");
        }
        std::fflush(stdout);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    options config;
    if (!parse_options(argc, argv, config)) {
        usage(argv[0]);
        return 2;
    }
    const size_t gate_bytes = static_cast<size_t>(config.intermediate_columns) *
        (config.input_columns / q4_0_block_elements) * q4_0_block_bytes;
    const size_t down_bytes = static_cast<size_t>(config.output_columns) *
        (config.intermediate_columns / q4_0_block_elements) * q4_0_block_bytes;
    const size_t activation_bytes = static_cast<size_t>(config.rows) *
        config.input_columns * sizeof(float);
    const std::vector<uint8_t> gate = read_exact_file(config.gate_path, gate_bytes);
    const std::vector<uint8_t> up = read_exact_file(config.up_path, gate_bytes);
    const std::vector<uint8_t> down = read_exact_file(config.down_path, down_bytes);
    const std::vector<uint8_t> activation = read_exact_file(
        config.activation_path, activation_bytes);
    if (gate.empty() || up.empty() || down.empty() || activation.empty()) {
        return 1;
    }
    graph_owner owner;
    if (!build_graph(owner, config, gate, up, down, activation)) {
        std::fprintf(stderr, "CPU_REPACK FFN graph construction failed\n");
        return 1;
    }
    if (config.serve) {
        return serve(owner, config);
    }
    std::vector<uint64_t> samples(config.samples);
    for (uint32_t iteration = 0; iteration < config.warmups + config.samples; ++iteration) {
        uint64_t elapsed = 0;
        if (!compute(owner, elapsed)) {
            return 1;
        }
        if (iteration >= config.warmups) {
            samples[iteration - config.warmups] = elapsed;
        }
    }
    if (!config.output_path.empty() && !dump_output(owner, config, config.output_path)) {
        return 1;
    }
    std::printf(
        "{\"schema_version\":1,\"kind\":\"llama-cpu-repack-ffn-samples\","
        "\"backend\":\"%s\","
        "\"placements\":{\"gate\":\"%s\",\"up\":\"%s\","
        "\"geglu\":\"%s\",\"down\":\"%s\"},"
        "\"input_columns\":%u,\"intermediate_columns\":%u,"
        "\"output_columns\":%u,\"rows\":%u,\"cpu_threads\":%u,"
        "\"warmups\":%u,\"retained_samples\":%u,\"complete_ns\":",
        config.use_blas ? "openblas" : "cpu-repack",
        config.use_blas ? "OpenBLAS" : "CPU_REPACK",
        config.use_blas ? "OpenBLAS" : "CPU_REPACK",
        "CPU",
        config.use_blas ? "OpenBLAS" : "CPU_REPACK",
        config.input_columns, config.intermediate_columns, config.output_columns,
        config.rows, config.cpu_threads, config.warmups, config.samples);
    print_samples(samples);
    std::printf("}\n");
    return 0;
}
