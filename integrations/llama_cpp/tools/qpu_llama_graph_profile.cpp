#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

struct options {
    std::filesystem::path model;
    std::filesystem::path output;
    std::filesystem::path fixture_dir;
    std::filesystem::path other_model;
    std::string prompt = "A deterministic graph profiling sentence records stable inference data. ";
    std::string fixture_pattern;
    int32_t batch = 1;
    int32_t context = 0;
    int32_t context_size = 8192;
    int32_t threads = 3;
    int32_t warmups = 1;
    int32_t iterations = 3;
    int64_t fixture_max_bytes = 64 * 1024 * 1024;
    bool mtp = false;
    bool quiet = false;
};

static int32_t parse_i32(const char * value, const char * name) {
    const long parsed = std::stol(value);
    if (parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string(name) + " is outside int32 range");
    }
    return static_cast<int32_t>(parsed);
}

static int64_t parse_i64(const char * value, const char * name) {
    try {
        return std::stoll(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
}

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " --model FILE --output FILE [options]\n"
        << "  --other-model FILE        target model that owns an MTP context's shared state\n"
        << "  --batch N                 target decode/verification width (default 1)\n"
        << "  --context N               exact populated context tokens (default 0)\n"
        << "  --context-size N          llama context capacity (default 8192)\n"
        << "  --threads N               CPU generation and batch threads (default 3)\n"
        << "  --warmups N               untimed graph warmups (default 1)\n"
        << "  --iterations N            profiled graph calls (default 3)\n"
        << "  --context-type default|mtp\n"
        << "  --prompt TEXT             deterministic token source\n"
        << "  --fixture-dir DIR         optionally capture bounded node tensors\n"
        << "  --fixture-pattern REGEX   node name/op selection for capture\n"
        << "  --fixture-max-bytes N     maximum bytes per captured tensor\n"
        << "  --quiet                   suppress non-error llama.cpp logs\n";
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            return argv[++i];
        };
        if (argument == "--model") {
            result.model = value();
        } else if (argument == "--other-model") {
            result.other_model = value();
        } else if (argument == "--output") {
            result.output = value();
        } else if (argument == "--batch") {
            result.batch = parse_i32(value(), "batch");
        } else if (argument == "--context") {
            result.context = parse_i32(value(), "context");
        } else if (argument == "--context-size") {
            result.context_size = parse_i32(value(), "context-size");
        } else if (argument == "--threads") {
            result.threads = parse_i32(value(), "threads");
        } else if (argument == "--warmups") {
            result.warmups = parse_i32(value(), "warmups");
        } else if (argument == "--iterations") {
            result.iterations = parse_i32(value(), "iterations");
        } else if (argument == "--context-type") {
            const std::string context_type = value();
            if (context_type == "mtp") {
                result.mtp = true;
            } else if (context_type != "default") {
                throw std::runtime_error("context-type must be default or mtp");
            }
        } else if (argument == "--prompt") {
            result.prompt = value();
        } else if (argument == "--fixture-dir") {
            result.fixture_dir = value();
        } else if (argument == "--fixture-pattern") {
            result.fixture_pattern = value();
        } else if (argument == "--fixture-max-bytes") {
            result.fixture_max_bytes = parse_i64(value(), "fixture-max-bytes");
        } else if (argument == "--quiet") {
            result.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument " + argument);
        }
    }
    if (result.model.empty() || result.output.empty()) {
        throw std::runtime_error("model and output are required");
    }
    if (result.batch <= 0 || result.context < 0 || result.context_size <= 0 ||
        result.context + result.batch >= result.context_size || result.threads <= 0 ||
        result.warmups < 0 || result.iterations <= 0 || result.fixture_max_bytes <= 0) {
        throw std::runtime_error("invalid numeric option");
    }
    if (!result.fixture_dir.empty() && result.fixture_pattern.empty()) {
        throw std::runtime_error("fixture-pattern is required with fixture-dir");
    }
    return result;
}

static void quiet_log_callback(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << text;
    }
}

static json tensor_metadata(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }
    json shape = json::array();
    json strides = json::array();
    for (int index = 0; index < GGML_MAX_DIMS; ++index) {
        shape.push_back(tensor->ne[index]);
        strides.push_back(tensor->nb[index]);
    }
    return {
        {"name", tensor->name},
        {"type", ggml_type_name(tensor->type)},
        {"shape", shape},
        {"strides", strides},
        {"bytes", ggml_nbytes(tensor)},
    };
}

static std::string safe_filename(std::string value) {
    for (char & character : value) {
        if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '-' ||
              character == '_')) {
            character = '_';
        }
    }
    if (value.empty()) {
        return "unnamed";
    }
    return value.substr(0, 80);
}

struct profile_state {
    bool active = false;
    int32_t run_index = 0;
    int64_t node_index = 0;
    int64_t fixture_max_bytes = 0;
    std::filesystem::path fixture_dir;
    std::regex fixture_filter;
    bool capture_fixtures = false;
    json nodes = json::array();
    json fixtures = json::array();
    std::unordered_map<ggml_tensor *, clock_type::time_point> starts;
};

static json dump_tensor(
    profile_state & state,
    const ggml_tensor * tensor,
    const std::string & role,
    int64_t node_index) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    const size_t total_bytes = ggml_nbytes(tensor);
    const size_t captured_bytes = std::min<size_t>(
        total_bytes,
        static_cast<size_t>(state.fixture_max_bytes));
    std::vector<uint8_t> data(captured_bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, captured_bytes);
    const std::string filename =
        "run" + std::to_string(state.run_index) + "-node" + std::to_string(node_index) + "-" +
        safe_filename(tensor->name) + "-" + role + ".bin";
    const std::filesystem::path path = state.fixture_dir / filename;
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open fixture " + path.string());
    }
    stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) {
        throw std::runtime_error("failed to write fixture " + path.string());
    }
    json record = tensor_metadata(tensor);
    record["role"] = role;
    record["path"] = std::filesystem::absolute(path).string();
    record["captured_bytes"] = captured_bytes;
    record["truncated"] = captured_bytes != total_bytes;
    return record;
}

static bool profile_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<profile_state *>(user_data);
    if (!state.active) {
        return false;
    }
    if (ask) {
        state.starts[tensor] = clock_type::now();
        return true;
    }
    const auto finished = clock_type::now();
    const auto found = state.starts.find(tensor);
    if (found == state.starts.end()) {
        throw std::runtime_error("profile callback lost node start time");
    }
    const int64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - found->second).count();
    state.starts.erase(found);
    const int64_t node_index = state.node_index++;
    json sources = json::array();
    for (int index = 0; index < GGML_MAX_SRC; ++index) {
        if (tensor->src[index] != nullptr) {
            sources.push_back(tensor_metadata(tensor->src[index]));
        }
    }
    json record = tensor_metadata(tensor);
    record["run_index"] = state.run_index;
    record["node_index"] = node_index;
    record["op"] = ggml_op_name(tensor->op);
    record["op_description"] = ggml_op_desc(tensor);
    record["duration_ns"] = duration_ns;
    record["sources"] = sources;

    const std::string selection = std::string(tensor->name) + " " + ggml_op_name(tensor->op);
    if (state.capture_fixtures && state.run_index == 0 &&
        std::regex_search(selection, state.fixture_filter)) {
        json fixture = {
            {"run_index", state.run_index},
            {"node_index", node_index},
            {"node", tensor_metadata(tensor)},
            {"tensors", json::array()},
        };
        fixture["tensors"].push_back(dump_tensor(state, tensor->src[0], "src0", node_index));
        fixture["tensors"].push_back(dump_tensor(state, tensor->src[1], "src1", node_index));
        fixture["tensors"].push_back(dump_tensor(state, tensor, "output", node_index));
        state.fixtures.push_back(std::move(fixture));
    }
    state.nodes.push_back(std::move(record));
    return true;
}

static std::vector<llama_token> tokenize_exact(
    const llama_vocab * vocab,
    const std::string & source,
    int32_t count) {
    std::string text = source;
    std::vector<llama_token> tokens;
    while (true) {
        const int32_t required = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            nullptr,
            0,
            true,
            true);
        if (required == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("tokenization size overflow");
        }
        const int32_t capacity = required < 0 ? -required : required;
        tokens.resize(static_cast<size_t>(capacity));
        const int32_t produced = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            capacity,
            true,
            true);
        if (produced < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize(static_cast<size_t>(produced));
        if (produced >= count) {
            tokens.resize(static_cast<size_t>(count));
            return tokens;
        }
        text += text;
    }
}

static int32_t decode_tokens(
    llama_context * context,
    llama_token * tokens,
    int32_t count,
    int32_t position_start,
    bool all_logits) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int32_t index = 0; index < count; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = position_start + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = all_logits ? 1 : 0;
    }
    const int32_t status = llama_decode(context, batch);
    llama_batch_free(batch);
    return status;
}

static int32_t decode_tokens_with_embeddings(
    llama_context * context,
    llama_token * tokens,
    const float * embeddings,
    int32_t count,
    int32_t embedding_columns,
    int32_t position_start) {
    llama_batch batch = llama_batch_init(count, embedding_columns, 1);
    batch.token = static_cast<llama_token *>(malloc(sizeof(llama_token) * static_cast<size_t>(count)));
    if (batch.token == nullptr) {
        llama_batch_free(batch);
        throw std::bad_alloc();
    }
    batch.n_tokens = count;
    std::copy(tokens, tokens + count, batch.token);
    std::copy(
        embeddings,
        embeddings + static_cast<size_t>(count) * embedding_columns,
        batch.embd);
    for (int32_t index = 0; index < count; ++index) {
        batch.pos[index] = position_start + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = 1;
    }
    const int32_t status = llama_decode(context, batch);
    llama_batch_free(batch);
    return status;
}

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        if (!std::filesystem::is_regular_file(opts.model)) {
            throw std::runtime_error("model file does not exist");
        }
        if (!opts.other_model.empty() && !std::filesystem::is_regular_file(opts.other_model)) {
            throw std::runtime_error("other model file does not exist");
        }
        std::filesystem::create_directories(opts.output.parent_path());
        if (!opts.fixture_dir.empty()) {
            std::filesystem::create_directories(opts.fixture_dir);
        }

        profile_state state;
        state.fixture_max_bytes = opts.fixture_max_bytes;
        state.fixture_dir = opts.fixture_dir;
        state.capture_fixtures = !opts.fixture_dir.empty();
        if (state.capture_fixtures) {
            state.fixture_filter = std::regex(opts.fixture_pattern, std::regex::optimize);
        }

        if (opts.quiet) {
            llama_log_set(quiet_log_callback, nullptr);
        }
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (model == nullptr) {
            llama_backend_free();
            throw std::runtime_error("failed to load model");
        }
        llama_model * other_model = nullptr;
        llama_context * other_context = nullptr;
        if (!opts.other_model.empty()) {
            other_model = llama_model_load_from_file(opts.other_model.c_str(), model_params);
            if (other_model == nullptr) {
                llama_model_free(model);
                llama_backend_free();
                throw std::runtime_error("failed to load other model");
            }
            llama_context_params other_params = llama_context_default_params();
            other_params.n_ctx = static_cast<uint32_t>(opts.context_size);
            other_params.n_batch = static_cast<uint32_t>(opts.context_size);
            other_params.n_ubatch = static_cast<uint32_t>(std::min(opts.context_size, 512));
            other_params.n_outputs_max = static_cast<uint32_t>(opts.batch);
            other_params.n_threads = opts.threads;
            other_params.n_threads_batch = opts.threads;
            other_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            other_context = llama_init_from_model(other_model, other_params);
            if (other_context == nullptr) {
                llama_model_free(other_model);
                llama_model_free(model);
                llama_backend_free();
                throw std::runtime_error("failed to create other context");
            }
            llama_set_embeddings_nextn(other_context, true, false);
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(opts.context_size);
        context_params.n_batch = static_cast<uint32_t>(opts.context_size);
        context_params.n_ubatch = static_cast<uint32_t>(std::min(opts.context_size, 512));
        context_params.n_outputs_max = static_cast<uint32_t>(opts.batch);
        context_params.n_threads = opts.threads;
        context_params.n_threads_batch = opts.threads;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.ctx_type = opts.mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
        context_params.cb_eval = profile_callback;
        context_params.cb_eval_user_data = &state;
        context_params.ctx_other = other_context;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            if (other_context != nullptr) {
                llama_free(other_context);
                llama_model_free(other_model);
            }
            llama_model_free(model);
            llama_backend_free();
            throw std::runtime_error("failed to create context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(other_model != nullptr ? other_model : model);
        std::vector<llama_token> tokens = tokenize_exact(vocab, opts.prompt, opts.context + opts.batch);
        const int32_t embedding_columns = llama_model_n_embd_inp(model);
        if (other_model != nullptr && llama_model_n_embd(other_model) != embedding_columns) {
            throw std::runtime_error("other model hidden size does not match MTP input size");
        }
        json decode_calls = json::array();
        const int32_t total_runs = opts.warmups + opts.iterations;
        for (int32_t run = 0; run < total_runs; ++run) {
            llama_memory_clear(llama_get_memory(context), true);
            state.active = false;
            std::vector<float> target_embeddings;
            if (other_context != nullptr) {
                llama_memory_clear(llama_get_memory(other_context), true);
                if (opts.context > 0 &&
                    decode_tokens(other_context, tokens.data(), opts.context, 0, false) != 0) {
                    throw std::runtime_error("other context prefill failed");
                }
                if (decode_tokens(
                        other_context,
                        tokens.data() + opts.context,
                        opts.batch,
                        opts.context,
                        true) != 0) {
                    throw std::runtime_error("other target decode failed");
                }
                const float * hidden = llama_get_embeddings_nextn(other_context);
                if (hidden == nullptr) {
                    throw std::runtime_error("other target produced no next-token embeddings");
                }
                target_embeddings.assign(
                    hidden,
                    hidden + static_cast<size_t>(opts.batch) * embedding_columns);
            } else if (opts.context > 0 &&
                       decode_tokens(context, tokens.data(), opts.context, 0, false) != 0) {
                throw std::runtime_error("context prefill failed");
            }
            state.run_index = run - opts.warmups;
            state.node_index = 0;
            state.active = run >= opts.warmups;
            const auto started = clock_type::now();
            const int32_t status = other_context == nullptr
                ? decode_tokens(
                      context,
                      tokens.data() + opts.context,
                      opts.batch,
                      opts.context,
                      true)
                : decode_tokens_with_embeddings(
                      context,
                      tokens.data() + opts.context,
                      target_embeddings.data(),
                      opts.batch,
                      embedding_columns,
                      opts.context + opts.batch);
            const int64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock_type::now() - started).count();
            state.active = false;
            if (status != 0) {
                throw std::runtime_error("profiled decode failed with status " + std::to_string(status));
            }
            if (run >= opts.warmups) {
                decode_calls.push_back(
                    {{"run_index", state.run_index}, {"wall_ns", wall_ns}});
            }
        }

        json result = {
            {"schema_version", 1},
            {"kind", "llama-cpp-serialized-node-profile"},
            {"measurement_contract",
             {
                 {"node_duration", "callback-bracketed graph view execution plus backend synchronization"},
                 {"serialization", "eval callback requests every node and therefore serializes node execution"},
                 {"fixture_timing_eligible", !state.capture_fixtures},
             }},
            {"model", std::filesystem::absolute(opts.model).string()},
            {"other_model",
             opts.other_model.empty()
                 ? json(nullptr)
                 : json(std::filesystem::absolute(opts.other_model).string())},
            {"configuration",
             {
                 {"batch", opts.batch},
                 {"context_tokens", opts.context},
                 {"context_size", opts.context_size},
                 {"threads", opts.threads},
                 {"warmups", opts.warmups},
                 {"iterations", opts.iterations},
                 {"context_type", opts.mtp ? "mtp" : "default"},
                 {"all_target_logits", true},
                 {"flash_attention", true},
             }},
            {"model_properties",
             {
                 {"embedding", llama_model_n_embd(model)},
                 {"layers", llama_model_n_layer(model)},
                 {"nextn_layers", llama_model_n_layer_nextn(model)},
                 {"size_bytes", llama_model_size(model)},
             }},
            {"decode_calls", decode_calls},
            {"nodes", state.nodes},
            {"fixtures", state.fixtures},
        };
        std::ofstream output(opts.output);
        if (!output) {
            throw std::runtime_error("failed to open output file");
        }
        output << result.dump(2) << '\n';
        if (!output) {
            throw std::runtime_error("failed to write output file");
        }
        llama_free(context);
        if (other_context != nullptr) {
            llama_free(other_context);
            llama_model_free(other_model);
        }
        llama_model_free(model);
        llama_backend_free();
        std::cout << "wrote " << opts.output << ": " << state.nodes.size()
                  << " node records\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qpu_llama_graph_profile: " << error.what() << '\n';
        return 1;
    }
}
