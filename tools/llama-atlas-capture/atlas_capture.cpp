#include "arg.h"
#include "common.h"
#include "llama-ext.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr std::uint64_t MAX_INPUT_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t MAX_LINE_BYTES = 1024ULL * 1024ULL;
constexpr std::size_t MAX_SAMPLES = 4096;
constexpr std::size_t MAX_TOKENS_PER_SAMPLE = 131072;

struct custom_args {
    fs::path tokens_jsonl;
    fs::path out_dir;
    std::vector<std::uint32_t> layers;
};

struct sample {
    std::string sample_id;
    std::string domain;
    std::vector<llama_token> token_ids;
};

class temp_dir_guard {
public:
    explicit temp_dir_guard(fs::path path) : path_(std::move(path)) {}
    temp_dir_guard(const temp_dir_guard &) = delete;
    temp_dir_guard & operator=(const temp_dir_guard &) = delete;
    ~temp_dir_guard() {
        if (armed_) {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
    }
    void release() noexcept { armed_ = false; }

private:
    fs::path path_;
    bool armed_ = true;
};

class backend_guard {
public:
    backend_guard() { llama_backend_init(); }
    backend_guard(const backend_guard &) = delete;
    backend_guard & operator=(const backend_guard &) = delete;
    ~backend_guard() { llama_backend_free(); }
};

class file_descriptor {
public:
    explicit file_descriptor(int value) : value_(value) {}
    file_descriptor(const file_descriptor &) = delete;
    file_descriptor & operator=(const file_descriptor &) = delete;
    ~file_descriptor() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }
    int get() const noexcept { return value_; }

private:
    int value_;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

std::string require_custom_value(
        int argc, char ** argv, int & index, const std::string & name, const std::string & prefix) {
    const std::string arg(argv[index]);
    if (arg.rfind(prefix, 0) == 0) {
        const std::string value = arg.substr(prefix.size());
        if (value.empty()) {
            fail(name + " requires a non-empty value");
        }
        return value;
    }
    if (arg == name) {
        if (index + 1 >= argc || std::string(argv[index + 1]).empty()) {
            fail(name + " requires a non-empty value");
        }
        return argv[++index];
    }
    fail("internal custom-argument parser error");
}

std::vector<std::uint32_t> parse_layers(const std::string & value) {
    std::vector<std::uint32_t> result;
    std::istringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos) {
            fail("--layers must be a comma-separated list of non-negative decimal integers");
        }
        std::size_t consumed = 0;
        const unsigned long parsed = std::stoul(item, &consumed, 10);
        if (consumed != item.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
            fail("--layers contains an out-of-range layer number");
        }
        result.push_back(static_cast<std::uint32_t>(parsed));
    }
    if (result.empty()) {
        fail("--layers must contain at least one layer");
    }
    if (!std::is_sorted(result.begin(), result.end())) {
        fail("--layers must be sorted in ascending order");
    }
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        fail("--layers must not contain duplicates");
    }
    return result;
}

std::pair<custom_args, std::vector<std::string>> split_custom_args(int argc, char ** argv) {
    custom_args custom;
    bool saw_tokens = false;
    bool saw_out = false;
    bool saw_layers = false;
    std::vector<std::string> common_args{argv[0]};

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--tokens-jsonl" || arg.rfind("--tokens-jsonl=", 0) == 0) {
            if (saw_tokens) {
                fail("--tokens-jsonl may be supplied only once");
            }
            custom.tokens_jsonl = require_custom_value(
                    argc, argv, i, "--tokens-jsonl", "--tokens-jsonl=");
            saw_tokens = true;
        } else if (arg == "--out-dir" || arg.rfind("--out-dir=", 0) == 0) {
            if (saw_out) {
                fail("--out-dir may be supplied only once");
            }
            custom.out_dir = require_custom_value(argc, argv, i, "--out-dir", "--out-dir=");
            saw_out = true;
        } else if (arg == "--layers" || arg.rfind("--layers=", 0) == 0) {
            if (saw_layers) {
                fail("--layers may be supplied only once");
            }
            custom.layers = parse_layers(
                    require_custom_value(argc, argv, i, "--layers", "--layers="));
            saw_layers = true;
        } else {
            common_args.push_back(arg);
        }
    }

    if (!saw_tokens || !saw_out || !saw_layers) {
        fail("required custom arguments: --tokens-jsonl PATH --out-dir PATH --layers CSV");
    }
    return {std::move(custom), std::move(common_args)};
}

void reject_symlink(const fs::path & path, const std::string & label) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        fail("cannot inspect " + label + ": " + error.message());
    }
    if (!error && fs::is_symlink(status)) {
        fail(label + " must not be a symlink: " + path.string());
    }
}

std::vector<sample> load_samples(const fs::path & path) {
    const int raw_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (raw_fd < 0) {
        fail("cannot open tokens JSONL without following symlinks: " +
             std::string(std::strerror(errno)));
    }
    file_descriptor input_fd(raw_fd);
    struct stat input_stat {};
    if (::fstat(input_fd.get(), &input_stat) != 0) {
        fail("cannot stat open tokens JSONL: " + std::string(std::strerror(errno)));
    }
    if (!S_ISREG(input_stat.st_mode)) {
        fail("tokens JSONL must be a regular file");
    }
    if (input_stat.st_size < 0 || static_cast<std::uint64_t>(input_stat.st_size) > MAX_INPUT_BYTES) {
        fail("tokens JSONL exceeds the 64 MiB input bound");
    }

    std::string contents;
    contents.reserve(static_cast<std::size_t>(input_stat.st_size));
    char buffer[64 * 1024];
    while (true) {
        if (contents.size() == MAX_INPUT_BYTES) {
            char extra = 0;
            const ssize_t extra_count = ::read(input_fd.get(), &extra, 1);
            if (extra_count < 0 && errno == EINTR) {
                continue;
            }
            if (extra_count < 0) {
                fail("failed while reading tokens JSONL: " + std::string(std::strerror(errno)));
            }
            if (extra_count > 0) {
                fail("tokens JSONL grew beyond the 64 MiB input bound");
            }
            break;
        }
        const std::size_t capacity =
                std::min<std::size_t>(sizeof(buffer), MAX_INPUT_BYTES - contents.size());
        const ssize_t count = ::read(input_fd.get(), buffer, capacity);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            fail("failed while reading tokens JSONL: " + std::string(std::strerror(errno)));
        }
        if (count == 0) {
            break;
        }
        contents.append(buffer, static_cast<std::size_t>(count));
    }

    std::istringstream input(contents);
    std::vector<sample> samples;
    std::set<std::string> sample_ids;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.size() > MAX_LINE_BYTES) {
            fail("tokens JSONL line " + std::to_string(line_number) + " exceeds 1 MiB");
        }
        if (line.empty()) {
            fail("tokens JSONL contains an empty line at " + std::to_string(line_number));
        }
        if (samples.size() >= MAX_SAMPLES) {
            fail("tokens JSONL exceeds the 4096-sample bound");
        }

        json value;
        try {
            value = json::parse(line);
        } catch (const json::exception & exc) {
            fail("invalid JSON on line " + std::to_string(line_number) + ": " + exc.what());
        }
        if (!value.is_object() || value.size() != 3 || !value.contains("sample_id") ||
                !value.contains("domain") || !value.contains("token_ids")) {
            fail("line " + std::to_string(line_number) +
                 " must contain exactly sample_id, domain, and token_ids");
        }
        if (!value["sample_id"].is_string() || value["sample_id"].get_ref<const std::string &>().empty()) {
            fail("line " + std::to_string(line_number) + " has an invalid sample_id");
        }
        if (!value["domain"].is_string() || value["domain"].get_ref<const std::string &>().empty()) {
            fail("line " + std::to_string(line_number) + " has an invalid domain");
        }
        if (!value["token_ids"].is_array() || value["token_ids"].size() < 2 ||
                value["token_ids"].size() > MAX_TOKENS_PER_SAMPLE) {
            fail("line " + std::to_string(line_number) +
                 " token_ids length must be between 2 and 131072");
        }

        sample parsed;
        parsed.sample_id = value["sample_id"].get<std::string>();
        parsed.domain = value["domain"].get<std::string>();
        if (!sample_ids.insert(parsed.sample_id).second) {
            fail("duplicate sample_id: " + parsed.sample_id);
        }
        parsed.token_ids.reserve(value["token_ids"].size());
        for (const json & token : value["token_ids"]) {
            if (!token.is_number_integer()) {
                fail("line " + std::to_string(line_number) + " contains a non-integer token ID");
            }
            const std::int64_t id = token.get<std::int64_t>();
            if (id < std::numeric_limits<llama_token>::min() ||
                    id > std::numeric_limits<llama_token>::max()) {
                fail("line " + std::to_string(line_number) + " contains an out-of-range token ID");
            }
            parsed.token_ids.push_back(static_cast<llama_token>(id));
        }
        samples.push_back(std::move(parsed));
    }
    if (samples.empty()) {
        fail("tokens JSONL must contain at least one sample");
    }
    return samples;
}

fs::path normalized_path(const fs::path & path, const std::string & label) {
    std::error_code error;
    const fs::path absolute = fs::absolute(path, error);
    if (error) {
        fail("cannot resolve " + label + ": " + error.message());
    }
    const fs::path normalized = fs::weakly_canonical(absolute, error);
    if (error) {
        fail("cannot canonicalize " + label + ": " + error.message());
    }
    return normalized;
}

bool is_ancestor_or_equal(const fs::path & ancestor, const fs::path & candidate) {
    auto ancestor_it = ancestor.begin();
    auto candidate_it = candidate.begin();
    while (ancestor_it != ancestor.end() && candidate_it != candidate.end()) {
        if (*ancestor_it != *candidate_it) {
            return false;
        }
        ++ancestor_it;
        ++candidate_it;
    }
    return ancestor_it == ancestor.end();
}

bool paths_overlap(const fs::path & left, const fs::path & right) {
    return is_ancestor_or_equal(left, right) || is_ancestor_or_equal(right, left);
}

fs::path temp_path_for(const fs::path & output) {
    const fs::path parent = output.has_parent_path() ? output.parent_path() : fs::path(".");
    return parent / ("." + output.filename().string() + ".atlas-capture.tmp");
}

void validate_capture_paths(
        const fs::path & tokens_jsonl, const fs::path & output, const fs::path & model_path) {
    if (model_path.empty()) {
        fail("a local --model path is required");
    }
    const fs::path input_normalized = normalized_path(tokens_jsonl, "tokens JSONL");
    const fs::path output_normalized = normalized_path(output, "output directory");
    const fs::path temp_normalized = normalized_path(temp_path_for(output), "temporary directory");
    const fs::path model_normalized = normalized_path(model_path, "model path");

    std::error_code error;
    const bool model_is_directory = fs::is_directory(model_normalized, error);
    if (error) {
        fail("cannot inspect model source path: " + error.message());
    }
    const fs::path model_source =
            model_is_directory ? model_normalized : model_normalized.parent_path();

    for (const auto & candidate : {output_normalized, temp_normalized}) {
        if (paths_overlap(candidate, input_normalized)) {
            fail("capture output must not overlap the tokens JSONL path");
        }
        if (is_ancestor_or_equal(model_source, candidate)) {
            fail("capture output must not be written inside the model source directory");
        }
        if (paths_overlap(candidate, model_normalized)) {
            fail("capture output must not overlap the model source path");
        }
    }
}

fs::path prepare_temp_dir(const fs::path & output) {
    if (output.empty() || output.filename().empty()) {
        fail("--out-dir must name a non-root output directory");
    }
    const fs::path parent = output.has_parent_path() ? output.parent_path() : fs::path(".");
    reject_symlink(output, "output directory");
    std::error_code error;
    if (fs::exists(output, error) || error) {
        fail("output directory already exists: " + output.string());
    }
    if (!fs::is_directory(parent, error) || error) {
        fail("output parent must be an existing directory: " + parent.string());
    }
    reject_symlink(parent, "output parent");

    const fs::path temp = temp_path_for(output);
    reject_symlink(temp, "temporary output directory");
    if (fs::exists(temp, error) || error) {
        fail("temporary output directory already exists: " + temp.string());
    }
    if (!fs::create_directory(temp, error) || error) {
        fail("cannot create temporary output directory: " + error.message());
    }
    if (::chmod(temp.c_str(), S_IRWXU) != 0) {
        const int saved_errno = errno;
        fs::remove(temp, error);
        fail("cannot make temporary output directory private: " +
             std::string(std::strerror(saved_errno)));
    }
    return temp;
}

void atomic_publish(const fs::path & temp, const fs::path & output) {
    const long rc = ::syscall(
            SYS_renameat2, AT_FDCWD, temp.c_str(), AT_FDCWD, output.c_str(), RENAME_NOREPLACE);
    if (rc != 0) {
        fail("cannot atomically publish capture directory: " + std::string(std::strerror(errno)));
    }
}

std::ofstream open_binary(const fs::path & path) {
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) {
        fail("cannot create output file: " + path.string());
    }
    return stream;
}

void write_floats(std::ofstream & stream, const float * values, std::size_t count) {
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    stream.write(reinterpret_cast<const char *>(values), bytes);
    if (!stream) {
        fail("failed while streaming float capture data");
    }
}

void close_checked(std::ofstream & stream, const std::string & label) {
    stream.flush();
    if (!stream) {
        fail("failed to flush " + label);
    }
    stream.close();
    if (!stream) {
        fail("failed to close " + label);
    }
}

std::string layer_file_name(std::uint32_t layer) {
    std::ostringstream name;
    name << "layer-" << std::setw(3) << std::setfill('0') << layer << ".f32";
    return name.str();
}

std::string hex_encode(const std::string & value) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(value.size() * 2);
    for (const unsigned char byte : value) {
        encoded.push_back(HEX[byte >> 4]);
        encoded.push_back(HEX[byte & 0x0f]);
    }
    return encoded;
}

void write_tokenizer(const fs::path & path, const llama_vocab * vocab, std::uint32_t n_vocab) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        fail("cannot create tokenizer.tsv");
    }
    for (std::uint32_t id = 0; id < n_vocab; ++id) {
        output << id << '\t'
               << hex_encode(common_token_to_piece(vocab, static_cast<llama_token>(id), true))
               << '\n';
        if (!output) {
            fail("failed while writing tokenizer.tsv");
        }
    }
    output.flush();
    if (!output) {
        fail("failed to flush tokenizer.tsv");
    }
    output.close();
    if (!output) {
        fail("failed to close tokenizer.tsv");
    }
}

void validate_model_bounds(
        const std::vector<sample> & samples,
        const std::vector<std::uint32_t> & layers,
        std::uint32_t n_vocab,
        std::uint32_t n_ctx,
        std::uint32_t n_layer) {
    for (const std::uint32_t layer : layers) {
        if (layer > n_layer) {
            fail("requested layer " + std::to_string(layer) + " exceeds n_layer " +
                 std::to_string(n_layer));
        }
    }
    for (const sample & item : samples) {
        if (item.token_ids.size() > n_ctx) {
            fail("sample " + item.sample_id + " exceeds the loaded context length " +
                 std::to_string(n_ctx));
        }
        for (const llama_token token : item.token_ids) {
            if (token < 0 || static_cast<std::uint32_t>(token) >= n_vocab) {
                fail("sample " + item.sample_id + " contains token ID outside [0, n_vocab)");
            }
        }
    }
}

int capture(const custom_args & custom, common_params & params) {
    const std::vector<sample> samples = load_samples(custom.tokens_jsonl);
    validate_capture_paths(custom.tokens_jsonl, custom.out_dir, params.model.path);
    const fs::path temp = prepare_temp_dir(custom.out_dir);
    temp_dir_guard cleanup(temp);

    backend_guard backend;
    llama_numa_init(params.numa);
    common_init_result_ptr initialized = common_init_from_params(params);
    llama_model * model = initialized ? initialized->model() : nullptr;
    llama_context * context = initialized ? initialized->context() : nullptr;
    if (model == nullptr || context == nullptr) {
        fail("unable to load model and context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::uint32_t n_vocab = static_cast<std::uint32_t>(llama_vocab_n_tokens(vocab));
    const std::uint32_t n_embd = static_cast<std::uint32_t>(llama_model_n_embd(model));
    const std::uint32_t n_layer = static_cast<std::uint32_t>(llama_model_n_layer(model));
    const std::uint32_t n_ctx = llama_n_ctx(context);
    const std::uint32_t n_batch = llama_n_batch(context);
    if (n_vocab == 0 || n_embd == 0 || n_batch == 0) {
        fail("loaded model reported invalid capture dimensions");
    }
    validate_model_bounds(samples, custom.layers, n_vocab, n_ctx, n_layer);

    for (const std::uint32_t layer : custom.layers) {
        llama_set_embeddings_layer_inp(context, layer, true);
    }

    std::ofstream logits = open_binary(temp / "logits.f32");
    std::vector<std::ofstream> layer_outputs;
    layer_outputs.reserve(custom.layers.size());
    for (const std::uint32_t layer : custom.layers) {
        layer_outputs.push_back(open_binary(temp / layer_file_name(layer)));
    }
    std::ofstream alignment(temp / "alignment.jsonl", std::ios::out | std::ios::trunc);
    if (!alignment) {
        fail("cannot create alignment.jsonl");
    }

    std::uint64_t row = 0;
    for (const sample & item : samples) {
        llama_memory_clear(llama_get_memory(context), true);
        std::size_t offset = 0;
        while (offset < item.token_ids.size()) {
            const std::size_t chunk_size =
                    std::min<std::size_t>(n_batch, item.token_ids.size() - offset);
            llama_batch batch = llama_batch_init(static_cast<std::int32_t>(chunk_size), 0, 1);
            if (batch.token == nullptr) {
                fail("failed to allocate capture batch");
            }
            for (std::size_t i = 0; i < chunk_size; ++i) {
                common_batch_add(
                        batch,
                        item.token_ids[offset + i],
                        static_cast<llama_pos>(offset + i),
                        {0},
                        true);
            }
            const std::int32_t decode_result = llama_decode(context, batch);
            if (decode_result != 0) {
                llama_batch_free(batch);
                fail("llama_decode failed for sample " + item.sample_id + " with code " +
                     std::to_string(decode_result));
            }

            const std::size_t rows_in_chunk =
                    std::min(chunk_size, item.token_ids.size() - 1 - offset);
            for (std::size_t i = 0; i < rows_in_chunk; ++i) {
                const float * row_logits =
                        llama_get_logits_ith(context, static_cast<std::int32_t>(i));
                if (row_logits == nullptr) {
                    llama_batch_free(batch);
                    fail("llama_get_logits_ith returned null");
                }
                write_floats(logits, row_logits, n_vocab);
            }
            for (std::size_t layer_index = 0; layer_index < custom.layers.size(); ++layer_index) {
                const float * hidden =
                        llama_get_embeddings_layer_inp(context, custom.layers[layer_index]);
                if (hidden == nullptr) {
                    llama_batch_free(batch);
                    fail("layer-input capture returned null");
                }
                write_floats(layer_outputs[layer_index], hidden, rows_in_chunk * n_embd);
            }
            llama_batch_free(batch);

            for (std::size_t i = 0; i < rows_in_chunk; ++i) {
                const std::size_t input_position = offset + i;
                const json record = {
                        {"row", row++},
                        {"sample_id", item.sample_id},
                        {"domain", item.domain},
                        {"input_position", input_position},
                        {"target_position", input_position + 1},
                        {"target_token_id", item.token_ids[input_position + 1]},
                };
                alignment << record.dump() << '\n';
                if (!alignment) {
                    fail("failed while writing alignment.jsonl");
                }
            }
            offset += chunk_size;
        }
    }

    close_checked(logits, "logits.f32");
    for (std::size_t i = 0; i < layer_outputs.size(); ++i) {
        close_checked(layer_outputs[i], layer_file_name(custom.layers[i]));
    }
    alignment.flush();
    if (!alignment) {
        fail("failed to flush alignment.jsonl");
    }
    alignment.close();
    if (!alignment) {
        fail("failed to close alignment.jsonl");
    }
    write_tokenizer(temp / "tokenizer.tsv", vocab, n_vocab);

    std::vector<std::string> layer_files;
    layer_files.reserve(custom.layers.size());
    for (const std::uint32_t layer : custom.layers) {
        layer_files.push_back(layer_file_name(layer));
    }
    const json manifest = {
            {"schema_version", 1},
            {"capture_mode", "teacher_forced"},
            {"vocab_size", n_vocab},
            {"hidden_size", n_embd},
            {"n_hidden_layers", n_layer},
            {"row_count", row},
            {"sample_count", samples.size()},
            {"layers", custom.layers},
            {"files",
             {
                     {"logits", "logits.f32"},
                     {"layer_inputs", layer_files},
                     {"alignment", "alignment.jsonl"},
                     {"tokenizer", "tokenizer.tsv"},
             }},
    };
    std::ofstream manifest_output(temp / "raw-capture.json", std::ios::out | std::ios::trunc);
    if (!manifest_output) {
        fail("cannot create raw-capture.json");
    }
    manifest_output << manifest.dump(2) << '\n';
    manifest_output.flush();
    if (!manifest_output) {
        fail("failed to flush raw-capture.json");
    }
    manifest_output.close();
    if (!manifest_output) {
        fail("failed to close raw-capture.json");
    }

    initialized.reset();
    atomic_publish(temp, custom.out_dir);
    cleanup.release();
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        auto [custom, common_storage] = split_custom_args(argc, argv);
        std::vector<char *> common_argv;
        common_argv.reserve(common_storage.size());
        for (std::string & arg : common_storage) {
            common_argv.push_back(arg.data());
        }

        common_params params;
        params.escape = false;
        params.warmup = false;
        common_init();
        if (!common_params_parse(
                    static_cast<int>(common_argv.size()),
                    common_argv.data(),
                    params,
                    LLAMA_EXAMPLE_RESULTS)) {
            return 2;
        }
        return capture(custom, params);
    } catch (const std::exception & exc) {
        std::cerr << "llama-atlas-capture: " << exc.what() << '\n';
        return 1;
    }
}
