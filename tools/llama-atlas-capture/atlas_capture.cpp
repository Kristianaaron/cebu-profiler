#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "llama-ext.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>
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
constexpr std::size_t MAX_CAPTURE_LAYERS = 64;
constexpr std::uint64_t MAX_GGUF_METADATA_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t MAX_ARTIFACT_BYTES = 32ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t MAX_AGGREGATE_BYTES = 64ULL * 1024ULL * 1024ULL * 1024ULL;

struct custom_args {
    fs::path tokens_jsonl;
    fs::path out_dir;
    std::vector<std::uint32_t> layers;
    std::string request_id;
    std::string model_sha256;
    std::string model_artifact_manifest_sha256;
    std::string tool_binary_sha256;
    std::string tool_build_contract_sha256;
    std::string forced_tokens_sha256;
    std::string held_out_manifest_sha256;
    std::string ordered_sample_ids_sha256;
    std::string profile_tokenizer_sha256;
    std::string runtime_argv_sha256;
    std::string role;
    std::string reference_kind;
};

struct sample {
    std::string sample_id;
    std::string domain;
    std::vector<llama_token> token_ids;
};

struct loaded_samples {
    std::vector<sample> records;
    std::string measured_sha256;
};

struct file_identity {
    dev_t device;
    ino_t inode;
    off_t size;
    timespec modified;
    timespec changed;
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
    file_descriptor(file_descriptor && other) noexcept : value_(other.value_) { other.value_ = -1; }
    file_descriptor & operator=(file_descriptor && other) noexcept {
        if (this != &other) {
            if (value_ >= 0) {
                ::close(value_);
            }
            value_ = other.value_;
            other.value_ = -1;
        }
        return *this;
    }
    ~file_descriptor() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }
    int get() const noexcept { return value_; }

private:
    int value_;
};

struct pinned_model {
    file_descriptor descriptor;
    file_identity identity;
    fs::path canonical_path;
    std::string proc_path;
    std::string measured_sha256;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

fs::path normalized_path(const fs::path & path, const std::string & label);

// FIPS 180-4 SHA-256, adapted from llama.cpp's public-domain OpenCL cache helper.
struct sha256_context {
    std::uint32_t state[8];
    std::uint64_t bit_length;
    std::uint8_t buffer[64];
    std::size_t buffer_length;
};

constexpr std::uint32_t SHA256_CONSTANTS[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

std::uint32_t rotate_right(std::uint32_t value, unsigned amount) {
    return (value >> amount) | (value << (32 - amount));
}

void sha256_compress(std::uint32_t state[8], const std::uint8_t block[64]) {
    std::uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
        words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                       (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                       (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                       static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t first = rotate_right(words[index - 15], 7) ^
                                    rotate_right(words[index - 15], 18) ^
                                    (words[index - 15] >> 3);
        const std::uint32_t second = rotate_right(words[index - 2], 17) ^
                                     rotate_right(words[index - 2], 19) ^
                                     (words[index - 2] >> 10);
        words[index] = words[index - 16] + first + words[index - 7] + second;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t sum1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + sum1 + choice + SHA256_CONSTANTS[index] + words[index];
        const std::uint32_t sum0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256_initialize(sha256_context & context) {
    const std::uint32_t initial[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    std::copy(std::begin(initial), std::end(initial), context.state);
    context.bit_length = 0;
    context.buffer_length = 0;
}

void sha256_update(sha256_context & context, const void * data, std::size_t length) {
    const auto * bytes = static_cast<const std::uint8_t *>(data);
    context.bit_length += static_cast<std::uint64_t>(length) * 8;
    if (context.buffer_length > 0) {
        const std::size_t copied = std::min(64 - context.buffer_length, length);
        std::memcpy(context.buffer + context.buffer_length, bytes, copied);
        context.buffer_length += copied;
        bytes += copied;
        length -= copied;
        if (context.buffer_length == 64) {
            sha256_compress(context.state, context.buffer);
            context.buffer_length = 0;
        }
    }
    while (length >= 64) {
        sha256_compress(context.state, bytes);
        bytes += 64;
        length -= 64;
    }
    if (length > 0) {
        std::memcpy(context.buffer, bytes, length);
        context.buffer_length = length;
    }
}

std::string sha256_finish(sha256_context & context) {
    const std::uint64_t bit_length = context.bit_length;
    context.buffer[context.buffer_length++] = 0x80;
    if (context.buffer_length > 56) {
        while (context.buffer_length < 64) {
            context.buffer[context.buffer_length++] = 0;
        }
        sha256_compress(context.state, context.buffer);
        context.buffer_length = 0;
    }
    while (context.buffer_length < 56) {
        context.buffer[context.buffer_length++] = 0;
    }
    for (int index = 7; index >= 0; --index) {
        context.buffer[context.buffer_length++] =
                static_cast<std::uint8_t>(bit_length >> (index * 8));
    }
    sha256_compress(context.state, context.buffer);

    static constexpr char HEX[] = "0123456789abcdef";
    std::string result(64, '0');
    for (int word = 0; word < 8; ++word) {
        for (int byte = 0; byte < 4; ++byte) {
            const std::uint8_t value =
                    static_cast<std::uint8_t>(context.state[word] >> (24 - byte * 8));
            result[(word * 4 + byte) * 2] = HEX[value >> 4];
            result[(word * 4 + byte) * 2 + 1] = HEX[value & 0x0f];
        }
    }
    return result;
}

std::string sha256_bytes(const void * data, std::size_t length) {
    sha256_context context;
    sha256_initialize(context);
    sha256_update(context, data, length);
    return sha256_finish(context);
}

file_identity identity_from_stat(const struct stat & value) {
    return {
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtim,
            value.st_ctim,
    };
}

bool same_identity(const file_identity & left, const file_identity & right) {
    return left.device == right.device && left.inode == right.inode && left.size == right.size &&
           left.modified.tv_sec == right.modified.tv_sec &&
           left.modified.tv_nsec == right.modified.tv_nsec &&
           left.changed.tv_sec == right.changed.tv_sec && left.changed.tv_nsec == right.changed.tv_nsec;
}

file_identity descriptor_identity(int descriptor, const std::string & label) {
    struct stat value {};
    if (::fstat(descriptor, &value) != 0) {
        fail("cannot fstat " + label + ": " + std::string(std::strerror(errno)));
    }
    if (!S_ISREG(value.st_mode) || value.st_size <= 0) {
        fail(label + " must be a non-empty regular file");
    }
    return identity_from_stat(value);
}

std::string sha256_descriptor(
        int descriptor, const file_identity & expected, const std::string & label) {
    sha256_context context;
    sha256_initialize(context);
    std::vector<std::uint8_t> buffer(4 * 1024 * 1024);
    off_t offset = 0;
    while (offset < expected.size) {
        const std::size_t wanted = static_cast<std::size_t>(
                std::min<off_t>(static_cast<off_t>(buffer.size()), expected.size - offset));
        const ssize_t count = ::pread(descriptor, buffer.data(), wanted, offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            fail(label + " changed or became unreadable while hashing");
        }
        sha256_update(context, buffer.data(), static_cast<std::size_t>(count));
        offset += count;
    }
    const file_identity after = descriptor_identity(descriptor, label);
    if (!same_identity(expected, after)) {
        fail(label + " identity changed while hashing");
    }
    return sha256_finish(context);
}

struct bounded_gguf_reader {
    int descriptor;
    std::uint64_t file_size;
};

std::size_t bounded_gguf_read(
        void * userdata, void * output, std::uint64_t offset, std::size_t length) {
    auto & reader = *static_cast<bounded_gguf_reader *>(userdata);
    if (length == 0 || offset > reader.file_size || length > reader.file_size - offset ||
            offset > MAX_GGUF_METADATA_BYTES || length > MAX_GGUF_METADATA_BYTES - offset) {
        return 0;
    }
    std::size_t completed = 0;
    while (completed < length) {
        const ssize_t count = ::pread(
                reader.descriptor,
                static_cast<std::uint8_t *>(output) + completed,
                length - completed,
                static_cast<off_t>(offset + completed));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            break;
        }
        completed += static_cast<std::size_t>(count);
    }
    return completed;
}

void validate_single_file_gguf(int descriptor, const file_identity & identity) {
    bounded_gguf_reader reader{
            descriptor,
            static_cast<std::uint64_t>(identity.size),
    };
    const gguf_init_params parameters{
            true,
            nullptr,
    };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(
                    bounded_gguf_read,
                    &reader,
                    16 * 1024 * 1024,
                    reader.file_size,
                    parameters),
            gguf_free);
    if (!metadata) {
        fail("model is not valid GGUF metadata within the 512 MiB preflight bound");
    }
    if (gguf_get_version(metadata.get()) != GGUF_VERSION ||
            gguf_get_data_offset(metadata.get()) > MAX_GGUF_METADATA_BYTES ||
            gguf_get_n_kv(metadata.get()) < 1 || gguf_get_n_kv(metadata.get()) > 1'000'000 ||
            gguf_get_n_tensors(metadata.get()) < 1 ||
            gguf_get_n_tensors(metadata.get()) > 1'000'000) {
        fail("GGUF metadata exceeds the reviewed v3 bounds");
    }
    for (const char * key : {"split.no", "split.count", "split.tensors.count"}) {
        if (gguf_find_key(metadata.get(), key) >= 0) {
            fail("capture schema v1 requires a GGUF with no split metadata keys");
        }
    }
    if (!same_identity(identity, descriptor_identity(descriptor, "model GGUF"))) {
        fail("model GGUF identity changed during metadata preflight");
    }
}

pinned_model open_pinned_model(const fs::path & original_path, const std::string & asserted_sha256) {
    const fs::path canonical = normalized_path(original_path, "model path");
    const int raw_descriptor = ::open(canonical.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (raw_descriptor < 0) {
        fail("cannot pin model GGUF without following symlinks: " +
             std::string(std::strerror(errno)));
    }
    file_descriptor descriptor(raw_descriptor);
    const file_identity identity = descriptor_identity(descriptor.get(), "model GGUF");
    validate_single_file_gguf(descriptor.get(), identity);
    const std::string measured = sha256_descriptor(descriptor.get(), identity, "model GGUF");
    if (measured != asserted_sha256) {
        fail("measured model SHA-256 does not match --model-sha256");
    }
    const std::string proc_path = "/proc/self/fd/" + std::to_string(descriptor.get());
    struct stat proc_stat {};
    if (::stat(proc_path.c_str(), &proc_stat) != 0 ||
            !same_identity(identity, identity_from_stat(proc_stat))) {
        fail("pinned model descriptor is not loadable through its verified procfs path");
    }
    return {
            std::move(descriptor),
            identity,
            canonical,
            proc_path,
            measured,
    };
}

std::string measured_tool_sha256(const std::string & asserted_sha256) {
    const int raw_descriptor = ::open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (raw_descriptor < 0) {
        fail("cannot open /proc/self/exe for native tool measurement");
    }
    file_descriptor descriptor(raw_descriptor);
    const file_identity identity = descriptor_identity(descriptor.get(), "capture executable");
    const std::string measured =
            sha256_descriptor(descriptor.get(), identity, "capture executable");
    if (measured != asserted_sha256) {
        fail("measured capture executable SHA-256 does not match --tool-binary-sha256");
    }
    return measured;
}

void verify_pinned_model_unchanged(const pinned_model & model) {
    const std::string measured =
            sha256_descriptor(model.descriptor.get(), model.identity, "model GGUF after capture");
    if (measured != model.measured_sha256) {
        fail("model GGUF bytes changed during capture");
    }
    struct stat path_stat {};
    if (::lstat(model.canonical_path.c_str(), &path_stat) != 0 || S_ISLNK(path_stat.st_mode) ||
            !same_identity(model.identity, identity_from_stat(path_stat))) {
        fail("model GGUF path identity changed during capture");
    }
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
    if (result.size() > MAX_CAPTURE_LAYERS) {
        fail("--layers exceeds the native 64-layer capture bound");
    }
    if (!std::is_sorted(result.begin(), result.end())) {
        fail("--layers must be sorted in ascending order");
    }
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        fail("--layers must not contain duplicates");
    }
    return result;
}

bool is_lowercase_sha256(const std::string & value) {
    return value.size() == 64 &&
           value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

void set_hash_once(
        int argc,
        char ** argv,
        int & index,
        const std::string & name,
        std::string & destination) {
    if (!destination.empty()) {
        fail(name + " may be supplied only once");
    }
    const std::string value = require_custom_value(argc, argv, index, name, name + "=");
    if (!is_lowercase_sha256(value)) {
        fail(name + " must be exactly 64 lowercase hexadecimal characters");
    }
    destination = value;
}

void validate_role_reference(const std::string & role, const std::string & reference_kind) {
    const bool valid =
            (role == "candidate" && reference_kind == "candidate") ||
            (role == "identity_control" && reference_kind == "identity_control") ||
            (role == "nvfp4_source_reference" && reference_kind == "nvfp4_source_relative") ||
            (role == "bf16_teacher" && reference_kind == "bf16");
    if (!valid) {
        fail("--role and --reference-kind do not form a reviewed capture pairing");
    }
}

std::pair<custom_args, std::vector<std::string>> split_custom_args(int argc, char ** argv) {
    custom_args custom;
    bool saw_tokens = false;
    bool saw_out = false;
    bool saw_layers = false;
    bool saw_role = false;
    bool saw_reference_kind = false;
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
        } else if (arg == "--request-id" || arg.rfind("--request-id=", 0) == 0) {
            set_hash_once(argc, argv, i, "--request-id", custom.request_id);
        } else if (arg == "--model-sha256" || arg.rfind("--model-sha256=", 0) == 0) {
            set_hash_once(argc, argv, i, "--model-sha256", custom.model_sha256);
        } else if (arg == "--model-artifact-manifest-sha256" ||
                   arg.rfind("--model-artifact-manifest-sha256=", 0) == 0) {
            set_hash_once(
                    argc,
                    argv,
                    i,
                    "--model-artifact-manifest-sha256",
                    custom.model_artifact_manifest_sha256);
        } else if (arg == "--tool-binary-sha256" ||
                   arg.rfind("--tool-binary-sha256=", 0) == 0) {
            set_hash_once(
                    argc, argv, i, "--tool-binary-sha256", custom.tool_binary_sha256);
        } else if (arg == "--build-contract-sha256" ||
                   arg.rfind("--build-contract-sha256=", 0) == 0) {
            set_hash_once(
                    argc,
                    argv,
                    i,
                    "--build-contract-sha256",
                    custom.tool_build_contract_sha256);
        } else if (arg == "--forced-tokens-sha256" ||
                   arg.rfind("--forced-tokens-sha256=", 0) == 0) {
            set_hash_once(
                    argc, argv, i, "--forced-tokens-sha256", custom.forced_tokens_sha256);
        } else if (arg == "--held-out-manifest-sha256" ||
                   arg.rfind("--held-out-manifest-sha256=", 0) == 0) {
            set_hash_once(
                    argc,
                    argv,
                    i,
                    "--held-out-manifest-sha256",
                    custom.held_out_manifest_sha256);
        } else if (arg == "--ordered-sample-ids-sha256" ||
                   arg.rfind("--ordered-sample-ids-sha256=", 0) == 0) {
            set_hash_once(
                    argc,
                    argv,
                    i,
                    "--ordered-sample-ids-sha256",
                    custom.ordered_sample_ids_sha256);
        } else if (arg == "--profile-tokenizer-sha256" ||
                   arg.rfind("--profile-tokenizer-sha256=", 0) == 0) {
            set_hash_once(
                    argc,
                    argv,
                    i,
                    "--profile-tokenizer-sha256",
                    custom.profile_tokenizer_sha256);
        } else if (arg == "--runtime-argv-sha256" ||
                   arg.rfind("--runtime-argv-sha256=", 0) == 0) {
            set_hash_once(
                    argc, argv, i, "--runtime-argv-sha256", custom.runtime_argv_sha256);
        } else if (arg == "--role" || arg.rfind("--role=", 0) == 0) {
            if (saw_role) {
                fail("--role may be supplied only once");
            }
            custom.role = require_custom_value(argc, argv, i, "--role", "--role=");
            saw_role = true;
        } else if (arg == "--reference-kind" || arg.rfind("--reference-kind=", 0) == 0) {
            if (saw_reference_kind) {
                fail("--reference-kind may be supplied only once");
            }
            custom.reference_kind =
                    require_custom_value(argc, argv, i, "--reference-kind", "--reference-kind=");
            saw_reference_kind = true;
        } else {
            common_args.push_back(arg);
        }
    }

    const bool hashes_present = !custom.request_id.empty() && !custom.model_sha256.empty() &&
                                !custom.model_artifact_manifest_sha256.empty() &&
                                !custom.tool_binary_sha256.empty() &&
                                !custom.tool_build_contract_sha256.empty() &&
                                !custom.forced_tokens_sha256.empty() &&
                                !custom.held_out_manifest_sha256.empty() &&
                                !custom.ordered_sample_ids_sha256.empty() &&
                                !custom.profile_tokenizer_sha256.empty() &&
                                !custom.runtime_argv_sha256.empty();
    if (!saw_tokens || !saw_out || !saw_layers || !saw_role || !saw_reference_kind ||
            !hashes_present) {
        fail("all capture paths, layers, receipt digests, role, and reference kind are required");
    }
    validate_role_reference(custom.role, custom.reference_kind);
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

loaded_samples load_samples(const fs::path & path) {
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
    const file_identity input_identity = identity_from_stat(input_stat);
    const file_identity input_after = descriptor_identity(input_fd.get(), "tokens JSONL");
    if (!same_identity(input_identity, input_after) ||
            contents.size() != static_cast<std::size_t>(input_identity.size)) {
        fail("tokens JSONL changed while reading from its pinned descriptor");
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
    return {
            std::move(samples),
            sha256_bytes(contents.data(), contents.size()),
    };
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

std::string canonical_option_name(const std::string & value) {
    if (value == "-m") {
        return "--model";
    }
    if (value == "-c") {
        return "--ctx-size";
    }
    if (value == "-b") {
        return "--batch-size";
    }
    if (value == "-ub") {
        return "--ubatch-size";
    }
    if (value == "-t") {
        return "--threads";
    }
    if (value == "-tb") {
        return "--threads-batch";
    }
    return value;
}

std::vector<std::string> normalized_runtime_argv(
        const std::vector<std::string> & common_args, const custom_args & custom) {
    std::vector<std::string> normalized;
    normalized.push_back(normalized_path("/proc/self/exe", "capture executable").string());
    for (std::size_t index = 1; index < common_args.size(); ++index) {
        std::string option = common_args[index];
        std::string inline_value;
        if (option.rfind("--", 0) == 0) {
            const std::size_t equals = option.find('=');
            if (equals != std::string::npos) {
                inline_value = option.substr(equals + 1);
                option.resize(equals);
            }
        }
        option = canonical_option_name(option);
        normalized.push_back(option);
        if (!inline_value.empty()) {
            normalized.push_back(
                    option == "--model"
                            ? normalized_path(inline_value, "model argument").string()
                            : inline_value);
        } else if (option == "--model" && index + 1 < common_args.size()) {
            normalized.push_back(
                    normalized_path(common_args[++index], "model argument").string());
        }
    }
    normalized.insert(
            normalized.end(),
            {
                    "--tokens-jsonl",
                    normalized_path(custom.tokens_jsonl, "tokens JSONL").string(),
                    "--out-dir",
                    normalized_path(custom.out_dir, "output directory").string(),
                    "--layers",
            });
    std::ostringstream layer_csv;
    for (std::size_t index = 0; index < custom.layers.size(); ++index) {
        if (index != 0) {
            layer_csv << ',';
        }
        layer_csv << custom.layers[index];
    }
    normalized.push_back(layer_csv.str());
    return normalized;
}

std::string split_mode_name(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:
            return "none";
        case LLAMA_SPLIT_MODE_LAYER:
            return "layer";
        case LLAMA_SPLIT_MODE_ROW:
            return "row";
        case LLAMA_SPLIT_MODE_TENSOR:
            return "tensor";
    }
    fail("common parser produced an unknown split mode");
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
    reject_symlink(model_path, "model path");
    if (!fs::is_regular_file(model_normalized, error) || error) {
        fail("v1 capture requires a primary single-file GGUF model");
    }
    if (model_normalized.extension() != ".gguf") {
        fail("v1 capture model must have the lowercase .gguf suffix");
    }
    static const std::regex SPLIT_GGUF_NAME(R"(.*-[0-9]{5}-of-[0-9]{5}\.gguf)");
    if (std::regex_match(model_normalized.filename().string(), SPLIT_GGUF_NAME)) {
        fail("split GGUF model names are not supported by capture schema v1");
    }
    const fs::path model_source = model_normalized.parent_path();

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

void fsync_regular_file(const fs::path & path) {
    const int raw_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (raw_descriptor < 0) {
        fail("cannot open capture artifact for fsync: " + path.string() + ": " +
             std::string(std::strerror(errno)));
    }
    file_descriptor descriptor(raw_descriptor);
    const file_identity identity = descriptor_identity(descriptor.get(), path.filename().string());
    if (::fsync(descriptor.get()) != 0) {
        fail("cannot fsync capture artifact " + path.string() + ": " +
             std::string(std::strerror(errno)));
    }
    if (!same_identity(
                identity, descriptor_identity(descriptor.get(), path.filename().string()))) {
        fail("capture artifact changed while it was being fsynced: " + path.string());
    }
}

file_descriptor open_directory_descriptor(const fs::path & path, const std::string & label) {
    const int raw_descriptor =
            ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (raw_descriptor < 0) {
        fail("cannot open " + label + " directory: " + std::string(std::strerror(errno)));
    }
    struct stat value {};
    if (::fstat(raw_descriptor, &value) != 0) {
        const int saved_errno = errno;
        ::close(raw_descriptor);
        fail("cannot fstat " + label + " directory: " + std::string(std::strerror(saved_errno)));
    }
    if (!S_ISDIR(value.st_mode)) {
        ::close(raw_descriptor);
        fail(label + " is not a directory");
    }
    return file_descriptor(raw_descriptor);
}

void fsync_directory(const fs::path & path, const std::string & label) {
    file_descriptor descriptor = open_directory_descriptor(path, label);
    if (::fsync(descriptor.get()) != 0) {
        fail("cannot fsync " + label + " directory: " + std::string(std::strerror(errno)));
    }
}

void atomic_publish(const fs::path & temp, const fs::path & output) {
    const fs::path parent = output.has_parent_path() ? output.parent_path() : fs::path(".");
    if (temp.parent_path() != parent) {
        fail("temporary and final capture directories do not share a parent");
    }
    file_descriptor parent_descriptor = open_directory_descriptor(parent, "capture parent");
    const long rc = ::syscall(
            SYS_renameat2,
            parent_descriptor.get(),
            temp.filename().c_str(),
            parent_descriptor.get(),
            output.filename().c_str(),
            RENAME_NOREPLACE);
    if (rc != 0) {
        fail("cannot atomically publish capture directory: " + std::string(std::strerror(errno)));
    }
    if (::fsync(parent_descriptor.get()) != 0) {
        const int saved_errno = errno;
        fail("capture was atomically published but parent directory fsync failed; durability is "
             "ambiguous and the output was left in place for manual handling: " +
             std::string(std::strerror(saved_errno)));
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
        if (layer >= n_layer) {
            fail("requested layer " + std::to_string(layer) + " is outside [0, n_layer) for " +
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

std::uint64_t checked_product(
        std::uint64_t left, std::uint64_t right, const std::string & label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        fail(label + " byte estimate overflowed");
    }
    return left * right;
}

void validate_capture_size_budget(
        const std::vector<sample> & samples,
        std::size_t layer_count,
        std::uint32_t n_vocab,
        std::uint32_t n_embd) {
    std::uint64_t rows = 0;
    for (const sample & item : samples) {
        const std::uint64_t item_rows = item.token_ids.size() - 1;
        if (item_rows > std::numeric_limits<std::uint64_t>::max() - rows) {
            fail("capture row estimate overflowed");
        }
        rows += item_rows;
    }
    const std::uint64_t logits = checked_product(
            checked_product(rows, n_vocab, "logits"), sizeof(float), "logits");
    const std::uint64_t layer = checked_product(
            checked_product(rows, n_embd, "layer input"), sizeof(float), "layer input");
    if (logits > MAX_ARTIFACT_BYTES || layer > MAX_ARTIFACT_BYTES) {
        fail("native capture estimate exceeds the 32 GiB per-artifact bound");
    }
    const std::uint64_t all_layers =
            checked_product(layer, layer_count, "aggregate layer inputs");
    if (logits > MAX_AGGREGATE_BYTES || all_layers > MAX_AGGREGATE_BYTES - logits) {
        fail("native capture estimate exceeds the 64 GiB aggregate FP32 bound");
    }
}

int capture(
        const custom_args & custom,
        common_params & params,
        const std::vector<std::string> & common_args) {
    const loaded_samples token_input = load_samples(custom.tokens_jsonl);
    if (token_input.measured_sha256 != custom.forced_tokens_sha256) {
        fail("measured forced-token SHA-256 does not match --forced-tokens-sha256");
    }
    const std::vector<sample> & samples = token_input.records;
    validate_capture_paths(custom.tokens_jsonl, custom.out_dir, params.model.path);
    const std::vector<std::string> runtime_argv = normalized_runtime_argv(common_args, custom);
    const std::string tool_measured_sha256 = measured_tool_sha256(custom.tool_binary_sha256);
    pinned_model model_pin = open_pinned_model(params.model.path, custom.model_sha256);
    params.model.path = model_pin.proc_path;
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
    const std::uint32_t n_ubatch = llama_n_ubatch(context);
    if (n_vocab == 0 || n_embd == 0 || n_batch == 0) {
        fail("loaded model reported invalid capture dimensions");
    }
    validate_model_bounds(samples, custom.layers, n_vocab, n_ctx, n_layer);
    validate_capture_size_budget(samples, custom.layers.size(), n_vocab, n_embd);

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
    verify_pinned_model_unchanged(model_pin);

    std::vector<std::string> layer_files;
    layer_files.reserve(custom.layers.size());
    for (const std::uint32_t layer : custom.layers) {
        layer_files.push_back(layer_file_name(layer));
    }
    std::vector<std::string> device_names;
    device_names.reserve(params.devices.size());
    for (const ggml_backend_dev_t device : params.devices) {
        if (device == nullptr) {
            fail("parsed runtime device list contains a null device");
        }
        device_names.emplace_back(ggml_backend_dev_name(device));
    }
    const json runtime_params = {
            {"model_path", model_pin.canonical_path.string()},
            {"tokens_jsonl", normalized_path(custom.tokens_jsonl, "tokens JSONL").string()},
            {"output_dir", normalized_path(custom.out_dir, "output directory").string()},
            {"layers", custom.layers},
            {"context_tokens", n_ctx},
            {"batch_tokens", n_batch},
            {"ubatch_tokens", n_ubatch},
            {"threads", llama_n_threads(context)},
            {"threads_batch", llama_n_threads_batch(context)},
            {"split_mode", split_mode_name(params.split_mode)},
            {"n_gpu_layers", params.n_gpu_layers},
            {"main_gpu", params.main_gpu},
            {"fit_params", params.fit_params},
            {"devices", device_names},
            {"warmup", params.warmup},
    };
    const json receipt = {
            {"request_id", custom.request_id},
            {"model_sha256", custom.model_sha256},
            {"measured_model_sha256", model_pin.measured_sha256},
            {"model_artifact_manifest_sha256", custom.model_artifact_manifest_sha256},
            {"tool_binary_sha256", custom.tool_binary_sha256},
            {"measured_tool_binary_sha256", tool_measured_sha256},
            {"tool_build_contract_sha256", custom.tool_build_contract_sha256},
            {"forced_tokens_sha256", custom.forced_tokens_sha256},
            {"measured_forced_tokens_sha256", token_input.measured_sha256},
            {"held_out_manifest_sha256", custom.held_out_manifest_sha256},
            {"ordered_sample_ids_sha256", custom.ordered_sample_ids_sha256},
            {"profile_tokenizer_sha256", custom.profile_tokenizer_sha256},
            {"runtime_argv_sha256", custom.runtime_argv_sha256},
            {"role", custom.role},
            {"reference_kind", custom.reference_kind},
            {"layers", custom.layers},
            {"normalized_runtime_argv", runtime_argv},
            {"runtime_params", runtime_params},
    };
    const json manifest = {
            {"schema_version", 1},
            {"capture_mode", "teacher_forced"},
            {"vocab_size", n_vocab},
            {"hidden_size", n_embd},
            {"n_hidden_layers", n_layer},
            {"row_count", row},
            {"sample_count", samples.size()},
            {"layers", custom.layers},
            {"receipt", receipt},
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

    fsync_regular_file(temp / "logits.f32");
    for (const std::string & layer_file : layer_files) {
        fsync_regular_file(temp / layer_file);
    }
    fsync_regular_file(temp / "alignment.jsonl");
    fsync_regular_file(temp / "tokenizer.tsv");
    fsync_regular_file(temp / "raw-capture.json");
    fsync_directory(temp, "temporary capture");

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
        return capture(custom, params, common_storage);
    } catch (const std::exception & exc) {
        std::cerr << "llama-atlas-capture: " << exc.what() << '\n';
        return 1;
    }
}
