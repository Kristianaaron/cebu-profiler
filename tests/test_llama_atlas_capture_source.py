from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools" / "llama-atlas-capture" / "atlas_capture.cpp"
CMAKE = ROOT / "tools" / "llama-atlas-capture" / "CMakeLists.txt"


def test_capture_tool_keeps_custom_arguments_out_of_common_parser() -> None:
    source = SOURCE.read_text()
    assert '"--tokens-jsonl"' in source
    assert '"--out-dir"' in source
    assert '"--layers"' in source
    assert "split_custom_args(argc, argv)" in source
    assert "common_argv.data()" in source


def test_capture_tool_is_forced_token_only_and_atomically_published() -> None:
    source = SOURCE.read_text()
    assert "llama_decode(context, batch)" in source
    assert "llama_get_logits_ith" in source
    assert "llama_get_embeddings_layer_inp" in source
    assert "llama_memory_clear" in source
    assert "RENAME_NOREPLACE" in source
    assert "common_sampler" not in source
    assert "llama_sampler" not in source


def test_capture_input_and_source_paths_fail_closed() -> None:
    source = SOURCE.read_text()
    assert "O_NOFOLLOW" in source
    assert "fstat(input_fd.get()" in source
    assert "contents.size() == MAX_INPUT_BYTES" in source
    assert "paths_overlap(candidate, input_normalized)" in source
    assert "is_ancestor_or_equal(model_source, candidate)" in source


def test_capture_schema_has_alignment_and_raw_tensor_files() -> None:
    source = SOURCE.read_text()
    for field in (
        '"schema_version"',
        '"capture_mode"',
        '"row_count"',
        '"sample_count"',
        '"logits.f32"',
        '"alignment.jsonl"',
        '"tokenizer.tsv"',
    ):
        assert field in source


def test_build_links_exact_pinned_library_sonames() -> None:
    cmake = CMAKE.read_text()
    assert "4df29be4f4c3673f428170fda944a5b19f743bb8" in cmake
    assert "libllama.so.0.1.0" in cmake
    assert "libllama-common.so.0.1.0" in cmake
    assert "libggml.so.0.20.0" in cmake
    assert "BUILD_RPATH" in cmake
