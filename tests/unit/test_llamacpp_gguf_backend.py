import hashlib
import json
import struct
import subprocess
from pathlib import Path

import pytest

from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.llamacpp_gguf import (
    CPU_QUANTIZER_RELATIVE_PATH,
    GENERIC_EXPERT_RULE,
    PINNED_COMMIT,
    LlamaCppGgufMixedAdapter,
    build_llamacpp_gguf_record,
    probe_llamacpp_gguf,
)
from model_atlas.backend.registry import BackendRegistry
from model_atlas.checkpoint.validators import _gguf_structure
from model_atlas.jobs.artifacts import source_manifest
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus
from model_atlas.recipe.compiler import RecipeCompiler
from model_atlas.recipe.schema import SourceIdentity, StageEffectClass
from model_atlas.recipes.artifact import CompiledPlanArtifact
from model_atlas.recipes.builtin import (
    GLM52_GGUF_TENSOR_PLAN_SHA256,
    llamacpp_gguf_mixed_recipe,
)

PLAN = "\n".join(
    (
        r"^blk\.2\.ffn_gate_exps\.weight$=NVFP4",
        r"^blk\.2\.ffn_up_exps\.weight$=NVFP4",
        r"^blk\.2\.ffn_down_exps\.weight$=NVFP4",
        GENERIC_EXPERT_RULE,
        "",
    )
)


def _write_fake_gguf(path: Path) -> None:
    name = b"weight"
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, 1, 0)
    header += struct.pack("<Q", len(name)) + name
    header += struct.pack("<I", 1)
    header += struct.pack("<Q", 1)
    header += struct.pack("<I", 0)  # F32
    header += struct.pack("<Q", 0)  # first tensor at data offset zero
    padding = (-len(header)) % 32
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + b"\0" * padding + struct.pack("<f", 1.0))


def _write_two_tensor_gguf(
    path: Path, *, names: tuple[bytes, bytes], offsets: tuple[int, int]
) -> None:
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 0))
    for name, offset in zip(names, offsets, strict=True):
        header += struct.pack("<Q", len(name)) + name
        header += struct.pack("<IQIQ", 1, 1, 0, offset)
    padding = (-len(header)) % 32
    payload_size = max(offsets) + 4
    path.write_bytes(bytes(header) + b"\0" * padding + b"\0" * payload_size)


def _fake_toolchain(tmp_path: Path, *, commit: str = PINNED_COMMIT) -> tuple[Path, Path]:
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    (root / ".git/HEAD").write_text(commit, encoding="utf-8")
    (root / "convert_hf_to_gguf.py").write_text("# fake converter\n", encoding="utf-8")
    quantizer = root / CPU_QUANTIZER_RELATIVE_PATH
    quantizer.parent.mkdir(parents=True)
    quantizer.write_text("#!/bin/sh\n", encoding="utf-8")
    quantizer.chmod(0o755)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    return root, python


def _tool_hashes(root: Path) -> dict[str, str]:
    return {
        "expected_converter_sha256": hashlib.sha256(
            (root / "convert_hf_to_gguf.py").read_bytes()
        ).hexdigest(),
        "expected_quantizer_sha256": hashlib.sha256(
            (root / CPU_QUANTIZER_RELATIVE_PATH).read_bytes()
        ).hexdigest(),
        "expected_python_sha256": hashlib.sha256(
            (root.parent / "venv/bin/python").read_bytes()
        ).hexdigest(),
    }


class FakeRunner:
    def __init__(self, *, fail_quantizer: bool = False) -> None:
        self.fail_quantizer = fail_quantizer
        self.calls: list[list[str]] = []

    def run(self, argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert cwd.name == "llama.cpp"
        self.calls.append(list(argv))
        if "convert_hf_to_gguf.py" in argv[1]:
            _write_fake_gguf(Path(argv[argv.index("--outfile") + 1]))
            return subprocess.CompletedProcess(argv, 0, "", "")
        if self.fail_quantizer:
            return subprocess.CompletedProcess(argv, 7, "", "synthetic quantizer failure")
        _write_fake_gguf(Path(argv[-3]))
        return subprocess.CompletedProcess(argv, 0, "", "")


def _source(tmp_path: Path) -> tuple[Path, SourceIdentity]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"glm_moe_dsa"}', encoding="utf-8")
    manifest = source_manifest(str(source))
    files = manifest["files"]
    assert isinstance(files, dict)
    identity = SourceIdentity(
        source_id="tiny-hf-source",
        checkpoint_path=str(source),
        checkpoint_revision="fake-v1",
        sha256={str(name): str(digest) for name, digest in files.items()},
    )
    return source, identity


def _context(source: Path, staging: Path, *, plan: str = PLAN) -> dict[str, object]:
    return {
        "source": str(source),
        "staging_dir": str(staging),
        "parameters": {"tensor_plan_content": plan, "threads": "3"},
    }


def test_probe_is_filesystem_only_and_rejects_wrong_commit(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path, commit="0" * 40)

    result = probe_llamacpp_gguf(root, python)

    assert not result.available
    evidence = json.loads(result.evidence)
    assert evidence["probe_executed_binaries"] is False
    assert evidence["converter_sha256"]
    assert evidence["quantizer_sha256"]
    assert evidence["quantizer"] == str((root / "build-atlas-cpu/bin/llama-quantize").resolve())
    assert evidence["quantizer_build_contract"] == {
        "GGML_CUDA": False,
        "GGML_RPC": False,
    }
    assert evidence["commit"] == "0" * 40
    record = build_llamacpp_gguf_record(
        toolchain_root=root, python_executable=python, **_tool_hashes(root)
    )
    registry = BackendRegistry({record.backend_id: record})
    assert not registry.is_backend_available(record.backend_id)
    assert record.runtime_compat == ()


def test_recipe_defaults_bind_regenerated_canonical_plan(tmp_path: Path) -> None:
    _source_path, identity = _source(tmp_path)

    recipe = llamacpp_gguf_mixed_recipe(identity)

    assert recipe.stages[0].parameters["tensor_plan_sha256"] == (GLM52_GGUF_TENSOR_PLAN_SHA256)


def test_fake_subprocess_job_uses_exact_no_pruning_argv_and_preserves_source(
    tmp_path: Path,
) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, identity = _source(tmp_path)
    before = source_manifest(str(source))
    runner = FakeRunner()
    record = build_llamacpp_gguf_record(
        toolchain_root=root,
        python_executable=python,
        runner=runner,
        **_tool_hashes(root),
    )
    registry = BackendRegistry({record.backend_id: record})
    recipe = llamacpp_gguf_mixed_recipe(identity, tensor_plan_content=PLAN, threads=3)
    compiled = RecipeCompiler(registry).compile(recipe)
    engine = JobEngine(compiled, registry, tmp_path / "work")

    job = engine.run(inputs={})

    assert job.status is JobStatus.COMPLETED
    assert recipe.stages[0].effect_class is StageEffectClass.QUANTIZATION
    assert recipe.constraints.no_pruning
    assert recipe.hardware.runtime_backend == "none"
    assert not recipe.publish.require_runtime_benchmarked
    assert source_manifest(str(source)) == before
    assert len(runner.calls) == 2
    converter, quantizer = runner.calls
    assert converter == [
        str(python),
        str((root / "convert_hf_to_gguf.py").resolve()),
        str(source.resolve()),
        "--outfile",
        str(engine.run_dir / "stage/llamacpp-gguf-mixed/llamacpp-work/source-auto.gguf"),
        "--outtype",
        "auto",
    ]
    assert quantizer[1:9] == [
        "--allow-requantize",
        "--tensor-type-file",
        str(engine.run_dir / "stage/llamacpp-gguf-mixed/llamacpp-work/tensor-types.txt"),
        "--output-tensor-type",
        "Q4_K",
        "--token-embedding-type",
        "Q4_K",
        str(engine.run_dir / "stage/llamacpp-gguf-mixed/llamacpp-work/source-auto.gguf"),
    ]
    assert quantizer[-3:] == [
        str(engine.run_dir / "stage/llamacpp-gguf-mixed/llamacpp-work/candidate/model.gguf"),
        "Q4_K",
        "3",
    ]
    assert not any(arg.startswith("--prune") for call in runner.calls for arg in call)
    outputs = {ref.name: ref for ref in job.stage("llamacpp-gguf-mixed").outputs}
    evidence = json.loads(engine.store.read(outputs["llamacpp-gguf-mixed.evidence.json"]))
    assert evidence["result"]["runtime_validated"] is False
    assert evidence["result"]["pruning"] is False
    assert (
        evidence["result"]["tensor_plan_sha256"] == hashlib.sha256(PLAN.encode("utf-8")).hexdigest()
    )


def test_tensor_plan_path_hash_mismatch_fails_before_subprocess(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, _identity = _source(tmp_path)
    plan = tmp_path / "plan.txt"
    plan.write_text(PLAN, encoding="utf-8")
    runner = FakeRunner()
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=runner,
        **_tool_hashes(root),
    )
    context = {
        "source": str(source),
        "staging_dir": str(tmp_path / "stage/staging"),
        "parameters": {
            "tensor_plan_path": str(plan),
            "tensor_plan_sha256": "0" * 64,
        },
    }
    adapter.prepare(context)

    with pytest.raises(BackendUnavailable, match="sha256 mismatch"):
        adapter.execute(context, "handle")
    assert runner.calls == []


def test_quantizer_failure_preserves_intermediate_and_resume_skips_conversion(
    tmp_path: Path,
) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, _identity = _source(tmp_path)
    staging = tmp_path / "stage/staging"
    failing = FakeRunner(fail_quantizer=True)
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=failing,
        **_tool_hashes(root),
    )
    context = _context(source, staging)
    adapter.prepare(context)

    with pytest.raises(BackendUnavailable, match="quantizer exited 7"):
        adapter.execute(context, "first")
    intermediate = staging.parent / "llamacpp-work/source-auto.gguf"
    assert intermediate.is_file()

    resumed = FakeRunner()
    adapter2 = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=resumed,
        **_tool_hashes(root),
    )
    result = adapter2.resume(context, "second")

    assert len(resumed.calls) == 1
    assert resumed.calls[0][0].endswith("llama-quantize")
    assert result["runtime_validated"] is False
    assert not intermediate.exists()
    assert _gguf_structure("llamacpp_gguf_mixed", staging, "gguf").ok
    assert adapter2.validate(context, {})["validated"] is True


def test_old_unscoped_generic_fallback_is_rejected(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, _identity = _source(tmp_path)
    runner = FakeRunner()
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=runner,
        **_tool_hashes(root),
    )
    bad_plan = PLAN.replace(GENERIC_EXPERT_RULE, r"ffn_(gate|up|down)_exps\.weight=Q1_0")
    context = _context(source, tmp_path / "stage/staging", plan=bad_plan)
    adapter.prepare(context)

    with pytest.raises(BackendUnavailable, match="must end with the generic"):
        adapter.execute(context, "handle")
    assert runner.calls == []


def test_bounded_gguf_validator_rejects_truncated_header(tmp_path: Path) -> None:
    _write_fake_gguf(tmp_path / "model.gguf")
    assert _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf").ok
    (tmp_path / "model.gguf").write_bytes(b"GGUF\x03")

    result = _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf")

    assert not result.ok
    assert "truncated" in result.detail


def test_probe_rejects_modified_toolchain_bytes_without_execution(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    expected = _tool_hashes(root)
    (root / "convert_hf_to_gguf.py").write_text("# modified\n", encoding="utf-8")

    result = probe_llamacpp_gguf(root, python, **expected)

    assert not result.available
    assert json.loads(result.evidence)["probe_executed_binaries"] is False


def test_compiled_artifact_binds_and_freshly_rechecks_tool_bytes(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    _source_path, identity = _source(tmp_path)
    record = build_llamacpp_gguf_record(
        toolchain_root=root,
        python_executable=python,
        **_tool_hashes(root),
    )
    registry = BackendRegistry({record.backend_id: record})
    recipe = llamacpp_gguf_mixed_recipe(identity, tensor_plan_content=PLAN, threads=3)
    compiled = RecipeCompiler(registry).compile(recipe)
    artifact = CompiledPlanArtifact.from_compiled(compiled, registry=registry)

    pin = artifact.resolved_pins[recipe.stages[0].id]
    assert len(pin["execution_identity_sha256"]) == 64
    artifact.verify_pins_against(registry)

    (root / "convert_hf_to_gguf.py").write_text("# drifted converter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no longer available|execution identity"):
        artifact.verify_pins_against(registry)


def test_validator_rejects_partial_and_unknown_payloads(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    _write_fake_gguf(model)
    raw = model.read_bytes()
    model.write_bytes(raw[:-1])
    assert not _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf").ok

    # Tensor type lives 12 bytes before the aligned data in this tiny fixture.
    raw = bytearray(raw)
    type_offset = len(raw) - 4 - ((-(len(raw) - 4)) % 32) - 12
    # Locate robustly from the known name and shape encoding.
    marker = b"weight" + struct.pack("<IQ", 1, 1)
    type_offset = raw.index(marker) + len(marker)
    struct.pack_into("<I", raw, type_offset, 999)
    model.write_bytes(raw)
    result = _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf")
    assert not result.ok
    assert "unsupported GGUF tensor type" in result.detail


def test_validator_rejects_duplicate_names_and_overlapping_payloads(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    _write_two_tensor_gguf(model, names=(b"same", b"same"), offsets=(0, 32))
    result = _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf")
    assert not result.ok
    assert "unique" in result.detail

    _write_two_tensor_gguf(model, names=(b"first", b"second"), offsets=(0, 0))
    result = _gguf_structure("llamacpp_gguf_mixed", tmp_path, "gguf")
    assert not result.ok
    assert "overlap" in result.detail


def test_scratch_may_not_overlap_immutable_source(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    source = tmp_path / "stage/llamacpp-work"
    source.mkdir(parents=True)
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root, python_executable=python, **_tool_hashes(root)
    )
    context = _context(source, tmp_path / "stage/staging")

    with pytest.raises(BackendUnavailable, match="scratch and immutable source"):
        adapter.prepare(context)

    safe_source = tmp_path / "safe-source"
    safe_source.mkdir()
    stage = tmp_path / "alias-stage"
    stage.mkdir()
    (stage / "llamacpp-work").symlink_to(safe_source, target_is_directory=True)
    alias_context = _context(safe_source, stage / "staging")
    with pytest.raises(BackendUnavailable, match="scratch must not be a symlink"):
        adapter.prepare(alias_context)


def test_resume_rejects_changed_plan_provenance_before_subprocess(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, _identity = _source(tmp_path)
    staging = tmp_path / "stage/staging"
    failing = FakeRunner(fail_quantizer=True)
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=failing,
        **_tool_hashes(root),
    )
    context = _context(source, staging)
    adapter.prepare(context)
    with pytest.raises(BackendUnavailable, match="quantizer exited"):
        adapter.execute(context, "first")

    changed = _context(source, staging, plan=PLAN.replace("blk\\.2", "blk\\.3"))
    resumed = FakeRunner()
    adapter2 = LlamaCppGgufMixedAdapter(
        toolchain_root=root,
        python_executable=python,
        runner=resumed,
        **_tool_hashes(root),
    )
    with pytest.raises(BackendUnavailable, match="resume provenance does not match"):
        adapter2.resume(changed, "second")
    assert resumed.calls == []

    changed_threads = _context(source, staging)
    parameters = changed_threads["parameters"]
    assert isinstance(parameters, dict)
    parameters["threads"] = "7"
    with pytest.raises(BackendUnavailable, match="resume provenance does not match"):
        adapter2.resume(changed_threads, "third")
    assert resumed.calls == []
