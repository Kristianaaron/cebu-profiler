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
    assert evidence["quantizer"] == str(
        (root / "build-atlas-cpu/bin/llama-quantize").resolve()
    )
    assert evidence["quantizer_build_contract"] == {
        "GGML_CUDA": False,
        "GGML_RPC": False,
    }
    assert evidence["commit"] == "0" * 40
    record = build_llamacpp_gguf_record(toolchain_root=root, python_executable=python)
    registry = BackendRegistry({record.backend_id: record})
    assert not registry.is_backend_available(record.backend_id)
    assert record.runtime_compat == ()


def test_recipe_defaults_bind_regenerated_canonical_plan(tmp_path: Path) -> None:
    _source_path, identity = _source(tmp_path)

    recipe = llamacpp_gguf_mixed_recipe(identity)

    assert recipe.stages[0].parameters["tensor_plan_sha256"] == (
        GLM52_GGUF_TENSOR_PLAN_SHA256
    )


def test_fake_subprocess_job_uses_exact_no_pruning_argv_and_preserves_source(
    tmp_path: Path,
) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, identity = _source(tmp_path)
    before = source_manifest(str(source))
    runner = FakeRunner()
    record = build_llamacpp_gguf_record(
        toolchain_root=root, python_executable=python, runner=runner
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
        "--no-nextn",
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
        str(engine.run_dir / "stage/llamacpp-gguf-mixed/staging/model.gguf"),
        "Q4_K",
        "3",
    ]
    assert not any(arg.startswith("--prune") for call in runner.calls for arg in call)
    outputs = {ref.name: ref for ref in job.stage("llamacpp-gguf-mixed").outputs}
    evidence = json.loads(engine.store.read(outputs["llamacpp-gguf-mixed.evidence.json"]))
    assert evidence["result"]["runtime_validated"] is False
    assert evidence["result"]["pruning"] is False
    assert evidence["result"]["tensor_plan_sha256"] == hashlib.sha256(
        PLAN.encode("utf-8")
    ).hexdigest()


def test_tensor_plan_path_hash_mismatch_fails_before_subprocess(tmp_path: Path) -> None:
    root, python = _fake_toolchain(tmp_path)
    source, _identity = _source(tmp_path)
    plan = tmp_path / "plan.txt"
    plan.write_text(PLAN, encoding="utf-8")
    runner = FakeRunner()
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=root, python_executable=python, runner=runner
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
        toolchain_root=root, python_executable=python, runner=failing
    )
    context = _context(source, staging)
    adapter.prepare(context)

    with pytest.raises(BackendUnavailable, match="quantizer exited 7"):
        adapter.execute(context, "first")
    intermediate = staging.parent / "llamacpp-work/source-auto.gguf"
    assert intermediate.is_file()

    resumed = FakeRunner()
    adapter2 = LlamaCppGgufMixedAdapter(
        toolchain_root=root, python_executable=python, runner=resumed
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
        toolchain_root=root, python_executable=python, runner=runner
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
