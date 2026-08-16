import hashlib
import json
import struct
from pathlib import Path

from model_atlas.backend.contract import BackendAdapter, BackendRecord
from model_atlas.backend.registry import BackendRegistry
from model_atlas.jobs.artifacts import ContentAddressedStore
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus
from model_atlas.recipe.compiler import RecipeCompiler
from model_atlas.recipe.schema import (
    CalibrationIdentity,
    CompressionRecipe,
    RecipeStage,
    RecipeStatus,
    SourceIdentity,
    StageBackendPin,
    StageEffectClass,
)


def test_safetensors_validator_accepts_f8_e4m3_bytes(tmp_path: Path) -> None:
    from model_atlas.checkpoint.validators import _safetensors_structure

    header = {
        "scale": {"dtype": "F8_E4M3", "shape": [3], "data_offsets": [0, 3]},
        "__metadata__": {"format": 0},
    }
    encoded = json.dumps(header).encode("utf-8")
    shard = tmp_path / "model.safetensors"
    with shard.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(b"\x00\x7f\xff")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"scale": shard.name}}), encoding="utf-8"
    )

    result = _safetensors_structure("modelopt_nvfp4", tmp_path, "safetensors")

    assert result.ok, result.detail
    assert result.tensor_count == 1


def test_cas_large_file_verification_never_uses_read_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"atlas-large-file" * 140_000)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    original_read_bytes = Path.read_bytes

    def reject_large_read_bytes(path: Path) -> bytes:
        if path.exists() and path.stat().st_size > 1 << 20:
            raise AssertionError("large-file comparison used Path.read_bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_large_read_bytes)
    store = ContentAddressedStore(tmp_path / "cas")

    ref = store.put_file(source.name, source)

    assert ref.sha256 == expected
    assert store.verify(ref)


class _ContextCaptureAdapter(BackendAdapter):
    backend_id = "context_capture"

    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def prepare(self, context: dict[str, object]) -> str:
        return "ready"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        self.captured.update(context)
        Path(str(context["staging_dir"]), "context.json").write_text("{}", encoding="utf-8")
        return {"captured": True}

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        return self.execute(context, handle)

    def validate(
        self, context: dict[str, object], outputs: dict[str, object]
    ) -> dict[str, object]:
        return {"validated": True}


def test_engine_passes_recipe_source_and_calibration_identity_to_adapter(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weight = source_dir / "weight.bin"
    weight.write_bytes(b"weights")
    captured: dict[str, object] = {}
    adapter = _ContextCaptureAdapter(captured)
    record = BackendRecord(
        backend_id=adapter.backend_id,
        display_name="Context capture",
        method_family="atlas",
        formats=("json",),
        architectures=("any",),
        compute_archs=("any",),
        topologies=("any",),
        runtime_compat=("vllm-modelopt",),
        status=RecipeStatus.VALIDATED,
        version="1.0.0",
        supported_formats=("json",),
        availability_probe=lambda: (True, "1.0.0", "test adapter"),
        adapter=adapter,
    )
    registry = BackendRegistry({adapter.backend_id: record})
    recipe = CompressionRecipe(
        name="context-bound",
        source=SourceIdentity(
            source_id="glm-5.2-teacher",
            checkpoint_path=str(source_dir),
            checkpoint_revision="rev-42",
            sha256={"weight.bin": hashlib.sha256(b"weights").hexdigest()},
        ),
        calibration=CalibrationIdentity(
            calibration_id="cal-7",
            corpus_name="atlas-calibration",
            seed=17,
            partition="held-out",
            corpus_records_path="/datasets/calibration.jsonl",
            tokenizer_sha256="a" * 64,
        ),
        stages=[
            RecipeStage(
                id="capture",
                name="capture",
                effect_class=StageEffectClass.PROFILING,
                backend=StageBackendPin(backend_id=adapter.backend_id, version="1.0.0"),
                produces_format=["json"],
                expected_outputs=["context.json"],
            )
        ],
    )

    compiled = RecipeCompiler(registry).compile(recipe)
    job = JobEngine(compiled, registry, tmp_path / "work").run(inputs={})

    assert job.status is JobStatus.COMPLETED
    assert captured["source"] == str(source_dir)
    assert captured["source_revision"] == "rev-42"
    assert captured["source_identity"] == recipe.source.model_dump(mode="json")
    assert captured["calibration"] == recipe.calibration.model_dump(mode="json")
