import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.nvfp4_width_slice import AtlasNvfp4WidthSliceAdapter
from model_atlas.backend.registry import build_default_registry
from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.jobs.artifacts import source_manifest
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus
from model_atlas.recipe.compiler import RecipeCompileError, RecipeCompiler
from model_atlas.recipe.schema import SourceIdentity, StageEffectClass
from model_atlas.recipes.builtin import nvfp4_width_slice_optin_recipe


def _tiny_glm_nvfp4(root: Path) -> Path:
    root.mkdir()
    hidden, full = 64, 32
    config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 1,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": hidden,
        "moe_intermediate_size": full,
        "vocab_size": 8,
        "quantization_config": {"quant_algo": "NVFP4"},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "._model.safetensors.index.json").write_bytes(b"AppleDouble junk")

    tensors: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [8, hidden],
            "bytes": b"\x01\x02" * (8 * hidden),
        },
        "lm_head.weight": {
            "dtype": "BF16",
            "shape": [8, hidden],
            "bytes": b"\x03\x04" * (8 * hidden),
        },
        "model.layers.0.input_layernorm.weight": {
            "dtype": "F32",
            "shape": [hidden],
            "bytes": struct.pack("<f", 1.0) * hidden,
        },
        "model.layers.0.mlp.gate.weight": {
            "dtype": "BF16",
            "shape": [1, hidden],
            "bytes": b"\x05\x06" * hidden,
        },
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        down = projection == "down_proj"
        weight_shape = [hidden, full // 2] if down else [full, hidden // 2]
        scale_shape = [hidden, full // 16] if down else [full, hidden // 16]
        prefix = f"model.layers.0.mlp.experts.0.{projection}"
        tensors[f"{prefix}.weight"] = {
            "dtype": "U8",
            "shape": weight_shape,
            "bytes": bytes(range(256)) * (weight_shape[0] * weight_shape[1] // 256),
        }
        tensors[f"{prefix}.weight_scale"] = {
            "dtype": "F8_E4M3",
            "shape": scale_shape,
            "bytes": b"\x7f" * (scale_shape[0] * scale_shape[1]),
        }
        tensors[f"{prefix}.weight_scale_2"] = {
            "dtype": "F32",
            "shape": [],
            "bytes": struct.pack("<f", 2.0),
        }
        tensors[f"{prefix}.input_scale"] = {
            "dtype": "F32",
            "shape": [],
            "bytes": struct.pack("<f", 3.0),
        }
    shard = root / "model-00001-of-00001.safetensors"
    write_safetensors(shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return root


def _identity(source: Path) -> SourceIdentity:
    manifest = source_manifest(str(source))
    files = manifest["files"]
    assert isinstance(files, dict)
    return SourceIdentity(
        source_id="tiny-glm-nvfp4",
        checkpoint_path=str(source),
        checkpoint_revision="synthetic-v1",
        sha256={str(name): str(digest) for name, digest in files.items()},
    )


def test_backend_record_and_recipe_are_explicit_pruning(tmp_path: Path) -> None:
    source = _tiny_glm_nvfp4(tmp_path / "source")
    registry = build_default_registry()
    record = registry.requires("atlas_nvfp4_width_slice")
    recipe = nvfp4_width_slice_optin_recipe(_identity(source), width=16)

    assert record.produces_derivative
    assert "pruning" in record.declared_capabilities
    assert record.runtime_compat == ()
    assert record.adapter is not None and record.adapter.produces_derivative
    assert recipe.stages[0].effect_class is StageEffectClass.PRUNING
    assert recipe.hardware.runtime_backend == "none"
    RecipeCompiler(registry).compile(recipe)

    record.produces_derivative = False
    with pytest.raises(RecipeCompileError, match="backend_not_derivative_producer"):
        RecipeCompiler(registry).compile(recipe)

    record.produces_derivative = True
    runtime_required = recipe.model_copy(
        update={
            "publish": recipe.publish.model_copy(
                update={"require_runtime_benchmarked": True}
            )
        }
    )
    with pytest.raises(RecipeCompileError, match="runtime_required_missing"):
        RecipeCompiler(registry).compile(runtime_required)


def test_checkpoint_validator_is_cold_importable() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from model_atlas.checkpoint.validators import _safetensors_structure",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_tiny_glm_width_slice_runs_through_job_engine_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = _tiny_glm_nvfp4(tmp_path / "source")
    before = source_manifest(str(source))
    registry = build_default_registry()
    recipe = nvfp4_width_slice_optin_recipe(_identity(source), width=16)
    compiled = RecipeCompiler(registry).compile(recipe)
    engine = JobEngine(compiled, registry, tmp_path / "work")

    job = engine.run(inputs={})

    assert job.status is JobStatus.COMPLETED
    assert source_manifest(str(source)) == before
    outputs = {ref.name: ref for ref in job.stage("width-slice").outputs}
    output_config = json.loads(engine.store.read(outputs["config.json"]))
    evidence = json.loads(engine.store.read(outputs["width-slice.evidence.json"]))
    assert output_config["moe_intermediate_size"] == 16
    assert evidence["result"]["runtime_validated"] is False
    assert "._model.safetensors.index.json" not in outputs
    assert "model-00001-of-00001.safetensors" in outputs


def test_adapter_rejects_output_inside_source(tmp_path: Path) -> None:
    source = _tiny_glm_nvfp4(tmp_path / "source")
    adapter = AtlasNvfp4WidthSliceAdapter()
    context: dict[str, object] = {
        "source": str(source),
        "staging_dir": str(source / "derived"),
        "parameters": {"width": "16"},
    }

    with pytest.raises(BackendUnavailable, match="must not overlap"):
        adapter.prepare(context)
