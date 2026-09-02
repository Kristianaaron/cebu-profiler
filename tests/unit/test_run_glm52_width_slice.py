"""Tests for the GLM-5.2 width-slice runner (scripts/run_glm52_width_slice.py).

Covers: keep-map basis parsing/ranking, dry-run purity, the execute happy path
(bundle pack + verified handoff), and the explicit gate step (accepted publish
vs rejected exit 2 with no publication). Fixtures mirror
tests/unit/test_nvfp4_width_slice_adapter.py / test_llamacpp_capture.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest

from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.prune.keep_map import KeepMapBasisError, parse_saliency_basis
from model_atlas.recommend.policy import AtlasProfile
from model_atlas.runtime_artifact_handoff import load_verified_width_slice_handoff

_MODULE_PATH = Path(__file__).parents[2] / "scripts" / "run_glm52_width_slice.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_glm52_width_slice", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_glm52_width_slice"] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glm_source(root: Path) -> Path:
    """Tiny GLM-style NVFP4 checkpoint (1 sparse layer x 1 expert, full=32)."""
    root.mkdir(parents=True, exist_ok=True)
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
        json.dumps({"metadata": {}, "weight_map": {n: shard.name for n in tensors}}),
        encoding="utf-8",
    )
    return root


def _profile(root: Path, source: Path) -> dict[str, object]:
    """Profile fixture mirroring the real builder's execution/evidence shape."""
    files = {
        str(path.relative_to(source)): _sha(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    manifest_digest = hashlib.sha256(
        json.dumps(
            {"type": "dir", "files": files, "file_stats": {}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    census_basis = hashlib.sha256(
        json.dumps(
            {
                "source_manifest_digest": manifest_digest,
                "config_sha256": _sha(source / "config.json"),
                "index_sha256": _sha(source / "model.safetensors.index.json"),
                "tensor_plan_sha256": _sha(source / "config.json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "profile_id": "glm52",
        "model": "glm-5.2",
        "routing_consistency_passed": True,
        "evidence": {
            "identity": {"kind": "measured", "present": True},
            "nvfp4_suitability": {
                "kind": "estimated",
                "present": True,
                "detail": f"risk_artifact_sha256={'e' * 64};tensor_plan_sha256={'f' * 64}",
            },
            "channel_saliency": {
                "kind": "estimated",
                "present": True,
                "detail": (
                    "basis=header_structural_census;"
                    f"basis_artifact_sha256={census_basis};"
                    "activation_profiling=not_yet_run"
                ),
            },
        },
        "execution": {
            "source_id": "tiny-glm",
            "checkpoint_path": str(source),
            "checkpoint_revision": "synthetic-v1",
            "source_manifest_digest": manifest_digest,
            "source_sha256": {},
            "tokenizer_hash": _sha(source / "tokenizer.json"),
        },
    }


def _metric_report(tmp_path: Path) -> Path:
    """A REAL CaptureMetricReport from two synthetic capture artifacts."""
    spec = importlib.util.spec_from_file_location(
        "test_glm52_width_slice_capture", _CAPTURE_TEST_PATH
    )
    assert spec is not None and spec.loader is not None
    capture_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture_test)
    candidate_request, candidate_root = capture_test._raw_capture(tmp_path / "candidate")
    candidate = capture_test.finalize_capture(candidate_request)
    reference_request, reference_root = capture_test._raw_capture(
        tmp_path / "reference", role=capture_test.CaptureRole.IDENTITY_CONTROL
    )
    reference = capture_test.finalize_capture(reference_request)
    from model_atlas.evaluation.capture_metrics import (
        _canonical_sha256,
        evaluate_capture_pair,
    )

    report = evaluate_capture_pair(
        reference_root=reference_root,
        reference=reference,
        candidate_root=candidate_root,
        candidate=candidate,
        cka_rows=3,
    )
    payload = report.model_dump(mode="json", exclude={"report_id"})
    path = tmp_path / "metric-report.json"
    path.write_text(json.dumps({**payload, "report_id": _canonical_sha256(payload)}))
    return path


_CAPTURE_TEST_PATH = Path(__file__).parents[2] / "tests" / "unit" / "test_llamacpp_capture.py"


def _runner_args(tmp_path: Path, profile_path: Path, **overrides: object) -> list[str]:
    args = [
        "--profile",
        str(profile_path),
        "--output",
        str(tmp_path / "out" / "deriv"),
        "--result",
        str(tmp_path / "result.json"),
        "--width",
        "16",
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        else:
            args.extend((flag, str(value)))
    return args


def _run(module: ModuleType, argv: list[str]) -> int:
    return module.main(argv)


def _execute(tmp_path: Path, argv: list[str]) -> tuple[int, str, str]:
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(_MODULE_PATH), *argv],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture()
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = _glm_source(tmp_path / "source")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(tmp_path, source)))
    return tmp_path, profile_path


# ---------------------------------------------------------------------------
# keep-map basis parsing + ranking
# ---------------------------------------------------------------------------


def test_parse_saliency_basis_structural_round_trip() -> None:
    detail = (
        "basis=header_structural_census;"
        f"basis_artifact_sha256={'a' * 64};"
        "activation_profiling=not_yet_run"
    )
    basis = parse_saliency_basis(detail)
    assert basis.ranking_basis == "structural_group_order"
    assert basis.activation_profiling_run is False
    assert basis.group_scores is None
    assert basis.basis_artifact_sha256 == "a" * 64


def test_parse_saliency_basis_measured_scores() -> None:
    detail = (
        "basis=activation_profile_v1;"
        f"basis_artifact_sha256={'b' * 64};"
        'activation_profiling=completed;saliency={"(0,0)": [1.0, 9.0]}'
    )
    basis = parse_saliency_basis(detail)
    assert basis.ranking_basis == "measured_group_saliency"
    assert basis.group_scores == {"(0,0)": [1.0, 9.0]}


@pytest.mark.parametrize(
    ("detail", "match"),
    [
        ("basis=x;activation_profiling=not_yet_run", "missing"),
        (
            "basis=x;basis_artifact_sha256=ZZ;activation_profiling=not_yet_run",
            "sha256",
        ),
        (
            "basis=x;basis_artifact_sha256="
            + "c" * 64
            + ';activation_profiling=not_yet_run;saliency={"(0,0)": [1.0]}',
            "not completed",
        ),
        (
            "basis=x;basis_artifact_sha256="
            + "d" * 64
            + ';activation_profiling=completed;saliency={"(0,0)": [1.0, 1e999]}',
            "finite",
        ),
        ("basis=x;;activation_profiling=not_yet_run", "malformed"),
    ],
)
def test_parse_saliency_basis_fail_closed(detail: str, match: str) -> None:
    with pytest.raises(KeepMapBasisError, match=match):
        parse_saliency_basis(detail)


def test_profile_channel_saliency_detail_round_trips_through_parser() -> None:
    """The emitter's detail format is exactly what the runner consumes."""
    detail = (
        "basis=header_structural_census;"
        f"basis_artifact_sha256={'0123456789abcdef' * 4};"
        "activation_profiling=not_yet_run"
    )
    basis = parse_saliency_basis(detail)
    assert basis.ranking_basis == "structural_group_order"


# ---------------------------------------------------------------------------
# StageEvidence fail-closed parseability of the emitted profile
# ---------------------------------------------------------------------------


def test_emitted_profile_parses_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real emitter's channel_saliency survives StageEvidence.from_dict."""
    import tempfile

    import model_atlas.glm52_source_profile as source_profile
    from model_atlas.glm52_source_profile import (
        build_glm52_mixed_gguf_profile,
        build_resumable_source_manifest,
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        source = tdp / "source"
        source.mkdir()
        (source / "config.json").write_text('{"architectures":["GlmForCausalLM"]}')
        (source / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        tokenizer = source / "tokenizer.json"
        tokenizer.write_text('{"version":"1.0"}')
        manifest_path = tdp / "manifest.json"
        build_resumable_source_manifest(
            source, checkpoint_path=tdp / "state.json", output_path=manifest_path
        )
        plan = tdp / "plan.txt"
        plan.write_text("^blk\\.5\\.ffn_gate_exps\\.weight$=NVFP4\n")
        risk = tdp / "risk.json"
        risk.write_text(
            json.dumps(
                {
                    "source": str(source.resolve()),
                    "config_sha256": _sha(source / "config.json"),
                    "index_sha256": _sha(source / "model.safetensors.index.json"),
                    "tensor_type_sha256": _sha(plan),
                    "tensor_type_lines": plan.read_text(encoding="utf-8").splitlines(),
                    "evidence_kind": "estimated",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(source_profile, "GLM52_GGUF_TENSOR_PLAN_SHA256", _sha(plan))
        profile = build_glm52_mixed_gguf_profile(
            manifest_path=manifest_path,
            source_path=source,
            source_revision="revision",
            tokenizer_path=tokenizer,
            risk_path=risk,
            tensor_plan_path=plan,
            output_path=tdp / "profile.json",
        )
        from model_atlas.recommend.policy import StageEvidence

        imported = AtlasProfile.from_dict(json.loads((tdp / "profile.json").read_text()))
        stage = imported.evidence["channel_saliency"]
        assert isinstance(stage, StageEvidence)
        assert stage.kind == "estimated"
        assert stage.present is True
        raw = profile["evidence"]
        assert isinstance(raw, dict)
        emitted = raw["channel_saliency"]
        assert isinstance(emitted, dict)
        assert StageEvidence.from_dict("channel_saliency", emitted) == stage


# ---------------------------------------------------------------------------
# runner: dry-run purity + execute happy path
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_writes_nothing(workspace: tuple[Path, Path]) -> None:
    module = _module()
    tmp, profile_path = workspace
    out = tmp / "out" / "deriv"
    result = tmp / "result.json"
    argv = _runner_args(tmp, profile_path)
    rc = _run(module, argv)
    assert rc == 0
    assert not out.exists()
    assert not result.exists()
    assert not list((tmp / "out").glob("*")) if (tmp / "out").exists() else True


def test_execute_happy_path_publishes_verified_handoff(
    workspace: tuple[Path, Path],
) -> None:
    module = _module()
    tmp, profile_path = workspace
    out = tmp / "out" / "deriv"
    bundle = tmp / "out" / "deriv.atlasbundle"
    result = tmp / "result.json"
    rc = _run(module, _runner_args(tmp, profile_path, execute=True))
    assert rc == 0
    assert out.is_dir()
    assert bundle.is_file()
    assert result.is_file()
    handoff = load_verified_width_slice_handoff(result)
    assert handoff.artifact_sha256 == _sha(bundle)
    assert handoff.evidence_size_bytes > 0
    payload = json.loads(result.read_text())
    assert payload["runtime_artifact"]["runtime_validated"] is False
    assert payload["runtime_claim"] == "artifact_only_unvalidated"
    evidence = json.loads(
        (Path(payload["runtime_artifact"]["path"]).parents[2])
        .joinpath(payload["runtime_artifact"]["evidence"]["relpath"])
        .read_text()
    )
    assert evidence["quality_claim"] is False
    assert evidence["runtime_validated"] is False
    assert evidence["keep_map"] == {"0:0": list(range(16, 32))}
    assert evidence["ranking_basis"] == "structural_group_order"


def test_execute_refuses_existing_outputs(workspace: tuple[Path, Path]) -> None:
    module = _module()
    tmp, profile_path = workspace
    (tmp / "out").mkdir()
    (tmp / "out" / "deriv.atlasbundle").write_bytes(b"x")
    with pytest.raises(module.WidthSliceRunnerError, match="already exists"):
        _run(module, _runner_args(tmp, profile_path, execute=True))


def test_width_or_memory_target_required_but_not_both(
    workspace: tuple[Path, Path],
) -> None:
    module = _module()
    tmp, profile_path = workspace
    source = Path(json.loads(profile_path.read_text())["execution"]["checkpoint_path"])
    with pytest.raises(module.WidthSliceRunnerError, match="not both"):
        module._resolve_width(source, 16, 0.001)
    argv = [
        "--profile",
        str(profile_path),
        "--output",
        str(tmp / "out" / "deriv"),
        "--result",
        str(tmp / "result.json"),
        "--memory-target-gib",
        "0.001",
    ]
    plan = json.loads(json.dumps(module._plan(module.parse_args(argv))))
    assert plan["width"] % 16 == 0


# ---------------------------------------------------------------------------
# runner: explicit gate step
# ---------------------------------------------------------------------------


def test_gate_accepted_records_decision_and_publishes(
    workspace: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> None:
    module = _module()
    tmp, profile_path = workspace
    report = _metric_report(tmp_path_factory.mktemp("gate-acc"))
    rc = _run(
        module,
        _runner_args(
            tmp,
            profile_path,
            execute=True,
            metric_report=report,
            mean_budget=0.01,
            worst_domain_budget=0.02,
            p99_budget=0.03,
            cka_floor=0.9,
        ),
    )
    assert rc == 0
    payload = json.loads((tmp / "result.json").read_text())
    evidence = json.loads(
        Path(payload["runtime_artifact"]["path"])
        .parents[2]
        .joinpath(payload["runtime_artifact"]["evidence"]["relpath"])
        .read_text()
    )
    assert evidence["gate_decision"] is not None
    assert evidence["gate_decision"]["status"] == "accepted"


def test_gate_rejection_exits_2_and_publishes_nothing(
    workspace: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> None:
    module = _module()
    tmp, profile_path = workspace
    report = _metric_report(tmp_path_factory.mktemp("gate-rej"))
    out = tmp / "out" / "deriv"
    bundle = tmp / "out" / "deriv.atlasbundle"
    result = tmp / "rejected.json"
    rc = _run(
        module,
        [
            "--profile",
            str(profile_path),
            "--output",
            str(out),
            "--result",
            str(result),
            "--width",
            "16",
            "--execute",
            "--metric-report",
            str(report),
            "--mean-budget",
            "0.01",
            "--worst-domain-budget",
            "0.02",
            "--p99-budget",
            "0.03",
            "--cka-floor",
            "1.5",
        ],
    )
    assert rc == 2
    assert not out.exists()
    assert not bundle.exists()
    decision = json.loads(result.read_text())
    assert decision["quality_gate"]["status"] == "rejected"
    assert decision["quality_gate"]["gate"]["accepted"] is False
    assert any("CKA" in failure for failure in decision["quality_gate"]["gate"]["failures"])


def test_gate_requires_budgets_together(workspace: tuple[Path, Path]) -> None:
    module = _module()
    tmp, profile_path = workspace
    with pytest.raises(module.WidthSliceRunnerError, match="all four budgets"):
        _run(
            module,
            _runner_args(
                tmp,
                profile_path,
                metric_report=tmp / "missing-report.json",
            ),
        )


def test_missing_channel_saliency_blocks_run(workspace: tuple[Path, Path]) -> None:
    module = _module()
    tmp, profile_path = workspace
    payload = json.loads(profile_path.read_text())
    del payload["evidence"]["channel_saliency"]
    profile_path.write_text(json.dumps(payload))
    with pytest.raises(module.WidthSliceRunnerError, match="channel_saliency"):
        _run(module, _runner_args(tmp, profile_path))
