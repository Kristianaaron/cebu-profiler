#!/usr/bin/env python3
"""GLM-5.2 NVFP4 width-slice runner: saliency-ranked keep-map -> derivative.

Dry-run by default: resolves the retention width from the header-level
sizing/planner, derives the deterministic saliency-ranked keep-map from the
source profile's ``channel_saliency`` evidence, prints the resulting plan
(width, per-layer keep counts, bundle path, gate step state) and writes
NOTHING. ``--execute`` performs the real work:

1. ``materialize_uniform_width`` with the ranked keep-map into the output
   directory (transactional exporter; the immutable source is never touched).
2. Structural validation through the REGISTERED ``atlas_nvfp4_width_slice``
   checkpoint validator (real safetensors structure, never a name check).
3. ``pack_derivative_bundle`` -> one deterministic ``.atlasbundle``.
4. Run-result JSON (run lineage + CAS-layout artifact/evidence slots) that
   ``load_verified_width_slice_handoff`` byte-verifies end-to-end.

The quality verdict is an EXPLICIT separate step: ``--metric-report`` accepts
a real ``CaptureMetricReport`` and runs the existing KLD/CKA gate semantics
(``run_kl_quality_gate``). On rejection the rejected decision JSON is written,
NO bundle/handoff is published, and the exit code is 2. Structural success is
never claimed as runtime validation; the bundle stays
``runtime_validated=False`` until a real load/forward canary passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.checkpoint.validators import get_checkpoint_validator
from model_atlas.evaluation.capture_metrics import CaptureMetricReport
from model_atlas.evaluation.quality_gate import QualityGateRejection, run_kl_quality_gate
from model_atlas.loader import materialize_uniform_width
from model_atlas.prune.bundle import pack_derivative_bundle
from model_atlas.prune.keep_map import KeepMapBasisError, build_keep_map, parse_saliency_basis
from model_atlas.prune.kl_gate import KLGateBudget, KLGateError
from model_atlas.prune.planner import plan_uniform_width
from model_atlas.prune.width_sizing import size_checkpoint_for_width
from model_atlas.recommend.policy import AtlasProfile
from model_atlas.runtime_artifact_handoff import load_verified_width_slice_handoff

SCHEMA_VERSION = 1
_METHOD = "atlas-nvfp4-width-slice"
_ARTIFACT_NAME = "model.safetensors.atlasbundle"
_EVIDENCE_NAME = f"{_METHOD}.evidence.json"
_MAX_REPORT_BYTES = 256 * 1024 * 1024


class WidthSliceRunnerError(RuntimeError):
    """Width-slice run cannot proceed without weakening an invariant."""


def _read_bounded(path: Path, limit: int = _MAX_REPORT_BYTES) -> bytes:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise WidthSliceRunnerError(f"metric report is unreadable: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or not info.st_size or info.st_size > limit:
        raise WidthSliceRunnerError("metric report is not a bounded regular file")
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise WidthSliceRunnerError("metric report exceeds its bounded read")
    return data


def _exclusive_json(path: Path, payload: dict[str, object]) -> str:
    """Write one JSON artifact exclusively + durably; returns its sha256."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _load_profile(path: Path) -> tuple[AtlasProfile, str]:
    raw = _read_bounded(path, 16 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WidthSliceRunnerError(f"profile is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WidthSliceRunnerError("profile must be a JSON object")
    profile = AtlasProfile.from_dict(payload)
    if profile.execution is None:
        raise WidthSliceRunnerError("profile lacks executable source identity")
    checkpoint = Path(profile.execution.checkpoint_path)
    if not checkpoint.is_dir():
        raise WidthSliceRunnerError(f"profile checkpoint path is not a directory: {checkpoint}")
    return profile, hashlib.sha256(raw).hexdigest()


def _saliency_evidence(profile: AtlasProfile) -> dict[str, object]:
    """Extract the channel_saliency StageEvidence dict from a profile.

    Fail-closed: the width-slice method requires this stage (policy
    ``evidence_stages=('channel_saliency',)``); a missing/absent claim means
    the run must not silently fall back to a fabricated ranking.
    """
    stage = profile.evidence.get("channel_saliency")
    if stage is None or not stage.present:
        raise WidthSliceRunnerError(
            "profile has no present channel_saliency evidence; the width-slice "
            "method requires it (never fabricate a ranking)"
        )
    return {
        "kind": stage.kind,
        "present": stage.present,
        "coverage": stage.coverage,
        "detail": stage.detail,
    }


def _resolve_width(
    source: Path, explicit: int | None, target_gib: float | None
) -> tuple[int, dict[str, object]]:
    """Resolve the retention width from --width or the sizing/planner pair."""
    if explicit is not None and target_gib is not None:
        raise WidthSliceRunnerError("pass either --width or --memory-target-gib, not both")
    if explicit is not None:
        if explicit <= 0 or explicit % 16 != 0:
            raise WidthSliceRunnerError(f"--width {explicit} must be a positive multiple of 16")
        return explicit, {"resolved_by": "explicit"}
    if target_gib is None:
        raise WidthSliceRunnerError("one of --width or --memory-target-gib is required")
    sizing = size_checkpoint_for_width(source)
    width = plan_uniform_width(
        expert_source_gib=sizing.expert_gib,
        protected_gib=sizing.protected_gib,
        target_gib=target_gib,
        full=sizing.full_width,
    )
    return width, {
        "resolved_by": "size_checkpoint_for_width+plan_uniform_width",
        "expert_gib": sizing.expert_gib,
        "protected_gib": sizing.protected_gib,
        "full_width": sizing.full_width,
        "target_gib": target_gib,
        "shards_scanned": sizing.shards_scanned,
    }


def _plan(args: argparse.Namespace) -> dict[str, object]:
    """Build the full dry-run/execute plan (no writes in either mode)."""
    profile, profile_sha256 = _load_profile(args.profile)
    execution = profile.execution
    assert execution is not None  # narrowed by _load_profile
    source = Path(execution.checkpoint_path).resolve()
    config = json.loads((source / "config.json").read_text())
    manifest_digest = execution.source_manifest_digest

    width, width_provenance = _resolve_width(source, args.width, args.memory_target_gib)

    saliency = _saliency_evidence(profile)
    try:
        basis = parse_saliency_basis(
            saliency["detail"] if isinstance(saliency["detail"], str) else ""
        )
        from model_atlas.loader import _infer_geometry

        geometry = load_manifest(str(source))
        full, n_exp, sparse_layers = _infer_geometry(geometry, config)
        keep_map = build_keep_map(
            basis, width=width, full=full, sparse_layers=sparse_layers, n_exp=n_exp
        )
    except KeepMapBasisError as exc:
        raise WidthSliceRunnerError(f"saliency keep-map derivation failed: {exc}") from exc

    per_layer_counts: dict[str, int] = {}
    for (layer, _expert), channels in sorted(keep_map.items()):
        per_layer_counts[str(layer)] = max(per_layer_counts.get(str(layer), 0), len(channels))
    output = args.output.resolve()
    bundle_path = (output.parent / f"{output.name}.atlasbundle").resolve()
    if _paths_overlap(source, output) or _paths_overlap(source, bundle_path):
        raise WidthSliceRunnerError("output and bundle paths must not overlap the source")
    return {
        "schema_version": SCHEMA_VERSION,
        "method": _METHOD,
        "profile_path": str(args.profile),
        "profile_sha256": profile_sha256,
        "profile_id": profile.profile_id_of(),
        "source": str(source),
        "source_manifest_digest": manifest_digest,
        "width": width,
        "width_provenance": width_provenance,
        "saliency_evidence": saliency,
        "ranking_basis": basis.ranking_basis,
        "basis_artifact_sha256": basis.basis_artifact_sha256,
        "activation_profiling_run": basis.activation_profiling_run,
        "keep_map_coverage": {
            "targets": len(keep_map),
            "channels_per_target": width,
            "per_layer_max_kept": per_layer_counts,
        },
        "output": str(output),
        "bundle_path": str(bundle_path),
        "quality_gate": {
            "metric_report": str(args.metric_report) if args.metric_report else None,
            "budgets": {
                "mean_kld": args.mean_budget,
                "worst_domain_kld": args.worst_domain_budget,
                "p99_kld": args.p99_kld_budget,
                "cka_floor": args.cka_floor,
            },
            "state": "pending" if args.metric_report else "not_requested",
            "note": (
                "quality verdict comes only from the KLD/CKA gate step; structural "
                "success here is never runtime validation"
            ),
        },
        "execute": args.execute,
    }


def _validate_structurally(output: Path) -> dict[str, object]:
    validator = get_checkpoint_validator("atlas_nvfp4_width_slice", "checkpoint")
    if validator is None:
        raise WidthSliceRunnerError(
            "atlas_nvfp4_width_slice checkpoint validator is not registered; fail closed"
        )
    result = validator("atlas_nvfp4_width_slice", output, "safetensors")
    if not result.ok:
        raise WidthSliceRunnerError(f"structural validation failed: {result.detail}")
    return result.to_dict()


def _write_handoff_result(
    *,
    run_dir: Path,
    run_id: str,
    plan: dict[str, object],
    bundle_path: Path,
    bundle_sha256: str,
    bundle_size: int,
    validation: dict[str, object],
    keep_map: dict[tuple[int, int], list[int]],
    gate_decision: dict[str, object] | None,
) -> Path:
    """Write the CAS-layout result JSON consumable by the width-slice handoff."""
    evidence_payload = {
        "stage": _METHOD,
        "structural_validation": validation,
        "keep_map": {f"{k[0]}:{k[1]}": list(v) for k, v in sorted(keep_map.items())},
        "width": plan["width"],
        "ranking_basis": plan["ranking_basis"],
        "basis_artifact_sha256": plan["basis_artifact_sha256"],
        "activation_profiling_run": plan["activation_profiling_run"],
        "runtime_validated": False,
        "quality_claim": False,
        "gate_decision": gate_decision,
    }
    evidence_encoded = json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode()
    evidence_sha256 = hashlib.sha256(evidence_encoded).hexdigest()
    evidence_slot = run_dir / "objects" / evidence_sha256[:2] / f"{evidence_sha256}.blob"
    evidence_slot.parent.mkdir(parents=True, exist_ok=True)
    evidence_tmp = evidence_slot.with_suffix(".blob.tmp")
    evidence_tmp.write_bytes(evidence_encoded)
    os.replace(evidence_tmp, evidence_slot)

    bundle_slot = run_dir / "objects" / bundle_sha256[:2] / f"{bundle_sha256}.blob"
    bundle_slot.parent.mkdir(parents=True, exist_ok=True)
    if bundle_slot.exists():
        if _sha256_file(bundle_slot) != bundle_sha256:
            raise WidthSliceRunnerError("CAS bundle slot collision with different content")
    else:
        bundle_tmp = bundle_slot.with_suffix(".blob.tmp")
        with bundle_path.open("rb") as src, bundle_tmp.open("wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256_file(bundle_tmp) != bundle_sha256:
            bundle_tmp.unlink(missing_ok=True)
            raise WidthSliceRunnerError("bundle bytes drifted during CAS staging")
        os.replace(bundle_tmp, bundle_slot)

    evidence_relpath = str(evidence_slot.relative_to(run_dir))
    bundle_relpath = str(bundle_slot.relative_to(run_dir))
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "method": _METHOD,
        "runtime_claim": "artifact_only_unvalidated",
        "run_id": run_id,
        "plan_id": plan["profile_id"],
        "recipe_sha256": plan["basis_artifact_sha256"],
        "profile_id": plan["profile_id"],
        "profile_sha256": plan["profile_sha256"],
        "recommendation_id": plan["profile_id"],
        "outputs": {
            "run_id": run_id,
            "outputs": [
                {
                    "stage": _METHOD,
                    "name": _EVIDENCE_NAME,
                    "sha256": evidence_sha256,
                    "size_bytes": len(evidence_encoded),
                    "relpath": evidence_relpath,
                },
                {
                    "stage": _METHOD,
                    "name": _ARTIFACT_NAME,
                    "sha256": bundle_sha256,
                    "size_bytes": bundle_size,
                    "relpath": bundle_relpath,
                },
            ],
        },
        "runtime_artifact": {
            "path": str(bundle_slot),
            "sha256": bundle_sha256,
            "size_bytes": bundle_size,
            "stage": _METHOD,
            "logical_name": _ARTIFACT_NAME,
            "relpath": bundle_relpath,
            "runtime_validated": False,
            "evidence": {
                "stage": _METHOD,
                "logical_name": _EVIDENCE_NAME,
                "sha256": evidence_sha256,
                "size_bytes": len(evidence_encoded),
                "relpath": evidence_relpath,
            },
        },
    }
    result_path = run_dir / "result.json"
    _exclusive_json(result_path, result)
    # end-to-end self-check: the published handoff must byte-verify NOW
    load_verified_width_slice_handoff(result_path)
    return result_path


def _run_gate(
    metric_report_path: Path,
    budgets: KLGateBudget,
) -> tuple[dict[str, object], bool]:
    """Run the EXPLICIT quality gate step over a real metric report.

    Returns (decision dict, accepted). The report is re-validated from bytes
    (content-digest report_id) exactly as scripts/run_glm52_quality_gate.py;
    any rejection yields the rejected decision — never an invented pass.
    """
    raw = _read_bounded(metric_report_path)
    report = CaptureMetricReport.model_validate_json(raw.decode("utf-8"))
    try:
        result = run_kl_quality_gate(report=report, budgets=budgets)
    except QualityGateRejection as rejection:
        gate = rejection.result
        return {
            "status": "rejected",
            "quality_claim": False,
            "metric_report_id": report.report_id,
            "metric_report_sha256": hashlib.sha256(raw).hexdigest(),
            "budgets": {
                "mean_kld": budgets.mean_kld,
                "worst_domain_kld": budgets.worst_domain_kld,
                "p99_kld": budgets.p99_kld,
                "cka_floor": budgets.cka_floor,
            },
            "gate": {
                "mean_kld": gate.mean_kld,
                "worst_domain_kld": gate.worst_domain_kld,
                "p99_kld": gate.p99_kld,
                "min_cka": gate.min_cka,
                "accepted": False,
                "failures": list(gate.failures),
            },
        }, False
    except KLGateError as exc:
        raise WidthSliceRunnerError(f"quality gate refused to run: {exc}") from exc
    return {
        "status": "accepted",
        "quality_claim": True,
        "metric_report_id": report.report_id,
        "metric_report_sha256": hashlib.sha256(raw).hexdigest(),
        "budgets": {
            "mean_kld": budgets.mean_kld,
            "worst_domain_kld": budgets.worst_domain_kld,
            "p99_kld": budgets.p99_kld,
            "cka_floor": budgets.cka_floor,
        },
        "gate": {
            "mean_kld": result.mean_kld,
            "worst_domain_kld": result.worst_domain_kld,
            "p99_kld": result.p99_kld,
            "min_cka": result.min_cka,
            "accepted": True,
            "failures": [],
        },
    }, True


def _execute(args: argparse.Namespace) -> int:
    plan = _plan(args)
    output = Path(str(plan["output"]))
    bundle_path = Path(str(plan["bundle_path"]))
    run_dir = Path(str(plan["bundle_path"])).parent / "runs" / f"run-{plan['profile_id']}"
    if output.exists() or bundle_path.exists() or args.result.exists() or run_dir.exists():
        raise WidthSliceRunnerError(
            "output/bundle/result/run path already exists; use a new attempt path"
        )
    profile, _profile_sha = _load_profile(args.profile)
    saliency = _saliency_evidence(profile)
    basis = parse_saliency_basis(str(saliency["detail"]))
    from model_atlas.loader import _infer_geometry

    geometry = load_manifest(str(plan["source"]))
    full, n_exp, sparse_layers = _infer_geometry(
        geometry, json.loads((Path(str(plan["source"])) / "config.json").read_text())
    )
    keep_map = build_keep_map(
        basis,
        width=int(plan["width"]),
        full=full,
        sparse_layers=sparse_layers,
        n_exp=n_exp,
    )

    # ---- explicit gate step FIRST on rejection: publish nothing ------------
    gate_decision: dict[str, object] | None = None
    if args.metric_report is not None:
        budgets = KLGateBudget(
            mean_kld=args.mean_budget,
            worst_domain_kld=args.worst_domain_budget,
            p99_kld=args.p99_kld_budget,
            cka_floor=args.cka_floor,
        )
        gate_decision, accepted = _run_gate(args.metric_report, budgets)
        if not accepted:
            _exclusive_json(args.result, {**plan, "quality_gate": {**gate_decision}})
            print(
                "quality gate REJECTED: " + "; ".join(gate_decision["gate"]["failures"]),  # type: ignore[union-attr]
                file=sys.stderr,
            )
            return 2

    export = materialize_uniform_width(
        str(plan["source"]),
        str(output),
        int(plan["width"]),
        keep_channels=keep_map,
    )
    if not export.promoted or not export.structurally_complete:
        raise WidthSliceRunnerError("exporter did not produce a promoted structural derivative")
    if export.runtime_validated:
        raise WidthSliceRunnerError("exporter must not claim runtime validation")
    validation = _validate_structurally(output)

    bundle_sha256, bundle_size = pack_derivative_bundle(output, bundle_path)
    run_id = f"run-{plan['profile_id']}"
    run_dir = bundle_path.parent / "runs" / run_id
    if run_dir.exists():
        raise WidthSliceRunnerError(f"run directory already exists: {run_dir}")
    result_path = _write_handoff_result(
        run_dir=run_dir,
        run_id=run_id,
        plan=plan,
        bundle_path=bundle_path,
        bundle_sha256=bundle_sha256,
        bundle_size=bundle_size,
        validation=validation,
        keep_map=keep_map,
        gate_decision=gate_decision,
    )
    public = {
        **plan,
        "structural_validation": validation,
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size,
        "run_id": run_id,
        "result_path": str(result_path),
        "quality_gate": (
            {**gate_decision, "state": "accepted"} if gate_decision else plan["quality_gate"]
        ),
    }
    # the reviewed handoff result is ALSO published at the operator's --result
    # path (exclusively); the run-dir copy stays the CAS-canonical one.
    result_public = json.loads(result_path.read_text())
    result_public["result_relpath"] = str(result_path.relative_to(run_dir))
    _exclusive_json(args.result, result_public)
    print(json.dumps(public, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="perform the real slice")
    parser.add_argument("--width", type=int, help="explicit aligned retention width")
    parser.add_argument("--memory-target-gib", type=float, help="plan width from this budget")
    parser.add_argument(
        "--metric-report", type=Path, help="explicit gate step: CaptureMetricReport JSON"
    )
    parser.add_argument("--mean-budget", type=float)
    parser.add_argument("--worst-domain-budget", type=float)
    parser.add_argument("--p99-budget", type=float, dest="p99_kld_budget")
    parser.add_argument("--cka-floor", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate_requested = args.metric_report is not None
    if gate_requested != (
        args.mean_budget is not None
        and args.worst_domain_budget is not None
        and args.p99_kld_budget is not None
        and args.cka_floor is not None
    ):
        raise WidthSliceRunnerError(
            "--metric-report requires all four budgets; budgets require --metric-report"
        )
    args.profile = args.profile.resolve(strict=True)
    args.result = args.result.resolve()
    if args.metric_report is not None:
        args.metric_report = args.metric_report.resolve(strict=True)
    if not args.execute:
        print(json.dumps(_plan(args), sort_keys=True))
        return 0
    return _execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WidthSliceRunnerError as error:
        print(f"width-slice runner failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
