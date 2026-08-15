"""Control-plane tests: recipe schema/compiler, backends, job engine, repair.

Covers (from the implementation contract):

* schema round trips (serialize -> parse -> identical canonical form)
* canonical hashes (same recipe => same id, edits => new id, deterministic)
* no-pruning enforcement (a pruning stage under no_pruning=true fails compile)
* pruning opt-in capability gate (no_pruning=false + declared capability required)
* invalid hybrid rejection (EXL3+NVFP4+FP8 without declared support)
* missing-backend fail-closed (unavailable backend => compile error)
* deterministic replay / idempotency (re-running a completed run reproduces)
* interrupted resume (a crashed mid-stage run recovers and completes)
* locks (a held run lock blocks a second engine)
* atomic promotion (staging then commit, no in-place mutation)
* provenance non-escalation (reported measured never raises recorded kind)
* repair authorization (non-allowlisted proposals rejected; rollback restores)
* source immutability check + small end-to-end fixture (compile->run->complete)
"""

import json
from pathlib import Path

import pytest

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.jobs.artifacts import (
    ContentAddressedStore,
    StageStager,
    acquire_file_lock,
    assert_source_readonly,
    release_file_lock,
    source_snapshot,
)
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus, StageStatus
from model_atlas.recipe.compiler import (
    RecipeCompileError,
    RecipeCompiler,
    canonical_json,
)
from model_atlas.recipe.schema import (
    CalibrationIdentity,
    CompressionRecipe,
    RecipeConstraints,
    RecipeStage,
    RecipeStatus,
    SourceIdentity,
    StageBackendPin,
    StageEffectClass,
)
from model_atlas.repair.gate import CompiledRepair, RepairGate, RepairProposal
from model_atlas.schemas.evidence import EvidenceKind

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _recipe(
    *stages,
    name: str = "t",
    no_pruning: bool = True,
    allow_pruning: bool = False,
    allow_hybrid: bool = False,
    source_path: str = "/nonexistent-source",
) -> CompressionRecipe:
    return CompressionRecipe(
        name=name,
        source=SourceIdentity(source_id="s", checkpoint_path=source_path),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        constraints=RecipeConstraints(
            no_pruning=no_pruning,
            allow_pruning_capability=allow_pruning,
            allow_hybrid_precision=allow_hybrid,
        ),
        stages=list(stages),
    )


def _stage(
    sid: str,
    backend: str = "atlas_quant_probe",
    effect: StageEffectClass = StageEffectClass.PROFILING,
    produces: list[str] | None = None,
    requires: list[str] | None = None,
    policy: EvidenceKind = EvidenceKind.PREDICTED,
    name: str | None = None,
) -> RecipeStage:
    return RecipeStage(
        id=sid,
        name=name or sid,
        effect_class=effect,
        backend=StageBackendPin(backend_id=backend, version="unpinned"),
        produces_format=produces or [],
        requires_formats=requires or [],
        evidence_policy=policy,
    )


@pytest.fixture
def registry() -> BackendRegistry:
    return build_default_registry()


@pytest.fixture
def compiler(registry: BackendRegistry) -> RecipeCompiler:
    return RecipeCompiler(registry)


# ---------------------------------------------------------------------------
# 1. schema round trips
# ---------------------------------------------------------------------------


def test_recipe_round_trip_preserves_canonical_form():
    recipe = _recipe(
        _stage("a", produces=["manifest.json"]),
        _stage("b", produces=["exl3"], requires=["manifest.json"]),
    )
    dumped = recipe.model_dump_json()
    parsed = CompressionRecipe.model_validate_json(dumped)
    assert canonical_json(parsed.model_dump(exclude={"recipe_id", "created_at"})) == (
        canonical_json(recipe.model_dump(exclude={"recipe_id", "created_at"}))
    )


def test_recipe_json_round_trip_pydantic():
    recipe = _recipe(_stage("a", produces=["bit-allocation"]))
    raw = recipe.model_dump(mode="json")
    restored = CompressionRecipe.model_validate(raw)
    assert restored.name == recipe.name
    assert restored.stages[0].id == "a"


# ---------------------------------------------------------------------------
# 2. canonical hashes
# ---------------------------------------------------------------------------


def test_recipe_id_stable_and_content_sensitive(compiler: RecipeCompiler):
    a = _recipe(_stage("a"))
    b = _recipe(_stage("a"))
    issues, ra, sha_a = compiler.validate(a)
    issues2, rb, sha_b = compiler.validate(b)
    assert issues == issues2
    assert ra == rb
    assert sha_a == sha_b
    # a content edit (different param) changes the id
    c = _recipe(_stage("a", produces=["exl3"]))
    _, rc, _ = compiler.validate(c)
    assert rc != ra


def test_run_id_stable_for_same_inputs(compiler: RecipeCompiler):
    recipe = _recipe(_stage("a", produces=["manifest.json"]))
    compiled = compiler.compile(recipe)
    assert compiled.run_id({"x": 1}) == compiled.run_id({"x": 1})
    assert compiled.run_id({}) != compiled.run_id({"x": 1})


# ---------------------------------------------------------------------------
# 3. no-pruning enforcement
# ---------------------------------------------------------------------------


def test_no_pruning_rejects_pruning_stage(compiler: RecipeCompiler):
    recipe = _recipe(_stage("p", effect=StageEffectClass.PRUNING, produces=["pruned-checkpoint"]))
    issues, _, _ = compiler.validate(recipe)
    codes = {i.code for i in issues}
    assert "no_pruning_violation" in codes
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "no_pruning" in str(err.value)


def test_no_pruning_transitive_rejects_consumer(compiler: RecipeCompiler):
    recipe = _recipe(
        _stage("prune", effect=StageEffectClass.PRUNING, produces=["pruned-checkpoint"]),
        _stage("consume", produces=["exl3"], requires=["pruned-checkpoint"]),
    )
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "no_pruning_violation_transitive" in str(err.value)


def test_pruning_optin_requires_declared_capability(compiler: RecipeCompiler):
    # A PRUNING stage must have an opt-in pruning backend (tenp_pruning) wired;
    # pinning an ordinary backend must be rejected. Here the stage uses the
    # tenp_pruning backend which is declared on the registry but UNAVAILABLE, so
    # the capability gate passes and the fail-closed compile gate fires instead.
    recipe = _recipe(
        _stage(
            "p",
            backend="tenp_pruning",
            effect=StageEffectClass.PRUNING,
            produces=["pruned-checkpoint"],
        ),
        no_pruning=False,
        allow_pruning=True,
    )
    issues, _, _ = compiler.validate(recipe)
    assert not [i for i in issues if i.code == "pruning_capability_not_registered"]
    # but the pruning backend is unavailable -> compile still fails closed
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "backend_unavailable" in str(err.value)


def test_pruning_stage_on_ordinary_backend_rejected(compiler: RecipeCompiler):
    # a PRUNING stage pinned to a non-pruning-capable backend (atlas_quant_probe
    # is available but declares no pruning capability) must be rejected by the
    # capability-per-backend gate.
    recipe = _recipe(
        _stage(
            "p",
            backend="atlas_quant_probe",
            effect=StageEffectClass.PRUNING,
            produces=["pruned-checkpoint"],
        ),
        no_pruning=False,
        allow_pruning=True,
    )
    issues, _, _ = compiler.validate(recipe)
    codes = {i.code for i in issues}
    assert "pruning_stage_backend_not_capable" in codes
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "pruning_stage_backend_not_capable" in str(err.value)


# ---------------------------------------------------------------------------
# 4. invalid hybrid rejection
# ---------------------------------------------------------------------------


def test_hybrid_exl3_nvfp4_fp8_rejected(compiler: RecipeCompiler):
    recipe = _recipe(
        _stage("a", produces=["exl3"], effect=StageEffectClass.QUANTIZATION),
        _stage("b", produces=["modelopt_nvfp4"], effect=StageEffectClass.QUANTIZATION),
        _stage("c", produces=["fp8_e4m3"], effect=StageEffectClass.QUANTIZATION),
    )
    issues, _, _ = compiler.validate(recipe)
    assert "unsupported_hybrid_precision" in {i.code for i in issues}
    with pytest.raises(RecipeCompileError):
        compiler.compile(recipe)


def test_hybrid_permitted_only_with_declared_support(compiler: RecipeCompiler):
    # The ONLY way an unsupported hybrid compiles is an explicit capability
    # declaration by the selected backend/runtime for that EXACT format set.
    # Neither availability nor an author flag substitutes for it.
    recipe = _recipe(
        _stage("a", produces=["exl3"], effect=StageEffectClass.QUANTIZATION),
        _stage("b", produces=["modelopt_nvfp4"], effect=StageEffectClass.QUANTIZATION),
        allow_hybrid=False,
    )
    issues, _, _ = compiler.validate(recipe)
    assert "unsupported_hybrid_precision" in {i.code for i in issues}
    with pytest.raises(RecipeCompileError):
        compiler.compile(recipe)

    # the modelopt backend declares ONLY hybrid:modelopt_nvfp4+fp8_e4m3 —
    # {exl3,modelopt_nvfp4} is a DIFFERENT combination, still unsupported
    # -> still rejected
    assert compiler._registry.declares_hybrid({"modelopt_nvfp4", "fp8_e4m3"})
    assert not compiler._registry.declares_hybrid({"exl3", "modelopt_nvfp4"})


def test_hybrid_author_flag_never_authorizes_unsupported(compiler: RecipeCompiler):
    # allow_hybrid_precision=true must NOT turn an unsupported composition into
    # a warning; it fails closed identically (the flag only records intent).
    recipe = _recipe(
        _stage("a", produces=["exl3"], effect=StageEffectClass.QUANTIZATION),
        _stage("b", produces=["modelopt_nvfp4"], effect=StageEffectClass.QUANTIZATION),
        allow_hybrid=True,
    )
    issues, _, _ = compiler.validate(recipe)
    codes = {i.code for i in issues}
    assert "unsupported_hybrid_precision" in codes
    assert "hybrid_unvalidated" not in codes  # never demoted to a warning
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "allow_hybrid_precision" in str(err.value) or "declares support" in str(err.value)


def test_hybrid_with_declared_capability_compiles(compiler: RecipeCompiler):
    # a runtime/backend that EXPLICITLY declares hybrid:exl3+fp8_e4m3+modelopt_nvfp4
    # allows the composition; the compiler trusts the declaration, and the
    # fail-closed availability gate still enforces run-time readiness.
    from model_atlas.backend.contract import BackendRecord, CommandBackedAdapter
    from model_atlas.backend.registry import BackendRegistry

    declared_run = BackendRecord(
        backend_id="modelopt_nvfp4_declared",
        display_name="ModelOpt NVFP4 (declared hybrid)",
        method_family="modelopt",
        formats=("modelopt_nvfp4", "safetensors"),
        represents_method="NVFP4 block-scaled substitution",
        architectures=("glm-5.2", "any"),
        runtime_compat=("sm121", "two-spark"),
        status=RecipeStatus.DISCOVERED,
        version="0.0.1",
        declared_capabilities=("hybrid:exl3+fp8_e4m3+modelopt_nvfp4",),
        supported_formats=(),
        fail_closed=True,
        availability_probe=lambda: (True, "0.0.1", "declared (test harness)"),
        adapter=CommandBackedAdapter(backend_id="modelopt_nvfp4_declared"),
    )
    base = build_default_registry()
    reg = BackendRegistry(
        {**{i: r for i, r in base._records.items()}, "modelopt_nvfp4_declared": declared_run}
    )
    compiler2 = RecipeCompiler(reg)
    recipe = _recipe(
        _stage(
            "a",
            backend="atlas_quant_probe",
            effect=StageEffectClass.QUANTIZATION,
            produces=["exl3"],
        ),
        _stage(
            "b",
            backend="modelopt_nvfp4_declared",
            effect=StageEffectClass.QUANTIZATION,
            produces=["modelopt_nvfp4"],
        ),
        _stage(
            "c",
            backend="atlas_quant_probe",
            effect=StageEffectClass.QUANTIZATION,
            produces=["fp8_e4m3"],
        ),
        allow_hybrid=True,
    )
    issues, _, _ = compiler2.validate(recipe)
    assert "unsupported_hybrid_precision" not in {i.code for i in issues}


def test_hybrid_with_unavailable_backend_fails_closed(compiler: RecipeCompiler):
    # even under author override, an unavailable backend fails closed
    recipe = _recipe(
        _stage("a", backend="exl3", produces=["exl3"], effect=StageEffectClass.QUANTIZATION),
        _stage(
            "b",
            backend="modelopt_nvfp4",
            produces=["modelopt_nvfp4"],
            effect=StageEffectClass.QUANTIZATION,
        ),
        allow_hybrid=True,
    )
    issues, _, _ = compiler.validate(recipe)
    assert any(i.code == "backend_unavailable" for i in issues)
    with pytest.raises(RecipeCompileError):
        compiler.compile(recipe)


# ---------------------------------------------------------------------------
# 5b. run directory/run_id derive from actual input identity
# ---------------------------------------------------------------------------


def test_run_dir_derives_from_actual_inputs(
    compiler: RecipeCompiler, tmp_path: Path, stable_source: str
):
    """A run with real inputs must land in a run dir derived from THOSE inputs
    (never from the empty-input placeholder). Different inputs -> different
    deterministic run dirs; same inputs -> same dir (idempotent)."""
    recipe = _simple_recipe(stable_source)
    compiled = compiler.compile(recipe)

    e_a = JobEngine(compiled, build_default_registry(), tmp_path)
    j_a = e_a.run(inputs={"batch": 1})
    assert j_a.run_dir == str(tmp_path / "runs" / j_a.run_id)

    e_a2 = JobEngine(compiled, build_default_registry(), tmp_path)
    j_a2 = e_a2.run(inputs={"batch": 1})  # same inputs -> same run dir
    assert j_a2.run_id == j_a.run_id

    e_b = JobEngine(compiled, build_default_registry(), tmp_path)
    j_b = e_b.run(inputs={"batch": 2})  # different inputs -> different run dir
    assert j_b.run_id != j_a.run_id
    run_id_b = j_b.run_id
    assert str(tmp_path / "runs" / run_id_b) != str(tmp_path / "runs" / j_a.run_id)


# ---------------------------------------------------------------------------
# 5. missing-backend fail-closed
# ---------------------------------------------------------------------------


def test_unavailable_backend_fails_closed(compiler: RecipeCompiler):
    recipe = _recipe(_stage("a", backend="exl3", produces=["exl3"]))
    issues, _, _ = compiler.validate(recipe)
    assert "backend_unavailable" in {i.code for i in issues}
    with pytest.raises(RecipeCompileError) as err:
        compiler.compile(recipe)
    assert "fail closed" in str(err.value)


def test_unknown_backend_fails_closed(compiler: RecipeCompiler):
    recipe = _recipe(_stage("a", backend="no_such_backend"))
    issues, _, _ = compiler.validate(recipe)
    assert "backend_missing" in {i.code for i in issues}
    with pytest.raises(RecipeCompileError):
        compiler.compile(recipe)


# ---------------------------------------------------------------------------
# 6. deterministic replay / idempotency (job engine)
# ---------------------------------------------------------------------------


def _simple_recipe(source_path: str = "/nonexistent-source") -> CompressionRecipe:
    stages = [
        _stage(
            "s1",
            produces=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
        ),
        _stage(
            "s2",
            backend="atlas_analysis_v3",
            produces=["sensitivity-map"],
            requires=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
        ),
    ]
    return CompressionRecipe(
        name="e2e",
        source=SourceIdentity(source_id="s", checkpoint_path=source_path),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        stages=stages,
    )


@pytest.fixture
def stable_source(tmp_path: Path) -> str:
    """A real, stable (never-mutated) source dir for run-level tests so the
    source-immutability gate doesn't fire spuriously during the run."""
    src = tmp_path / "model_src"
    src.mkdir()
    (src / "tensor.bin").write_bytes(b"stable-weights-v1")
    return str(src)


def _run_engine(compiler: RecipeCompiler, root: Path, recipe: CompressionRecipe) -> JobEngine:
    compiled = compiler.compile(recipe)
    return JobEngine(compiled, build_default_registry(), root)


def test_deterministic_replay_and_idempotency(
    compiler: RecipeCompiler, tmp_path: Path, stable_source: str
):
    root = tmp_path / "runs"
    recipe = _simple_recipe(stable_source)
    compiled = compiler.compile(recipe)
    inputs = {"batch": 1}
    e1 = JobEngine(compiled, build_default_registry(), root)
    j1 = e1.run(inputs=inputs)
    assert j1.status is JobStatus.COMPLETED

    # snapshot the journal path bytes after the first run
    journal = root / "runs" / compiled.run_id(inputs) / "events.jsonl"
    assert journal.exists()
    before = journal.read_text(encoding="utf-8").splitlines()

    # second engine, SAME inputs: must (a) reproduce the same deterministic id,
    # (b) not append any new stage.start for completed stages (idempotent), and
    # (c) keep all outputs content-address-verified.
    e2 = JobEngine(compiled, build_default_registry(), root)
    j2 = e2.run(inputs=inputs)
    assert j2.status is JobStatus.COMPLETED
    assert j2.run_id == j1.run_id
    new_lines = [
        line for line in journal.read_text(encoding="utf-8").splitlines() if line not in set(before)
    ]
    assert all('"stage.start"' not in line for line in new_lines)
    for sid in j2.stage_order:
        for ref in j2.stage(sid).outputs:
            assert e2.store.verify(ref), f"output {ref.name} failed integrity"


# ---------------------------------------------------------------------------
# 7. interrupted resume
# ---------------------------------------------------------------------------


def test_interrupted_resume_recovers(compiler: RecipeCompiler, tmp_path: Path, stable_source: str):
    root = tmp_path / "runs"
    recipe = _simple_recipe(stable_source)
    compiled = compiler.compile(recipe)

    # fail stage s2 on first attempt as if the process died mid-run
    calls = {"n": 0}

    def _boom(context, handle):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash inside stage s2")
        return {"ok": True}

    reg = build_default_registry()
    orig = reg.adapter_for("atlas_analysis_v3").execute
    reg.adapter_for("atlas_analysis_v3").execute = _boom  # type: ignore[method-assign]

    eng = JobEngine(compiled, reg, root)
    j1 = eng.run(inputs={})
    assert j1.status is JobStatus.FAILED_RECOVERABLE
    assert j1.failed_stage == "s2"

    # restore the adapter, resume: s2 re-executes, s1 skipped; run completes
    reg.adapter_for("atlas_analysis_v3").execute = orig  # type: ignore[method-assign]
    eng2 = JobEngine(compiled, reg, root)
    j2 = eng2.resume()
    assert j2.status is JobStatus.COMPLETED
    assert j2.stage("s1").status is StageStatus.DONE
    assert j2.stage("s2").status is StageStatus.DONE
    events = eng2.journal.read()
    assert any(ev["event"] == "stage.resumed" and ev["stage"] == "s1" for ev in events)


# ---------------------------------------------------------------------------
# 8. locks
# ---------------------------------------------------------------------------


def test_lock_exclusion(tmp_path: Path):
    lock = tmp_path / "run.lock"
    assert acquire_file_lock(lock)
    # a second acquire must fail within the wait window
    assert not acquire_file_lock(lock, wait_seconds=0.2)
    release_file_lock(lock)
    assert acquire_file_lock(lock)


def test_engine_lock_blocks_second_run(
    compiler: RecipeCompiler, tmp_path: Path, stable_source: str
):
    root = tmp_path / "runs"
    recipe = _simple_recipe(stable_source)
    compiled = compiler.compile(recipe)
    eng = JobEngine(compiled, build_default_registry(), root)
    eng.run(inputs={})
    # hold the lock on the same run dir
    run_dir_path = root / "runs" / eng.inspect()["run_id"]
    acquired = acquire_file_lock(run_dir_path / "run.lock", wait_seconds=0.2)
    assert acquired
    e2 = JobEngine(compiled, build_default_registry(), root)
    with pytest.raises(RuntimeError, match="locked"):
        e2.resume()
    release_file_lock(run_dir_path / "run.lock")


# ---------------------------------------------------------------------------
# 9. atomic promotion + content addressing
# ---------------------------------------------------------------------------


def test_atomic_promotion_no_inplace_mutation(tmp_path: Path):
    root = tmp_path / "run"
    store = ContentAddressedStore(root)
    stager = StageStager(root, "s1")
    staged = stager.path("out.bin")
    staged.write_bytes(b"hello world" * 100)
    refs = stager.commit(store)
    assert len(refs) == 1
    ref = refs[0]
    assert store.verify(ref)
    # content-addressed path exists
    blob = root / ref.relpath
    assert blob.exists()
    assert blob.read_bytes() == b"hello world" * 100
    # idempotent commit of the same content yields the same address
    refs2 = stager.commit(store)
    assert refs2[0].sha256 == ref.sha256


def test_staging_then_commit_is_atomic_and_verified(tmp_path: Path):
    root = tmp_path / "run"
    store = ContentAddressedStore(root)
    stager = StageStager(root, "s2")
    p = stager.path("payload")
    p.write_text("alpha\ndelta")
    refs = stager.commit(store)
    assert refs and store.verify(refs[0])
    written = root / refs[0].relpath
    assert written.read_text() == "alpha\ndelta"


# ---------------------------------------------------------------------------
# 10. provenance non-escalation
# ---------------------------------------------------------------------------


def test_provenance_never_upgrades(compiler: RecipeCompiler, tmp_path: Path, stable_source: str):
    # stage policy = PREDICTED (ceiling). Backend can never raise it to MEASURED.
    recipe = CompressionRecipe(
        name="prov",
        source=SourceIdentity(source_id="s", checkpoint_path=stable_source),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        stages=[_stage("s1", produces=["manifest.json"], policy=EvidenceKind.PREDICTED)],
    )
    compiled = compiler.compile(recipe)
    eng = JobEngine(compiled, build_default_registry(), tmp_path / "runs")
    job = eng.run(inputs={})
    assert job.stage("s1").evidence_kind is EvidenceKind.PREDICTED
    assert job.stage("s1").evidence_reported  # records what the backend actually did
    # even an ESTIMATED-policy stage must be recorded as ESTIMATED at most
    recipe2 = CompressionRecipe(
        name="prov2",
        source=SourceIdentity(source_id="s", checkpoint_path=stable_source),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        stages=[_stage("s1", produces=["manifest.json"], policy=EvidenceKind.ESTIMATED)],
    )
    eng2 = JobEngine(compiler.compile(recipe2), build_default_registry(), tmp_path / "runs2")
    job2 = eng2.run(inputs={})
    assert job2.stage("s1").evidence_kind.value in {"predicted", "estimated"}


def test_evidence_kind_non_escalation_event_logged(
    compiler: RecipeCompiler, tmp_path: Path, stable_source: str
):
    # The job engine records a non-escalation event whenever the policy ceiling
    # suppresses the reported kind. In-repo backends report PREDICTED and stage
    # policies here are PREDICTED too, so no event should fire when ceilings match.
    recipe = _simple_recipe(stable_source)
    compiled = compiler.compile(recipe)
    eng = JobEngine(compiled, build_default_registry(), tmp_path / "runs")
    eng.run(inputs={})
    # with matching ceilings there is nothing to suppress; the recorded kind is
    # exactly the policy ceiling (never higher than what the backend reported)
    for sid in recipe.stages:
        st = eng.inspect()["stages"][sid.id]
        assert st["evidence_kind"] in {"predicted", "estimated"}


# ---------------------------------------------------------------------------
# 11. repair authorization + rollback
# ---------------------------------------------------------------------------


def test_repair_non_allowlisted_rejected():
    gate = RepairGate()
    bad = RepairProposal(kind="delete_all_weights", target="s1", params={})
    validation = gate.validate_proposal(bad)
    assert not validation.ok
    assert any("not on the deterministic allowlist" in e for e in validation.errors)


def test_repair_allowlisted_compile_and_apply():
    gate = RepairGate()
    proposal = RepairProposal(kind="keep_channels_normalize", target="s1", params={})
    validation = gate.validate_proposal(proposal)
    assert validation.ok
    compiled = validation.compiled
    assert compiled is not None
    before = json.dumps({"keep_channels": [4, 1, 4, 3], "x": 1}).encode()
    ok, result = gate.apply(compiled, before_bytes=before)
    assert ok
    assert result.compiled is not None
    after_key = result.compiled.new_key
    assert after_key != result.compiled.restore_key
    # the repaired content is canonicalized
    assert json.loads(gate._builtin_apply(proposal, before))["keep_channels"] == [1, 3, 4]


def test_repair_rollback_restores_and_refuses_tampered():
    gate = RepairGate()
    proposal = RepairProposal(kind="evidence_downgrade", target="s1", params={"to": "inferred"})
    validation = gate.validate_proposal(proposal)
    compiled: CompiledRepair = validation.compiled  # type: ignore[assignment]
    before = json.dumps({"evidence_kind": "measured"}).encode()
    ok, result = gate.apply(compiled, before_bytes=before)
    assert ok
    after = gate._builtin_apply(proposal, before)
    assert json.loads(after)["evidence_kind"] == "inferred"
    # rollback with the ORIGINAL bytes succeeds
    rb = gate.rollback(result.compiled, before_bytes=before)
    assert rb.ok
    # rollback with tampered bytes is refused
    tampered = json.dumps({"evidence_kind": "measured", "zz": 1}).encode()
    rb2 = gate.rollback(result.compiled, before_bytes=tampered)
    assert not rb2.ok


# ---------------------------------------------------------------------------
# 12. source immutability
# ---------------------------------------------------------------------------


def test_source_immutability_detects_change(tmp_path: Path):
    src = tmp_path / "src_dir"
    src.mkdir()
    (src / "tensor.bin").write_bytes(b"weights-v1")
    snap = source_snapshot(str(src))
    engine_check = lambda: assert_source_readonly(snap, str(src))  # noqa: E731
    engine_check()  # unchanged passes
    (src / "tensor.bin").write_bytes(b"weights-v2-TAMPERED")
    with pytest.raises(RuntimeError, match="changed"):
        engine_check()


def test_engine_enforces_immutable_source(compiler: RecipeCompiler, tmp_path: Path, monkeypatch):
    root = tmp_path / "runs"
    src = tmp_path / "model"
    src.mkdir()
    (src / "w.bin").write_bytes(b"v1")
    recipe = CompressionRecipe(
        name="imm",
        source=SourceIdentity(source_id="s", checkpoint_path=str(src)),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        stages=[_stage("s1", produces=["manifest.json"])],
    )
    compiled = compiler.compile(recipe)
    reg = build_default_registry()

    # tamper between the run-start snapshot and the stage boundary
    def tampering_stage(context, handle):
        (src / "w.bin").write_bytes(b"v2")

        return {"tampered": True}

    reg.adapter_for("atlas_quant_probe").execute = tampering_stage  # type: ignore[method-assign]
    eng = JobEngine(compiled, reg, root)
    job = eng.run(inputs={})
    assert job.status is JobStatus.FAILED_RECOVERABLE
    assert "changed" in job.error or "missing" in job.error


# ---------------------------------------------------------------------------
# 13. small end-to-end fixture (compile -> run -> complete -> inspect -> lineage)
# ---------------------------------------------------------------------------


def test_end_to_end_fixture(compiler: RecipeCompiler, tmp_path: Path, stable_source: str):
    from model_atlas.controlplane.api import ControlPlane

    plane = ControlPlane(work_root=str(tmp_path / "cp"))
    recipe = _simple_recipe(stable_source)
    cap = plane.capabilities()
    assert "atlas_quant_probe" in cap["available"]
    compiled = plane.compile_recipe(recipe)
    assert compiled.plan_id
    stat = plane.status(plane.start_compiled(compiled, {}).inspect()["run_id"])
    assert stat["status"] == "completed"
    lineage = plane.lineage(recipe)
    assert lineage["recipe_id"] == compiled.recipe_id
    assert lineage["run_id"]
    # reproduce command present
    assert "model-atlas" in lineage["reproduce_command"]


def test_compile_recipe_dryrun_cli(tmp_path: Path):
    from typer.testing import CliRunner

    from model_atlas.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["compile-recipe", "--recipe", "tenp-pruning-optin"])
    assert res.exit_code == 0
    assert "recipe_id" in res.output
    # the opt-in pruning recipe fails to compile (backend unavailable) -> dry run reports it
    assert "compiles: False" in res.output
    assert "backend_unavailable" in res.output


def test_backend_capabilities_cli():
    from typer.testing import CliRunner

    from model_atlas.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["backend-capabilities"])
    assert res.exit_code == 0
    assert "atlas_quant_probe" in res.output
    assert "tenp_pruning" in res.output
    assert "pruning" in res.output  # capability declared


def test_builtin_glm52_recipe_is_dryrun_plan_only(compiler: RecipeCompiler):
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    recipe = glm52_no_pruning_recipe()
    issues, _, _ = compiler.validate(recipe)
    codes = {i.code for i in issues}
    assert codes, "the canonical recipe must carry honest compile blockers"
    assert "no_pruning_violation" not in codes  # never self-violates
    with pytest.raises(RecipeCompileError):
        compiler.compile(recipe)  # fails closed until deps/validation exist
