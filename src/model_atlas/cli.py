"""model-atlas command-line interface."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import typer

from model_atlas import __version__
from model_atlas.analysis import (
    build_corpus_semantic_map,
)
from model_atlas.atlas.export import export_run
from model_atlas.atlas.reap import make_synthetic_corpus, run_calibration
from model_atlas.atlas.runtime import build_mini_moe
from model_atlas.atlas.v3_pipeline import run_v3_pipeline
from model_atlas.candidates import CandidateGraph, CandidateNode, CandidateStage
from model_atlas.census.census import build_manifest
from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.checkpoint.structural_graph import build_structural_graph
from model_atlas.dashboard import write_dashboard
from model_atlas.experiments.pareto_v3 import restrict_frontier
from model_atlas.planning.memory_planner import GIB, assess
from model_atlas.planning.realbytes import account_manifest, plan_candidates, report
from model_atlas.preflight import write_preflight
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.evidence import EvidenceKind

if False:
    from model_atlas.controlplane.api import ControlPlane  # noqa: F401 (annotation only)

app = typer.Typer(no_args_is_help=True, help="Model-agnostic Atlas platform CLI.")


# ---------------------------------------------------------------------------
# Atlas control plane (compression recipes / jobs) — see docs/control-plane.md
# ---------------------------------------------------------------------------


def _default_control_plane() -> ControlPlane:
    from model_atlas.controlplane.api import ControlPlane

    return ControlPlane()


@app.command("backend-capabilities")
def backend_capabilities(
    out: str = typer.Option("", "--out", help="write capabilities JSON here"),
) -> None:
    """List registered compression backends, capabilities, availability."""
    plane = _default_control_plane()
    cap = plane.capabilities()
    print(json.dumps(cap, indent=2, sort_keys=True))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(cap, indent=2, sort_keys=True))
        print(f"wrote: {out}")


@app.command("compile-recipe")
def compile_recipe(
    recipe: str = typer.Option("glm52-no-pruning", "--recipe", help="builtin recipe family name"),
    dry_run: bool = typer.Option(True, "--dry-run", help="dry-run compile (non-strict)"),
    out: str = typer.Option("", "--out", help="write compiled plan JSON here"),
) -> None:
    """Compile a canonical recipe (dry-run by default; fails closed on errors
    only with --no-dry-run which is the immutable compile path)."""
    from model_atlas.recipes.builtin import (
        glm52_no_pruning_recipe,
        tenp_pruning_optin_recipe,
    )

    recipes = {
        "glm52-no-pruning": glm52_no_pruning_recipe,
        "tenp-pruning-optin": tenp_pruning_optin_recipe,
    }
    builder = recipes.get(recipe)
    if builder is None:
        raise typer.BadParameter(f"unknown recipe {recipe!r}; known: {', '.join(sorted(recipes))}")
    r = builder()
    plane = _default_control_plane()
    if dry_run:
        report = plane.dry_run(r)
        print(f"recipe: {recipe}")
        print(f"  recipe_id: {report['recipe_id']}")
        print(f"  compiles: {report['compiles']}")
        issues = report["issues"]
        issue_list = issues if isinstance(issues, list) else []
        for issue in issue_list:
            if isinstance(issue, dict):
                print(f"  [{issue.get('severity')}] {issue.get('code')}: {issue.get('message')}")
    else:
        compiled = plane.compile_recipe(r)
        print(f"recipe: {recipe}")
        print(f"  compiled plan_id: {compiled.plan_id} (immutable)")
        print(f"  recipe_sha256: {compiled.recipe_sha256}")
        if out:
            from model_atlas.recipes import CompiledPlanArtifact

            artifact = CompiledPlanArtifact.from_compiled(
                compiled,
                inputs={},
                registry=plane.registry,
            )
            artifact.verify()
            artifact.verify_pins_against(plane.registry)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(artifact.to_plain_dict(), indent=2, sort_keys=True))
            print(f"wrote (verified compiled-plan artifact): {out}")
        else:
            print("  note: pass --out to write a versioned compiled-plan artifact")


@app.command("job")
def job(
    action: str = typer.Argument(..., help="start|status|resume|validate|cancel|lineage"),
    recipe: str = typer.Option("glm52-no-pruning", "--recipe", help="builtin recipe family name"),
    plan: str = typer.Option("", "--plan", help="path to a saved compiled plan JSON"),
    run_id: str = typer.Option("", "--run-id", help="run id for status/resume/validate/cancel"),
    stage: str = typer.Option("", "--stage", help="stage id for validate"),
    out: str = typer.Option("controlplane_runs", "--out", help="work root"),
    reason: str = typer.Option("operator cancel", "--reason"),
) -> None:
    """Control-plane job lifecycle: start / status / resume / validate / cancel /
    lineage. `start` compiles then executes the recipe (fails closed when a
    backend dependency is unavailable)."""
    from model_atlas.controlplane.api import ControlPlane
    from model_atlas.recipes.builtin import (
        glm52_no_pruning_recipe,
        tenp_pruning_optin_recipe,
    )

    plane = ControlPlane(work_root=out)
    if action == "start":
        if plan:
            # Functional --plan path: load + VERIFY the versioned compiled-plan
            # artifact (recipe, ids, pins, run_id from CANONICAL inputs), then
            # start the run with those inputs; fail closed on any fault.
            from model_atlas.recipes import CompiledPlanArtifact

            try:
                artifact = CompiledPlanArtifact.model_validate_json(
                    Path(plan).read_text(encoding="utf-8")
                )
                artifact.verify()
                # compare the artifact's recorded pins (backend id, exact
                # version, adapter identity, status, capability hash) against the
                # LIVE registry — never discard pin metadata + recompile-only
                artifact.verify_pins_against(plane.registry)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL-CLOSED: cannot load/verify plan {plan!r}: {exc}")
                raise typer.Exit(1) from exc
            try:
                engine = plane.start(
                    artifact.recipe, inputs=dict(artifact.inputs), verify_artifact=artifact
                )
            except Exception as exc:  # noqa: BLE001 — fail-closed is the CLI contract
                print(f"FAIL-CLOSED: could not start run from plan: {exc}")
                raise typer.Exit(1) from exc
            st = engine.inspect()
            print(f"run_id: {st['run_id']}")
            print(f"status: {st['status']} (started from verified plan {plan!r})")
            return
        recipes = {
            "glm52-no-pruning": glm52_no_pruning_recipe,
            "tenp-pruning-optin": tenp_pruning_optin_recipe,
        }
        builder = recipes.get(recipe)
        if builder is None:
            raise typer.BadParameter(
                f"unknown recipe {recipe!r}; known: {', '.join(sorted(recipes))}"
            )
        r = builder()
        try:
            engine = plane.start(r, inputs={})
        except Exception as exc:  # noqa: BLE001 — fail-closed is the CLI contract
            print(f"FAIL-CLOSED: could not start run: {exc}")
            raise typer.Exit(1) from exc
        st = engine.inspect()
        print(f"run_id: {st['run_id']}")
        print(f"status: {st['status']}")
    elif action == "lineage":
        r = glm52_no_pruning_recipe()
        print(json.dumps(plane.lineage(r), indent=2, sort_keys=True))
    else:
        if not run_id:
            raise typer.BadParameter("--run-id required for status/resume/validate/cancel")
        if action == "status":
            print(json.dumps(plane.status(run_id), indent=2, sort_keys=True))
        elif action == "resume":
            print(json.dumps(plane.resume(run_id), indent=2, sort_keys=True))
        elif action == "validate":
            if not stage:
                raise typer.BadParameter("--stage required for validate")
            print(json.dumps(plane.validate(run_id, stage), indent=2, sort_keys=True))
        elif action == "cancel":
            print(json.dumps(plane.cancel(run_id, reason), indent=2, sort_keys=True))
        else:
            raise typer.BadParameter(f"unknown job action {action!r}")


@app.command()
def doctor() -> None:
    """Check the runtime environment and package health."""
    ok = True
    try:
        import pydantic  # noqa: F401

        pydantic_ver = pydantic.VERSION
    except ImportError:
        pydantic_ver = "MISSING"
        ok = False
    try:
        import yaml  # noqa: F401
    except ImportError:
        yaml_ver = "MISSING"
        ok = False
    else:
        yaml_ver = yaml.__version__

    print(f"model-atlas {__version__}")
    print(f"python {platform.python_version()} ({platform.machine()})")
    print(f"pydantic {pydantic_ver}")
    print(f"pyyaml {yaml_ver}")
    print(f"architectures: {', '.join(get_registry().names())}")
    print("OK" if ok else "FAIL")
    if not ok:
        raise typer.Exit(1)


@app.command()
def export(
    eval_lab_root: str = typer.Option(
        ..., "--eval-lab-root", help="eval-lab repo root with tasks/ tree"
    ),
    out: str = typer.Option(
        "atlas_runs", "--out", help="output root that receives atlas_runs/<run_id>/"
    ),
    build: bool = typer.Option(False, "--build", help="also build + register a derivative"),
    arch: str = typer.Option("k3-mini", "--arch"),
    seed: int = typer.Option(0, "--seed"),
    keep_per_layer: int = typer.Option(4, "--keep-per-layer"),
) -> None:
    """Export an atlas run dir over an eval-lab task corpus (atlas-bridge)."""
    result = export_run(
        out,
        eval_lab_root=eval_lab_root,
        arch_name=arch,
        seed=seed,
        keep_per_layer=keep_per_layer,
        build=build,
    )
    print(f"wrote atlas run: {result['run_dir']}")
    print(f"  plans: {', '.join(result['plan_names'])}")


@app.command()
def dashboard(
    out: str = typer.Option("atlas_dashboard.html", "--out", help="output HTML path"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Render the Atlas Lab interactive dashboard from measured data."""
    path = write_dashboard(out, seed=seed)
    print(f"wrote interactive dashboard: {path}")


@app.command("list-architectures")
def list_architectures() -> None:
    """List registered architectures."""
    for name in get_registry().names():
        spec = get_registry().get(name)
        flag = " [needs source measurement]" if spec.needs_source_measurement else ""
        print(f"{name}{flag}")
        print(
            f"  layers={spec.num_text_layers} experts={spec.moe.num_routed_experts} "
            f"top_k={spec.moe.top_k} latent={spec.moe.latent_dim} hidden={spec.hidden_dim}"
        )


@app.command()
def inspect(
    checkpoint_dir: str,
    write_graph: bool = typer.Option(
        False,
        "--write-graph",
        help="Write checkpoint_manifest.json + structural_model_graph.json into the checkpoint dir",
    ),
) -> None:
    """Census a real checkpoint directory (headers only) into a structural graph."""
    manifest = load_manifest(checkpoint_dir)
    graph = build_structural_graph(manifest)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"  shards: {len(manifest.shards)}")
    print(f"  tensors: {manifest.tensor_count}")
    print(f"  total bytes: {manifest.total_bytes}")
    print(f"  coverage: {graph.coverage:.3f}")
    print(f"  valid: {graph.valid}")
    for u in graph.unclassified:
        print(f"  UNCLASSIFIED: {u}")
    if write_graph and graph.valid:
        root = Path(checkpoint_dir)
        (root / "checkpoint_manifest.json").write_text(manifest.model_dump_json(indent=2))
        (root / "structural_model_graph.json").write_text(graph.model_dump_json(indent=2))
        print("  wrote checkpoint_manifest.json + structural_model_graph.json")
    if not graph.valid:
        raise typer.Exit(1)


@app.command()
def calibrate(
    arch: str = typer.Option("k3-mini", "--arch"),
    seed: int = typer.Option(0, "--seed"),
    samples: int = typer.Option(16, "--samples"),
    seq_len: int = typer.Option(8, "--seq-len"),
    topk: int = typer.Option(4, "--topk"),
) -> None:
    """Run a streamed REAP calibration over a synthetic mini-MoE."""
    spec = get_registry().get(arch)
    if spec.needs_source_measurement:
        print(f"{arch}: no measured tensors; use a synthetic architecture like k3-mini")
        raise typer.Exit(1)
    model = build_mini_moe(spec, seed=seed)
    corpus, labels, _ = make_synthetic_corpus(
        n_samples=samples,
        seq_len=seq_len,
        vocab=spec.vocabulary_size or 1000,
        seed=seed,
    )
    acc = run_calibration(model, corpus, top_k=topk)
    print(
        f"calibrated {arch} over {samples} samples "
        f"(layers={spec.num_text_layers}, experts={spec.moe.num_routed_experts}, top_k={topk})"
    )
    for label in labels[:6]:
        ranked = acc.rank(label, topk=5)
        top = ", ".join(f"L{lay}E{exp}={score:.5f}" for lay, exp, score in ranked[:5])
        print(f"  {label.value}: {top}")


@app.command()
def census(arch: str) -> None:
    """Enumerate every tensor of an architecture into an ownership manifest."""
    spec = get_registry().get(arch)
    manifest = build_manifest(spec)
    if manifest.status == "needs_source_measurement":
        print(f"{arch}: no deterministic tensor sizes -> needs source checkpoint measurement")
        return
    by_node = manifest.bytes_by_node()
    print(f"{arch}: status={manifest.status} tensors={len(manifest.records)}")
    print(f"  total bytes={manifest.total_bytes():.0f}")
    for loc, b in sorted(by_node.items(), key=lambda kv: kv[0].value):
        if b > 0:
            print(f"  {loc.value}: {b:.0f} bytes")
    role_totals = manifest.bytes_by_role()
    for role, b in sorted(role_totals.items(), key=lambda kv: -kv[1]):
        print(f"  role {role.value}: {b:.0f}")


@app.command("real-candidates")
def real_candidates(
    checkpoint_dir: str,
    envelopes: str = typer.Option(
        "190,210,225", "--envelopes", help="Comma-separated GiB envelopes"
    ),
) -> None:
    """Plan real-bytes derivative candidates at the given resident envelopes."""
    manifest = load_manifest(checkpoint_dir)
    acc = account_manifest(manifest)
    envs = tuple(float(x) for x in envelopes.split(","))
    print(report(plan_candidates(acc, envelopes=envs), acc))


@app.command()
def plan(
    arch: str,
    budget_a_gb: float = typer.Option(..., "--node-a-gb", help="Node A resident budget (GiB)"),
    budget_b_gb: float = typer.Option(..., "--node-b-gb", help="Node B resident budget (GiB)"),
    reserve_gb: float = typer.Option(30.0, "--reserve-gb", help="Runtime reserve (GiB)"),
) -> None:
    """Byte-accurate memory plan with per-node go/no-go."""
    spec = get_registry().get(arch)
    manifest = build_manifest(spec)
    result = assess(
        spec,
        manifest,
        budget_a_gb=budget_a_gb,
        budget_b_gb=budget_b_gb,
        runtime_reserve_gb=reserve_gb,
    )
    print(f"{arch}:")
    print(f"  node A resident: {result.node_a_resident_gib():.2f} GiB")
    print(f"  node B resident: {result.node_b_resident_gib():.2f} GiB")
    print(f"  stored:          {result.stored_bytes / GIB:.2f} GiB")
    print(f"  active exp/tok:  {result.active_expert_bytes_per_token:.0f} bytes")
    print(f"  SAFE: {result.safe}")
    for f in result.failures:
        print(f"  - {f}")
    if not result.safe:
        raise typer.Exit(1)


@app.command()
def analyze(
    arch: str = typer.Option("k3-mini", "--arch"),
    seed: int = typer.Option(0, "--seed"),
    samples: int = typer.Option(12, "--samples"),
    seq_len: int = typer.Option(6, "--seq-len"),
) -> None:
    """Run fidelity-first v3 analyzers: spectral/shared/conditional/routing/
    bit-budget/NVFP4/KV/fallback over a synthetic model."""
    spec = get_registry().get(arch)
    if spec.needs_source_measurement:
        print(f"{arch}: requires measured tensor sizes; use synthetic arch like k3-mini")
        raise typer.Exit(1)
    model = build_mini_moe(spec, seed=seed)
    corpus = make_synthetic_corpus(
        n_samples=samples, seq_len=seq_len, vocab=spec.vocabulary_size or 1000, seed=seed
    )[0]
    run = run_v3_pipeline(model, corpus, seed=seed)
    print(f"v3 analyzers on {arch} (seed {seed}):")
    print(f"  stages: {', '.join(run.stages_run)}")
    print(
        "  routing-consistency identity gate: "
        f"{'PASS' if run.routing_consistency_passed else 'FAIL'}"
    )
    nv = run.nvfp4
    print(f"  nvfp4 candidates accepted: {getattr(nv, 'accepted_count', 0) if nv else 0}")
    pf = run.pareto
    print(f"  pareto frontier: {', '.join(getattr(pf, 'frontier_ids', []) if pf else [])}")


@app.command("v3-pareto")
def v3_pareto(
    arch: str = typer.Option("k3-mini", "--arch"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Compute the v3 Pareto frontier + knee region over an idealized candidate sweep."""
    from model_atlas.experiments.pareto_v3 import FrontierPoint

    pts = [
        FrontierPoint(
            candidate_id="A",
            values={
                "quality": 0.99,
                "resident_gib": 214.0,
                "decode_tps": 21.0,
                "context": 256000,
            },
            evidence_kind=EvidenceKind.PREDICTED,
        ),
        FrontierPoint(
            candidate_id="B",
            values={"quality": 0.995, "resident_gib": 196.0, "decode_tps": 26.0, "context": 384000},
            evidence_kind=EvidenceKind.PREDICTED,
        ),
        FrontierPoint(
            candidate_id="C",
            values={"quality": 0.96, "resident_gib": 150.0, "decode_tps": 33.0, "context": 720000},
            evidence_kind=EvidenceKind.PREDICTED,
        ),
        FrontierPoint(
            candidate_id="D",
            values={"quality": 0.97, "resident_gib": 160.0, "decode_tps": 30.0, "context": 700000},
            evidence_kind=EvidenceKind.PREDICTED,
        ),
    ]
    r = restrict_frontier(pts)
    print("v3 pareto (IDEALIZED demo points — tagged PREDICTED, not measured):")
    print(f"  frontier: {', '.join(r.frontier_ids)}")
    print(f"  knee region: {', '.join(r.knee_region)}")
    for cid, deltas in r.neighbor_deltas.items():
        for d in deltas:
            print(
                f"  {cid} -> {d.direction} {d.candidate_id}: "
                f"dQ {d.dquality} dGiB {d.dresident_gib} dQ/GiB {d.quality_per_gib}"
            )


@app.command("v3-candidates")
def v3_candidates() -> None:
    """Print the v3 candidate graph lineage + predicted/measured discipline."""
    g = CandidateGraph(source_teacher_id="teacher")
    g.add(
        CandidateNode(
            candidate_id="teacher",
            name="BF16 teacher",
            stage=CandidateStage.P0_REFERENCE,
            predicted=False,
            deployed=True,
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3",
            name="EXL3 global allocation",
            parent_ids=["teacher"],
            stage=CandidateStage.P4_EXL3,
            predicted=True,
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3-nvfp4",
            name="+NVFP4 substitution",
            parent_ids=["mk-exl3"],
            stage=CandidateStage.P6_SM121_ALLOCATION,
            predicted=True,
        )
    )
    print("v3 candidate graph:")
    for node in g.nodes.values():
        status = "deployable" if node.deployed else ("predicted" if node.predicted else "measured")
        print(f"  {node.candidate_id}: {node.name} [{status}] parents={node.parent_ids}")


@app.command("v3-corpus")
def v3_corpus(
    arch: str = typer.Option("k3-mini", "--arch"),
    seed: int = typer.Option(0, "--seed"),
    samples: int = typer.Option(12, "--samples"),
) -> None:
    """Build the corpus-semantic bidirectional map + coverage gate."""
    from model_atlas.schemas.coverage import EvidenceGate

    spec = get_registry().get(arch)
    model = build_mini_moe(spec, seed=seed)
    corpus = make_synthetic_corpus(
        n_samples=samples, seq_len=6, vocab=spec.vocabulary_size or 1000, seed=seed
    )[0]
    rep = build_corpus_semantic_map(model, corpus, top_k=2, gate=EvidenceGate())
    print("v3 corpus evidence:")
    print(f"  clusters: {', '.join(c.cluster_id for c in rep.clusters)}")
    print(f"  insufficient-evidence cluster-cells: {len(rep.insufficient_clusters)}")
    print(f"  per-cluster rows: {len(rep.cluster_expert_coverage)}")


@app.command()
def preflight(
    out: str = typer.Option("capability_report.json", "--out"),
    model_dir: str = typer.Option("", "--model-dir", help="extra mounted model dir to size"),
) -> None:
    """Write a machine-readable capability/preflight report (measured)."""
    model_paths = [model_dir] if model_dir else None
    path = write_preflight(out, model_paths)
    print(f"wrote preflight/capability report: {path}")


@app.command("validate-bodies")
def validate_bodies(
    checkpoint_dir: str,
    out: str = typer.Option("", "--out", help="write JSON scan report here"),
) -> None:
    """Bounded, read-only body validation over a real checkpoint (GLM-5.2 NVFP4).

    Reads only a few reference tensors + one NVFP4 expert's constituents; never
    materializes the source. GPU/mount untouched.
    """
    from model_atlas.checkpoint.realbody import validate_real_bodies

    scan = validate_real_bodies(checkpoint_dir)
    print(json.dumps(scan.as_dict(), indent=2, sort_keys=True))
    if out:
        Path(out).write_text(json.dumps(scan.as_dict(), indent=2, sort_keys=True))
        print(f"wrote: {out}")


@app.command("canary")
def canary(
    out: str = typer.Option("", "--out", help="write canary-status JSON here"),
) -> None:
    """Real-GLM canary status: metadata census + bounded body validation done,
    forward trace honestly reported blocked (no torch/NVFP4 decoder in venv)."""
    from model_atlas.canary import canary_status

    st = canary_status()
    print(json.dumps(st.to_dict(), indent=2, sort_keys=True))
    if out:
        Path(out).write_text(json.dumps(st.to_dict(), indent=2, sort_keys=True))
        print(f"wrote: {out}")


@app.command("backends")
def backends() -> None:
    """Probe real quantization backend support (EXL3 / ModelOpt NVFP4 / vllm)."""
    from model_atlas.quantbackends import all_backend_probes

    for b in all_backend_probes():
        print(
            f"{b.backend_id}: installed={b.installed} version={b.version} support={b.support.value}"
        )
        print(f"  {b.note}")
        if b.setup:
            for s in b.setup:
                print(f"    setup: {s}")


@app.command("two-node")
def two_node(
    out: str = typer.Option("", "--out", help="write inventory JSON here"),
) -> None:
    """Probe the two DGX-Spark nodes (spark + gx10-ac63), non-evasive."""
    from model_atlas.twonode import run_inventory

    inv = run_inventory()
    print("reachable nodes:", ", ".join(inv.reachable()))
    for host, n in inv.nodes.items():
        print(
            f"  {host}: gpu={n.gpu_name} cap={n.compute_cap} "
            f"host_GiB={n.host_mem_total_gib:.1f} avail={n.host_mem_available_gib:.1f} "
            f"production_occupied_GiB={n.production_occupied_gib:.1f} "
            f"torch={n.torch} svc={n.active_gpu_services}"
        )
    if out:
        Path(out).write_text(json.dumps(inv.to_dict(), indent=2, sort_keys=True))
        print(f"wrote: {out}")


@app.command("runtime-contract")
def runtime_contract(
    context: int = typer.Option(8192, "--context", help="target context tokens"),
    kv_scheme: str = typer.Option("fp8", "--kv-scheme"),
) -> None:
    """Print the SM121/MTP/KV/runtime contract for the two-Spark experiment."""
    from model_atlas.runtimecontracts import build_runtime_contract

    rc = build_runtime_contract(context_tokens=context, kv_scheme=kv_scheme)
    print(json.dumps(rc.to_dict(), indent=2, sort_keys=True))


@app.command("recommend")
def recommend_cli(
    profile: str = typer.Option("", "--profile", help="profile model/id to recommend for"),
    profiles_dir: str = typer.Option("profiles", "--profiles-dir"),
    out: str = typer.Option("", "--out", help="write recommendation JSON here"),
    memory_target_gib: float = typer.Option(115.0, "--memory-target-gib"),
) -> None:
    """Profile -> Recommend: deterministic versioned recommendation (no_pruning
    default); writes machine-readable recommendation JSON."""
    from model_atlas.recommend import RecommendationService, RecTarget

    svc = RecommendationService(profile_root=profiles_dir)
    rec = svc.recommend(profile or "default", RecTarget(memory_target_gib=memory_target_gib))
    payload = rec.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote: {out}")


@app.command("recommend-gui")
def recommend_gui(
    out: str = typer.Option("atlas_recommend.html", "--out", help="output HTML path"),
    profiles_dir: str = typer.Option("profiles", "--profiles-dir"),
    work_root: str = typer.Option("controlplane_runs", "--work-root"),
) -> None:
    """Profile -> Recommend -> Review -> Compress -> Monitor-> Output local GUI
    (server-less single-file HTML). Run: `model-atlas recommend-gui --out
    atlas_recommend.html` then open the file in a browser."""
    from model_atlas.recommend import RecommendationService, write_gui

    svc = RecommendationService(profile_root=profiles_dir, work_root=work_root)
    path = write_gui(out, svc)
    print(f"wrote (open in your browser): {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
