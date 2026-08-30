"""model-atlas command-line interface."""

from __future__ import annotations

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
from model_atlas.kernels import (
    KernelManifestRequest,
    KernelPhase,
    KernelQuery,
    build_execution_manifest,
    load_catalog,
    rankability_reasons,
    write_catalog,
)
from model_atlas.planning.memory_planner import GIB, assess
from model_atlas.planning.realbytes import account_manifest, plan_candidates, report
from model_atlas.registry.architectures import get_registry

app = typer.Typer(no_args_is_help=True, help="Model-agnostic Atlas platform CLI.")


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
    kernel_receipt: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--kernel-receipt",
        help="Runtime-kernel receipt or catalog; repeat to import several",
    ),
) -> None:
    """Render the Atlas Lab interactive dashboard from measured data."""
    path = write_dashboard(out, seed=seed, kernel_receipts=kernel_receipt)
    print(f"wrote interactive dashboard: {path}")


@app.command("kernel-import")
def kernel_import(
    receipts: list[str] = typer.Argument(..., help="Receipt/catalog JSON files"),  # noqa: B008
    out: str = typer.Option("kernel-evidence.json", "--out", help="Normalized catalog path"),
) -> None:
    """Validate runtime receipts and write a deterministic Atlas catalog."""
    catalog = load_catalog(receipts)
    path = write_catalog(out, catalog)
    eligible = sum(not rankability_reasons(receipt) for receipt in catalog.receipts)
    print(f"wrote kernel evidence catalog: {path}")
    print(f"  receipts: {len(catalog.receipts)}")
    print(f"  measured and ranking-eligible: {eligible}")


@app.command("kernel-query")
def kernel_query(
    catalog_path: str = typer.Option(..., "--catalog"),
    device_name: str = typer.Option(..., "--device"),
    compute_capability: str = typer.Option(..., "--compute-capability"),
    cuda_version: str = typer.Option(..., "--cuda-version"),
    driver_version: str = typer.Option(..., "--driver-version"),
    representation_format: str = typer.Option(..., "--format"),
    abi_name: str = typer.Option(..., "--abi"),
    abi_version: int = typer.Option(1, "--abi-version"),
    phase: KernelPhase = typer.Option(KernelPhase.DECODE, "--phase"),  # noqa: B008
    m: int = typer.Option(..., "--m"),
    n: int = typer.Option(..., "--n"),
    k: int = typer.Option(..., "--k"),
    tp_world_size: int = typer.Option(1, "--tp"),
    grouped_moe: bool = typer.Option(False, "--grouped-moe"),
    backend_commit: str | None = typer.Option(None, "--backend-commit"),
    allow_bucket_estimate: bool = typer.Option(False, "--allow-bucket-estimate"),
) -> None:
    """Ask the fail-closed kernel oracle about one exact workload."""
    catalog = load_catalog([catalog_path])
    result = catalog.query(
        KernelQuery(
            device_name=device_name,
            compute_capability=compute_capability,
            cuda_version=cuda_version,
            driver_version=driver_version,
            representation_format=representation_format,
            abi_name=abi_name,
            abi_version=abi_version,
            phase=phase,
            m=m,
            n=n,
            k=k,
            tp_world_size=tp_world_size,
            grouped_moe=grouped_moe,
            backend_commit=backend_commit,
        ),
        allow_bucket_estimate=allow_bucket_estimate,
    )
    print(result.model_dump_json(indent=2))
    if not result.eligible:
        raise typer.Exit(2)


@app.command("kernel-manifest")
def kernel_manifest(
    catalog_path: str = typer.Option(..., "--catalog"),
    request_path: str = typer.Option(..., "--request"),
    out: str = typer.Option("kernel-execution-manifest.json", "--out"),
) -> None:
    """Bind a candidate to exact measured kernels or refuse the export."""
    request = KernelManifestRequest.model_validate_json(Path(request_path).read_bytes())
    manifest = build_execution_manifest(
        request.candidate_id,
        request.requirements,
        load_catalog([catalog_path]),
    )
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote measured kernel execution manifest: {destination}")


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
                "quality": 0.99, "resident_gib": 214.0, "decode_tps": 21.0, "context": 256000,
            },
        ),
        FrontierPoint(
            candidate_id="B",
            values={"quality": 0.995, "resident_gib": 196.0, "decode_tps": 26.0, "context": 384000},
        ),
        FrontierPoint(
            candidate_id="C",
            values={"quality": 0.96, "resident_gib": 150.0, "decode_tps": 33.0, "context": 720000},
        ),
        FrontierPoint(
            candidate_id="D",
            values={"quality": 0.97, "resident_gib": 160.0, "decode_tps": 30.0, "context": 700000},
        ),
    ]
    r = restrict_frontier(pts)
    print("v3 pareto:")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
