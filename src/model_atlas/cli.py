"""model-atlas command-line interface."""

from __future__ import annotations

import platform

import typer

from model_atlas import __version__
from model_atlas.census.census import build_manifest
from model_atlas.planning.memory_planner import GIB, assess
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
