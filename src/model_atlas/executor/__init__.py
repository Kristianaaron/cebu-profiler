"""Structural executor consuming compression manifests (blueprint §12)."""

from model_atlas.executor.structural import (
    apply_manifest,
    build_clone,
    dry_run,
    orders_from_manifest,
    reorder_channels,
)

__all__ = [
    "apply_manifest",
    "build_clone",
    "dry_run",
    "orders_from_manifest",
    "reorder_channels",
]
