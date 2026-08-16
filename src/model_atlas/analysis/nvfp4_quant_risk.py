"""Bounded weight-only risk profile for mixed-bit GLM NVFP4 conversion.

This is ESTIMATED evidence.  It samples a fixed number of rows and experts from
every routed-expert layer/projection without materializing a tensor or shard.
The result allocates the most sensitive *layer/projection* GGUF tensors to
NVFP4 and sends the remainder to Q1_0.  GGUF stacks experts, so this contract
does not pretend to provide per-expert allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_atlas.backend.nvfp4_sample import decode_nvfp4_rows

_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_PROJECTION_GGUF = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
_MAX_ROWS = 4
_MAX_EXPERTS = 16
_MAX_INDEX_BYTES = 32 * 1024 * 1024
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_HEADER_CACHE_BYTES = 512 * 1024 * 1024
_MAX_SAMPLE_BYTES = 1 << 20
_MAX_DECODED_ELEMENTS = 1 << 17


@dataclass(frozen=True)
class QuantRiskRow:
    layer: int
    projection: str
    sampled_experts: tuple[int, ...]
    sampled_rows: int
    rms_mean: float
    max_abs_mean: float
    outlier_ratio_mean: float
    expert_dispersion: float
    risk_score: float
    retained_type: str


@dataclass(frozen=True)
class NVFP4QuantRiskReport:
    schema_version: int
    source: str
    evidence_kind: str
    index_sha256: str
    config_sha256: str
    sample_sha256: str
    sample_experts: int
    sample_rows: int
    sensitive_fraction: float
    rows: tuple[QuantRiskRow, ...]
    tensor_type_lines: tuple[str, ...]
    tensor_type_sha256: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rows"] = [asdict(row) for row in self.rows]
        data["tensor_type_lines"] = list(self.tensor_type_lines)
        return data


class _CheckpointReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "model.safetensors.index.json"
        if self.index_path.stat().st_size > _MAX_INDEX_BYTES:
            raise ValueError("checkpoint index exceeds metadata bound")
        index_bytes = self.index_path.read_bytes()
        index = json.loads(index_bytes)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        self.weight_map = {str(k): str(v) for k, v in weight_map.items()}
        self.index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        config_path = root / "config.json"
        if config_path.stat().st_size > _MAX_CONFIG_BYTES:
            raise ValueError("checkpoint config exceeds metadata bound")
        config_bytes = config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._header_cache_bytes = 0

    def _header(self, shard_name: str) -> tuple[int, dict[str, Any]]:
        cached = self._headers.get(shard_name)
        if cached is not None:
            return cached
        path = self.root / shard_name
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"{shard_name}: truncated safetensors header")
            header_size = struct.unpack("<Q", raw)[0]
            if header_size > _MAX_HEADER_BYTES:
                raise ValueError(f"{shard_name}: safetensors header exceeds bound")
            if self._header_cache_bytes + header_size > _MAX_HEADER_CACHE_BYTES:
                raise ValueError("safetensors header cache exceeds total bound")
            header_bytes = handle.read(header_size)
        if len(raw) != 8:
            raise ValueError(f"{shard_name}: truncated safetensors header")
        if len(header_bytes) != header_size:
            raise ValueError(f"{shard_name}: truncated safetensors header JSON")
        try:
            header = json.loads(header_bytes)
        except ValueError as exc:
            raise ValueError(f"{shard_name}: invalid safetensors header JSON") from exc
        if not isinstance(header, dict):
            raise ValueError(f"{shard_name}: safetensors header is not an object")
        base = 8 + header_size
        self._header_cache_bytes += header_size
        cached = (base, header)
        self._headers[shard_name] = cached
        return cached

    def read_rows(self, name: str, rows: int) -> tuple[tuple[float, ...], bytes]:
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        scale2_name = name.removesuffix(".weight") + ".weight_scale_2"
        try:
            shards = {self.weight_map[n] for n in (name, scale_name, scale2_name)}
        except KeyError as exc:
            raise ValueError(f"missing NVFP4 constituent {exc.args[0]}") from exc
        if len(shards) != 1:
            raise ValueError(f"{name}: NVFP4 constituents span shards")
        shard_name = shards.pop()
        base, header = self._header(shard_name)
        weight = _spec(header, name, "U8", 2)
        scale = _spec(header, scale_name, "F8_E4M3", 2)
        scale2 = _spec(header, scale2_name, "F32", 0)
        weight_rows, packed_columns = (int(v) for v in weight["shape"])
        scale_rows, scale_columns = (int(v) for v in scale["shape"])
        if weight_rows != scale_rows or packed_columns * 2 != scale_columns * 16:
            raise ValueError(f"{name}: invalid NVFP4 geometry")
        if weight_rows < rows:
            raise ValueError(f"{name}: tensor has fewer than the requested {rows} rows")
        take = rows
        decoded_elements = take * packed_columns * 2
        sample_bytes = take * (packed_columns + scale_columns) + 4
        if decoded_elements > _MAX_DECODED_ELEMENTS:
            raise ValueError(f"{name}: decoded sample exceeds absolute element bound")
        if sample_bytes > _MAX_SAMPLE_BYTES:
            raise ValueError(f"{name}: sample exceeds absolute byte-read bound")
        shard = self.root / shard_name
        packed = _read(shard, base, weight, take * packed_columns)
        scales = _read(shard, base, scale, take * scale_columns)
        scale2_bytes = _read(shard, base, scale2, 4)
        factor = struct.unpack("<f", scale2_bytes)[0]
        values = decode_nvfp4_rows(
            packed,
            scales,
            factor,
            rows=take,
            packed_columns=packed_columns,
        )
        return values, packed + scales + scale2_bytes


def profile_nvfp4_quant_risk(
    checkpoint_dir: str | Path,
    *,
    sample_experts: int = 8,
    sample_rows: int = 4,
    sensitive_fraction: float = 0.28,
) -> NVFP4QuantRiskReport:
    if not 1 <= sample_experts <= _MAX_EXPERTS:
        raise ValueError(f"sample_experts must be in [1, {_MAX_EXPERTS}]")
    if not 1 <= sample_rows <= _MAX_ROWS:
        raise ValueError(f"sample_rows must be in [1, {_MAX_ROWS}]")
    if not 0.0 < sensitive_fraction < 1.0:
        raise ValueError("sensitive_fraction must be between zero and one")
    root = Path(checkpoint_dir).resolve()
    reader = _CheckpointReader(root)
    groups: dict[tuple[int, str], list[int]] = {}
    names: dict[tuple[int, str, int], str] = {}
    for name in reader.weight_map:
        match = _EXPERT_WEIGHT.match(name)
        if match is None:
            continue
        prefix = name.removesuffix(".weight")
        if (
            f"{prefix}.weight_scale" not in reader.weight_map
            or f"{prefix}.weight_scale_2" not in reader.weight_map
        ):
            # Appended draft/MTP or mixed-precision tensors are not part of the
            # ModelOpt NVFP4 routed-expert allocation being measured here.
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        groups.setdefault((layer, projection), []).append(expert)
        names[(layer, projection, expert)] = name
    if not groups:
        raise ValueError("checkpoint contains no routed-expert NVFP4 weights")

    raw_rows: list[dict[str, Any]] = []
    sample_digest = hashlib.sha256()
    for layer, projection in sorted(groups):
        experts = sorted(set(groups[(layer, projection)]))
        selected = _even_sample(experts, sample_experts)
        rms_values: list[float] = []
        maxima: list[float] = []
        ratios: list[float] = []
        for expert in selected:
            values, raw = reader.read_rows(names[(layer, projection, expert)], sample_rows)
            sample_digest.update(names[(layer, projection, expert)].encode())
            sample_digest.update(raw)
            rms = math.sqrt(math.fsum(v * v for v in values) / len(values))
            maximum = max(abs(v) for v in values)
            rms_values.append(rms)
            maxima.append(maximum)
            ratios.append(maximum / max(rms, 1e-30))
        mean_rms = statistics.fmean(rms_values)
        dispersion = (
            statistics.pstdev(rms_values) / max(mean_rms, 1e-30)
            if len(rms_values) > 1
            else 0.0
        )
        raw_rows.append(
            {
                "layer": layer,
                "projection": projection,
                "sampled_experts": tuple(selected),
                "sampled_rows": sample_rows,
                "rms_mean": mean_rms,
                "max_abs_mean": statistics.fmean(maxima),
                "outlier_ratio_mean": statistics.fmean(ratios),
                "expert_dispersion": dispersion,
            }
        )

    for metric in ("rms_mean", "outlier_ratio_mean", "expert_dispersion"):
        ranks = _percentile_ranks([float(row[metric]) for row in raw_rows])
        for row, rank in zip(raw_rows, ranks, strict=True):
            row[f"{metric}_rank"] = rank
    for row in raw_rows:
        row["risk_score"] = (
            0.45 * row["rms_mean_rank"]
            + 0.35 * row["outlier_ratio_mean_rank"]
            + 0.20 * row["expert_dispersion_rank"]
        )
    keep_count = max(1, math.ceil(len(raw_rows) * sensitive_fraction))
    sensitive = {
        (int(row["layer"]), str(row["projection"]))
        for row in sorted(
            raw_rows,
            # An exact budget can split a tied score.  The secondary key is an
            # explicit stable policy, never an accidental insertion-order rank.
            key=lambda row: (
                float(row["risk_score"]),
                float(row["rms_mean"]),
                int(row["layer"]),
                str(row["projection"]),
            ),
            reverse=True,
        )[:keep_count]
    }
    result_rows = tuple(
        QuantRiskRow(
            layer=int(row["layer"]),
            projection=str(row["projection"]),
            sampled_experts=tuple(row["sampled_experts"]),
            sampled_rows=int(row["sampled_rows"]),
            rms_mean=float(row["rms_mean"]),
            max_abs_mean=float(row["max_abs_mean"]),
            outlier_ratio_mean=float(row["outlier_ratio_mean"]),
            expert_dispersion=float(row["expert_dispersion"]),
            risk_score=float(row["risk_score"]),
            retained_type=(
                "NVFP4"
                if (int(row["layer"]), str(row["projection"])) in sensitive
                else "Q1_0"
            ),
        )
        for row in sorted(raw_rows, key=lambda row: (int(row["layer"]), str(row["projection"])))
    )
    sensitive_lines = [
        rf"^blk\.{row.layer}\.ffn_{_PROJECTION_GGUF[row.projection]}_exps\.weight$=NVFP4"
        for row in result_rows
        if row.retained_type == "NVFP4"
    ]
    lines = tuple(
        sensitive_lines
        + [r"blk\..*\.ffn_(gate|up|down)_exps\.weight=Q1_0"]
    )
    plan_bytes = ("\n".join(lines) + "\n").encode()
    return NVFP4QuantRiskReport(
        schema_version=1,
        source=str(root),
        evidence_kind="estimated",
        index_sha256=reader.index_sha256,
        config_sha256=reader.config_sha256,
        sample_sha256=sample_digest.hexdigest(),
        sample_experts=sample_experts,
        sample_rows=sample_rows,
        sensitive_fraction=sensitive_fraction,
        rows=result_rows,
        tensor_type_lines=lines,
        tensor_type_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        note=(
            "bounded weight-only risk; no activation/Hessian/KLD claim; GGUF allocation "
            "is layer/projection-granular because experts are stacked"
        ),
    )


def _even_sample(values: list[int], count: int) -> list[int]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def _percentile_ranks(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        midrank = ((start + end - 1) / 2) / (len(values) - 1)
        for position in range(start, end):
            ranks[order[position]] = midrank
        start = end
    return ranks


def _spec(header: dict[str, Any], name: str, dtype: str, dimensions: int) -> dict[str, Any]:
    spec = header.get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"missing tensor {name}")
    if spec.get("dtype") != dtype or len(spec.get("shape") or []) != dimensions:
        raise ValueError(f"unexpected dtype/shape for {name}")
    offsets = spec.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"invalid offsets for {name}")
    return spec


def _read(path: Path, base: int, spec: dict[str, Any], size: int) -> bytes:
    start, end = (int(v) for v in spec["data_offsets"])
    if size < 0 or size > end - start:
        raise ValueError("requested sample exceeds tensor span")
    with path.open("rb") as handle:
        handle.seek(base + start)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError("short safetensors sample read")
    return data
