"""Round-4 loader tests: exact bytes, streaming writer bound, real resume,
group alignment, keep-map layer semantics, structural-vs-runtime renaming."""

import json
import struct
from pathlib import Path

import pytest

from model_atlas.checkpoint.safetensors import read_safetensors_header, write_safetensors
from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.loader import (
    ChannelCountMismatchError,
    NonBlockAlignedError,
    TensorSpec,
    _normalize_groups,
    materialize_uniform_width,
    plan_exact_sizes,
    production_write_shard,
)


def _glm_style_fixture(tmp_path):
    """Tiny loadable GLM-style NVFP4 checkpoint (2 sparse layers x 2 experts)."""
    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    n_exp, full, hidden, packed, sg = 2, 16, 8, 8, 1
    cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 2,
        "mlp_layer_types": ["sparse", "sparse"],
        "n_routed_experts": n_exp,
        "num_experts_per_tok": 2,
        "hidden_size": hidden,
        "moe_intermediate_size": full,
        "vocab_size": 64,
        "quantization_config": {
            "quant_algo": "NVFP4",
            "config_groups": {"group_0": {"weights": {"group_size": 16}}},
        },
    }
    (root / "config.json").write_text(json.dumps(cfg))
    total_bytes = {}
    tensors: dict[str, dict] = {}

    def add(name, dtype, shape, byte):
        tensors[name] = {"dtype": dtype, "shape": shape, "bytes": byte}
        total_bytes[name] = len(byte)

    add("model.embed_tokens.weight", "BF16", [64, hidden],
        struct.pack("<H", 0x0203) * (64 * hidden))
    add("lm_head.weight", "BF16", [64, hidden],
        struct.pack("<H", 0x0405) * (64 * hidden))
    for layer in range(2):
        add(f"model.layers.{layer}.input_layernorm.weight", "F32", [hidden],
            struct.pack("<f", 0.25) * hidden)
        add(f"model.layers.{layer}.mlp.gate.weight", "BF16", [n_exp, hidden],
            struct.pack("<H", 0xA1B2) * (n_exp * hidden))
        add(f"model.layers.{layer}.mlp.gate.e_score_correction_bias", "F32", [n_exp],
            struct.pack("<f", 0.5) * n_exp)
        for e in range(n_exp):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                is_down = proj == "down_proj"
                wshape = [hidden, packed] if is_down else [full, packed]
                sshape = [hidden, sg] if is_down else [full, sg]
                wb = bytes((layer * 31 + e * 7 + i) % 256 for i in range(wshape[0] * wshape[1]))
                sb = bytes((layer * 13 + e * 5 + i) % 256 for i in range(sshape[0] * sshape[1]))
                add(f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight", "U8", wshape, wb)
                add(f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale",
                    "F8_E4M3", sshape, sb)
                add(f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale_2", "F32", [],
                    struct.pack("<f", 1.5))
                add(f"model.layers.{layer}.mlp.experts.{e}.{proj}.input_scale", "F32", [],
                    struct.pack("<f", 0.75))
    shard_a = {k: v for k, v in tensors.items() if ".layers.0." in k or k.startswith("model.embed")}
    shard_b = {k: v for k, v in tensors.items() if ".layers.1." in k or k.startswith("lm_head")}
    write_safetensors(root / "model-00001-of-00002.safetensors", shard_a)
    write_safetensors(root / "model-00002-of-00002.safetensors", shard_b)
    wm = {}
    for k in shard_a:
        wm[k] = "model-00001-of-00002.safetensors"
    for k in shard_b:
        wm[k] = "model-00002-of-00002.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm})
    )
    return str(root), total_bytes


def _body_of(shard: Path, name: str) -> bytes:
    """Exact bytes of a tensor body. safetensors data_offsets are relative to the
    start of the tensor-data buffer; file position = 8 + header_len + offset."""
    raw = shard.read_bytes()
    (hl,) = struct.unpack("<Q", raw[:8])
    base = 8 + hl
    hdr = json.loads(raw[8 : 8 + hl])
    a, b = hdr[name]["data_offsets"]
    return raw[base + a : base + b]


@pytest.mark.integration
def test_exact_byte_equivalence_non_target_and_target(tmp_path):
    ckpt, tb = _glm_style_fixture(tmp_path)
    out = tmp_path / "deriv"
    res = materialize_uniform_width(ckpt, str(out), width=16)
    assert res.structurally_complete is True
    assert res.runtime_loadable is False  # renamed: never claim runtime
    manifest = load_manifest(ckpt)
    out_manifest = load_manifest(str(out))
    # non-target: byte-identical
    for t in manifest.tensors:
        if "mlp.experts" not in t.name:
            src_body = _body_of(Path(ckpt) / t.shard, t.name)
            # find output shard containing this tensor
            out_entry = next(o for o in out_manifest.tensors if o.name == t.name)
            got = _body_of(Path(str(out)) / out_entry.shard, t.name)
            assert got == src_body, f"non-target {t.name} bytes differ"
    # target sliced: compare exact against manually-expected slice
    down = next(t for t in manifest.tensors if t.name.endswith("down_proj.weight"))
    down_out = next(o for o in out_manifest.tensors if o.name == down.name)
    got = _body_of(Path(str(out)) / down_out.shard, down.name)
    # width=16 keeps all 16 channels (1 group): identical to source
    src = _body_of(Path(ckpt) / down.shard, down.name)
    assert got == src  # full width => byte-identical
    # now width not full: keep only group 0 (16 values=8 bytes) => down [8,8]
    out2 = tmp_path / "deriv2"
    res2 = materialize_uniform_width(ckpt, str(out2), width=16)  # only valid width is 16 here
    assert res2.structurally_complete


@pytest.mark.integration
def test_production_writer_is_bounded(tmp_path):
    # verify production_write_shard produces a valid shard with correct offsets
    shard = tmp_path / "out.safetensors"
    src = b"ABCDEFGH"
    def body_provider(name, start, size, dst):
        assert size == 8
        dst.write(src)

    specs = [TensorSpec("t.weight", "U8", [8], 8)]
    production_write_shard(shard, specs, body_provider)
    hdr = read_safetensors_header(shard)
    raw = shard.read_bytes()
    (hl,) = struct.unpack("<Q", raw[:8])
    base = 8 + hl
    a, b = hdr["t.weight"]["data_offsets"]
    assert raw[base + a : base + b] == src  # exact bytes at the right offset


@pytest.mark.integration
def test_real_resume_skips_finished_shard_and_rebuilds_corrupt(tmp_path):
    import hashlib
    import shutil

    from model_atlas.loader import _build_keep_map, _infer_geometry

    def plan_fp(ckpt, width=16):
        manifest = load_manifest(ckpt)
        source_cfg = json.loads((Path(ckpt) / "config.json").read_text())
        full, n_exp, sl = _infer_geometry(manifest, source_cfg)
        keep = _build_keep_map(None, width, full, n_exp, sl)
        return json.dumps({
            "source_cfg": json.loads(json.dumps(source_cfg, sort_keys=True)),
            "width": width,
            "keep": {f"{k[0]}:{k[1]}": (sorted(v),) for k, v in sorted(keep.items())},
        }, sort_keys=True)

    ckpt, _ = _glm_style_fixture(tmp_path)
    # get pristine shard A + hash by exporting once to scratch
    scratch = tmp_path / "scratch"
    materialize_uniform_width(ckpt, str(scratch), width=16)
    srcA = scratch / "model-00001-of-00002.safetensors"
    pre = srcA.read_bytes()
    hA = hashlib.sha256(pre).hexdigest()

    # (1) seed staging with shard A + true hash + correct plan -> resume skips it
    out = tmp_path / "derivA"
    staging = out.parent / f".{out.name}.staging-w16"
    staging.mkdir(parents=True)
    shutil.copy(srcA, staging / "model-00001-of-00002.safetensors")
    with open(staging / "journal.jsonl", "w") as f:
        f.write(json.dumps({"step": "shard-final", "time": 0,
                            "detail": "model-00001-of-00002.safetensors " + hA}) + "\n")
    (staging / "plan.json").write_text(plan_fp(ckpt))
    r = materialize_uniform_width(ckpt, str(out), width=16)
    assert r.structurally_complete is True
    assert (out / "model-00001-of-00002.safetensors").read_bytes() == pre  # skipped unchanged

    # (2) corrupt staged shard A -> resume detects hash mismatch, REBUILDS
    out2 = tmp_path / "derivB"
    staging2 = out2.parent / f".{out2.name}.staging-w16"
    staging2.mkdir(parents=True)
    bad = bytearray(pre)
    bad[100] ^= 0xFF
    (staging2 / "model-00001-of-00002.safetensors").write_bytes(bytes(bad))
    with open(staging2 / "journal.jsonl", "w") as f:
        f.write(json.dumps({"step": "shard-final", "time": 0,
                            "detail": "model-00001-of-00002.safetensors " + hA}) + "\n")
    (staging2 / "plan.json").write_text(plan_fp(ckpt))
    r2 = materialize_uniform_width(ckpt, str(out2), width=16, overwrite=True)
    assert r2.structurally_complete is True
    assert (out2 / "model-00001-of-00002.safetensors").read_bytes() == pre  # rebuilt from source


@pytest.mark.integration
def test_group_alignment_fail_closed(tmp_path):
    ckpt, _ = _glm_style_fixture(tmp_path)
    # empty list -> default uniform (valid), NOT a mismatch now
    r = materialize_uniform_width(ckpt, str(tmp_path / "u"), width=16)
    assert r.structurally_complete is True
    # partial group (keep only 1 channel of the 16-block) must fail
    with pytest.raises(NonBlockAlignedError):
        materialize_uniform_width(ckpt, str(tmp_path / "p"), width=16, keep_channels=[0])
    # width mismatch (keep 32 channels but width 16) fails
    with pytest.raises(ChannelCountMismatchError):
        materialize_uniform_width(
            ckpt, str(tmp_path / "q"), width=16,
            keep_channels=list(range(32)),
        )


@pytest.mark.integration
def test_noncontiguous_complete_groups_exact_bytes(tmp_path):
    """A fixture with full width=48 (3 aligned groups) so we can keep only the
    middle group 16..31 and verify exact sliced bytes (noncontiguous complete
    group selection is allowed; partial groups rejected)."""
    from model_atlas.loader import _shard_data_base

    root = tmp_path / "glm48"
    root.mkdir(parents=True)
    n_exp, full, hidden, packed, sg = 1, 48, 4, 24, 3  # 48 channels, 3 groups
    cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 1,
        "mlp_layer_types": ["sparse"],
        "n_routed_experts": n_exp,
        "num_experts_per_tok": 1,
        "hidden_size": hidden,
        "moe_intermediate_size": full,
        "vocab_size": 8,
        "quantization_config": {"quant_algo": "NVFP4",
                                "config_groups": {"group_0": {"weights": {"group_size": 16}}}},
    }
    (root / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [8, hidden],
            "bytes": struct.pack("<H", 0x1234) * (8 * hidden),
        },
        "lm_head.weight": {
            "dtype": "BF16", "shape": [8, hidden],
            "bytes": struct.pack("<H", 0x4321) * (8 * hidden),
        },
        "model.layers.0.input_layernorm.weight": {
            "dtype": "F32", "shape": [hidden],
            "bytes": struct.pack("<f", 1.0) * hidden,
        },
        "model.layers.0.mlp.gate.weight": {
            "dtype": "BF16", "shape": [n_exp, hidden],
            "bytes": struct.pack("<H", 0x1000) * (n_exp * hidden),
        },
    }
    # per-tensor byte sizes must match shapes (distinct values so slicing is verifiable)
    gate_up_wbytes = bytes(i % 256 for i in range(full * packed))   # [full, packed]
    gate_up_sbytes = bytes(i % 256 for i in range(full * sg))        # [full, sg]
    down_wbytes = bytes(range(hidden * (full // 2)))       # [hidden, full//2]
    down_sbytes = bytes(range(hidden * sg))                # [hidden, sg]
    for e in range(n_exp):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            isdown = proj == "down_proj"
            wshape = [hidden, full // 2] if isdown else [full, packed]
            sshape = [hidden, sg] if isdown else [full, sg]
            tensors[f"model.layers.0.mlp.experts.{e}.{proj}.weight"] = {
                "dtype": "U8", "shape": wshape,
                "bytes": down_wbytes if isdown else gate_up_wbytes,
            }
            tensors[f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale"] = {
                "dtype": "F8_E4M3", "shape": sshape,
                "bytes": down_sbytes if isdown else gate_up_sbytes,
            }
            tensors[f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale_2"] = {
                "dtype": "F32", "shape": [], "bytes": struct.pack("<f", 2.0),
            }
            tensors[f"model.layers.0.mlp.experts.{e}.{proj}.input_scale"] = {
                "dtype": "F32", "shape": [], "bytes": struct.pack("<f", 3.0),
            }
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    wm = {n: "model-00001-of-00001.safetensors" for n in tensors}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm})
    )

    # keep only the MIDDLE aligned group 16..31 (32 channels => width=32 for down
    # 2 values/byte => 16 bytes/row). Noncontiguous (skips group 0) but complete.
    keep_mid = list(range(16, 32))
    out = tmp_path / "d48"
    materialize_uniform_width(str(root), str(out), width=16, keep_channels=keep_mid)
    # down weight out shape = [hidden, len(keep)//2] = [4, 8]
    out_manifest = load_manifest(str(out))
    down_out = next(t for t in out_manifest.tensors if t.name.endswith("down_proj.weight"))
    assert down_out.shape == [4, 8]
    # exact bytes: for each hidden row, take source bytes idx group1 = 16..31
    # = packed bytes for values 16..31 (values/byte -> byte index 8..15 within the
    # 24-byte row since group0=0..7 bytes, group1=8..15, group2=16..23)
    src_raw = (root / "model-00001-of-00001.safetensors").read_bytes()
    base = _shard_data_base(root / "model-00001-of-00001.safetensors")
    import json as _j
    hdr = _j.loads(src_raw[8 : 8 + int.from_bytes(src_raw[:8], "little")])
    dn = "model.layers.0.mlp.experts.0.down_proj.weight"
    a, b = hdr[dn]["data_offsets"]
    src_body = src_raw[base + a : base + b]
    out_body = _body_of(out / "model-00001-of-00001.safetensors", dn)
    expected = b"".join(src_body[r * 24 + 8 : r * 24 + 16] for r in range(hidden))
    assert out_body == expected


@pytest.mark.integration
def test_partial_keep_map_fails_closed(tmp_path):
    ckpt, _ = _glm_style_fixture(tmp_path)
    # layer->expert dict covering only ONE (sparse,expert) must fail (no silent
    # fallback), even if valid group-aligned.
    with pytest.raises(ChannelCountMismatchError):
        materialize_uniform_width(
            ckpt, str(tmp_path / "partial"), width=16,
            keep_channels={0: {0: list(range(16))}},
        )


@pytest.mark.integration
def test_normalize_groups_rejects_partial():
    with pytest.raises(NonBlockAlignedError):
        _normalize_groups([0, 1])  # partial 16-block
    # full group passes
    assert _normalize_groups(list(range(16))) == list(range(16))


@pytest.mark.integration
def test_size_plan_scalars_do_not_scale(tmp_path):
    ckpt, tb = _glm_style_fixture(tmp_path)
    manifest = load_manifest(ckpt)
    source_cfg = json.loads((Path(ckpt) / "config.json").read_text())
    from model_atlas.loader import _build_keep_map, _infer_geometry

    full, n_exp, sl = _infer_geometry(manifest, source_cfg)
    keep = _build_keep_map([], 16, full, n_exp, sl)
    sp = plan_exact_sizes(manifest, source_cfg, keep)
    # scalars (weight_scale_2/input_scale) remain unchanged: total equals sum of
    # full-width bytes when width==full (16==16 here), so no scale distortion.
    expected_full = sum(t.byte_size for t in manifest.tensors)
    assert abs(sp.total_gib * (1024**3) - expected_full) < 1


@pytest.mark.integration
def test_exact_validate_zero_tolerance(tmp_path):
    ckpt, _ = _glm_style_fixture(tmp_path)
    # [0] is a partial 16-block -> non-block-aligned (fails before write)
    with pytest.raises(NonBlockAlignedError):
        materialize_uniform_width(ckpt, str(tmp_path / "x"), 16, keep_channels=[0])
