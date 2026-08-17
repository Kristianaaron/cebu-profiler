from pathlib import Path

import pytest

from model_atlas.prune.bundle import (
    BundleError,
    pack_derivative_bundle,
    unpack_derivative_bundle,
)


def _tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"moe_intermediate_size":16}', encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        '{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"\x00\x01" * 100)
    (root / "model-00002-of-00002.safetensors").write_bytes(b"\xfe\xff" * 50)
    return root


def test_pack_is_deterministic(tmp_path: Path) -> None:
    a = pack_derivative_bundle(_tree(tmp_path / "one"), tmp_path / "b1.atlasbundle")
    b = pack_derivative_bundle(_tree(tmp_path / "two"), tmp_path / "b2.atlasbundle")
    assert a == b
    sha, size = a
    assert len(sha) == 64
    assert size > 0


def test_pack_unpack_round_trip(tmp_path: Path) -> None:
    source = _tree(tmp_path / "src")
    bundle = tmp_path / "deriv.atlasbundle"
    pack_derivative_bundle(source, bundle)
    out = tmp_path / "out"
    written = unpack_derivative_bundle(bundle, out)
    assert written == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]
    assert (out / "config.json").read_text() == source.joinpath("config.json").read_text()
    assert (out / "model-00001-of-00002.safetensors").read_bytes() == (
        source / "model-00001-of-00002.safetensors"
    ).read_bytes()


def test_unpack_rejects_tampered_member(tmp_path: Path) -> None:
    source = _tree(tmp_path / "src")
    bundle = tmp_path / "deriv.atlasbundle"
    pack_derivative_bundle(source, bundle)

    # Re-pack the same archive but flip one byte inside a member body,
    # keeping the manifest and record sizes identical (a real tamper).
    tampered = _tamper_member(bundle, "model-00002-of-00002.safetensors",
                              tmp_path / "tampered.atlasbundle")
    assert tampered.stat().st_size == bundle.stat().st_size  # byte-neutral flip
    with pytest.raises(BundleError, match="sha256 mismatch"):
        unpack_derivative_bundle(tampered, tmp_path / "badout")


def _tamper_member(bundle: Path, target: str, out: Path) -> Path:
    import io
    import tarfile

    with tarfile.open(bundle, "r:*") as src, tarfile.open(
        out, "w", format=tarfile.PAX_FORMAT
    ) as dst:
        for member in src:
            if member.isdir():
                continue
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == target and data is not None:
                flipped = bytearray(data)
                flipped[0] ^= 0xFF
                data = bytes(flipped)
            new = tarfile.TarInfo(member.name)
            if data is not None:
                new.size = len(data)
                new.mtime = 0
                new.uid = 0
                new.gid = 0
                new.mode = 0o644
                dst.addfile(new, io.BytesIO(data))
            else:
                dst.addfile(new)
    return out


def test_unpack_rejects_missing_member(tmp_path: Path) -> None:
    source = _tree(tmp_path / "src")
    bundle = tmp_path / "deriv.atlasbundle"
    pack_derivative_bundle(source, bundle)
    out = tmp_path / "out"
    written = unpack_derivative_bundle(bundle, out)
    assert (out / "config.json").exists()
    assert len(written) == 4


def test_pack_rejects_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(BundleError):
        pack_derivative_bundle(tmp_path / "empty", tmp_path / "x.atlasbundle")


def test_unpack_rejects_non_bundle(tmp_path: Path) -> None:
    f = tmp_path / "junk.atlasbundle"
    f.write_bytes(b"not a real bundle")
    with pytest.raises(BundleError):
        unpack_derivative_bundle(f, tmp_path / "out")
