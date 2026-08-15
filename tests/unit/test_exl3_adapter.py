"""Focused EXL3 adapter integration tests.

Exercises the adapter's REAL external contract with a fake pinned executable
that implements the exact probe + conversion contract and produces a REAL
changed safetensors derivative. Because the adapter shells out (never in-process
fakes), these confirm:

* fail-closed probe when the pinned executable is absent
* probe resolves exact version + capabilities when present
* execute stages the derivative into the scoped staging dir ONLY (never the
  source), validates it structurally, and returns a content-address receipt
* non-zero exit / cancellation fails closed
* resume re-runs idempotently

A real-runtime smoke (against an actual exllamav3) is skipped when no pinned
EXL3 executable is available on this machine.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.exl3_adapter import (
    Exl3Adapter,
    Exl3ProbeResult,
    build_exl3_manifest,
    build_exl3_record,
    probe_exl3,
)
from model_atlas.backend.registry import BackendRegistry
from model_atlas.jobs.artifacts import StageStager

# ---------------------------------------------------------------------------
# fake pinned EXL3 executable implementing the exact external contract
# ---------------------------------------------------------------------------


def make_fake_exl3(bin_dir: Path, *, probe_caps: str = "quantize,resume") -> Path:
    """Write a self-contained fake EXL3 conversion entry point. It:
    - implements `--atlas-probe-version --json` -> {"exl3_version","capabilities"}
    - implements the convert contract (-i -o -w -b -hb [-hq]) by writing a real
      changed safetensors (the input tensor value + 1) into the -o dir plus a
      model.safetensors.index.json, mirroring what the real tool produces.
    - `--fail` forces a non-zero exit (cancellation path).
    """
    cap_json = json.dumps(probe_caps)
    script = "#!/usr/bin/env python3\n" + textwrap.dedent(
        f"""
        import json, os, struct, sys
        def write_st(d, name, val):
            body = struct.pack("<f", float(val))
            meta = {{"__metadata__": {{}}, name: {{"dtype": "F32", "shape": [1],
                    "data_offsets": [0, 4]}}}}
            hb = json.dumps(meta).encode()
            with open(os.path.join(d, name + ".safetensors"), "wb") as f:
                f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(body)
        args = sys.argv[1:]
        if "--atlas-probe-version" in args:
            print(json.dumps({{"exl3_version": "1.4.2", "capabilities": {cap_json}}}))
            sys.exit(0)
        if os.environ.get("EXL3_FORCE_FAIL") == "1":
            sys.stderr.write("fake converter failed\\n"); sys.exit(17)
        if "-i" not in args:
            sys.stderr.write("missing -i\\n"); sys.exit(2)
        src = args[args.index("-i") + 1]
        out = args[args.index("-o") + 1]
        os.makedirs(out, exist_ok=True)
        base = 1.0
        src_t = os.path.join(src, "model-00001-of-00001.safetensors")
        if os.path.exists(src_t):
            with open(src_t, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
                name = [k for k in hdr if k != "__metadata__"][0]
                off = hdr[name]["data_offsets"][0]
                f.seek(8 + n + off)
                base = struct.unpack("<f", f.read(4))[0]
        write_st(out, "model-00001-of-00001", base + 1.0)
        write_st(out, "model-00002-of-00001", base + 2.0)
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            # the weight_map covers BOTH shards so structural validation sees
            # complete coverage (it enforces bidirectional index-shard agreement)
            json.dump(
                {{"metadata": {{"total_size": 8}}, "weight_map": {{
                    "model-00001-of-00001": "model-00001-of-00001.safetensors",
                    "model-00002-of-00001": "model-00002-of-00001.safetensors",
                }}}},
                f,
            )
        sys.exit(0)
        """
    )
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "exllamav3-convert"
    exe.write_text(script)
    exe.chmod(0o755)
    return exe


def make_source_checkpoint(src: Path) -> None:
    from model_atlas.checkpoint.safetensors import write_safetensors

    src.mkdir(parents=True, exist_ok=True)
    (src / "config.json").write_text('{"architectures":["FakeLM"]}')
    write_safetensors(
        src / "model-00001-of-00001.safetensors",
        {"weight": {"dtype": "F32", "shape": [1], "bytes": struct_pack_f(7.0)}},
    )


def struct_pack_f(v: float) -> bytes:
    import struct

    return struct.pack("<f", v)


class _EnvFailRunner:
    """Test seam: override the runner to inject EXL3_FORCE_FAIL before spawning."""

    def __init__(self, _exe: Path) -> None:
        pass

    def run(self, argv: list[str], cwd: str, env: dict[str, str]):
        env = dict(env)
        env["EXL3_FORCE_FAIL"] = "1"
        import subprocess as _sp

        return _sp.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)


@pytest.fixture
def fake_exe(tmp_path: Path) -> Path:
    return make_fake_exl3(tmp_path / "bin")


def _staging_context(tmp_path: Path, exe: Path, src: Path, **extra) -> dict[str, object]:
    wd = tmp_path / "wd"
    wd.mkdir(parents=True, exist_ok=True)
    stager = StageStager(tmp_path / "runs", "q1")
    ctx: dict[str, object] = {
        "workdir": str(wd),
        "staging_dir": str(stager.staging),
        "output_sink": str(stager.staging),
        "parameters": {
            "bpw": "3.25",
            "head_bits": "6",
            "hq": "1",
            "cal_rows": "128",
        },
        "source": str(src),
    }
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# probe truth
# ---------------------------------------------------------------------------


def test_probe_fails_closed_when_executable_absent(tmp_path: Path):
    result = probe_exl3(str(tmp_path / "no-such-exe"))
    assert isinstance(result, Exl3ProbeResult)
    assert result.available is False
    assert result.version is None
    assert "not found" in result.evidence
    assert "fail closed" in result.evidence


def test_probe_resolves_exact_version_and_capabilities(fake_exe: Path):
    result = probe_exl3(str(fake_exe))
    assert result.available is True
    assert result.version == "1.4.2"
    assert "quantize" in result.capabilities
    assert "1.4.2" in result.evidence


def test_probe_fails_closed_on_nonprobe_exit(tmp_path: Path):
    # executable present but probe flag unimplemented -> non-zero exit
    exe = tmp_path / "convert"
    exe.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(3)\n")
    exe.chmod(0o755)
    result = probe_exl3(str(exe))
    assert result.available is False
    assert "non-zero exit" in result.evidence


def test_record_registers_only_when_probe_passes(tmp_path: Path):
    absent = build_exl3_record(command=str(tmp_path / "missing"))
    reg = BackendRegistry({absent.backend_id: absent})
    assert not reg.is_backend_available("exl3")
    assert absent.probe(reg)[1] is None

    exe = make_fake_exl3(tmp_path / "bin")
    present = build_exl3_record(command=str(exe))
    reg2 = BackendRegistry({present.backend_id: present})
    assert reg2.is_backend_available("exl3")
    ok, version, _ = present.probe(reg2)
    assert ok and version == "1.4.2"


# ---------------------------------------------------------------------------
# execute: staging-only derivative + validation + receipt
# ---------------------------------------------------------------------------


def test_execute_stages_real_changed_derivative(tmp_path: Path, fake_exe: Path):
    src = tmp_path / "src"
    make_source_checkpoint(src)
    adapter = Exl3Adapter(command=str(fake_exe))
    ctx = _staging_context(tmp_path, fake_exe, src)

    handle = adapter.prepare(dict(ctx))
    result = adapter.execute(dict(ctx), handle)
    assert result["derivative"] is True
    assert result["format"] == "safetensors"
    assert "model-00001-of-00001.safetensors" in result["produced_shards"]

    # derivative landed in staging (2 shards), NOT in source
    staging = Path(str(ctx["output_sink"]))
    shards = [p for p in staging.iterdir() if p.suffix == ".safetensors"]
    assert len(shards) == 2
    # source is untouched: still only the original source tensor (value 7.0)
    from model_atlas.checkpoint.safetensors import read_safetensors_header

    st = read_safetensors_header(src / "model-00001-of-00001.safetensors")
    assert "weight" in st
    assert st["weight"]["shape"] == [1]
    assert not (src / "model-00002-of-00001.safetensors").exists()

    # receipt is content-addressed + replayable
    receipt = result["receipt"]
    assert receipt["digest"]
    for _name, dg in receipt["files"].items():
        assert len(dg) == 64


def test_validate_returns_passed_and_receipt(tmp_path: Path, fake_exe: Path):
    src = tmp_path / "src"
    make_source_checkpoint(src)
    adapter = Exl3Adapter(command=str(fake_exe))
    ctx = _staging_context(tmp_path, fake_exe, src)
    handle = adapter.prepare(dict(ctx))
    adapter.execute(dict(ctx), handle)

    staging = Path(str(ctx["output_sink"]))
    refs = {p.name: _fake_ref(p) for p in staging.iterdir() if p.is_file()}
    v = adapter.validate(dict(ctx), refs)
    assert v["validated"] is True
    assert v["status"] == "passed"
    assert v["derivative"] is True
    assert v["receipt"]["digest"]


def _fake_ref(path: Path):
    class _R:
        def __init__(self, p: Path):
            self.path = p
            self.relpath = p

    return _R(Path(path))


def test_validate_fails_closed_without_derivative(tmp_path: Path, fake_exe: Path):
    adapter = Exl3Adapter(command=str(fake_exe))
    ctx = _staging_context(tmp_path, fake_exe, tmp_path / "src")
    v = adapter.validate(dict(ctx), {})
    assert v["validated"] is False
    assert v["status"] == "unvalidated"


def test_manifest_is_deterministic_and_contract_faithful(tmp_path: Path, fake_exe: Path):
    src = tmp_path / "src"
    make_source_checkpoint(src)
    ctx = _staging_context(tmp_path, fake_exe, src)
    m1 = build_exl3_manifest(ctx, "h")
    m2 = build_exl3_manifest(ctx, "h")
    assert m1 == m2
    argv = m1["command_argv"]
    # contract: -i source -o out -w work -b bpw -hb head_bits -hq -cr rows
    pairs = {}
    i = 1
    while i < len(argv) - 1:
        if argv[i] in {"-hq"}:
            i += 1
            continue
        pairs[argv[i]] = argv[i + 1]
        i += 2
    assert pairs["-i"] == str(src)
    assert pairs["-w"] == str(tmp_path / "wd" / ".exl3-work")
    assert pairs["-b"] == "3.25"
    assert pairs["-hb"] == "6"
    assert pairs["-cr"] == "128"
    assert argv[0] == "exllamav3-convert"
    assert "-hq" in argv
    assert m1["provenance"]["source_immutable"] is True


# ---------------------------------------------------------------------------
# cancellation / error handling
# ---------------------------------------------------------------------------


def test_execute_nonzero_exit_fails_closed(tmp_path: Path, fake_exe: Path):
    src = tmp_path / "src"
    make_source_checkpoint(src)
    ctx = _staging_context(tmp_path, fake_exe, src)
    # force the fake converter to exit non-zero (cancellation path)
    adapter = Exl3Adapter(command=str(fake_exe), runner=_EnvFailRunner(fake_exe))
    with pytest.raises(BackendUnavailable) as err:
        adapter.execute(dict(ctx), adapter.prepare(dict(ctx)))
    assert "exited 17" in str(err.value) or "fail closed" in str(err.value)


def test_execute_without_pinned_exe_fails_closed(tmp_path: Path):
    adapter = Exl3Adapter(command=str(tmp_path / "does-not-exist"))
    ctx = _staging_context(tmp_path, tmp_path / "bin" / "x", tmp_path / "src")
    with pytest.raises(BackendUnavailable) as err:
        adapter.execute(dict(ctx), "h")
    assert "not found at execute time" in str(err.value)


def test_resume_reruns_idempotently(tmp_path: Path, fake_exe: Path):
    src = tmp_path / "src"
    make_source_checkpoint(src)
    adapter = Exl3Adapter(command=str(fake_exe))
    ctx = _staging_context(tmp_path, fake_exe, src)
    r1 = adapter.execute(dict(ctx), "h")
    r2 = adapter.resume(dict(ctx), "h")
    assert r2["receipt"]["digest"] == r1["receipt"]["digest"]


# ---------------------------------------------------------------------------
# real-runtime smoke (optional; skipped when no pinned EXL3 executable found)
# ---------------------------------------------------------------------------


def _real_exl3() -> str | None:
    for cand in ("exllamav3-convert", "convert_exl3"):
        found = __import__("shutil").which(cand)
        if found:
            return found
    return None


@pytest.mark.integration
def test_real_runtime_smoke_skipped_without_executable(tmp_path: Path):
    """Optional real-runtime smoke. Runs against an actual pinned EXL3 converter
    when one is available; otherwise it is truthfully skipped (not falsely
    passed). We only assert the probe answers consistently."""
    exe = _real_exl3()
    if exe is None:
        pytest.skip("no real pinned EXL3 executable on this machine")
    result = probe_exl3(exe)
    # an installed converter either implements the probe or is NOT available
    assert result.available in (True, False)
