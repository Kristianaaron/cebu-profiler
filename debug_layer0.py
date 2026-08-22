"""Pinpoint where old/new engines diverge at layer 0, token 0."""

from __future__ import annotations

import importlib.util
import math
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD = load_module("old_engine", "engine_old/model_atlas/atlas/runtime.py")
NEW = load_module("new_engine", "engine_new/model_atlas/atlas/runtime.py")

from model_atlas.registry.architectures import get_registry  # noqa: E402

arch = get_registry().get("k3-mini")
mo = OLD.build_mini_moe(arch, seed=0)
mn = NEW.build_mini_moe(arch, seed=0)

print("embed equal:", mo.embed == mn.embed)
print("router l0 equal:", mo.layers[0].router == mn.layers[0].router)
print("gate e0 equal:", mo.layers[0].experts[0]["gate"] == mn.layers[0].experts[0]["gate"])

tokens = [1, 42, 7]
# old layer-0 internals for token 0
h = list(mo.embed[tokens[0]])
lw = mo.layers[0]
ln_o = [v * w for v, w in zip(h, lw.ln_w)]
logits_o = OLD._matvec(lw.router, ln_o)
gate_o = OLD._matvec(lw.experts[0]["gate"], ln_o)
up_o = OLD._matvec(lw.experts[0]["up"], ln_o)
gu_o = [g * u for g, u in zip(gate_o, up_o)]
eo_o = OLD._matvec(lw.experts[0]["down"], gu_o)

# new layer-0 internals
import numpy as np

H = np.asarray([mn.embed[t] for t in tokens], dtype=np.float64)
W = NEW._model_np(mn)["layers"][0]
ln_n = H * W["ln_w"]
logits_n = ln_n @ W["router"].T
gate_n = np.einsum("th,emh->tem", ln_n, W["gate"])[0, 0]
up_n = np.einsum("th,emh->tem", ln_n, W["up"])[0, 0]
gu_n = gate_n * up_n
eo_n = np.einsum("tem,ehm->teh", np.einsum("th,emh->tem", ln_n, W["gate"]) * np.einsum("th,emh->tem", ln_n, W["up"]), W["down"])[0, 0]

print("ln equal:", ln_o == list(ln_n[0]))
print("logits equal:", logits_o == list(logits_n[0]))
print("gate e0 equal:", gate_o == list(gate_n))
print("gate e0 old[:3]:", [round(v, 12) for v in gate_o[:3]])
print("gate e0 new[:3]:", [round(float(v), 12) for v in gate_n[:3]])
print("gate max abs diff:", max(abs(a - float(b)) for a, b in zip(gate_o, gate_n)))
print("expert_out e0 max abs diff:", max(abs(a - float(b)) for a, b in zip(eo_o, eo_n)))
print("expert_norm e0: old", OLD._l2_norm(eo_o), "new", float(np.linalg.norm(eo_n)))
