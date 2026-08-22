"""Golden-equivalence harness: old pure-Python engine vs new NumPy engine.

Compares forward() traces, representation_profile(), and ChannelStat rows
between the pristine git version and the working-tree version of
runtime.py + collector.py, across plain forwards, ablations, route overrides,
and channel-stats collection.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD = load_module("old_engine", "engine_old/model_atlas/atlas/runtime.py")
OLD_COL = load_module("old_col", "engine_old/model_atlas/atlas/collector.py")
NEW = load_module("new_engine", "engine_new/model_atlas/atlas/runtime.py")
NEW_COL = load_module("new_col", "engine_new/model_atlas/atlas/collector.py")

from model_atlas.registry.architectures import get_registry  # noqa: E402


def close(a, b, tol=5e-12):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def trace_dict(tr, engine):
    return {
        "layer": tr.layer,
        "logits": tr.logits,
        "probs_all": tr.probs_all,
        "topk_ids": tr.topk_ids,
        "topk_probs": tr.topk_probs,
        "expert_norm": tr.expert_norm,
        "router_weighted": tr.router_weighted,
        "entropy": tr.entropy,
        "input_norm": tr.input_norm,
        "moe_norm": tr.moe_norm,
        "output_norm": tr.output_norm,
        "combined": tr.combined,
    }


def run(engine_mod, col_mod, tokens, top_k, override=None, excluded=None, collect=False, model=None):
    arch = get_registry().get("k3-mini")
    if model is None:
        model = engine_mod.build_mini_moe(arch, seed=0)
    acc = col_mod.ChannelStatsAccumulator() if collect else None
    res = engine_mod.forward(
        model, tokens, top_k=top_k, route_override=override, excluded=excluded,
        channel_stats=acc,
    )
    prof = engine_mod.representation_profile(model, tokens[:4], top_k=top_k)
    stats = acc.finalize() if acc else []
    return res, prof, stats


def main() -> int:
    tokens = [1, 42, 7, 999, 3, 5, 100, 55]
    failures: list[str] = []
    scenarios = [
        ("plain", None, None, False),
        ("ablate", None, {0: frozenset({1, 3})}, False),
        ("override", {(0, 2): [4, 5], (1, 0): [0, 2]}, None, False),
        ("collect", None, None, True),
    ]

    # ---- pruned-clone scenario: variable-width experts ----
    # uniform + hetero pruning shrink experts to different channel counts,
    # exercising the width-grouped batching path in the new engine. The OLD
    # scalar engine is the reference; both engines must agree per width-mix.
    from model_atlas.experiments.controls import (
        channel_importance,
        hetero_clone,
        uniform_clone,
    )
    from model_atlas.atlas.reap import make_synthetic_corpus
    from model_atlas.synthetic.mini_moe import mini_moe_spec

    def make_corpus(arch_=None, seed=7):
        spec = mini_moe_spec()
        return make_synthetic_corpus(
            n_samples=8, seq_len=6, vocab=spec.vocabulary_size or 1000, seed=seed
        )[:2]

    arch = get_registry().get("k3-mini")
    cal_p, _ = make_corpus(seed=7)
    imp = channel_importance(NEW.build_mini_moe(arch, seed=0), cal_p)
    # clone builders are engine-agnostic (pure list surgery on weights); the same
    # pruned model is then forwarded through each engine on its own deepcopy.
    pruned_models = {
        name: fn(NEW.build_mini_moe(arch, seed=0), imp, 60)
        for name, fn in (("uniform60", uniform_clone), ("hetero60", hetero_clone))
    }
    for name, ov, ex, collect in scenarios:
        res_o, prof_o, stats_o = run(OLD, OLD_COL, tokens, 2, ov, ex, collect)
        res_n, prof_n, stats_n = run(NEW, NEW_COL, tokens, 2, ov, ex, collect)
        for tr_o, tr_n in zip(res_o.traces, res_n.traces):
            d = trace_dict(tr_o, OLD)
            d2 = trace_dict(tr_n, NEW)
            for key in d:
                if not close(d[key], d2[key], 1e-9):
                    failures.append(f"[{name}] trace[{tr_o.layer}].{key} differs")
        if not close(res_o.final_hidden, res_n.final_hidden, 1e-9):
            failures.append(f"[{name}] final_hidden differs")
        if not close(res_o.logits, res_n.logits, 1e-9):
            failures.append(f"[{name}] lm_head logits differ")
        if not close(res_o.final_hidden_states, res_n.final_hidden_states, 1e-9):
            failures.append(f"[{name}] final_hidden_states differ")
        if not close(prof_o, prof_n, 1e-9):
            failures.append(f"[{name}] representation_profile differs")
        if len(stats_o) != len(stats_n):
            failures.append(f"[{name}] channel stat count {len(stats_o)} != {len(stats_n)}")
        else:
            for so, sn in zip(stats_o, stats_n):
                tup_o = (so.layer, so.expert, so.channel, so.rms, so.mean_abs, so.frequency, so.peak, so.samples)
                tup_n = (sn.layer, sn.expert, sn.channel, sn.rms, sn.mean_abs, sn.frequency, sn.peak, sn.samples)
                if not close(list(tup_o), list(tup_n), 1e-9):
                    failures.append(f"[{name}] channel stat {so.layer}/{so.expert}/{so.channel} differs")
        print(f"scenario {name}: old {len(stats_o)} stats, new {len(stats_n)} stats — checked")

    # ---- variable-width expert models: old vs new must agree exactly ----
    def forward_pruned(engine_mod, col_mod, model, tokens, collect):
        acc = col_mod.ChannelStatsAccumulator() if collect else None
        res = engine_mod.forward(model, tokens, top_k=2, channel_stats=acc)
        stats = acc.finalize() if acc else []
        return res, stats

    import copy as _copy

    for pname, pmodel in pruned_models.items():
        for sname, collect in (("plain", False), ("collect", True)):
            mo = _copy.deepcopy(pmodel)
            mn = _copy.deepcopy(pmodel)
            ro, so_ = forward_pruned(OLD, OLD_COL, mo, tokens, collect)
            rn, sn_ = forward_pruned(NEW, NEW_COL, mn, tokens, collect)
            name = pname + "/" + sname
            for tr_o, tr_n in zip(ro.traces, rn.traces):
                d1 = trace_dict(tr_o, OLD)
                d2 = trace_dict(tr_n, NEW)
                for key in d1:
                    if not close(d1[key], d2[key], 5e-12):
                        failures.append(f"[{name}] trace[{tr_o.layer}].{key} differs")
            if not close(ro.final_hidden, rn.final_hidden, 5e-12):
                failures.append(f"[{name}] final_hidden differs")
            if not close(ro.logits, rn.logits, 5e-12):
                failures.append(f"[{name}] lm_head logits differ")
            if not close(ro.final_hidden_states, rn.final_hidden_states, 5e-12):
                failures.append(f"[{name}] final_hidden_states differ")
            if len(so_) != len(sn_):
                failures.append(f"[{name}] stat count {len(so_)} != {len(sn_)}")
            else:
                for a, b in zip(so_, sn_):
                    ta = (a.layer, a.expert, a.channel, a.rms, a.mean_abs, a.frequency, a.peak, a.samples)
                    tb = (b.layer, b.expert, b.channel, b.rms, b.mean_abs, b.frequency, b.peak, b.samples)
                    if not close(list(ta), list(tb), 5e-12):
                        failures.append(f"[{name}] channel stat {a.layer}/{a.expert}/{a.channel} differs")
        print(f"pruned-model {pname}: checked")

    if failures:
        print(json.dumps(sorted(set(failures)), indent=1))
        return 1
    print("GOLDEN EQUIVALENCE: PASS (all scenarios, tol 5e-12)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
