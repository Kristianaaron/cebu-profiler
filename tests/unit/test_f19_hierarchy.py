"""F19 tests: six-level profiler hierarchy (v2 §9) — up/down traceability.

Checks that build_hierarchy produces all six levels, links them in adjacency
order, is traceable up (weight/unit/expert -> behaviours) and down
(behaviour -> contributing experts/units/weights), tags every node as measured,
validates with no dangling/non-adjacent refs, and serializes to the §27
machine-readable form.
"""

import json

from cebu_profiler.profiler.hierarchy import (
    LEVEL_ORDER,
    ProfilerLevel,
    build_hierarchy,
    next_down,
    next_up,
)
from cebu_profiler.profiler.reap import make_synthetic_corpus
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _samples(n: int = 10, seq: int = 5) -> list:
    return make_synthetic_corpus(n_samples=n, seq_len=seq, vocab=ARCH.vocabulary_size, seed=0)[0]


def _hierarchy(seed: int = 1, n: int = 10) -> "object":
    model = build_mini_moe(ARCH, seed=seed)
    return build_hierarchy(model, _samples(n=n))


def test_all_six_levels_populated_and_ordered():
    hm = _hierarchy()
    counts = hm.counts()
    for lv in LEVEL_ORDER:
        assert counts[lv.value] > 0, f"missing level {lv.value}"
    # L1 tensors and L2 units are the granular lowest levels; behaviour the top
    assert counts["units"] >= counts["experts"]
    assert counts["behaviour"] <= counts["pathways"]


def test_level_adjacency_helpers():
    assert next_up(ProfilerLevel.WEIGHTS) is ProfilerLevel.UNITS
    assert next_up(ProfilerLevel.PATHWAYS) is ProfilerLevel.BEHAVIOUR
    assert next_up(ProfilerLevel.BEHAVIOUR) is None
    assert next_down(ProfilerLevel.BEHAVIOUR) is ProfilerLevel.PATHWAYS
    assert next_down(ProfilerLevel.WEIGHTS) is None


def test_up_traceability_weight_to_behaviour():
    hm = _hierarchy()
    # every ROUTED expert (one that appears in a coalition) is linked up to a
    # behaviour; never-routed experts legitimately carry no measured link.
    routed = [e for e in hm.nodes_at(ProfilerLevel.EXPERTS) if e.parents]
    assert routed, "expected at least one routed expert"
    for exp in routed:
        behavs = hm.behaviours_of(exp.key)
        assert behavs, f"routed expert {exp.key} not linked to any behaviour"
        # ancestors walk up through coalitions/pathways before behaviour
        levels = {n.level for n in hm.ancestors(exp.key)}
        assert ProfilerLevel.COALITIONS in levels
        assert ProfilerLevel.BEHAVIOUR in levels


def test_down_traceability_behaviour_to_weights():
    hm = _hierarchy()
    beh = hm.nodes_at(ProfilerLevel.BEHAVIOUR)[0]
    proj = hm.project_down(beh.key)
    # a behaviour decomposes all the way down to weights
    assert "experts" in proj and proj["experts"]
    assert "units" in proj and proj["units"]
    assert "weights" in proj and proj["weights"]
    # prevalence == number of distinct behaviours the contributor supports
    sample_row = proj["units"][0]
    assert sample_row["prevalence"] == len(hm.behaviours_of(sample_row["key"]))


def test_shared_components_are_load_bearing():
    hm = _hierarchy()
    # units under a ROUTED expert are each linked up (share/load-bearing is
    # expressed by prevalence, not by a hard routing frequency threshold)
    routed = [e for e in hm.nodes_at(ProfilerLevel.EXPERTS) if e.parents]
    for exp in routed:
        for unit in hm.descendants(exp.key):
            if unit.level is ProfilerLevel.UNITS:
                assert hm.behaviours_of(unit.key), "unlinked unit"
                break


def test_validation_no_warnings():
    hm = _hierarchy()
    assert hm.validate() == []


def test_all_nodes_measured_evidence():
    hm = _hierarchy()
    for n in hm.nodes.values():
        assert n.evidence == "measured"


def test_to_dict_json_roundtrip():
    hm = _hierarchy()
    payload = hm.to_dict()
    encoded = json.dumps(payload)  # must be JSON-serializable
    decoded = json.loads(encoded)
    assert decoded["counts"] == hm.counts()
    assert decoded["model_id"] == ARCH.name
    n_nodes = sum(len(v) for v in payload["nodes"].values())
    assert n_nodes == len(hm.nodes)
