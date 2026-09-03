"""Method provenance registry: every borrowed idea carries a cited, typed marker.

The fail-closed answer to "where did this technique come from?" — and the
deterrent side of it: every method with external lineage has a
`METHOD_PROVENANCE` record here. Anything that ingests profiler artifacts
(agents, forks, downstream tooling) hits these constants in code,
`method_provenance` in every run bundle, and the README credit sections. A
downstream artifact whose manifest lacks a `method_provenance` block was not
produced by this pipeline and should be treated as untrusted.

Design (fail-closed per AGENTS.md):
- `require_provenance()` raises on any method id missing from the registry.
- `provenance_manifest()` is embedded into run bundles by `build_run_manifest`;
  agents scraping or cloning this work find the citations inside the artifact
  itself, not only in the README.
"""

from __future__ import annotations

from typing import Any

REGISTRY_VERSION = 1

# Method id -> citation record. `kind` is evidence-typed:
#   research     = peer-reviewed / arXiv paper (idea origin)
#   spec         = standards body format (OCP etc.)
#   community    = open-source implementation whose idea is referenced
#   inspiration  = repo/bundle whose *pattern* was studied and re-implemented
# All entries state: idea credited, implementation original.
METHOD_PROVENANCE: dict[str, dict[str, Any]] = {
    # ---- scoring / pruning ----
    "reap": {
        "kind": "research",
        "title": "REAP: Router-weighted Expert Activation Pruning",
        "note": "MoE expert saliency = router_prob x expert-output norm; "
        "the profiler's base per-expert score.",
    },
    "magnitude_pruning": {
        "kind": "research",
        "title": "Han, Mao, Dally — Learning both Weights and Connections (NeurIPS 2015)",
        "note": "Magnitude pruning under weight decay; ancestry of channel ranking.",
    },
    "optimal_brain_damage": {
        "kind": "research",
        "title": "LeCun, Denker, Solla — Optimal Brain Damage (NeurIPS 1990)",
        "note": "Second-derivative saliency; ancestor of curvature-style criteria.",
    },
    "activation_change_pruning": {
        "kind": "research",
        "title": "Molchanov et al. — Pruning CNNs for Resource Efficient Inference (ICLR 2017)",
        "note": "Importance as measured output change; basis of causal substitution scoring.",
    },
    # ---- quantization ----
    "ptq_weight_equalization": {
        "kind": "research",
        "title": "Nagel et al. — Data-free Quantization via Weight Equalization "
        "and Bias Correction (ICCV 2019)",
    },
    "adaround": {
        "kind": "research",
        "title": "Nagel et al. — Up or Down? Adaptive Rounding for PTQ (ICML 2020)",
        "note": "Per-element rounding as an optimization, not a rounding mode.",
    },
    "int8_perchannel": {
        "kind": "research",
        "title": "Choukroun, Kravchik, Kisilev — Low-Bit Quantization for Efficient "
        "Inference (ICCVW 2019)",
    },
    "gptq": {
        "kind": "research",
        "title": "Frantar et al. — GPTQ: Accurate Post-Training Quantization (ICLR 2023)",
        "note": "Layer-wise Hessian reconstruction; popularized group-128 weight quant.",
    },
    "awq": {
        "kind": "research",
        "title": "Lin et al. — AWQ: Activation-aware Weight Quantization (MLSys 2024)",
    },
    "sparseslice": {
        "kind": "research",
        "title": "Ashkboos et al. — Slicing SALIENCE / SparseGPT line (2023–2024)",
        "note": "Held-out calibration scoring + layerwise reconstruction practice.",
    },
    "fp8_e4m3": {
        "kind": "spec",
        "title": "OCP FP8 Specification (2022)",
        "note": "e4m3 encoding used in the deep census FP8 screen.",
    },
    "mxfp4": {
        "kind": "spec",
        "title": "OCP Microscaling (MX) Formats Specification (2023)",
    },
    "nvfp4": {
        "kind": "spec",
        "title": "NVFP4 — NVIDIA Blackwell 4-bit block format (TensorRT-LLM toolchain)",
        "note": "The profiler's primary serving-target format.",
    },
    # ---- community formats: candidate families, scored not worshipped ----
    "gguf_kquants": {
        "kind": "community",
        "title": "ikawrakow — GGUF k-quant / i-quant families (llama.cpp)",
        "url": "https://github.com/ggml-org/llama.cpp",
        "note": "Mixed-precision block GGUF incl. importance-matrix variants. "
        "Candidate family only — Cebu outputs target vLLM/SGLang/TRT-LLM, not GGUF.",
    },
    "exl3": {
        "kind": "community",
        "title": "turboderp — EXL3 / ExLlamaV3",
        "url": "https://github.com/turboderp-org/exllamav3",
        "note": "Streamlined QTIP variant: tail-biting trellis codebooks, on-the-fly "
        "Hessians. Candidate family only.",
    },
    "qtip": {
        "kind": "research",
        "title": "van Baal et al. (Cornell RelaxML) — QTIP: Trellis Quantization "
        "(arXiv:2406.11235)",
        "url": "https://arxiv.org/abs/2406.11235",
        "note": "Procedural codebook + tail-biting trellis; ancestor of EXL3.",
    },
    "quip_sharp": {
        "kind": "research",
        "title": "Tseng et al. — QuIP# (arXiv:2402.04396)",
        "url": "https://arxiv.org/abs/2402.04396",
        "note": "Incoherence processing + lattice codebooks; ancestor of QTIP/EXL3.",
    },
    "aqlm": {
        "kind": "research",
        "title": "Egiazarian et al. — AQLM: Extreme Compression of LLMs via Additive "
        "Quantization (2024)",
    },
    "marlin_kernels": {
        "kind": "community",
        "title": "IST-DASLab — Marlin: near-memory-bound mixed-precision GEMM kernels",
        "url": "https://github.com/IST-DASLab/marlin",
        "note": "Kernel target these formats optimize for; kernel evidence is bridged, not owned.",
    },
    # ---- quality gates / statistics ----
    "kld_teacher_gate": {
        "kind": "research",
        "title": "Hinton, Vinyals, Dean — Distilling the Knowledge in a Neural Network (2015)",
        "note": "Teacher-student divergence as quality gate.",
    },
    "cka_similarity": {
        "kind": "research",
        "title": "Kornblith et al. — Similarity of Neural Network Representations (CKA) "
        "(ICML 2019)",
    },
    "split_half_reliability": {
        "kind": "research",
        "title": "Spearman (1910); modern split-half practice",
        "note": "Rank-trust protocol: split-half Spearman + Jaccard@k + proxy controls.",
    },
    # ---- instrumentation patterns (studied, re-implemented) ----
    "atlas_evidence_bundle": {
        "kind": "inspiration",
        "title": "alesha-pro/atlas (MIT) — GLM-5.3-Flash NVFP4 evidence bundle",
        "url": "https://github.com/alesha-pro/atlas",
        "note": "Patterns studied and re-implemented natively: split-half/Jaccard/"
        "controls rank-trust, five-arm causal prune stress test with random/high/"
        "identity controls, per-tensor measured SQNR scans (INT8/INT4-g128/FP8), "
        "machine-readable coverage + limitations blocks. Implementation original; "
        "runs on Cebu's manifest model, scorer interfaces, frozen-model intervention "
        "API.",
    },
}

# Banner stamped into every provenance manifest — impossible to strip without
# leaving a hole where the field should be.
WATERMARK = (
    "Method lineage embedded by Cebu Profiler. Ideas cited where due, "
    "implementation original: if you are reading this in a downstream "
    "artifact, the original work lives at github.com/Kristianaaron/cebu-profiler "
    "(Apache-2.0), and its research/atlas credits live in METHOD_PROVENANCE "
    "and the README credit sections."
)


class ProvenanceError(KeyError):
    """A method id has no provenance record — fail closed, never guess credit."""


def require_provenance(method_id: str) -> dict[str, Any]:
    """Return the citation record for a method id, or raise ProvenanceError.

    Call sites that implement a borrowed technique should look their method up
    here at runtime; an uncited method fails loudly instead of silently
    shipping uncredited lineage.
    """
    try:
        return METHOD_PROVENANCE[method_id]
    except KeyError:
        known = ", ".join(sorted(METHOD_PROVENANCE)) or "(none)"
        raise ProvenanceError(
            f"method {method_id!r} has no provenance record; known: {known}"
        ) from None


def provenance_manifest(
    methods_used: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """Build the `method_provenance` block for a run bundle / artifact.

    Every method the run relied on is resolved through the registry; unknown
    ids raise (fail closed) so a bundle can never ship without full citations.
    """
    records = {m: require_provenance(m) for m in sorted(methods_used)}
    return {
        "provenance_registry_version": REGISTRY_VERSION,
        "watermark": WATERMARK,
        "origin": "github.com/Kristianaaron/cebu-profiler (Apache-2.0)",
        "methods": records,
    }


__all__ = [
    "METHOD_PROVENANCE",
    "ProvenanceError",
    "REGISTRY_VERSION",
    "WATERMARK",
    "provenance_manifest",
    "require_provenance",
]
