# Kernel Evidence Bridge

Status: Milestone 0 Atlas integration, 2026-08-30

## Purpose and boundary

Atlas owns model profiling, representation choices, candidate evidence, and
promotion gates. Runtime repositories own CUDA kernels, kernel builds, and
benchmarks. The bridge between them is a versioned benchmark receipt.

This split keeps runtime optimization connected to the profiler without moving
CUDA infrastructure into Atlas or allowing unverified speed estimates to leak
onto the candidate decision surface.

The first runtime producer is the Atlas Runtime Compiler / SM121 Kernel Lab in
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`. It reuses ExLlamaV3's K3 MCG ABI:

- ABI name: `exllamav3.exl3.mcg`
- ABI version: `1`
- matrix convention: `A[M,K] @ W[K,N]`
- packed trellis: `int16[K/16,N/16,16*bits]`
- scales: `suh=float16[K]`, `svh=float16[N]`
- MCG marker: `int32(0xCBAC1FED)`
- representative Milestone 0 layout: integer K3 (3 trellis bits/weight)

No incompatible Atlas-specific EXL3 format is introduced.

## What landed

`model_atlas.kernels` provides four pieces:

1. `atlas.kernel-benchmark/v1`, a strict receipt schema that pins hardware,
   representation ABI, exact workload, backend commit, metrics, and provenance.
2. A deterministic catalog importer, including a conservative adapter for the
   existing Kernel Lab `schema_version: 1` receipt.
3. A kernel oracle keyed by hardware, ABI, phase, M tuning bucket, exact M/N/K,
   tensor parallelism, grouped-MoE state, and optional backend commit.
4. A fail-closed execution-manifest builder. Every tensor/group requirement
   must resolve to an exact-shape, measured, direct-packed receipt.

Candidate selection can consume `CandidateKernelAssessment`, whose runtime
objective is the sum of exact measured required-kernel latencies. It does not
invent end-to-end tokens/second from component timings; a model-level speed
claim still requires its own measured serving run.

The Atlas dashboard has a **Runtime Kernels** panel. It distinguishes:

- `measured`: eligible for speed ranking and manifest binding;
- `estimated`: nearby M-bucket evidence, visible but ineligible;
- `compatibility_only`: correctness/reference evidence, visible but ineligible;
- `failed` / `unsupported` / `unmeasured`: explicit blockers.

## Eligibility rules

A receipt may affect runtime selection only when all of these are true:

- evidence kind is `measured`;
- the run passed;
- device name and compute capability are present;
- CUDA and driver versions are pinned and exactly matched;
- the backend commit is pinned;
- the producer uses the canonical receipt schema and pins its runtime source commit;
- the path is `direct_packed`;
- reconstruction is fused with compute;
- no full dequantized weight tensor was materialized;
- positive latency exists at the exact requested M/N/K shape.

An observation from another M value in the same tuning bucket can be returned
as an estimate, but it cannot satisfy an execution manifest. This prevents a
measurement at M=2 from silently becoming a claimed measurement at M=4.

The existing CPU oracle receipt therefore imports as `compatibility_only`. It
validates packed reconstruction correctness but cannot change Atlas speed
rankings. The GB10 compatibility checks also do not become K3 latency evidence.

## Commands

Normalize one or more runtime receipts:

```text
model-atlas kernel-import receipt.json --out kernel-evidence.json
```

Render the profiler with imported evidence:

```text
model-atlas dashboard \
  --kernel-receipt kernel-evidence.json \
  --out atlas_dashboard.html
```

Ask the oracle for an exact direct-EXL3 decode shape:

```text
model-atlas kernel-query \
  --catalog kernel-evidence.json \
  --device "NVIDIA GB10" --compute-capability 12.1 \
  --cuda-version 13.0 --driver-version DRIVER_VERSION \
  --format exl3 --abi exllamav3.exl3.mcg --abi-version 1 \
  --phase decode --m 2 --n 6144 --k 1024 --tp 2
```

`kernel-query` exits with code 2 when the result is not execution-eligible.
`kernel-manifest` accepts a JSON request containing `candidate_id` and a list of
`{tensor_or_group, query}` requirements. It writes a manifest only if every
requirement resolves to exact measured evidence.

## Milestone 0 evidence and decision

Measured so far:

- deterministic K3 reference pack/unpack and streaming matmul correctness;
- portable CPU oracle maximum absolute error about `1.99e-7` for the small
  `M=2,N=128,K=128` identity case;
- official ExLlamaV3 CUDA packer byte-equivalence on GB10;
- exact MCG decoder equivalence over all 65,536 tested states on GB10.

Not measured:

- isolated K3 latency, effective bandwidth, or reconstruction overhead for the
  real GLM-5.2 / Hy4 expert shapes on SM121.

Both available Sparks were occupied by an unrelated two-node service, so the
current evidence remains compatibility-only. Atlas must continue to show the
direct K3 path as unmeasured for performance.

## Next measured action

Run the checked-in Kernel Lab TP2 decode sweep at M=1/2/4/8 for the real expert
projections during an isolated GB10 window, then emit canonical
`atlas.kernel-benchmark/v1` receipts with the runtime repository commit and
hardware identity.

- Reconstruction-bound: optimize the direct decode schedule before grouped MoE.
- Packed-memory-bound with reconstruction hidden: move next to grouped MoE and
  route packing.
- Near dense-FP16 latency: add tactic/autotuning coverage, then grouped MoE.
- Unable to beat reconstruct-then-matmul: revisit the representation.

Mixed EXL3/NVFP4 dispatch, grouped MoE, prefill, vLLM/SGLang adapters, and
autotuning remain explicit future consumers of the same receipt/oracle contract;
they are not implemented in this milestone.

## Upstream references

- [ExLlamaV3 format and kernels](https://github.com/turboderp-org/exllamav3/tree/c5d9c657966ffeeaa9353f0cc899f18629da4a13)
- [QTIP paper](https://arxiv.org/abs/2406.11235)
- [CUTLASS SM12x grouped NVFP4 example](https://github.com/NVIDIA/cutlass/blob/dc45f979ae336a235da1676b311f35efeb30149a/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)
- [CUTLASS CuTe DSL Blackwell persistent GEMM](https://github.com/NVIDIA/cutlass/blob/dc45f979ae336a235da1676b311f35efeb30149a/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py)
- [FlashInfer SM12x W4A16 MoE kernel](https://github.com/flashinfer-ai/flashinfer/blob/231f70828dfe93f5bbba7f0360a64435a7a846be/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py)
- [TensorRT-LLM low-M dispatcher](https://github.com/NVIDIA/TensorRT-LLM/blob/6c1ce33e7fbb1730c3d85150e6ac73ebc1b76deb/tensorrt_llm/_torch/modules/low_m_gemm.py)
- [GLM-5.2 model config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [Hy4-preview model config](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)
