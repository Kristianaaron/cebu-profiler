# Method Catalog

The **MethodSpec catalog** in `src/model_atlas/recommend/policy.py` is a single
fail-closed authority for which Atlas compression methods exist, what they do,
and how they may be combined. It is **frozen/versioned**: changing it alters the
catalog digest, which flows into recommendation ids, invalidating previously
issued recommendations. Add or modify a `MethodSpec` only as an explicit,
versioned contract change.

- Policy version: `policy-v2-catalog`
- Catalog version: `1`

## Current methods

All backends are **declared**, not verified to have run. A `planning_only`
method runs in-repo and produces a plan; a non-planning (derivative-producing)
method additionally requires an available, version-pinned, derivative-producing
backend before it can execute (see *Planning-only vs derivative-producing*).

| method | family | backend | effect classes | memory dir | routing-dependent | planning-only | status |
|---|---|---|---|---|---|---|---|
| `teacher-identity` | analysis | `atlas_analysis_v3` | identity | same | no | yes | declared |
| `calibration` | analysis | `atlas_analysis_v3` | profiling | same | no | yes | declared |
| `sensitivity` | analysis | `atlas_analysis_v3` | sensitivity | down | no | yes | declared |
| `bit-allocation` | allocation | `atlas_analysis_v3` | allocation | down | no | yes | declared |
| `nvfp4-substitute` | quantization | `modelopt_nvfp4` | quantization | down | yes | no | declared |
| `kv-optimization` | kv | `atlas_analysis_v3` | kv | down | no | yes | declared |
| `exl3-primary` | quantization | `exl3` | quantization | down | yes | no | declared |
| `llm-compressor` | quantization | `llm_compressor` | quantization | down | yes | no | declared |
| `modelopt-nvfp4` | quantization | `modelopt_nvfp4` | quantization | down | yes | no | declared |

Compatible intents: every method accepts the `quantize_only`, `hybrid`, and
`custom` intents; **none** accepts `prune_only`.

## Planning-only vs derivative-producing

The catalog distinguishes two roles:

- **Planning-only** (`planning_only=True`, families `analysis`/`allocation`/
  `kv` here): run inside the repo on evidence already present in the profile.
  They produce plans/decisions, never a modified model derivative. They are not
  gated on backend availability and never claim a `QUANTIZATION` or `PRUNING`
  effect.
- **Derivative-producing** (quantization family here): require a registered,
  available, version-pinned backend that actually produces a derivative, plus —
  because they operate on router-indexed expert tensors — a PASSED
  routing-consistency gate. Until those conditions hold they are blocked, never
  recommended for execution.

## Fail-closed unknowns

`method_spec(name)` raises `KeyError` for any name that is not an explicit
catalog entry. An unknown name — including pruning-looking strings that have no
catalogued `MethodSpec` (e.g. `tenp-pruning-not-catalogued`) — never gains a
family or an implicit effect. It is simply unknown.

## Effects vs intents

- **Effect classes** describe what a stage actually changes in the model
  (`IDENTITY`, `PROFILING`, `SENSITIVITY`, `ALLOCATION`, `QUANTIZATION`, `KV`,
  `PRUNING`, …). A method must declare the effect it performs.
- **Compatible intents** (`quantize_only`, `prune_only`, `hybrid`, `custom`)
  describe the user's overall compression goal a method may serve.

These are never conflated. Currently **no** entry declares a `PRUNING` effect or
accepts the `prune_only` intent; `hybrid` is a visible future intent that today
has no executable pruning `MethodSpec`.

## Pruning / hybrid: visible future intents

`StageEffectClass.PRUNING` exists and `prune_only`/`hybrid` are valid intents,
but the frozen v1 catalog contains **no executable pruning `MethodSpec`** — no
entry is in the `PRUNING` family, none declares a `PRUNING` effect, and none
accepts `prune_only`. Pruning stays immutable-by-default (`no_pruning=True`)
and cannot be enabled without a separately registered, available, version-pinned
pruning-capable backend. Neither pruning nor hybrid has been run by any backend.

## Catalog digest

`method_catalog_digest()` is a SHA-256 hex digest of canonical JSON over the
sorted per-method `identity_dict` payload plus `catalog_version`. It is:

- 64 lowercase hex characters, stable across repeated calls;
- independent of the source tuple's ordering (sorted by method id);
- sensitive to every execution-identity field (`method`, `family`,
  `backend_id`, `evidence_stages`, `recipe_stage_ids`, `effect_classes`,
  `compatible_intents`, `memory_direction`, `routing_dependent`,
  `planning_only`, `provenance_ids`).

The digest is embedded in every recommendation id, so a catalog change — an
added, removed, or edited `MethodSpec` — invalidates all prior recommendation
ids for the same profile/target. Tests in `tests/unit/test_method_catalog.py`
reconstruct the canonical payload from an arbitrary ordering and from mutated
specs to verify this property.
