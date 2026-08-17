# Adding a compression backend / method to the Atlas control plane

All Atlas compression is **interface-agnostic**: the plane never assumes a
method family. It scores any method against the same evidence base and refuses
to claim a method "works" from another method's run. This guide walks a new
backend from registration to validated use.

## 1. Contract you implement

A backend is two pieces: a declarative `BackendRecord` (capabilities, formats,
status) and an executable `BackendAdapter` (prepare/execute/resume/validate),
optionally backed by a pinned external command.

```python
# src/model_atlas/backend/contract.py (API surface)
@dataclass
class BackendRecord:
    backend_id: str
    method_family: str            # exl3 | modelopt | llm_compressor | eval_lab | atlas | custom
    formats: tuple[str, ...]
    status: RecipeStatus          # unavailable | discovered | experimental | validated | recommended
    version: str
    declared_capabilities: tuple[str, ...]   # "pruning", "hybrid:<f1>+<f2>…"
    availability_probe: AvailabilityProbe | None
    parameters: tuple[ParameterSpec, ...]    # validated before execute
    adapter: BackendAdapter | None

class BackendAdapter(ABC):
    def prepare(self, context) -> str                    # idempotent
    def execute(self, context, handle) -> dict[str, object]
    def resume(self, context, handle) -> dict[str, object]
    def validate(self, context, outputs) -> dict[str, object]
```

## 2. Lifecycle status (never self-promote)

| Status | Meaning | How you get there |
|---|---|---|
| `unavailable` | no dependency/adapter wired | default |
| `discovered` | a working reference (binary/source) was found; rehearsal possible | call `record.note_discovered(evidence)` |
| `experimental` | results reproduced on controlled small runs | `note_experimental(evidence)` |
| `validated` | the plan's gate passed for a real output (eq/held-out/runtime) | `note_validated(evidence)` |
| `recommended` | validated on the canonical product recipe | `note_recommended(evidence)` |

Status movements are one-way on evidence. Naming a method `validated` from a
comment is prohibited; the gate must have passed in a recorded run.

## 3. Register a record

Add the record to `_builtin_records()` in `backend/registry.py`, or (preferred
for a third-party method) drop a plugin module in a plugin dir exposed to
`load_backend_plugins(root)`:

```python
# my_exotic_quant.py
from model_atlas.backend.contract import BackendRecord, CommandBackedAdapter
from model_atlas.recipe.schema import RecipeStatus

def register_backends() -> dict[str, BackendRecord]:
    return {
        "my_exotic_quant": BackendRecord(
            backend_id="my_exotic_quant", display_name="…", method_family="custom",
            formats=("my_fmt", "safetensors"), status=RecipeStatus.DISCOVERED,
            version="unpinned", declared_capabilities=(),
            availability_probe=lambda: (False, None, "not wired (fail closed)"),
            adapter=CommandBackedAdapter(backend_id="my_exotic_quant"),  # no cmd yet
        ),
    }
```

`register_backends()` modules override the defaults by backend_id.

## 4. Make it honest + fail-closed

* Until a real pinned command/module is wired, keep `availability_probe` a
  `_probe_never` (or `_probe_module`/`command_exists`, returning `False` when
  absent). An available probe that lies is worse than unavailable.
* `CommandBackedAdapter` only *drives* a genuine dependency. Give it a
  `run_cmd` only when you have integration + tests that prove it. Without one,
  `execute` raises `BackendUnavailable` — never fabricate output.
* Executable stages require an **exact resolved version**. Set a concrete
  `version` on the record; a stage pinned `unpinned` is dry-run-only and
  non-executable (the compiler and the engine both enforce this).
* Declare hardware axes separately: `architectures` (model family),
  `compute_archs` (GPU/CPU compute), `topologies` (node layout),
  `runtime_compat` (serving runtime). Never compare glm-5.2 to gb10-sm121 or
  vllm-modelopt to sm121.
* Set `produces_derivative=True` ONLY for an adapter that produces a REAL
  derivative checkpoint. A compression stage (quantization / refinement /
  residual / conditioning) requires the adapter AND record to both report
  `produces_derivative=True` **and** a staged non-evidence weight file; a
  probe/analysis-only backend (e.g. `atlas_quant_probe`) can never serve one.
* Declare hybrid support **only** for the exact format combination the SELECTED
  (available + version-resolved) FORMAT-PRODUCING backend can run and has
  tested, using `hybrid:<sorted,+-joined formats>`. A declaration on an
  unrelated, unavailable, or non-producing record never authorizes a recipe
  hybrid.
* Declare `pruning` only for a real TENP/FlexMoE structural-pruning backend.
  Pruning recipes then require `no_pruning=false` + `allow_pruning_capability`
  + a capability-declaring backend for each pruning stage.

## 5. Validate parameter schema

`ParameterSpec` entries are checked before `execute`. Provide `type`, `enum`,
`minimum`/`maximum` (e.g. `bpw` 2–6 for EXL3). Mismatched stage parameters fail
at compile/execute time, not silently.

## 6. Test it

* Probe truth: an uninstalled dependency reports `unavailable`/fail-closed.
* A stage pinned to the backend refuses to run and the job goes
  `FAILED_TERMINAL` (`[fail-closed]`).
* Deterministic replay: same inputs → same run dir, same outputs, no re-fire.
* Resolve your method's `evidence_kind` from its declared policy ceiling.

## 7. Promotion rules

Publishing (`PublishRule`) requires all stages validated, `no_pruning`
per-capability, `MEASURED` evidence minimum, runtime benchmarked, and
repair/validated. Those are per-recipe; the plane never auto-publishes.

## 8. Contribute a *method*, not just a backend (the catalog seam)

A backend alone is not selectable — the recommendation layer only considers
methods that have a `MethodSpec` in the catalog (`METHOD_CATALOG`). Since the
catalog is core, new methods are contributed as **plugins** through the same
plugin-dir mechanism, but with a `register_methods()` entry point.

Set `ATLAS_METHOD_PLUGIN_DIR` to a directory of `*.py` modules. Each module may
expose:

```python
# my_method.py  (one plugin = backend + method)
from model_atlas.backend.contract import BackendRecord, CommandBackedAdapter
from model_atlas.recommend.policy import (
    CompressionIntent, MethodFamily, MethodSpec, StageEffectClass,
)
from model_atlas.recipe.schema import RecipeStatus

def register_backends() -> dict[str, BackendRecord]:
    return {
        "my_method": BackendRecord(
            backend_id="my_method", display_name="…", method_family="custom",
            formats=("safetensors",), status=RecipeStatus.DISCOVERED,
            version="unpinned", declared_capabilities=("pruning",),
            availability_probe=lambda: (False, None, "not wired (fail closed)"),
            adapter=CommandBackedAdapter(backend_id="my_method"),
        ),
    }

def register_methods() -> dict[str, MethodSpec]:
    return {
        "my-method": MethodSpec(
            "my-method", 900, MethodFamily.PRUNING, "my_method",
            ("channel_saliency",), ("width-slice",),
            (StageEffectClass.PRUNING,), (CompressionIntent.PRUNE_ONLY,),
            "down", routing_dependent=False, provenance_ids=("my-method-v1",),
        ),
    }
```

Notes on the seam:

* `MethodSpec(method, priority, family, backend_id, evidence_stages,
  recipe_stage_ids, effect_classes, compatible_intents, memory_direction,
  routing_dependent=False, planning_only=False, provenance_ids=())`.
* `backend_id` must match a registered (builtin or plugin) backend for the
  method to be executable.
* Plugin methods are appended in **deterministic** order (sorted module, then
  method id) and folded into `method_catalog_digest`, so two processes loading
  the same plugin dir derive the same catalog and recommendation identity.
* With `ATLAS_METHOD_PLUGIN_DIR` unset this is a no-op: the catalog is the
  builtin set and the digest is unchanged — a third-party method never changes
  core behavior until an operator opts in.
* Backends and methods are independent contributions. A plugin may add a method
  without a backend (it stays blocked until a backend exists), or a backend
  without a new method (it powers an existing catalog method).
