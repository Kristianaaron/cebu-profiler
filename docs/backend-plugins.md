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
* Declare hybrid support **only** for the exact format combination you can run
  and have tested, using `hybrid:<sorted,+-joined formats>` (see
  `modelopt_nvfp4` declaring `hybrid:fp8_e4m3+modelopt_nvfp4`).
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
