# Method lifecycle + evidence discipline

The Atlas records how a method was produced and what gate it reached. This
drives which results a UI may advertise and which may be published. The ladder
is one-way; nothing regresses silently, and nothing upgrades silently.

## Lifecycle

```
unavailable ──(working reference found)──▶ discovered
   discovered ──(controlled small-run reproduction)──▶ experimental
        experimental ──(plan gate passed on a real output)──▶ validated
             validated ──(canonical product recipe passed)──▶ recommended
```

* `unavailable` — no dependency/adapter/probe; fail-closed default.
* `discovered` — the reference exists (a binary, an import, a pinned command).
  Rehearsal allowed, claims withheld.
* `experimental` — results were reproduced on controlled runs; not yet
  deployable.
* `validated` — the plan's validation gate(s) passed on a **real** materialized
  output (including eq-control / identity-control where declared).
* `recommended` — validated on the canonical no-pruning GLM-5.2 recipe path.

## Evidence kinds

`predicted < estimated < measured < causally_tested`. Recap:

* `predicted` — planned/allocation without execution.
* `estimated` — offline surrogate over recorded stats.
* `measured` — materialized artifact + held-out + runtime benchmark.
* `causally_tested` — an intervention/ablation designed as causal.

Rules:

1. Recorded kind is `min(stage policy, reported kind)`. Never upgrade silently.
2. A `MEASURED` claim requires the full chain; `FrontierRecorder` in the repo
   already encodes this for candidates.
3. Resource estimates are `predicted` unless following a measured profile.

## UI/consumer contract

Any consumer (dashboard, agent, report) may show `status`, `evidence_kind`,
`compiles`, and `issues` as recorded; it must **not** infer `recommended` from a
`validated` backend, nor `measured` from a `predicted` row. The `capabilities`
endpoint returns these truthfully.
