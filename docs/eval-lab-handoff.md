# Candidate-only Eval Lab handoff

Atlas emits a content-addressed `EvalLabRequest` for the pinned Eval Lab
revision `5ee2f7cc33627b6259c0b10100d84932e676f36c`. The adapter resolves Git
HEAD from filesystem metadata, verifies the suite and task-tree bytes, and
only emits the real `eval-lab run suite ... --json` argv. It never starts an
endpoint or evaluation process and passes no nonexistent revision/request
options to the CLI.

The request binds the candidate artifact bytes, a non-secret endpoint
configuration identity and credential-free endpoint URL, absolute suite,
tasks, runs and database paths, the frozen task suite and task definitions, held-out
corpus sample IDs, tokenizer and template hashes, and deterministic generation
parameters. Its ID excludes timestamps. Calibration/unset partitions,
calibration/evaluation sample overlap, and tasks outside the pinned suite are
rejected before handoff.

The pinned CLI does not accept request-wide sampling, seed, or timeout options.
Atlas checks its current effective defaults (no seed, temperature 0,
max-tokens 4096) and the task timeout from pinned YAML, rejecting mismatches.
However, because the CLI cannot bind the complete request contract and the
direct runner does not enforce the task timeout around the endpoint call, the
handoff remains non-executable with `request_parameters_not_cli_bound` until
Eval Lab closes that seam. Task YAML not explicitly marked
`held_out_evaluation` adds a second typed blocker.

Output layout is `<root>/<request-id>/{request.json,candidate-task-report.json,result.json}`.
The result envelope binds the report path and SHA-256. Atlas re-hashes and
parses the returned report before accepting it.

This bridge is candidate-only: it may report task scores and runtime
performance, but `teacher_relative=false`, `token_kld=null`, and `cka=null` are
enforced. It records explicit blockers for the missing BF16 teacher, full
logits, and hidden activations. Genuine teacher-relative KLD/CKA remains a
separate evaluation requiring those measured inputs.
