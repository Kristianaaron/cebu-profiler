# Candidate-only Eval Lab handoff

Atlas emits a content-addressed `EvalLabRequest` for the pinned Eval Lab
revision `5ee2f7cc33627b6259c0b10100d84932e676f36c`. The adapter only emits the
reviewable argv; it never starts an endpoint or evaluation process.

The request binds the candidate artifact bytes, a non-secret endpoint
configuration identity, the frozen task suite and task definitions, held-out
corpus sample IDs, tokenizer and template hashes, and deterministic generation
parameters. Its ID excludes timestamps. Calibration/unset partitions,
calibration/evaluation sample overlap, and tasks outside the pinned suite are
rejected before handoff.

Output layout is `<root>/<request-id>/{request.json,candidate-task-report.json,result.json}`.
The result envelope binds the report path and SHA-256. Atlas re-hashes and
parses the returned report before accepting it.

This bridge is candidate-only: it may report task scores and runtime
performance, but `teacher_relative=false`, `token_kld=null`, and `cka=null` are
enforced. It records explicit blockers for the missing BF16 teacher, full
logits, and hidden activations. Genuine teacher-relative KLD/CKA remains a
separate evaluation requiring those measured inputs.
