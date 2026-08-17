# Candidate-only Eval Lab handoff

Atlas emits a content-addressed `EvalLabRequest` for the pinned Eval Lab
revision `a20da6c6b9cbf872f7c083bffe66afde40c2c8f2`. The adapter resolves Git
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

The pinned CLI now binds an explicit seed, temperature, maximum output tokens,
HTTP request timeout, task deadline, and `--require-held-out`. Atlas proves the
selected task order, direct runner, held-out partition, and identical task
timeout before emitting an executable argv. A missing seed, non-direct task,
timeout mismatch, or task YAML not explicitly marked `held_out_evaluation`
fails closed before execution.

Output layout is `<root>/<request-id>/{request.json,candidate-task-report.json,result.json}`.
The result envelope binds the report path and SHA-256. Atlas re-hashes and
parses the returned report before accepting it.

This bridge is candidate-only: it may report task scores and runtime
performance, but `teacher_relative=false`, `token_kld=null`, and `cka=null` are
enforced. It records explicit blockers for the missing BF16 teacher, full
logits, and hidden activations. Genuine teacher-relative KLD/CKA remains a
separate evaluation requiring those measured inputs.
