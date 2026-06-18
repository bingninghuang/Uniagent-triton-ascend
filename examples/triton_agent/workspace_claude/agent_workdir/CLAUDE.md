# Triton Ascend KernelBench Rollout

This workspace is already a prepared KernelBench operator task. Phase 1 task
extraction from the upstream triton-op-generator workflow is complete.

Read `INSTRUCTIONS.md` first. The concrete task file is under `src/`, and the
implementation target is `src/<op_name>_triton_ascend_impl.py`.

Use only local files in this workspace. Do not fetch, clone, or search remote
repositories for skills or references.

Use the PR205 local skills only as needed:
- `triton-op-designer`
- `triton-op-coding`
- `triton-op-verifier`
- `triton-latency-optimizer` only after correctness passes

Do not invoke `triton-task-extractor` for this benchmark path. The source task
and JSON cases have already been staged in `src/`.

Do not use task-management or planning tools, including `Task`, `TaskCreate`,
`TaskUpdate`, `TaskList`, `TaskGet`, `TaskOutput`, `TaskStop`, `Workflow`,
`EnterPlanMode`, `ExitPlanMode`, or `AskUserQuestion`. Use real `Read`, `Edit`,
`Write`, `Bash`, and `Skill` calls directly.

Do not read the full `AGENTS.md` during rollout unless explicitly debugging the
upstream plugin prompt. It is kept as the PR205 source reference, not as the
runtime checklist.

Use this compact validation loop before reading the full verifier skill docs:
- AST check: `python3 .claude/skills/triton-op-verifier/scripts/validate_triton_impl.py src/<op_name>_triton_ascend_impl.py --json`
- If AST check fails, edit the implementation immediately. Do not run verify or
  benchmark, remove kernels, or claim an environment issue.
- The original `src/<op_name>.py` is the torch reference. Before NPU verify,
  create a verify directory and copy files to the module names expected by the
  upstream verifier:
  `mkdir -p output/verify && cp src/<op_name>.py output/verify/<op_name>_torch.py && cp src/<op_name>_triton_ascend_impl.py output/verify/<op_name>_triton_ascend_impl.py`.
- NPU verify/benchmark Python: `PY="${OPERATOR_PYTHON:-python3}"`. Run NPU
  verifier/benchmark commands through `bash tools/run_npu_command.sh "$PY" ...`.
  This wrapper preserves the same evaluator Python, NPU visibility, and root
  execution environment used by the old pipeline while still calling the local
  PR205 verifier scripts directly.
- Verify: `bash tools/run_npu_command.sh "$PY" .claude/skills/triton-op-verifier/scripts/verify.py --op_name <op_name> --verify_dir output/verify --triton_impl_name triton_ascend_impl --timeout 900 --output output/verify/verify_result.json`.
- Benchmark only after verify reports all correctness cases passed.

Leave upstream verifier artifacts such as `verify_result.json`,
`perf_result.json`, `summary.json`, and final generated code in the workspace.

Validation output is authoritative. Any nonzero validation command is failure:
repair and rerun it, or finish as failed. Do not write success summaries, wait
for background tasks, or claim success until correctness passes.
