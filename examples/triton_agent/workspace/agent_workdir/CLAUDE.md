# Triton Ascend KernelBench Rollout

This workspace contains one prepared KernelBench operator task. Use
`INSTRUCTIONS.md` as the source of truth.

> NOTE: This rollout does NOT use the Claude Code CLI. The agent drives a
> standard OpenAI tool-call ReAct loop with five tools:
> `str_replace_editor`, `list_skills`, `read_skill`, `run_verify`, `submit`.
> The Claude-Code-style `Read`/`Write`/`Edit`/`Bash`/`Skill` tools do not apply.

## Working set

Read only the small working set by default:
- `INSTRUCTIONS.md` (workflow + paths)
- `src/<op_name>.py` (reference implementation)
- `src/<op_name>.json` (test cases) - read implicitly by `run_verify`
- `output/verify/verify_result.json` / `output/perf_result.json` (written by
  `run_verify`; do not read full raw logs unless the compact summary is missing)

Do not explore the workspace. Avoid broad file discovery such as `tools/**`,
`.claude/**`, `.skills/**` (use the `list_skills`/`read_skill` tools instead),
or `**/*`. Do not read verifier source, full `verify_result.raw.log`, or full
skill/reference docs unless the compact `run_verify` summary is missing and the
exact file is necessary.

## Workflow

1. `list_skills` then `read_skill` on demand for the relevant operator type.
2. `str_replace_editor create` the implementation at
   `src/<op_name>_triton_ascend_impl.py`. The public class MUST be `ModelNew`,
   not `Model`.
3. `run_verify` (AST + correctness + perf). Repair from the compact summary
   with the smallest targeted `str_replace_editor str_replace`.
4. Repeat `run_verify` until `verified_success: true` (all cases pass).
5. `submit`.

## Hard constraints

- Keep reasoning and prose short. No markdown reports or status files.
- Do not write literal tool-call markup - invoke the real tool.
- Do not run benchmark or latency tools manually; `run_verify` measures perf.
- Do not edit, replace, or wrap `tools/run_npu_command.sh` or files under
  `tools/triton-op-verifier/`; they are trusted read-only infrastructure.
- Preserve partial best attempts; avoid broad rewrites unless the structure is
  clearly impossible.
- If the same static/compiler error repeats twice, simplify the kernel shape
  handling instead of adding another special case.
- `run_verify` output is authoritative. AST success is only a precheck;
  correctness requires `passed_cases == total_cases`.

## Common rejected patterns

- Passing `.data_ptr()` to kernels.
- Using `tl.to(...)` instead of `value.to(tl.float32)`.
- Tuning Ascend-rejected kwargs such as `num_warps`, `num_ctas`, or
  `num_stages`.
- Naming the impl class `Model` instead of `ModelNew`.
- Host-side core computation in `ModelNew.forward()`.
- Runtime Python control flow inside `@triton.jit` kernels.