# Current Task — Triton Ascend NPU Operator Implementation

This workspace contains one prepared KernelBench operator task. The agent
implements the operator as a Triton Ascend kernel and verifies it.

> NOTE: This file is a workspace reference. The runtime conversation prompt is
> injected in code by the runner (synth system + user prompt), so this file
> only needs to stay consistent with that prompt. It is archived with the
> rollout artifacts.

## Workspace layout

- Workspace root: `{workspace_dir}` (env `TRITON_WORKSPACE_DIR`). ALL file
  paths must start with `{workspace_dir}/`.
- Reference implementation: `{workspace_dir}/src/{op_name}.py`
- Implementation target (YOU write this): `{workspace_dir}/src/{op_name}_triton_ascend_impl.py`
- Test cases: `{workspace_dir}/src/{op_name}.json`
- Skills (on-demand knowledge): `{workspace_dir}/.skills/` (read via the
  `list_skills` / `read_skill` tools, NOT by direct file reads)

## Available tools

- `list_skills`: list all available skills with name/description/path.
- `read_skill`: read a skill's SKILL.md (directory+summary by default,
  `--section` for a chapter, `--full` for the entire file, `--file` for a
  reference file).
- `str_replace_editor`: view, create, and edit files.
- `run_verify`: run AST check + correctness + performance benchmark on your
  implementation. No arguments needed — call it after writing code. It writes
  `output/verify/verify_result.json` and (when all cases pass)
  `output/perf_result.json` for the speedup reward.
- `submit`: finish the task. Only call after `run_verify` reports all cases
  passed.

## Workflow (explore on demand, verify early, verify often)

1. Call `list_skills` to see available skills and their purposes.
2. Read the reference implementation (`src/{op_name}.py`) to understand the
   operator type (elementwise / reduce / matmul / attention / ...).
3. Use `read_skill` to gather knowledge ON DEMAND — read only what is relevant
   to this operator, not entire skills. Don't over-explore; each read
   consumes context and leaves less room for code and fixes.
   - `read_skill --skill hardware-specs --file references/hw-{arch}.md`:
     hardware specs for the target architecture.
   - `read_skill --skill hardware-specs --section "关键硬件约束"`: key hardware
     constraints.
   - Use `list_skills` to discover other skills (api-reference,
     ascend-extension, examples) and `read_skill` to explore them.
4. Write the implementation using `str_replace_editor create` at
   `{workspace_dir}/src/{op_name}_triton_ascend_impl.py`. The class MUST be
   named `ModelNew` (not `Model`). Mirror the reference `Model` constructor
   and `forward` signature, then launch `@triton.jit` kernels from `forward`.
5. Call `run_verify` to check AST + correctness + perf. The tool outputs a
   structured summary: `ast_valid`, `passed_cases`/`total_cases`, `pass_rate`,
   `verified_success`, `speedup_vs_torch`, `error_groups`.
6. If errors exist, fix them with `str_replace_editor str_replace`, then call
   `run_verify` again. Repeat until all cases pass.
7. Only after `run_verify` reports `verified_success: true` (all cases passed),
   call `submit` to finish.

## Rules

- The impl class name is `ModelNew`, never `Model`. Copying the reference
  `class Model` into the impl file fails AST check.
- You may take multiple turns to explore skills before writing code, but read
  only what is relevant.
- Pass tensors directly to Triton kernels. Do not use `.data_ptr()`.
- Prefer module-level `@triton.jit` kernels launched as `kernel[grid](...)`.
- NEVER read files under `tools/` — they are infrastructure scripts.
- Call `run_verify` after EVERY code change — do NOT assume correctness.
- Do NOT loop on thinking. Write code, verify, fix, verify, submit.
- Do not benchmark or tune latency manually — `run_verify` measures perf for
  the reward; focus on correctness.
