#!/usr/bin/env python3
"""Run standalone OpenHands KernelBench baselines.

This script is intentionally separate from the RL training entrypoints. It
creates an isolated workspace per (baseline, operator), runs exactly one
OpenHands container/session for that pair, and records metrics for offline
baseline analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


BASELINES = ("current", "ascend_full", "generator_verifier")

_ASCEND_FULL_REFERENCE_BUNDLE: dict[str, tuple[str, ...]] = {
    "triton-op-designer": (
        "references/sketch-design.md",
        "references/cases/elemwise-broadcast-2d.md",
        "references/cases/elemwise-broadcast-3d.md",
        "references/cases/matmul-swizzle2d.md",
        "references/cases/reduction-amax-large.md",
        "references/cases/reduction-sum-large.md",
        "references/cases/index-put.md",
    ),
    "triton-op-coding": (
        "references/triton-ascend-fundamentals.md",
        "references/triton-ascend-examples.md",
        "references/triton-ascend-elementwise.md",
        "references/triton-ascend-matmul.md",
        "references/triton-ascend-reduce.md",
        "references/triton-ascend-attention.md",
        "references/triton-ascend-sort-select.md",
        "references/triton-ascend-interpolate.md",
    ),
    "npu-arch": (
        "references/npu-arch-guide-triton.md",
        "references/npu-hardware-params.md",
    ),
    "triton-latency-optimizer": (
        "references/constexpr_parameters.md",
        "references/load-order.md",
        "references/checklist.md",
        "references/tiling_optimization.md",
    ),
}

_ASCEND_FULL_REFERENCE_MARKER = "<!-- ASCEND_FULL_INLINE_REFERENCES -->"


CURRENT_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`
Suggested operator class from static pre-scan: {op_class_hint}
Immediate action: your first file operation should create
`src/{op_name}_triton_ascend_impl.py` with complete executable Python code.
Do not write analysis or code in assistant messages. Use tools directly.
Do not inspect `tools/` or `tools/operator_pipeline.sh`.

Quick implementation rules, to use silently:
- Elementwise/broadcast: one vectorized tile kernel; `diag(A) @ B` is
  `C[i,j] = A[i] * B[i,j]`.
- Matmul/linear: tiled `tl.dot`; pass tensors directly, never `.data_ptr()`;
  derive logical M/N/K after any reference transpose, and do not cast dtype
  unless the reference does. For 2D launch grids use `program_id(0)` for M
  tiles and `program_id(1)` for N tiles; for 1D swizzled grids launch the
  product of tile counts. Do not mix these patterns.
- Reductions: tile reduction, or partial-reduce plus final kernel for large axes.
- Layout/slice/transpose: write exact output indexing in Triton.
- On errors: read `metrics.json`; `int64 tl.load` means remove `.data_ptr()`;
  block mask pointer errors mean make the pointer expression block-shaped;
  `ub overflow` means reduce tile sizes.

Reference PyTorch code:
```python
{task_code}
```

Requirements:
1. Create `src/{op_name}_triton_ascend_impl.py`.
2. Define `ModelNew` with the same constructor and forward signature.
3. Keep core computation in module-scope Triton kernels.
4. Run:
   ```bash
   bash tools/operator_pipeline.sh --op_name {op_name}
   ```
5. Read `metrics.json` after every run. If success, do not edit again; run:
   ```bash
   cp -f src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
   cp -f metrics.json metrics_best.json
   ```
   Then stop. If a finish tool is not listed in Tools Available, end with a
   brief final assistant message instead of calling a tool.
"""


ASCEND_FULL_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`
Suggested operator class from static pre-scan: {op_class_hint}

Use the workspace `AGENTS.md` and local `.agents/skills` as the primary
instructions. Follow the AscendOpGenAgent-style Triton full-flow in this
single OpenHands session: designer sketch, generator implementation, verifier
pipeline, targeted repair, save best on success.

Initial phase keyword: sketch.

Use only tools listed in Tools Available. If a native skill-invocation tool is
actually listed, it may be used. Otherwise, use the skill manifest in
`AGENTS.md` and `file_editor.view` to read only the skill for the current
phase:

- Sketch phase: read `triton-op-designer/SKILL.md` and the required designer refs
  named by that skill for this operator before writing `src/sketch.txt`.
- Generator phase: after `src/sketch.txt` exists, read
  `triton-op-coding/SKILL.md` and the required generator refs named by that
  skill before writing the implementation.
- Verifier phase: after the implementation exists, read
  `triton-op-verifier/SKILL.md`; run the pipeline and read `metrics.json`.

Do not read unrelated skills or all references up front.

Suggested designer refs from static pre-scan:
{designer_ref_hint}

Suggested generator refs from static pre-scan:
{generator_ref_hint}

If these suggestions disagree with executable `Model.forward()`, `get_inputs()`
or `get_input_groups()`, or `get_init_inputs()`, trust the executable code and
select the corrected refs.

Run:
```bash
bash tools/operator_pipeline.sh --op_name {op_name}
```

Reference PyTorch code:
```python
{task_code}
```
"""


ASCEND_FULL_AGENTS_MD = """# AscendOpGenAgent Triton Full-Flow Benchmark

This workspace benchmarks a single-session OpenHands approximation of the
Just-it/AscendOpGenAgent Triton operator generation workflow. Treat local
workspace instructions and `.agents/skills/*` as the source of behavior; the
user request only supplies the operator, target architecture, output file, and
reference code.

Use only tools listed in Tools Available. If a native skill-invocation tool is
actually listed, it may be used. Otherwise, use the local skill manifest below
and file tools to read the relevant local `SKILL.md` file for the current phase
only. Read only selected `references/...`, `scripts/...`, or `assets/...` files
named by that skill and relevant to the current operator.
If a skill marks refs as required for the current operator pattern, open those
exact refs before creating or editing the corresponding artifact.

## Local Skill Manifest

Skill bodies are intentionally not injected into the prompt. Load them through
file tools only when the phase asks for them:

- `triton-op-designer`
  - Location: `.agents/skills/triton-op-designer/SKILL.md`
  - Use only in Phase 1 to design `src/sketch.txt`.
- `triton-op-coding`
  - Location: `.agents/skills/triton-op-coding/SKILL.md`
  - Use only in Phase 2 after `src/sketch.txt` exists.
- `triton-op-verifier`
  - Location: `.agents/skills/triton-op-verifier/SKILL.md`
  - Use only in Phase 3 and repair loops after the implementation exists.

Do not load `triton-latency-optimizer` in this single-session baseline unless the user
explicitly requests a post-success optimizer stage.

## Contract

- Backend: Ascend NPU, target architecture from the task.
- Suggested operator class from static pre-scan: `{op_class_hint}`. Treat this
  as a hint only; executable reference code is authoritative.
- DSL: Triton / Triton-Ascend.
- Output file: `src/{op_name}_triton_ascend_impl.py`.
- Required class: `ModelNew(nn.Module)` with constructor and `forward`
  compatible with the reference `Model`.
- Define `@triton.jit` kernels at module scope.
- Source of truth: executable `Model.forward`, `get_inputs()` or
  `get_input_groups()`, and
  `get_init_inputs()`. Ignore stale comments/docstrings if they disagree.
- Keep target computation in Triton kernels; no PyTorch fallback compute in
  `ModelNew.forward`.
- Pass tensor objects directly to kernels; never pass `.data_ptr()` or integer
  pointer values.
- Do not modify `tools/`, install packages, download files, add test-only code,
  print spam, placeholders, `pass`, or `if __name__ == "__main__"` blocks.

Allowed support operations in `forward`: allocation, shape/dtype/device
inspection, semantics-preserving reshape, view, transpose, permute, expand, and
contiguous calls when they only prepare layout for Triton kernels, plus Triton
launches.

## Full-Flow Stages

Follow these stages through tools. Keep chat messages short; put code in file
edits, not assistant text.

### Phase 0: Task Confirmation

Extract the operator name, output path, target architecture, init inputs,
forward inputs, tensor shapes, dtypes, layouts, reductions, broadcasts, and
reference output contract from the executable code. Confirm or correct the
static pre-scan class hint before opening type-specific references.

### Phase 1: Designer / Sketch

Load or read `triton-op-designer` using the visible skill mechanism, then read only
the sketch or case references needed by this operator from the skill location.
First classify the operator from the executable reference as one of:
elementwise/broadcast, matmul/linear/batched-matmul, convolution/correlation,
pooling/stencil, interpolate/resample, normalization, softmax/logsoftmax, reduction/statistical,
scan/prefix/cumulative, layout/index/slice/concat/transpose,
gather/scatter/embedding, sort/topk/arg, loss/distance, attention, or fused.
Open `sketch-design.md` plus only the case refs matching that class. Examples:
matmul -> `cases/matmul-swizzle2d.md`; broadcast elementwise ->
`cases/elemwise-broadcast-2d.md` or `cases/elemwise-broadcast-3d.md`; fused
elementwise plus scalar reduction -> `cases/reduction-sum-fused.md` and a
size-matched reduction case; indexing -> `cases/index-put.md` or
`cases/index-histogram.md`.
Do not read `triton-op-coding` or `triton-op-verifier` in this phase.
Create a compact `src/sketch.txt` when useful. The sketch is strategy only, not
executable Python. Include:

- op class: elementwise, matmul, reduction, pooling, layout/index, or fused
- exact indexing formula and output shape
- grid mapping and tile/block strategy
- dtype/accumulation policy
- correctness risks and likely compiler risks

### Phase 2: Generator

After `src/sketch.txt` exists, load or read `triton-op-coding` using the visible
skill mechanism, then read only the target architecture and op-type docs needed
for this operator from the skill location. Generate the implementation file.
Prefer the simplest correct Triton implementation first. Do not read verifier
references in this phase.
Always open the target hardware ref and `triton-ascend-fundamentals.md`. Then
open only the op-type refs matching the sketch:
elementwise/broadcast -> `triton-ascend-elementwise.md`;
matmul/linear/batched-matmul -> `triton-ascend-matmul.md`;
convolution/correlation -> fundamentals plus matmul and/or reduce refs,
depending on whether the sketch uses im2col/tiled dot or direct stencil loops;
reduction/statistical -> `triton-ascend-reduce.md`;
normalization, softmax/logsoftmax, loss/distance, and fused elementwise plus
reduction -> both elementwise and reduce refs;
scan/prefix/cumulative, pooling/stencil, interpolate/resample, sort/topk/arg, gather/scatter/
embedding, and layout/index -> fundamentals plus the closest examples/cases
named by the skill;
attention -> `triton-ascend-attention.md`; layout/index uses fundamentals plus
any matching examples/cases named by the skill. If the sketch classification
does not match the reference code, re-open the reference task and correct the
classification before writing kernels.

Static pre-scan suggested refs for this task:

Designer:
{designer_ref_hint}

Generator:
{generator_ref_hint}

### Phase 3: Verifier

Load or read `triton-op-verifier` using the visible skill mechanism. Run exactly:

```bash
bash tools/operator_pipeline.sh --op_name {op_name}
```

After every run, read `metrics.json`. If it has `error_truncated` or an
`error_file`, read the referenced file before editing.

### Phase 4: Repair Loop

Patch the smallest failing block and rerun. Avoid whole-file replacement after
the initial implementation create. If the failure is environment-only, stop and
record the metrics error.

### Phase 5: Success Freeze

When the pipeline succeeds or `metrics.json` has `"success": true`, save best
files and stop:

```bash
cp -f src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
cp -f metrics.json metrics_best.json
```

Low speedup is still success in this single-session baseline. Do not keep
tuning after the first successful benchmark unless the task explicitly asks for
a post-success optimizer stage.
Do not call a finish tool unless it is explicitly listed in Tools Available.

## Triton-Ascend Rules

- `tl.program_id` axes are only `0`, `1`, and `2`. Flatten higher-rank output
  spaces into one or more linear program ids and recover coordinates inside the
  kernel.
- Prefer vectorized output tiles. For rank > 3 tensors, flatten
  `N*C*D*H*W`-style domains rather than launching Python loops per dimension.
- Avoid Python chained boolean expressions inside `@triton.jit`. Build tensor
  masks with bitwise operators: `mask = (cond1) & (cond2) & (cond3)`.
- Avoid Python `if` for element validity in kernels. Use `tl.load(...,
  mask=mask, other=...)`, `tl.where`, and `tl.maximum`/`tl.minimum`.
- If pointer/mask block shapes conflict, make pointer expressions block-shaped.
- If `Unsupported ptr type ... int64 in tl.load` appears, remove `.data_ptr()`
  and pass tensors directly.
- If compile reports `ub overflow`, reduce tile sizes first.
- Do not run ad-hoc `python` or `python3` tests for operator debugging. The
  default runner Python may not have torch. Use only the requested pipeline.
- Never return dummy constants or code that only satisfies AST checks.
- Large flattened output spaces must not launch a grid larger than 65535 on any
  axis. Use a fixed core grid with an in-kernel stride loop when needed.
- For broadcasting, derive every input offset from the output index and the
  concrete shapes/strides from `get_inputs()` or `get_input_groups()` before
  writing code.
- Scalar reductions must be produced by Triton kernels; do not use `.item()`,
  host-side division, or PyTorch tensor construction for target compute.
- If terminal output says `Process still running (soft timeout)`, the command is
  still alive. Do not edit code or judge failure from that message. Poll the
  same terminal until the command exits, then read `metrics.json`.

## Operator Recipes

- Elementwise/broadcast: one vectorized output-tile kernel.
  `diag(A) @ B` is `C[i,j] = A[i] * B[i,j]`; never materialize `diag(A)`.
- Fused elementwise + reduction: flatten or tile the fused output/reduction
  domain, derive every input offset from the reference shapes, accumulate in
  the reference-compatible dtype, then run final Triton kernels for any scalar
  or reduced outputs.
- Matmul/linear: tiled `tl.dot`; use fp32 accumulation when needed; keep dtype
  behavior consistent with the reference. For transposed operands such as
  `A.T`, `B.T`, or `transpose(-1, -2)`, derive logical M/N/K from the
  post-transpose operands and either create semantics-only contiguous layout
  views before launch or use exact original strides in the kernel. Do not
  assume the storage order of the original tensor equals the logical matmul
  operand order. Do not mix a 2D launch grid with 1D swizzled program-id
  decoding. If operands are made contiguous in `forward`, prefer direct
  contiguous offsets using M/N/K instead of passing runtime strides.
- Reductions: reduce in Triton. For large axes, use partial reduction plus a
  final reduction kernel. Scalar final outputs must also be produced by Triton.
- Pooling/stencil: flatten output elements, recover coordinates, enumerate the
  fixed window with vectorized masked loads, and reduce with `tl.max` or
  repeated `tl.maximum`.
- Layout/slice/transpose: allocate the output and write exact indexed layout.
"""


GENERATOR_VERIFIER_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`

Use a lightweight generator + verifier flow in this single OpenHands session.
Do not output analysis. First file operation must create the implementation.

Generate one simple correct Triton implementation, then run:
```bash
bash tools/operator_pipeline.sh --op_name {op_name}
```
Read `metrics.json`; repair only concrete failures and rerun. When success is
true, copy best files and stop. Do not perform performance tuning after first
success.

Reference PyTorch code:
```python
{task_code}
```
"""


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_records(parquet_path: Path, *, max_rows: int | None, start: int = 0) -> list[dict[str, Any]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}. Run prepare_kernelbench_openhands_data.py first."
        )
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if start:
        df = df.iloc[start:]
    if max_rows is not None:
        df = df.head(max_rows)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        extra = row.get("extra_info")
        if isinstance(extra, str):
            extra = json.loads(extra)
        if not isinstance(extra, dict):
            continue
        records.append(extra)
    return records


def _parquet_meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".meta.json")


def _normalize_dataset_for_meta(hf_dataset: str) -> str:
    path = Path(hf_dataset)
    return str(path.resolve()) if path.exists() else hf_dataset


def _desired_parquet_meta(*, levels: str, hf_dataset: str, filter_mode: str) -> dict[str, str]:
    return {
        "levels": levels,
        "hf_dataset": _normalize_dataset_for_meta(hf_dataset),
        "filter_mode": filter_mode,
    }


def _read_parquet_meta(parquet_path: Path) -> dict[str, Any] | None:
    meta_path = _parquet_meta_path(parquet_path)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_parquet_meta(parquet_path: Path, meta: dict[str, str]) -> None:
    _parquet_meta_path(parquet_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parquet_matches_request(parquet_path: Path, meta: dict[str, str]) -> bool:
    return _read_parquet_meta(parquet_path) == meta


def _workspace_for_run(root: Path, baseline: str, op_name: str) -> Path:
    safe_op = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in op_name)[:120]
    return root / baseline / f"{safe_op}-{uuid.uuid4().hex[:8]}"


def _inline_ascend_full_skill_references(agent_dir: Path, *, arch: str) -> None:
    """Append selected skill reference docs into SKILL.md for prompt-only mode.

    Claude Code can discover a skill and read its reference files on demand.
    OpenHands SDK sessions may expose skills only as prompt metadata plus file
    locations. This inliner is kept as an optional fallback for SDK images where
    prompt-only skill activation does not provide enough reference context.
    """

    skills_root = agent_dir / ".agents" / "skills"
    if not skills_root.is_dir():
        return

    del arch
    for skill_name, rel_refs in _ASCEND_FULL_REFERENCE_BUNDLE.items():
        skill_dir = skills_root / skill_name
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.is_file():
            continue

        try:
            skill_text = skill_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _ASCEND_FULL_REFERENCE_MARKER in skill_text:
            continue

        parts = [
            "",
            _ASCEND_FULL_REFERENCE_MARKER,
            "",
            "## Inline Reference Pack for OpenHands Ascend Full Baseline",
            "",
            "The following reference files are inlined because this OpenHands "
            "runner does not automatically expand `references/...` files when "
            "a skill is activated.",
        ]
        for rel in rel_refs:
            ref_path = skill_dir / rel
            if not ref_path.is_file():
                continue
            try:
                ref_text = ref_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            parts.extend(
                [
                    "",
                    f"### {skill_name}/{rel}",
                    "",
                    ref_text,
                ]
            )

        skill_path.write_text(skill_text.rstrip() + "\n" + "\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _infer_operator_class(task: dict[str, Any]) -> str:
    """Best-effort static hint; the agent must verify it from executable code."""

    op_name = str(task.get("op_name", "")).lower()
    code = str(task.get("task_code", "")).lower()
    text = f"{op_name}\n{code}"

    has_reduction = bool(re.search(r"\b(torch\.)?(sum|mean|max|min|prod|amax|amin|var|std)\s*\(", text))
    has_pointwise = bool(
        re.search(
            r"\b(clamp|relu|sigmoid|tanh|gelu|silu|exp|log|sqrt|abs|where)\s*\(",
            text,
        )
        or any(op in code for op in [" + ", " - ", " * ", " / "])
    )

    if re.search(r"\b(scaled_dot_product_attention|multiheadattention|attention)\b", text):
        return "attention"
    if re.search(r"\b(conv[123]d|conv_transpose|convolution|correlation)\b", text):
        return "convolution/correlation"
    if re.search(r"\b(matmul|matrix_multiplication|bmm|mm|linear|addmm|einsum)\b", text) or "@" in code:
        return "matmul/linear/batched-matmul"
    if re.search(r"\b(layernorm|layer_norm|batchnorm|batch_norm|groupnorm|group_norm|instancenorm|instance_norm|rmsnorm|normalize)\b", text):
        return "normalization"
    if re.search(r"\b(log_softmax|logsoftmax|softmax)\b", text):
        return "softmax/logsoftmax"
    if re.search(r"\b(maxpool|avgpool|adaptiveavgpool|adaptivemaxpool|pool|unfold|stencil)\b", text):
        return "pooling/stencil"
    if re.search(r"\b(interpolate|upsample|resize|grid_sample)\b", text):
        return "interpolate/resample"
    if re.search(r"\b(cumsum|cumprod|cummax|cummin|prefix|scan)\b", text):
        return "scan/prefix/cumulative"
    if re.search(r"\b(embedding|gather|scatter|index_select|take|where|nonzero|masked_select|index_add|index_put)\b", text):
        return "gather/scatter/embedding"
    if re.search(r"\b(sort|topk|argsort|argmax|argmin|kthvalue)\b", text):
        return "sort/topk/arg"
    if re.search(r"\b(loss|mse|l1_loss|cross_entropy|nll_loss|binary_cross_entropy|hinge|cosine_similarity|pairwise_distance|cdist)\b", text):
        return "loss/distance"
    if re.search(r"\b(permute|transpose|reshape|view|slice|narrow|cat|concat|split|chunk|unsqueeze|squeeze|flatten|repeat|expand|roll|flip)\b", text):
        return "layout/index/slice/transpose"
    if has_reduction and has_pointwise:
        return "fused elementwise+reduction"
    if has_reduction:
        return "reduction/statistical"
    if has_pointwise or "broadcast" in text:
        return "elementwise/broadcast"
    return "fused/other"


def _suggest_refs_for_class(op_class: str, arch: str) -> tuple[str, str]:
    hw_ref = ".agents/skills/npu-arch/references/npu-arch-guide-triton.md"
    designer = [".agents/skills/triton-op-designer/references/sketch-design.md"]
    generator = [
        hw_ref,
        ".agents/skills/triton-op-coding/references/triton-ascend-fundamentals.md",
    ]

    if "matmul" in op_class or "linear" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/matmul-swizzle2d.md")
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-matmul.md")
    elif "convolution" in op_class or "correlation" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/matmul-swizzle2d.md")
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-matmul.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-reduce.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-examples.md",
        ])
    elif "attention" in op_class:
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-attention.md")
    elif (
        ("reduction" in op_class and "elementwise" in op_class)
        or "normalization" in op_class
        or "softmax" in op_class
        or "loss" in op_class
        or "distance" in op_class
    ):
        designer.extend([
            ".agents/skills/triton-op-designer/references/cases/reduction-sum-fused.md",
            ".agents/skills/triton-op-designer/references/cases/reduction-sum-large.md",
        ])
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-elementwise.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-reduce.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-examples.md",
        ])
    elif "reduction" in op_class or "statistical" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/reduction-sum-large.md")
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-reduce.md")
    elif (
        "layout" in op_class
        or "index" in op_class
        or "slice" in op_class
        or "transpose" in op_class
        or "gather" in op_class
        or "scatter" in op_class
        or "embedding" in op_class
    ):
        designer.append(".agents/skills/triton-op-designer/references/cases/index-put.md")
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-examples.md")
    elif "pooling" in op_class or "stencil" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/reduction-amax-large.md")
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-elementwise.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-reduce.md",
        ])
    elif "interpolate" in op_class or "resample" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/elemwise-broadcast-2d.md")
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-interpolate.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-elementwise.md",
        ])
    elif "scan" in op_class or "prefix" in op_class or "cumulative" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/reduction-sum-large.md")
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-reduce.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-examples.md",
        ])
    elif "sort" in op_class or "topk" in op_class or "arg" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/reduction-amax-large.md")
        generator.extend([
            ".agents/skills/triton-op-coding/references/triton-ascend-sort-select.md",
            ".agents/skills/triton-op-coding/references/triton-ascend-examples.md",
        ])
    elif "elementwise" in op_class or "broadcast" in op_class:
        designer.append(".agents/skills/triton-op-designer/references/cases/elemwise-broadcast-2d.md")
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-elementwise.md")
    else:
        generator.append(".agents/skills/triton-op-coding/references/triton-ascend-examples.md")

    return (
        "\n".join(f"- `{path}`" for path in designer),
        "\n".join(f"- `{path}`" for path in generator),
    )


def _task_prompt_values(task: dict[str, Any]) -> dict[str, str]:
    arch = str(task.get("arch", "ascend910b1"))
    op_class = _infer_operator_class(task)
    designer_refs, generator_refs = _suggest_refs_for_class(op_class, arch)
    return {
        "op_name": str(task["op_name"]),
        "arch": arch,
        "task_code": str(task.get("task_code", "")),
        "op_class_hint": op_class,
        "designer_ref_hint": designer_refs,
        "generator_ref_hint": generator_refs,
    }


def _setup_workspace(workspace: Path, task: dict[str, Any], instruction: str, *, baseline: str = "") -> None:
    sdk_dir = _script_dir()
    workspace_pkg = sdk_dir / "workspace"
    shutil.copytree(workspace_pkg, workspace, dirs_exist_ok=True)

    lock_src = sdk_dir / "distributed_npu_lock.py"
    lock_dst = workspace / "agent_workdir" / "tools" / "scripts" / "distributed_npu_lock.py"
    if lock_src.exists():
        lock_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lock_src, lock_dst)

    agent_dir = workspace / "agent_workdir"
    src_dir = agent_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "INSTRUCTIONS.md").write_text(instruction, encoding="utf-8")
    if baseline == "ascend_full":
        agents_text = ASCEND_FULL_AGENTS_MD.format(**_task_prompt_values(task))
        (agent_dir / "AGENTS.md").write_text(agents_text, encoding="utf-8")
        claude_md = agent_dir / "CLAUDE.md"
        if claude_md.exists():
            claude_md.unlink()
        agents_bak_md = agent_dir / "AGENTS_bak.md"
        if agents_bak_md.exists():
            agents_bak_md.unlink()
        if os.environ.get("OPENHANDS_INLINE_SKILL_REFS", "0") in ("1", "true", "True", "yes"):
            _inline_ascend_full_skill_references(
                agent_dir,
                arch=str(task.get("arch", "ascend910b1")),
            )
    (src_dir / f"{task['op_name']}.py").write_text(task["task_code"], encoding="utf-8")
    support_files = task.get("support_files")
    if isinstance(support_files, dict):
        for rel_name, content in support_files.items():
            safe_name = Path(str(rel_name)).name
            if safe_name and isinstance(content, str):
                (src_dir / safe_name).write_text(content, encoding="utf-8")


def _prompt_for_baseline(
    baseline: str,
    task: dict[str, Any],
    *,
    full_enable_optimizer: bool,
    optimizer_attempts: int,
) -> str:
    values = _task_prompt_values(task)
    if baseline == "current":
        return CURRENT_PROMPT.format(**values)
    if baseline == "generator_verifier":
        return GENERATOR_VERIFIER_PROMPT.format(**values)
    if baseline == "ascend_full":
        del full_enable_optimizer, optimizer_attempts
        return ASCEND_FULL_PROMPT.format(**values)
    raise ValueError(f"unknown baseline: {baseline}")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_result(
    *,
    baseline: str,
    task: dict[str, Any],
    workspace: Path,
    exit_code: int,
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    agent_dir = workspace / "agent_workdir"
    metrics = _read_json(agent_dir / "metrics.json") or {}
    best_metrics = _read_json(agent_dir / "metrics_best.json")
    perf = metrics.get("perf_data") or {}
    best_perf = (best_metrics or {}).get("perf_data") or {}

    return {
        "baseline": baseline,
        "op_name": task.get("op_name"),
        "kernelbench_problem_id": task.get("kernelbench_problem_id"),
        "kernelbench_name": task.get("kernelbench_name"),
        "workspace": str(workspace),
        "exit_code": exit_code,
        "duration_s": round(finished_at - started_at, 3),
        "success": bool(metrics.get("success", False)),
        "ast_check_ok": bool(metrics.get("ast_check_ok", False)),
        "correctness_ok": bool(metrics.get("correctness_ok", False)),
        "speedup_vs_torch": perf.get("speedup_vs_torch"),
        "best_success": bool((best_metrics or {}).get("success", False)),
        "best_speedup_vs_torch": best_perf.get("speedup_vs_torch"),
        "error_type": metrics.get("error_type"),
        "error": str(metrics.get("error") or "")[:1000],
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline in sorted({r["baseline"] for r in results}):
        rows = [r for r in results if r["baseline"] == baseline]
        total = len(rows)
        success = sum(1 for r in rows if r.get("success"))
        correctness = sum(1 for r in rows if r.get("correctness_ok"))
        ast_ok = sum(1 for r in rows if r.get("ast_check_ok"))
        speeds = [
            float(r["speedup_vs_torch"])
            for r in rows
            if r.get("speedup_vs_torch") is not None
        ]
        summary[baseline] = {
            "total": total,
            "ast_pass_rate": round(ast_ok / total, 4) if total else 0.0,
            "correctness_pass_rate": round(correctness / total, 4) if total else 0.0,
            "success_rate": round(success / total, 4) if total else 0.0,
            "speedup_mean": round(sum(speeds) / len(speeds), 4) if speeds else None,
            "speedup_gt_1_rate": round(sum(1 for s in speeds if s > 1.0) / len(speeds), 4) if speeds else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default=os.environ.get(
            "OPENHANDS_KERNELBENCH_PARQUET",
            str(_script_dir() / "kernelbench_openhands.parquet"),
        ),
        help="KernelBench OpenHands parquet produced by prepare_kernelbench_openhands_data.py",
    )
    parser.add_argument("--work-dir", default=str(_script_dir() / "baseline_runs"))
    parser.add_argument("--prepare", action="store_true", help="Create/overwrite the parquet before running")
    parser.add_argument("--levels", default=os.environ.get("OPENHANDS_KERNELBENCH_LEVELS", "all"))
    parser.add_argument(
        "--hf-dataset",
        default=os.environ.get(
            "OPENHANDS_KERNELBENCH_DATASET",
            str(_script_dir() / "benchmarks" / "NPUKernelBench"),
        ),
    )
    parser.add_argument(
        "--filter-mode",
        default=os.environ.get("OPENHANDS_KERNELBENCH_FILTER_MODE", "all"),
        help="Passed to prepare_kernelbench_openhands_data.py; use all for NPUKernelBench level0-4 baseline",
    )
    parser.add_argument("--baselines", default=",".join(BASELINES), help=f"Comma list from {BASELINES}")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--op-filter", default="", help="Substring filter on op_name/name/problem id")
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENHANDS_BASE_URL", os.environ.get("LLM_BASE_URL", "http://127.0.0.1:5000/v1")))
    parser.add_argument("--model", default=os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model"))
    parser.add_argument("--image", default=os.environ.get("OPENHANDS_IMAGE", "openhands-triton-env:v1"))
    parser.add_argument("--remote-eval-url", default=os.environ.get("OPENHANDS_REMOTE_EVAL_URL", ""))
    parser.add_argument("--container-host-alias", default=os.environ.get("OPENHANDS_CONTAINER_HOST_ALIAS", "127.0.0.1"))
    parser.add_argument("--eval-device-ids", default=os.environ.get("OPENHANDS_EVAL_DEVICE_IDS", ""))
    parser.add_argument("--eval-device-count", default=os.environ.get("OPENHANDS_EVAL_DEVICE_COUNT", ""))
    parser.add_argument("--container-timeout", type=int, default=int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "1800")))
    parser.add_argument("--max-iterations", type=int, default=int(os.environ.get("OPENHANDS_BASELINE_MAX_ITERATIONS", os.environ.get("OPENHANDS_MAX_ITERATIONS", "12"))))
    parser.add_argument(
        "--keep-full-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep full OpenHands container logs in agent_workdir/conversation.log, including remote-eval runs.",
    )
    parser.add_argument("--full-enable-optimizer", action="store_true", help="Deprecated for single-session ascend_full; use run_kernelbench_orchestrated_baseline.py --enable-optimizer")
    parser.add_argument("--optimizer-attempts", type=int, default=2)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    selected_baselines = tuple(b.strip() for b in args.baselines.split(",") if b.strip())
    unknown = [b for b in selected_baselines if b not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline(s): {unknown}; choices={BASELINES}")

    parquet_path = Path(args.parquet)
    parquet_meta = _desired_parquet_meta(
        levels=args.levels,
        hf_dataset=args.hf_dataset,
        filter_mode=args.filter_mode,
    )
    if args.prepare or not parquet_path.is_file() or not _parquet_matches_request(parquet_path, parquet_meta):
        from examples.openhands_sdk.prepare_kernelbench_openhands_data import _parse_levels, create_parquet

        os.environ["OPENHANDS_KERNELBENCH_FILTER_MODE"] = args.filter_mode
        create_parquet(
            str(parquet_path),
            levels=_parse_levels(args.levels),
            max_rows=None,
            hf_dataset=args.hf_dataset,
        )
        _write_parquet_meta(parquet_path, parquet_meta)

    import examples.openhands_sdk.openhands_agent as agent_mod

    eval_device_count = args.eval_device_count or str(
        len([x for x in args.eval_device_ids.split(",") if x.strip()])
        if args.eval_device_ids
        else 1
    )
    agent_mod._OPENHANDS_IMAGE = args.image
    agent_mod._MODEL_NAME = args.model
    agent_mod._MAX_ITERATIONS = args.max_iterations
    agent_mod._CONTAINER_TIMEOUT = args.container_timeout
    agent_mod._OPENHANDS_REMOTE_EVAL_URL = args.remote_eval_url.strip().rstrip("/")
    agent_mod._OPENHANDS_CONTAINER_HOST_ALIAS = args.container_host_alias
    agent_mod._OPENHANDS_EVAL_DEVICE_IDS = args.eval_device_ids
    agent_mod._OPENHANDS_EVAL_DEVICE_COUNT = eval_device_count

    os.environ["OPENHANDS_REMOTE_RETURN_FULL_LOG"] = "1" if args.keep_full_logs else "0"

    records = _load_records(parquet_path, max_rows=args.max_rows, start=args.start)
    if args.op_filter:
        needle = args.op_filter.lower()
        records = [
            r for r in records
            if needle in " ".join(
                str(r.get(k, "")) for k in ("op_name", "kernelbench_name", "kernelbench_problem_id")
            ).lower()
        ]

    run_root = Path(args.work_dir) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.jsonl"
    summary_path = run_root / "summary.json"

    print(f"[baseline] rows={len(records)} baselines={selected_baselines} run_root={run_root}")
    print(f"[baseline] llm_base_url={args.llm_base_url} model={args.model} max_iterations={args.max_iterations}")

    results: list[dict[str, Any]] = []
    with results_path.open("a", encoding="utf-8") as out:
        for task in records:
            for baseline in selected_baselines:
                instruction = _prompt_for_baseline(
                    baseline,
                    task,
                    full_enable_optimizer=args.full_enable_optimizer,
                    optimizer_attempts=args.optimizer_attempts,
                )
                workspace = _workspace_for_run(run_root, baseline, task["op_name"])
                _setup_workspace(workspace, task, instruction, baseline=baseline)

                started = time.time()
                print(f"[baseline] start baseline={baseline} op={task['op_name']} workspace={workspace}")
                prev_agent_mode = os.environ.get("OPENHANDS_AGENT_MODE")
                prev_enable_agent_skills = os.environ.get("OPENHANDS_ENABLE_AGENT_SKILLS")
                if baseline == "ascend_full":
                    os.environ["OPENHANDS_AGENT_MODE"] = "ascend_full"
                    os.environ["OPENHANDS_ENABLE_AGENT_SKILLS"] = "1"
                else:
                    os.environ.pop("OPENHANDS_AGENT_MODE", None)
                    os.environ["OPENHANDS_ENABLE_AGENT_SKILLS"] = "0"
                try:
                    exit_code = agent_mod._run_remote_eval_worker(
                        str(workspace),
                        agent_mod._to_container_url(args.llm_base_url),
                        instruction,
                        task=task,
                    )
                except Exception as exc:
                    exit_code = -1
                    (workspace / "agent_workdir" / "baseline_exception.txt").write_text(
                        repr(exc),
                        encoding="utf-8",
                    )
                    print(f"[baseline] exception baseline={baseline} op={task['op_name']}: {exc!r}")
                finally:
                    if prev_agent_mode is None:
                        os.environ.pop("OPENHANDS_AGENT_MODE", None)
                    else:
                        os.environ["OPENHANDS_AGENT_MODE"] = prev_agent_mode
                    if prev_enable_agent_skills is None:
                        os.environ.pop("OPENHANDS_ENABLE_AGENT_SKILLS", None)
                    else:
                        os.environ["OPENHANDS_ENABLE_AGENT_SKILLS"] = prev_enable_agent_skills

                finished = time.time()
                result = _collect_result(
                    baseline=baseline,
                    task=task,
                    workspace=workspace,
                    exit_code=int(exit_code),
                    started_at=started,
                    finished_at=finished,
                )
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    "[baseline] done "
                    f"baseline={baseline} op={task['op_name']} "
                    f"success={result['success']} correctness={result['correctness_ok']} "
                    f"speedup={result['speedup_vs_torch']} exit={exit_code}"
                )

    summary = _summarize(results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[baseline] wrote {results_path}")
    print(f"[baseline] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
