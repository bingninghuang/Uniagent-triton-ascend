"""run_verify tool: model-invoked AST check + correctness verification + benchmark.

The model calls this tool autonomously after writing its implementation.
The script runs inside the sandbox, auto-detects the op_name from
``src/*_triton_ascend_impl.py``, runs the AST check and (if it passes)
the correctness verifier, then (if all cases pass) the performance
benchmark, and prints a compact structured summary.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Run the verification pipeline (AST check + correctness verify + performance
benchmark) for your current implementation. Call this AFTER you have written
or edited the implementation file using str_replace_editor. The tool
auto-detects the op_name from src/*_triton_ascend_impl.py - no arguments
needed.

When all correctness cases pass, the tool also runs benchmark.py and reports
speedup_vs_torch in the summary. If AST fails or any case fails, the
benchmark step is skipped.

Output: a structured summary with ast_valid, passed_cases/total_cases,
pass_rate, error_groups (error_type + reason + count for each failure
group), error_preview (when verification fails), and speedup_vs_torch
(when all cases pass). Use the speedup to decide whether to optimize
further or call submit.

Call run_verify after every code change. Keep optimizing until speedup
reaches the target (default 2.0x), then call submit.
""".strip()


class RunVerifyArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": []})

    max_error_chars: int = Field(
        default=2000,
        description="Maximum characters of error preview to include in the summary when verification fails. Default 2000.",
    )


@register_tool("run_verify")
class RunVerifyTool(AbstractTool):
    @property
    def name(self) -> str:
        return "run_verify"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "run_verify"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=RunVerifyArguments,
        )

    def get_install_command(self) -> str:
        return None