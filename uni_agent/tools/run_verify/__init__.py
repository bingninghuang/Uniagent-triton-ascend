"""run_verify tool: AST check + correctness verify + performance benchmark."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Run AST check, correctness verification, and performance benchmark for the
current Triton Ascend implementation. No arguments needed.

The tool:
1. Validates the implementation is not a PyTorch fallback (AST check)
2. Stages files and runs NPU correctness verification (verify.py)
3. Runs performance benchmark when all cases pass (benchmark.py)
4. Prints a structured SUMMARY with ast_valid, verified_success, pass_rate,
   error_groups, speedup_vs_torch, bottleneck_hint, and kernel_metrics

Call after writing or editing code. Benchmark is skipped if AST fails or any
verify case fails.
""".strip()


class RunVerifyArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": []})


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
