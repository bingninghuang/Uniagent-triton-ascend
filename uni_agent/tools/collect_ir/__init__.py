"""collect_ir tool: collect Triton compiler IR for the current operator.

The model calls this tool after run_verify reports verified_success: true.
The script runs inside the sandbox, auto-detects the op_name from
``src/*_triton_ascend_impl.py``, generates a temporary runner script that
imports the implementation and triggers Triton compilation with
TRITON_DEBUG=1, extracts IR files (ttir, ttadapter, last_pass) via
bishengir-compile, and prints a structured summary with quick bottleneck
identification.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Collect Triton compiler IR (Intermediate Representation) for your current
implementation. Call this ONLY AFTER run_verify reports verified_success: true.

The tool:
1. Compiles your operator with TRITON_DEBUG=1 to dump compiler IR
2. Runs bishengir-compile to extract the final-stage IR (last_pass.mlir)
3. Outputs a structured summary with kernel names, IR file paths, and quick
   bottleneck identification (HIVM intrinsic ratio, memory ops, scalar ops,
   sync operations)

Use the IR file paths to read specific IR files for deeper analysis with
str_replace_editor. The summary suggests which optimization points to check
based on IR patterns.

No arguments needed - the tool auto-detects the op_name from
src/*_triton_ascend_impl.py.
""".strip()


class CollectIrArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": []})


@register_tool("collect_ir")
class CollectIrTool(AbstractTool):
    @property
    def name(self) -> str:
        return "collect_ir"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "collect_ir"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=CollectIrArguments,
        )

    def get_install_command(self) -> str:
        return None
