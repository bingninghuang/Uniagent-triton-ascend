"""collect_profiling tool: collect msprof profiling metrics for the current operator.

The model calls this tool after run_verify reports verified_success: true
and after collect_ir has been used to identify initial bottlenecks. The
script runs msprof op on the operator, parses op_summary CSV for key
performance metrics, and provides bottleneck analysis with optimization
suggestions.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Collect msprof profiling metrics for your current implementation. Call this
ONLY AFTER run_verify reports verified_success: true, typically after
collect_ir to validate IR-based findings.

The tool:
1. Runs msprof op to profile the Triton kernel execution
2. Parses op_summary CSV for key performance metrics
3. Provides bottleneck analysis with actionable optimization suggestions

Key metrics analyzed:
- aiv_vec_ratio: Vector utilization (ideal >80%)
- aiv_mte2_ratio: Memory transfer ratio (ideal <50%)
- aiv_scalar_ratio: Scalar operation ratio (ideal <20%)
- aic_cube_ratio: Cube utilization (ideal >80%)
- aic_mte1_ratio: L1->L0 transfer ratio

Use the bottleneck analysis to identify which optimization points to
check in the latency-optimizer skill.

No arguments needed - the tool auto-detects the op_name from
src/*_triton_ascend_impl.py.
""".strip()


class CollectProfilingArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": []})


@register_tool("collect_profiling")
class CollectProfilingTool(AbstractTool):
    @property
    def name(self) -> str:
        return "collect_profiling"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "collect_profiling"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=CollectProfilingArguments,
        )

    def get_install_command(self) -> str:
        return None
