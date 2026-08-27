# ruff: noqa
"""
Scaffold tools.
"""

from pydantic import BaseModel

from .collect_ir import CollectIrTool
from .collect_profiling import CollectProfilingTool
from .finish import FinishTool
from .registry import get_tool, AbstractTool
from .execute_bash import ExecuteBashTool
from .lark_cli import LarkCliTool
from .list_skills import ListSkillsTool
from .read_skill import ReadSkillTool
from .search_arxiv import SearchArxivTool
from .search import SearchWikiTool
from .str_replace_editor import StrReplaceEditorTool
from .submit import SubmitTool


class ToolConfig(BaseModel):
    name: str

    def get_tool(self) -> AbstractTool:
        """Return a tool instance (for env.install_tools / init_for_interaction)."""
        return get_tool(self.name)


__all__ = [
    "ToolConfig",
    "CollectIrTool",
    "CollectProfilingTool",
    "ExecuteBashTool",
    "FinishTool",
    "LarkCliTool",
    "ListSkillsTool",
    "ReadSkillTool",
    "SearchArxivTool",
    "SearchWikiTool",
    "StrReplaceEditorTool",
    "SubmitTool",
]
