"""Search skills tool — search local skill reference documents by keywords."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Search skill reference documents for relevant sections matching the given
keywords. Returns the most relevant snippets (up to 5) with source file and
heading. Use this instead of reading skill files directly — it is faster and
returns only the most relevant content.
""".strip()


class SearchSkillsArguments(BaseModel):
    query: str = Field(
        description=(
            "Space-separated keywords to search for, e.g. "
            '"reduction mean fp16 precision". Use English keywords that match '
            "the concepts you need help with."
        ),
    )


@register_tool("search_skills")
class SearchSkillsTool(AbstractTool):
    @property
    def name(self) -> str:
        return "search_skills"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "search_skills"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=SearchSkillsArguments,
        )

    def get_install_command(self) -> str:
        return None
