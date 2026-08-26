"""List skills tool — list all available skills with metadata."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
List all available skills in the knowledge base. Returns each skill's name,
description, path, and reference file count. Call this first to see what
skills are available, then use `read_skill` to read details of a specific
skill.
""".strip()


class ListSkillsArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": []})


@register_tool("list_skills")
class ListSkillsTool(AbstractTool):
    @property
    def name(self) -> str:
        return "list_skills"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "list_skills"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=ListSkillsArguments,
        )

    def get_install_command(self) -> str:
        return None
