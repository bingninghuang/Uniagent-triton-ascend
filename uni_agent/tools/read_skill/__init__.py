"""Read skill tool — read a skill's SKILL.md or reference files."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Read a skill's content. By default returns the SKILL.md metadata
(description + argument-hint) and the directory (all ## and ###
headings, each followed by the first sentence under it). Use --section
to read a specific ## or ### heading, --file to read a references/
file, or --full to read the entire SKILL.md. Always call `list_skills`
first to discover skill names, then call this to read details.
""".strip()


class ReadSkillArguments(BaseModel):
    skill: str = Field(
        description="Skill name, e.g. hardware-specs"
    )
    section: str = Field(
        default="",
        description="Optional ## or ### heading to read. If empty, returns metadata+directory+first sentences.",
    )
    file: str = Field(
        default="",
        description="Optional references/ file path, e.g. references/hw-ascend910b4.md or references/matmul.md",
    )
    full: bool = Field(
        default=False,
        description="If true, read the entire SKILL.md without truncation.",
    )


@register_tool("read_skill")
class ReadSkillTool(AbstractTool):
    @property
    def name(self) -> str:
        return "read_skill"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "read_skill"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=ReadSkillArguments,
        )

    def get_install_command(self) -> str:
        return None
