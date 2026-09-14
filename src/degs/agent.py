from __future__ import annotations

from typing import Any

from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
from spreadsheet_agent.system_prompts import render_full_system_prompt

from .bundle import EXPERIENCE_FORMAT, METHOD_FAMILY
from .provider import DEGSExperienceProvider


class DEGSExperienceAgent(CLIOnlyAgent):
    """The frozen CLI Agent with one Experience-SimGRAG context per task."""

    def __init__(
        self,
        *args: Any,
        experience_provider: DEGSExperienceProvider,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if type(experience_provider) is not DEGSExperienceProvider:
            raise ValueError("DEGS Agent requires a verified DEGS experience provider")
        self.experience_provider = experience_provider
        self._experience_content = ""

    @property
    def name(self) -> str:
        return "degs_experience_simgrag_agent"

    def get_system_template(self) -> str:
        return render_full_system_prompt(
            "preloaded_experience_full_system_v1.txt",
            experience_content=self._experience_content,
        )

    def run(self, context: Any) -> dict[str, Any]:
        payload = self.experience_provider.for_instance(context.instance_id)
        if (
            payload.metadata.get("format") != EXPERIENCE_FORMAT
            or payload.metadata.get("method_family") != METHOD_FAMILY
        ):
            raise ValueError("task experience is outside the method boundary")
        self._experience_content = payload.experience
        self._agent = None
        return super().run(context)


__all__ = ["DEGSExperienceAgent"]
