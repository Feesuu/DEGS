from __future__ import annotations

from typing import Any

from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
from spreadsheet_agent.system_prompts import render_full_system_prompt

from .provider import (
    EIR_GUIDANCE_FORMAT,
    EIR_METHOD_FAMILY,
    EIRGuidanceProvider,
)


class DEGSExperienceAgent(CLIOnlyAgent):
    """The fixed CLI Agent with one evidence-bounded guidance payload per task."""

    def __init__(
        self,
        *args: Any,
        experience_provider: EIRGuidanceProvider,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(experience_provider, EIRGuidanceProvider):
            raise ValueError("DEGS Agent requires a verified EIR guidance provider")
        self.experience_provider = experience_provider
        self._experience_content = ""

    @property
    def name(self) -> str:
        return "degs_eir_contextual_guidance_agent"

    def get_system_template(self) -> str:
        return render_full_system_prompt(
            "preloaded_experience_full_system_v1.txt",
            experience_content=self._experience_content,
        )

    def run(self, context: Any) -> dict[str, Any]:
        payload = self.experience_provider.for_instance(context.instance_id)
        identity = (
            payload.metadata.get("format"),
            payload.metadata.get("method_family"),
        )
        if identity != (EIR_GUIDANCE_FORMAT, EIR_METHOD_FAMILY):
            raise ValueError("task experience is outside the method boundary")
        self._experience_content = payload.experience
        self._agent = None
        return super().run(context)


__all__ = ["DEGSExperienceAgent"]
