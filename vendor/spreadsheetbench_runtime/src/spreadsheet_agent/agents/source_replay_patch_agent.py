from __future__ import annotations

from sb_adapter.experience import ExperienceProvider

from .cli_only_agent import CLIOnlyAgent
from ..system_prompts import render_full_system_prompt


class SourceReplayPatchAgent(CLIOnlyAgent):
    """Baseline CLI Agent with one exact task-local temporary patch."""

    def __init__(
        self,
        *args,
        patch_provider: ExperienceProvider,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_provider = patch_provider
        self._patch_content = ""

    @property
    def name(self) -> str:
        return "source_replay_patch_agent"

    def get_system_template(self) -> str:
        return render_full_system_prompt(
            "source_replay_patch_full_system_v1.txt",
            patch_content=self._patch_content,
        )

    def run(self, context):
        payload = self.patch_provider.for_instance(context.instance_id)
        if not payload.experience.strip():
            raise ValueError(
                f"source replay patch is missing for task {context.instance_id}"
            )
        if payload.metadata.get("format") != "degs_source_replay_patch_v1":
            raise ValueError("source replay patch has the wrong format identity")
        if payload.metadata.get("task_id") != context.instance_id:
            raise ValueError("source replay patch task identity does not match")
        self._patch_content = payload.experience
        self._agent = None
        return super().run(context)
