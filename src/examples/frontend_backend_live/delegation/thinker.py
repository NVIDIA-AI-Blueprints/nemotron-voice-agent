# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The thinker: delegated, tool-using work on a backend model.

Wraps a ``Backend`` with the prompt set's voice-channel note. The note describes how the reply will be
spoken; the backend's own task instructions pass through unchanged.
"""

from collections.abc import AsyncIterator

from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.models import BACKENDS, Backend
from examples.frontend_backend_live.prompts import PromptSet, get_prompt_set


class Thinker:
    """A backend model plus the prompt set's voice-channel note."""

    def __init__(self, backend: Backend, prompts: PromptSet):
        """Bind the backend model and the prompt set."""
        self.backend, self.prompts = backend, prompts

    @property
    def stateful(self) -> bool:
        """Whether the backend keeps the conversation itself."""
        return self.backend.stateful

    def instructions(self, task_instructions: str | None) -> str | None:
        """Return the task instructions plus the prompt set's voice-channel note, if any."""
        notes = self.prompts.backend_notes.strip()
        if not notes:
            return task_instructions
        return ((task_instructions or "").rstrip() + "\n" + notes).strip()

    def _payload(self, payload: dict) -> dict:
        instructions = self.instructions(payload.get("instructions"))
        return payload if instructions is None else {**payload, "instructions": instructions}

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield the Responses events of one backend round."""
        async for event in self.backend.stream(self._payload(payload)):
            yield event


def build_thinker(config: LiveConfig, backend_endpoint: ModelEndpoint, *, backend_client=None) -> Thinker:
    """Build the backend model wrapper named by ``config.backend``.

    Raises:
        ValueError: The configured backend provider is not registered.
    """
    backend_cls = BACKENDS.get(config.backend.provider)
    backend = (
        backend_cls(backend_endpoint, config.reliability, backend_client)
        if backend_client is not None
        else backend_cls(backend_endpoint, config.reliability)
    )
    return Thinker(backend, get_prompt_set(config.prompt_version))
