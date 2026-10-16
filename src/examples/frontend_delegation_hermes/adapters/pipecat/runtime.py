# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide state for the Pipecat adapter's sessions on the repository's shared server.

What FDH's own server (``app.py``, removed in ``shared-stack-plan.md`` step 3) built at startup, built
once per configuration instead: the delegation runtime (frontend model client, gateway check), the
event log, routing sink and filler log, and the session admission (``server.max_sessions``, refused
with ``server_busy`` and close code 1013, as before).

The shared server has no per-example startup hook, so the runtime gets ready on its first session
(:meth:`FdhRuntime.ensure_ready`, the Frontend/Backend Verdict example's rule): the session waits for
the frontend warm-up and the gateway check before it is admitted, so its first turn runs on a warm
connection, as it did after FDH's own server started. A failed start is reported to that session
(``server is starting``) and retried by the next one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.frontend_delegation_hermes.adapters.pipecat.profiles import (
    PROFILES,
    REALTIME_PROFILE,
    ProfileError,
    keepalive_mismatch,
    profile_name,
)
from examples.frontend_delegation_hermes.adapters.pipecat.settings import PipecatHeadSettings, load_pipecat_settings
from examples.frontend_delegation_hermes.core.config import DelegationConfig, load_delegation_config
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.frontend.llm import ChatModel
from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime
from examples.frontend_delegation_hermes.core.server_common import SessionSlots, health_body
from examples.frontend_delegation_hermes.core.wire import server_events as ev

if TYPE_CHECKING:
    from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SpeechStack

#: Builds one session's speech stack (``None``: Silero and the NVIDIA services); tests pass stubs.
SpeechFactory = Callable[[Any], "SpeechStack"]


_EVENT_LOGS: dict[tuple[str, bool], EventLog] = {}


def shared_event_log(path: str, *, redact_content: bool) -> EventLog:
    """One :class:`EventLog` (and its lock) per file, shared by every runtime of this process."""
    key = (path, redact_content)
    log = _EVENT_LOGS.get(key)
    if log is None:
        log = _EVENT_LOGS[key] = EventLog(path, redact_content=redact_content)
    return log


class FdhRuntime:
    """The adapter's state for one configuration (profile, plus a browser session's selections)."""

    def __init__(
        self,
        config: DelegationConfig,
        settings: PipecatHeadSettings,
        *,
        profile: str,
        chat_model: ChatModel | None = None,
        speech_factory: SpeechFactory | None = None,
    ) -> None:
        """Build the shared objects; nothing connects until :meth:`ensure_ready`."""
        voice = config.voice
        self.config = config
        self.settings = settings
        self.profile = profile
        self.speech_factory = speech_factory
        self.delegation = DelegationRuntime(config, chat_model=chat_model)
        event_log = shared_event_log(voice.logging.event_log, redact_content=voice.logging.redact_content)
        self.routing_sink = SessionRoutingSink(event_log)
        self.filler_log = FillerLog(voice.filler.log_path, event_log=event_log)
        self.slots = SessionSlots(max_sessions=voice.server.max_sessions)
        self.ready = False
        self.startup_error = ""
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> bool:
        """Warm the frontend connection and check the gateway, once."""
        if self.ready:
            return True
        async with self._lock:
            if self.ready:
                return True
            config = self.config
            for warning in config.voice.warnings:
                logger.warning(warning)
            try:
                await self.delegation.startup()
            except Exception as exc:  # noqa: BLE001 - reported to this session; the next one retries
                self.startup_error = str(exc)
                logger.error(f"frontend-delegation-hermes-agent: startup failed: {exc}")
                return False
            self.startup_error = ""
            self.ready = True
            logger.info(
                f"frontend-delegation-hermes-agent ready (adapter=pipecat, profile={self.profile}): loaded "
                f"{', '.join(str(path) for path in config.source_files)} (config hash {config.config_hash})"
            )
            logger.info(f"Pipecat adapter settings {self.settings.summary()}")
            logger.info(f"health {self.health()}")
            logger.info(
                f"frontend={config.frontend.decider}:{config.frontend.llm.model} backend={config.backend.link}:"
                f"{config.backend.url} tools={config.tools.executor}"
            )
            return True

    async def admit(self, websocket: Any) -> bool:
        """Today's admission rule: not ready or at ``max_sessions`` -> ``server_busy`` + 1013."""
        ready = await self.ensure_ready()
        reason = self.slots.refusal(ready)
        if reason:
            await websocket.send_text(json.dumps(ev.error(reason, code="server_busy")))
            await websocket.close(code=1013, reason=reason)
            return False
        self.slots.acquire()
        return True

    def release(self) -> None:
        """One admitted session ended."""
        self.slots.release()

    def health(self) -> dict[str, Any]:
        """What FDH's own ``/health`` reported; logged once ready (the shared server serves its own ``/health``)."""
        body, _ = health_body(
            ready=self.ready,
            slots=self.slots,
            startup_error=self.startup_error,
            extra=lambda: {
                **self.delegation.health(),
                "adapter": "pipecat",
                "profile": self.profile,
                "pipecat": self.settings.summary(),
            },
        )
        return body

    async def shutdown(self) -> None:
        """Close the frontend model client."""
        await self.delegation.shutdown()


_REALTIME: dict[str, FdhRuntime] = {}


def realtime_runtime(environ: Mapping[str, str] | None = None) -> FdhRuntime:
    """The runtime of every OpenAI Realtime session: always the ``eval`` profile (``FDH_PROFILE`` is still checked)."""
    profile_name(environ)
    runtime = _REALTIME.get(REALTIME_PROFILE)
    if runtime is None:
        profile = PROFILES[REALTIME_PROFILE]
        config = load_delegation_config(profile.delegation_path)
        mismatch = keepalive_mismatch(config.voice.server.ws_ping_interval_s, environ)
        if mismatch:
            raise ProfileError(mismatch)
        runtime = FdhRuntime(config, load_pipecat_settings(profile.pipecat_settings), profile=REALTIME_PROFILE)
        _REALTIME[REALTIME_PROFILE] = runtime
    return runtime
