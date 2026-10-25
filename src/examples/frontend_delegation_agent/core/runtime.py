# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The head-neutral delegation runtime: what every head needs to build a session's turn manager.

A head builds one :class:`DelegationRuntime` per process and calls
:meth:`DelegationRuntime.turn_manager` with each session's
:class:`~examples.frontend_delegation_agent.core.head_api.TurnContext`.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from examples.frontend_delegation_agent.core.config import (
    DelegationConfig,
)
from examples.frontend_delegation_agent.core.engine.backend_link import BackendLink, WebSocketBackendLink
from examples.frontend_delegation_agent.core.engine.spelling_hold import SpellingHoldPredicate
from examples.frontend_delegation_agent.core.engine.turn_manager import (
    DelegationTurnManager,
    SessionParts,
    SessionRules,
)
from examples.frontend_delegation_agent.core.frontend.decider import Decider, LLMDecider, RuleDecider
from examples.frontend_delegation_agent.core.frontend.llm import ChatModel, OpenAIChatModel
from examples.frontend_delegation_agent.core.frontend.prompts import PromptCatalog, load_prompts
from examples.frontend_delegation_agent.core.frontend.status_verbalizer import (
    LLMVerbalizer,
    StatusVerbalizer,
    TemplateVerbalizer,
)
from examples.frontend_delegation_agent.core.head_api import TurnContext
from examples.frontend_delegation_agent.core.normalization.arguments import ArgumentNormalizer, ResultHintSettings
from examples.frontend_delegation_agent.core.normalization.schema_rules import SchemaRules, build_schema_rules
from examples.frontend_delegation_agent.core.normalization.transcript import TranscriptNormalizer
from examples.frontend_delegation_agent.core.text.prompts import load_catalog
from examples.frontend_delegation_agent.core.tools.format_retry import FormatRetry
from examples.frontend_delegation_agent.core.tools.identity_ledger import (
    IdentityLedger,
    LedgerSettings,
    identification_tools,
)
from examples.frontend_delegation_agent.core.tools.relay import FormatRetryMessages
from examples.frontend_delegation_agent.core.tools.result_hints import ResultHints


class DelegationRuntime:
    """Resources shared by every session of one server (clients, prompts, normalizer templates)."""

    def __init__(self, config: DelegationConfig, *, chat_model: ChatModel | None = None) -> None:
        """Load prompts and build the shared model client (injectable for tests)."""
        self.config = config
        self.prompts: PromptCatalog = load_prompts(
            config.delegation.prompts_path, prompt_features=config.prompt_features
        )
        needs_llm = config.frontend.decider == "llm" or config.backend.verbalizer_mode == "llm"
        self.chat_model = (
            chat_model if chat_model is not None else (OpenAIChatModel(config.frontend.llm) if needs_llm else None)
        )
        voice = config.voice
        self._script = _load_script(config.frontend.script)
        self._argument_templates = ("", "", "")
        self._hint_templates = ("", "")  # result_hints: (message_key, escalate_message_key) texts
        self._ledger_template = ""  # F8: result_hints.ledger_message_key text
        self.format_messages = FormatRetryMessages()  # format_retry: hint, found, not-found texts
        arguments = voice.normalization.tool_arguments
        if arguments.enabled:
            catalog = load_catalog(voice.agent.prompts_path, voice.agent.prompts.inline)
            guard = arguments.retry_guard
            escalate = arguments.escalate_invalid_message_key
            self._argument_templates = (
                catalog.get(arguments.invalid_message_key),
                catalog.get(guard.message_key) if guard.enabled else "",
                catalog.get(escalate) if escalate else "",
            )
            hints = arguments.result_hints
            if hints.enabled:
                self._hint_templates = (
                    catalog.get(hints.message_key),
                    catalog.get(hints.escalate_message_key) if hints.escalate_message_key else "",
                )
                if hints.ledger:
                    self._ledger_template = catalog.get(hints.ledger_message_key)
            retry = arguments.format_retry
            if retry.enabled:
                self.format_messages = FormatRetryMessages(
                    hint=catalog.get(retry.hint_message_key),
                    found=catalog.get(retry.found_message_key),
                    not_found=catalog.get(retry.not_found_message_key),
                )
        self.gateway_health: dict[str, Any] = {}

    def session_rules(self, tools: Sequence[Mapping[str, Any]] | None = None) -> SessionRules:
        """The tool-dependent collaborators; ``auto`` switches derive their part from ``tools``.

        Before the client's tools are known (``tools`` is None) the ``auto`` parts are empty.
        """
        config, voice = self.config, self.config.voice
        schema: SchemaRules | None = build_schema_rules(tools) if tools is not None else None
        settings = voice.normalization.tool_arguments
        if schema is not None and settings.schema_rules:
            settings = replace(settings, rules=settings.rules + schema.rules)
        arguments = (
            ArgumentNormalizer(
                settings,
                transcript=voice.normalization.transcript,
                invalid_template=self._argument_templates[0],
                already_failed_template=self._argument_templates[1],
                escalate_invalid_template=self._argument_templates[2],
            )
            if settings.enabled
            else None
        )
        hints = settings.result_hints
        result_hints = (
            ResultHints(
                hints,
                message=self._hint_templates[0],
                escalate_message=self._hint_templates[1],
                read_tools=schema.read_tools if schema is not None else (),
                ledger=self._ledger(hints, tools),
            )
            if arguments is not None and hints.enabled
            else None
        )
        format_retry = (
            FormatRetry(
                settings.format_retry,
                fields=schema.format_retry_fields() if schema is not None else {},
                read_only_tools=schema.read_only_tools if schema is not None else (),
            )
            if arguments is not None and settings.format_retry.enabled
            else None
        )
        hold = config.delegation.spelling_hold
        spelling_hold = (
            SpellingHoldPredicate(
                voice.normalization.transcript,
                complete_patterns=hold.complete_patterns,
                schema_patterns=schema.complete_patterns if schema is not None and hold.auto_patterns else (),
                arguments=arguments,
            )
            if hold.enabled
            else None
        )
        return SessionRules(
            argument_normalizer=arguments,
            result_hints=result_hints,
            spelling_hold=spelling_hold,
            schema=schema,
            format_retry=format_retry,
        )

    def _ledger(self, hints: ResultHintSettings, tools: Sequence[Mapping[str, Any]] | None) -> IdentityLedger | None:
        """F8: the session's identification ledger (fix-plan-f1-f2-f4-f8.md section 4), or None when off."""
        if not hints.ledger:
            return None
        schemas = tuple(tools or ())
        names = identification_tools(schemas) if hints.auto_identification_tools else hints.identification_tools
        return IdentityLedger(
            names,
            tools=schemas,
            failure_pattern=hints.failure_pattern,
            template=self._ledger_template,
            settings=LedgerSettings(
                max_age_turns=hints.ledger_max_age_turns,
                correction_cues=hints.ledger_correction_cues,
                max_notes=hints.max_hints,
            ),
        )

    def parts(self, context: TurnContext) -> SessionParts:
        """Per-session collaborators (plan 12.1: every one chosen by config)."""
        config, voice = self.config, self.config.voice
        rules = self.session_rules()
        return SessionParts(
            decider=self._decider(),
            verbalizer=self._verbalizer(),
            link=self._link(),
            prompts=self.prompts,
            transcript_normalizer=TranscriptNormalizer(voice.normalization.transcript)
            if voice.normalization.transcript.enabled
            else None,
            argument_normalizer=rules.argument_normalizer,
            result_hints=rules.result_hints,
            local_tools=tuple(voice.tools.config_tool_specs) if config.tools.executor == "local" else (),
            spelling_hold=rules.spelling_hold,
            rebuild=self.session_rules if any(config.schema_auto.values()) else None,
            format_retry=rules.format_retry,
            format_messages=self.format_messages,
        )

    def turn_manager(self, context: TurnContext) -> DelegationTurnManager:
        """The ``SessionHooks.turn_manager_factory``."""
        return DelegationTurnManager(context, self.config, self.parts(context))

    def _decider(self) -> Decider:
        frontend = self.config.frontend
        if frontend.decider == "scripted":
            return RuleDecider(self._script)
        assert self.chat_model is not None  # noqa: S101 - built when decider is llm
        delegation = self.config.delegation
        return LLMDecider(
            self.chat_model,
            self.prompts,
            timeout_ms=frontend.timeout_ms,
            max_repair_attempts=delegation.max_repair_attempts,
            on_contract_error=delegation.on_contract_error,
            tool_choice=frontend.tool_choice,
            parallel_tool_calls=frontend.parallel_tool_calls,
            hedge_after_ms=frontend.hedge_after_ms,
        )

    def _verbalizer(self) -> StatusVerbalizer:
        backend = self.config.backend
        if backend.verbalizer_mode == "llm" and self.chat_model is not None:
            return LLMVerbalizer(self.chat_model, self.prompts, timeout_ms=backend.verbalizer_timeout_ms)
        return TemplateVerbalizer(self.prompts)

    def _link(self) -> BackendLink:
        backend = self.config.backend
        if backend.link == "in_process":
            # The gateway and the worker run in this process: the LLM backend, or the emulator.
            from examples.frontend_delegation_agent.core.sidecar.gateway_config import load_gateway_config
            from examples.frontend_delegation_agent.core.sidecar.inprocess_link import InProcessBackendLink

            assert backend.gateway_config is not None  # noqa: S101 - validated in config
            return InProcessBackendLink(load_gateway_config(backend.gateway_config))
        return WebSocketBackendLink(backend.url, open_timeout_s=min(10.0, backend.open_timeout_s))

    async def startup(self) -> None:
        """Warm the frontend connection; check the gateway (reachable, enough capacity)."""
        await self._warmup()
        backend = self.config.backend
        if backend.link != "websocket":
            return
        url = _health_url(backend.url)
        try:
            self.gateway_health = await asyncio.to_thread(_get_json, url)
        except Exception as exc:  # noqa: BLE001 - the gateway may start later; sessions will report it
            logger.warning(f"backend gateway not reachable at {url} ({exc}); sessions will fail until it is up")
            return
        capacity = int(self.gateway_health.get("max_sessions") or 0)
        wanted = self.config.voice.server.max_sessions
        if capacity and capacity < wanted:
            raise RuntimeError(
                f"server.max_sessions ({wanted}) exceeds the gateway's max_sessions ({capacity}); "
                "lower FDA_MAX_SESSIONS here or raise it on the gateway"
            )
        logger.info(f"backend gateway {url}: {self.gateway_health}")

    async def _warmup(self) -> None:
        """First request of a fresh process pays connection setup (measured: >4 s): pay it here."""
        if not self.config.frontend.warmup or self.chat_model is None or self.config.frontend.decider != "llm":
            return
        started = time.perf_counter()
        try:
            await asyncio.wait_for(self.chat_model.complete([{"role": "user", "content": "Say OK."}]), timeout=20)
        except Exception as exc:  # noqa: BLE001 - a failed warm-up only costs the first turn's latency
            logger.warning(f"frontend warm-up failed ({type(exc).__name__}: {exc}); the first turn may be slow")
            return
        logger.info(f"frontend warm-up done in {time.perf_counter() - started:.2f}s")

    async def shutdown(self) -> None:
        """Close the shared model client."""
        close = getattr(self.chat_model, "aclose", None)
        if close is not None:
            await close()

    def health(self) -> dict[str, Any]:
        """Extra ``/health`` fields."""
        backend = self.config.backend
        return {
            "prototype": "frontend-delegation-agent",
            "config_hash": self.config.config_hash,
            "features": self.config.features,
            "backend": {"link": backend.link, "url": backend.url if backend.link == "websocket" else ""},
            "frontend": {"decider": self.config.frontend.decider, "model": self.config.frontend.llm.model},
        }


def _health_url(ws_url: str) -> str:
    base = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    return base.split("/v1/", 1)[0].rstrip("/") + "/health"


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - operator-configured local URL
        return json.loads(response.read().decode())


def _load_script(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("frontend.script must be a JSON list of decisions")
    return data
