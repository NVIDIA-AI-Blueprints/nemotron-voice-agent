# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-server configuration of the delegation prototype (plan section 11).

``load_delegation_config(path)`` reads a delegation YAML chain (``extends:``,
``${VAR:-default}``, strict keys against :data:`DEFAULTS`), resolves ``voice_profile``
against the declaring file and loads it with the voice package's
``load_voice_config`` unchanged. The gateway has its own file (``gateway.yaml``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_delegation_agent.core.frontend.turn_class import ENGLISH, Lexicon, lexicon_from_mapping
from examples.frontend_delegation_agent.core.legacy_env import legacy_env_error
from examples.frontend_delegation_agent.core.prompt_features import FRONTEND_FEATURES, features
from examples.frontend_delegation_agent.core.text.config import interpolate_env
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig, deep_merge, load_voice_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "delegation_agent.yaml"
PROFILES_DIR = CONFIG_DIR / "profiles"

#: Keys resolved against the file that declares them.
IN_PATH_KEYS = ("voice_profile", "delegation.prompts.path", "backend.gateway_config")
#: Mappings whose keys are not checked (passed through as data).
_FREE_FORM = ("frontend.llm.extra_body", "delegation.guards.lexicon")
#: F1 replay triggers (fix-plan-f1-f2-f4-f8.md section 2.4); status_intent is the M2 path.
REPLAY_TRIGGERS = ("status_intent", "backchannel", "ping", "repeat")

DEFAULTS: dict[str, Any] = {
    "voice_profile": "voice/realtime_eval.yaml",
    "frontend": {
        "decider": "llm",
        "script": "",
        "llm": {
            "model": "nvidia/nvidia/nemotron-3.5-lightning",
            "base_url": "https://inference-api.nvidia.com/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 256,
            "timeout_s": 10,
            "extra_body": {},
        },
        "tool_choice": "named",
        "parallel_tool_calls": False,
        "timeout_ms": 4000,
        "hedge_after_ms": 1500,
        "warmup": True,
        "history": {"max_groups": 20},
        "backend_activity": {"max_items": 6, "preview_chars": 120},
    },
    "delegation": {
        "on_contract_error": "delegate",
        "max_repair_attempts": 1,
        "speak_when_delegating": True,
        # Single-turn evaluations: delegated fillers and proactive status lines are logged, not spoken (default off).
        "silent_frontend": False,
        "guards": {
            "backend_question_needs_delegate": True,
            "backchannel_words": [
                "mm-hmm",
                "mhm",
                "uh-huh",
                "uh huh",
                "hmm",
                "mm",
                "okay",
                "ok",
                "okay sure",
                "sure",
                "alright",
                "all right",
                "right",
                "thanks",
                "thank you",
                "got it",
                "cool",
            ],
            # fix-plan-f1-f2-f4-f8.md section 2.3 (F2), each off by default.
            "liveness_reply": False,  # R1: a silent reply to a ping or repeat request is delegated (or replayed)
            "question_never_silent": False,  # R2: a silent reply to a question is delegated
            "ping_as_status": False,  # R3: a ping while WORKING is a status request, never a steer
            # R4: a backchannel while the backend works gets a short acknowledgement line (off by default).
            "acknowledge_caller_while_working": False,
            "acknowledgement_max_per_run": 2,  # per backend run (fillers it replaces count too)
            "acknowledgement_min_interval_s": 5.0,  # between two acknowledgement lines
            "acknowledgement_min_since_agent_s": 0.0,  # since the agent last spoke (0: no extra gap)
            # Phrase lists per language (BCP-47 tag or base language); the session's ASR language picks one.
            # The backchannel list is always delegation.guards.backchannel_words.
            "lexicon": {"en": {key: value for key, value in ENGLISH.items() if key != "backchannel"}},
        },
        "prompts": {"path": "prompts/generic/frontend.yaml"},
        # Every behaviour below is off by default (the baseline).
        "spelling_hold": {"enabled": False, "hold_ms": 1500, "complete_patterns": []},
        "replay_unheard_answer": {
            "enabled": False,
            "ttl_s": 30,
            "triggers": ["status_intent"],
            "whole_if_heard_below": 0.0,
        },
        "filler_dedupe": {"enabled": False, "recent": 3},
    },
    "backend": {
        "link": "websocket",
        "url": "ws://localhost:8790/v1/backend",
        "open_timeout_s": 45,
        "gateway_config": "gateway.emulator.yaml",
        "simulated_delay": {"seconds": 0, "where": "per_delegation"},
        "steer_mode": "auto",
        "status_verbalizer": {"mode": "llm", "timeout_ms": 3000},
    },
    "session": {"update_after_start": "error"},
    "output": {
        "hold_max_ms": 4000,
        "on_stale": {"backend_answer": "keep", "status": "drop", "filler": "drop", "apology": "keep"},
        "proactive_status": {"enabled": False, "after_s": 15, "max_per_run": 1},
        "clean_answers": False,
    },
    "tools": {"executor": "wire", "result_timeout_s": 120, "batch_window_ms": 30},
    "instructions": {"apply_to": ["backend"]},
    # Frontend prompt variants ({% if features.<name> %} in prompts/<profile>/frontend.yaml); all off = the baseline.
    "prompt_features": dict(FRONTEND_FEATURES),
}

_ENUMS: dict[str, tuple[str, ...]] = {
    "frontend.decider": ("llm", "scripted"),
    "frontend.tool_choice": ("named", "required", "auto"),
    "delegation.on_contract_error": ("delegate", "direct_empty"),
    "backend.link": ("websocket", "in_process"),
    "backend.simulated_delay.where": ("per_delegation", "per_tool_call"),
    "backend.steer_mode": ("auto", "steer_only"),
    "backend.status_verbalizer.mode": ("llm", "template"),
    "session.update_after_start": ("error", "ignore"),
    "tools.executor": ("wire", "local"),
}
_STALE_KINDS = ("backend_answer", "status", "filler", "apology")


class DelegationConfigError(ValueError):
    """Invalid delegation configuration."""


@dataclass(frozen=True, slots=True)
class FrontendLLMConfig:
    """The frontend model endpoint."""

    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout_s: float
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    """The ``frontend`` section."""

    decider: str
    script: str
    llm: FrontendLLMConfig
    tool_choice: str
    parallel_tool_calls: bool
    timeout_ms: int
    hedge_after_ms: int
    warmup: bool
    history_max_groups: int
    activity_max_items: int
    activity_preview_chars: int


@dataclass(frozen=True, slots=True)
class SpellingHoldSettings:
    """``delegation.spelling_hold`` (M3.2): wait before deciding a turn that ends mid-spelling."""

    enabled: bool = False
    hold_ms: int = 1500
    #: Full-match patterns of a complete spelled value (e.g. a 6-character code): not held.
    complete_patterns: tuple[str, ...] = ()
    #: ``complete_patterns: auto``: derived per session from the identifier examples in the tool schemas.
    auto_patterns: bool = False


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    """``delegation.replay_unheard_answer`` (M2; F1 adds the triggers and the whole-answer threshold)."""

    enabled: bool = False
    ttl_s: float = 30.0
    #: When a replay is tried: status_intent (M2, the model's "status" in IDLE), backchannel, ping, repeat.
    triggers: tuple[str, ...] = ("status_intent",)
    #: Replay the whole answer when less than this share of its characters was heard (0 = the unheard part).
    whole_if_heard_below: float = 0.0


@dataclass(frozen=True, slots=True)
class LivenessSettings:
    """``delegation.guards`` F2 rules R1-R3 and the phrase lists (fix-plan-f1-f2-f4-f8.md section 2.3)."""

    liveness_reply: bool = False
    question_never_silent: bool = False
    ping_as_status: bool = False
    lexicons: dict[str, Lexicon] = field(default_factory=dict)

    @property
    def any_rule(self) -> bool:
        """Whether any of R1-R3 is on."""
        return self.liveness_reply or self.question_never_silent or self.ping_as_status


@dataclass(frozen=True, slots=True)
class AcknowledgeSettings:
    """``delegation.guards`` R4: a short acknowledgement line for a backchannel while the backend works."""

    enabled: bool = False
    max_per_run: int = 2
    min_interval_s: float = 5.0
    min_since_agent_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FillerDedupeSettings:
    """``delegation.filler_dedupe`` (G1)."""

    enabled: bool = False
    recent: int = 3


@dataclass(frozen=True, slots=True)
class DelegationSettings:
    """The ``delegation`` section."""

    on_contract_error: str
    max_repair_attempts: int
    speak_when_delegating: bool
    backend_question_needs_delegate: bool
    backchannel_words: tuple[str, ...]
    prompts_path: Path
    spelling_hold: SpellingHoldSettings = field(default_factory=SpellingHoldSettings)
    replay: ReplaySettings = field(default_factory=ReplaySettings)
    filler_dedupe: FillerDedupeSettings = field(default_factory=FillerDedupeSettings)
    liveness: LivenessSettings = field(default_factory=LivenessSettings)
    acknowledge: AcknowledgeSettings = field(default_factory=AcknowledgeSettings)
    silent_frontend: bool = False

    @property
    def classifies_turns(self) -> bool:
        """Whether a rule needs the turn class (F2 R1-R3 or an F1 trigger other than status_intent)."""
        replay_classes = self.replay.enabled and any(t != "status_intent" for t in self.replay.triggers)
        return self.liveness.any_rule or replay_classes


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """The ``backend`` section."""

    link: str
    url: str
    open_timeout_s: float
    gateway_config: Path | None
    delay_seconds: float
    delay_where: str
    steer_mode: str
    verbalizer_mode: str
    verbalizer_timeout_ms: int


@dataclass(frozen=True, slots=True)
class ProactiveStatusSettings:
    """``output.proactive_status`` (M1): one short line while the backend works in silence."""

    enabled: bool = False
    after_s: float = 15.0
    max_per_run: int = 1


@dataclass(frozen=True, slots=True)
class OutputSettings:
    """The ``output`` section."""

    hold_max_ms: int
    on_stale: dict[str, str]
    proactive_status: ProactiveStatusSettings = field(default_factory=ProactiveStatusSettings)
    #: G2: strip markdown and turn list items into sentences before an answer is spoken.
    clean_answers: bool = False


@dataclass(frozen=True, slots=True)
class ToolSettings:
    """The ``tools`` section (``tools.source`` itself lives in the voice profile)."""

    executor: str
    result_timeout_s: float
    batch_window_ms: int


@dataclass(frozen=True, slots=True)
class DelegationConfig:
    """Everything the delegation voice server needs."""

    voice: VoiceConfig
    frontend: FrontendConfig
    delegation: DelegationSettings
    backend: BackendSettings
    update_after_start: str
    output: OutputSettings
    tools: ToolSettings
    instructions_apply_to: tuple[str, ...]
    source_files: tuple[Path, ...] = ()
    effective: dict[str, Any] = field(default_factory=dict)
    prompt_features: dict[str, bool] = field(default_factory=lambda: dict(FRONTEND_FEATURES))

    @property
    def features(self) -> dict[str, Any]:
        """Every prompt variant and behaviour switch, for the deployed fingerprint (fda_session_start)."""
        dl, out = self.delegation, self.output
        return {
            "prompt_features": dict(self.prompt_features),
            "proactive_status": out.proactive_status.enabled,
            "clean_answers": out.clean_answers,
            "spelling_hold": dl.spelling_hold.enabled,
            "replay_unheard_answer": dl.replay.enabled,
            "replay_triggers": list(dl.replay.triggers) if dl.replay.enabled else [],
            "filler_dedupe": dl.filler_dedupe.enabled,
            "liveness_reply": dl.liveness.liveness_reply,
            "question_never_silent": dl.liveness.question_never_silent,
            "ping_as_status": dl.liveness.ping_as_status,
            "spelled_runs": self.voice.normalization.transcript.spelled_runs.enabled,
            "result_hints": self.voice.normalization.tool_arguments.enabled
            and self.voice.normalization.tool_arguments.result_hints.enabled,
            "field_provenance": self._field_provenance,
            "identity_ledger": self._identity_ledger,
            "format_retry": self._format_retry_mode,
            **{f"schema_{name}": on for name, on in self.schema_auto.items()},
        }

    @property
    def _field_provenance(self) -> bool:
        arguments = self.voice.normalization.tool_arguments
        return arguments.enabled and arguments.result_hints.enabled and arguments.result_hints.field_provenance

    @property
    def _identity_ledger(self) -> bool:
        """F8 (fix-plan-f1-f2-f4-f8.md section 4): the identification ledger's note after a failed lookup."""
        arguments = self.voice.normalization.tool_arguments
        return arguments.enabled and arguments.result_hints.enabled and arguments.result_hints.ledger

    @property
    def _format_retry_mode(self) -> str | bool:
        """``False`` when off, else the ``format_retry.mode`` in use (``call`` / ``hint``)."""
        arguments = self.voice.normalization.tool_arguments
        return arguments.format_retry.mode if arguments.enabled and arguments.format_retry.enabled else False

    @property
    def schema_auto(self) -> dict[str, bool]:
        """Which collaborators are derived per session from the tool schemas (``auto`` in config)."""
        arguments = self.voice.normalization.tool_arguments
        return {
            "rules": arguments.enabled and arguments.schema_rules,
            "result_hint_tools": arguments.enabled
            and arguments.result_hints.enabled
            and arguments.result_hints.auto_tools,
            "spelling_hold_patterns": self.delegation.spelling_hold.enabled
            and self.delegation.spelling_hold.auto_patterns,
            "format_retry_fields": arguments.enabled and arguments.format_retry.enabled,
        }

    @property
    def config_hash(self) -> str:
        """Short hash of the effective settings (secrets excluded), for grouping runs."""
        blob = json.dumps(self.effective, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_delegation_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DelegationConfig:
    """Load a delegation config file (or profile) and its voice profile."""
    if message := legacy_env_error():
        raise DelegationConfigError(message)
    merged, files = _load_chain(Path(path), ())
    _check_unknown(merged, DEFAULTS, files)
    return build_delegation_config(merged, files)


def build_delegation_config(merged: dict[str, Any], files: tuple[Path, ...] = ()) -> DelegationConfig:
    """Validate a merged mapping (paths already resolved) and load its voice profile."""
    if _get(merged, "backend.link") == "in_process_fake":
        raise DelegationConfigError("backend.link: in_process_fake was renamed to in_process")
    for dotted, allowed in _ENUMS.items():
        value = _get(merged, dotted)
        if value not in allowed:
            raise DelegationConfigError(f"{dotted}: {value!r} is not one of {list(allowed)}")
    voice_path = Path(str(merged["voice_profile"]))
    if not voice_path.is_absolute():
        voice_path = (CONFIG_DIR / voice_path).resolve()
    voice = load_voice_config(voice_path)
    fr, dl, be, out, tl = (merged[k] for k in ("frontend", "delegation", "backend", "output", "tools"))
    llm = fr["llm"]
    frontend = FrontendConfig(
        decider=fr["decider"],
        script=str(fr.get("script") or ""),
        llm=FrontendLLMConfig(
            model=str(llm["model"]),
            base_url=str(llm["base_url"]),
            api_key=str(llm.get("api_key") or ""),
            temperature=_float(llm["temperature"], "frontend.llm.temperature"),
            max_tokens=_int(llm["max_tokens"], "frontend.llm.max_tokens"),
            timeout_s=_float(llm["timeout_s"], "frontend.llm.timeout_s"),
            extra_body=dict(llm.get("extra_body") or {}),
        ),
        tool_choice=fr["tool_choice"],
        parallel_tool_calls=_bool(fr["parallel_tool_calls"], "frontend.parallel_tool_calls"),
        timeout_ms=_int(fr["timeout_ms"], "frontend.timeout_ms", minimum=1),
        hedge_after_ms=_int(fr["hedge_after_ms"], "frontend.hedge_after_ms"),
        warmup=_bool(fr["warmup"], "frontend.warmup"),
        history_max_groups=_int(fr["history"]["max_groups"], "frontend.history.max_groups", minimum=1),
        activity_max_items=_int(fr["backend_activity"]["max_items"], "frontend.backend_activity.max_items"),
        activity_preview_chars=_int(
            fr["backend_activity"]["preview_chars"], "frontend.backend_activity.preview_chars", minimum=10
        ),
    )
    delegation = DelegationSettings(
        on_contract_error=dl["on_contract_error"],
        max_repair_attempts=_int(dl["max_repair_attempts"], "delegation.max_repair_attempts"),
        speak_when_delegating=_bool(dl["speak_when_delegating"], "delegation.speak_when_delegating"),
        silent_frontend=_bool(dl["silent_frontend"], "delegation.silent_frontend"),
        backend_question_needs_delegate=_bool(
            dl["guards"]["backend_question_needs_delegate"], "delegation.guards.backend_question_needs_delegate"
        ),
        backchannel_words=tuple(str(word).lower() for word in dl["guards"]["backchannel_words"] or ()),
        prompts_path=_path(dl["prompts"]["path"]),
        spelling_hold=SpellingHoldSettings(
            enabled=_bool(dl["spelling_hold"]["enabled"], "delegation.spelling_hold.enabled"),
            hold_ms=_int(dl["spelling_hold"]["hold_ms"], "delegation.spelling_hold.hold_ms"),
            complete_patterns=()
            if dl["spelling_hold"]["complete_patterns"] == "auto"
            else tuple(
                _regex(pattern, "delegation.spelling_hold.complete_patterns")
                for pattern in dl["spelling_hold"]["complete_patterns"] or ()
            ),
            auto_patterns=dl["spelling_hold"]["complete_patterns"] == "auto",
        ),
        replay=_replay_settings(dl["replay_unheard_answer"]),
        filler_dedupe=FillerDedupeSettings(
            enabled=_bool(dl["filler_dedupe"]["enabled"], "delegation.filler_dedupe.enabled"),
            recent=_int(dl["filler_dedupe"]["recent"], "delegation.filler_dedupe.recent", minimum=1),
        ),
        liveness=_liveness_settings(dl["guards"]),
        acknowledge=_acknowledge_settings(dl["guards"]),
    )
    gateway_config = str(be.get("gateway_config") or "")
    backend = BackendSettings(
        link=be["link"],
        url=str(be["url"]),
        open_timeout_s=_float(be["open_timeout_s"], "backend.open_timeout_s", minimum=1),
        gateway_config=_path(gateway_config) if gateway_config else None,
        delay_seconds=_float(be["simulated_delay"]["seconds"], "backend.simulated_delay.seconds"),
        delay_where=be["simulated_delay"]["where"],
        steer_mode=be["steer_mode"],
        verbalizer_mode=be["status_verbalizer"]["mode"],
        verbalizer_timeout_ms=_int(be["status_verbalizer"]["timeout_ms"], "backend.status_verbalizer.timeout_ms"),
    )
    on_stale = dict(out["on_stale"])
    for kind, value in on_stale.items():
        if kind not in _STALE_KINDS or value not in ("keep", "drop"):
            raise DelegationConfigError(f"output.on_stale.{kind}: {value!r} (kinds {_STALE_KINDS}, keep | drop)")
    proactive = out["proactive_status"]
    output = OutputSettings(
        hold_max_ms=_int(out["hold_max_ms"], "output.hold_max_ms", minimum=1),
        on_stale=on_stale,
        proactive_status=ProactiveStatusSettings(
            enabled=_bool(proactive["enabled"], "output.proactive_status.enabled"),
            after_s=_float(proactive["after_s"], "output.proactive_status.after_s", minimum=0.1),
            max_per_run=_int(proactive["max_per_run"], "output.proactive_status.max_per_run", minimum=1),
        ),
        clean_answers=_bool(out["clean_answers"], "output.clean_answers"),
    )
    tools = ToolSettings(
        executor=tl["executor"],
        result_timeout_s=_float(tl["result_timeout_s"], "tools.result_timeout_s", minimum=1),
        batch_window_ms=_int(tl["batch_window_ms"], "tools.batch_window_ms"),
    )
    apply_to = tuple(merged["instructions"]["apply_to"] or ())
    if any(role not in ("backend", "frontend") for role in apply_to):
        raise DelegationConfigError(f"instructions.apply_to: {list(apply_to)} (allowed: backend, frontend)")
    # Cross-checks (plan 11.1).
    if tools.executor == "local" and voice.tools.source != "config":
        raise DelegationConfigError("tools.executor: local requires the voice profile's tools.source: config")
    if tools.executor == "wire" and voice.tools.source != "client":
        raise DelegationConfigError("tools.executor: wire requires the voice profile's tools.source: client")
    if backend.link == "in_process" and backend.gateway_config is None:
        raise DelegationConfigError("backend.link: in_process needs backend.gateway_config")
    try:
        prompt_features = dict(features(merged.get("prompt_features"), FRONTEND_FEATURES))
    except ValueError as exc:
        raise DelegationConfigError(f"prompt_features: {exc}") from exc
    if (
        delegation.replay.enabled
        and "status_intent" in delegation.replay.triggers
        and not prompt_features["replay_intent"]
    ):
        raise DelegationConfigError(
            "delegation.replay_unheard_answer.enabled needs prompt_features.replay_intent with the status_intent "
            "trigger (the frontend must be told when its last answer was not heard)"
        )
    effective = copy.deepcopy(merged)
    effective["frontend"]["llm"]["api_key"] = "***" if frontend.llm.api_key else ""
    if not delegation.silent_frontend:
        # Off: absent, so the config hash of every run without the switch stays unchanged.
        effective["delegation"].pop("silent_frontend", None)
    if not delegation.acknowledge.enabled:
        for key in ACKNOWLEDGE_KEYS:  # the same: R4 off leaves the config hash unchanged
            effective["delegation"]["guards"].pop(key, None)
    return DelegationConfig(
        voice=voice,
        frontend=frontend,
        delegation=delegation,
        backend=backend,
        update_after_start=merged["session"]["update_after_start"],
        output=output,
        tools=tools,
        instructions_apply_to=apply_to,
        source_files=(*files, *voice.source_files),
        effective=effective,
        prompt_features=prompt_features,
    )


def _replay_settings(section: dict[str, Any]) -> ReplaySettings:
    key = "delegation.replay_unheard_answer"
    triggers = tuple(str(t) for t in section.get("triggers") or ())
    unknown = [t for t in triggers if t not in REPLAY_TRIGGERS]
    if unknown:
        raise DelegationConfigError(f"{key}.triggers: {unknown} (allowed: {list(REPLAY_TRIGGERS)})")
    whole = _float(section.get("whole_if_heard_below", 0.0), f"{key}.whole_if_heard_below")
    if whole > 1.0:
        raise DelegationConfigError(f"{key}.whole_if_heard_below: {whole} is above 1")
    return ReplaySettings(
        enabled=_bool(section["enabled"], f"{key}.enabled"),
        ttl_s=_float(section["ttl_s"], f"{key}.ttl_s"),
        triggers=triggers,
        whole_if_heard_below=whole,
    )


#: The R4 keys of ``delegation.guards`` (left out of ``effective`` while R4 is off).
ACKNOWLEDGE_KEYS = (
    "acknowledge_caller_while_working",
    "acknowledgement_max_per_run",
    "acknowledgement_min_interval_s",
    "acknowledgement_min_since_agent_s",
)


def _acknowledge_settings(guards: dict[str, Any]) -> AcknowledgeSettings:
    key = "delegation.guards"
    return AcknowledgeSettings(
        enabled=_bool(guards["acknowledge_caller_while_working"], f"{key}.acknowledge_caller_while_working"),
        max_per_run=_int(guards["acknowledgement_max_per_run"], f"{key}.acknowledgement_max_per_run", minimum=1),
        min_interval_s=_float(guards["acknowledgement_min_interval_s"], f"{key}.acknowledgement_min_interval_s"),
        min_since_agent_s=_float(
            guards["acknowledgement_min_since_agent_s"], f"{key}.acknowledgement_min_since_agent_s"
        ),
    )


def _liveness_settings(guards: dict[str, Any]) -> LivenessSettings:
    backchannel = [str(word).lower() for word in guards["backchannel_words"] or ()]
    lexicons: dict[str, Lexicon] = {}
    for language, lists in (guards.get("lexicon") or {}).items():
        if not isinstance(lists, dict):
            raise DelegationConfigError(f"delegation.guards.lexicon.{language}: must be a mapping of phrase lists")
        try:
            lexicons[str(language)] = lexicon_from_mapping({**lists, "backchannel": backchannel})
        except ValueError as exc:
            raise DelegationConfigError(f"delegation.guards.lexicon.{language}: {exc}") from exc
    return LivenessSettings(
        liveness_reply=_bool(guards["liveness_reply"], "delegation.guards.liveness_reply"),
        question_never_silent=_bool(guards["question_never_silent"], "delegation.guards.question_never_silent"),
        ping_as_status=_bool(guards["ping_as_status"], "delegation.guards.ping_as_status"),
        lexicons=lexicons,
    )


# -- loading -----------------------------------------------------------------------------


def _load_chain(path: Path, seen: tuple[Path, ...]) -> tuple[dict[str, Any], tuple[Path, ...]]:
    path = path.expanduser().resolve()
    if path in seen:
        raise DelegationConfigError("extends cycle: " + " -> ".join(str(p) for p in (*seen, path)))
    if not path.exists():
        raise DelegationConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DelegationConfigError(f"{path}: config root must be a mapping")
    raw = interpolate_env(raw)
    for key in IN_PATH_KEYS:
        value = _get(raw, key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            _set(raw, key, str(candidate.resolve()))
    parent = raw.pop("extends", "") or ""
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged, files = _load_chain(parent_path, (*seen, path))
    else:
        merged, files = copy.deepcopy(DEFAULTS), ()
    return deep_merge(merged, raw, source=str(path), defaults=DEFAULTS), (*files, path)


def _check_unknown(merged: dict[str, Any], schema: dict[str, Any], files: tuple[Path, ...], prefix: str = "") -> None:
    for key, value in merged.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            raise DelegationConfigError(
                f"unknown key {dotted!r} (loaded from {', '.join(str(f) for f in files) or '<defaults>'})"
            )
        if dotted in _FREE_FORM:
            continue
        if isinstance(schema[key], dict) and isinstance(value, dict):
            _check_unknown(value, schema[key], files, dotted)


def _get(mapping: dict[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = mapping
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (CONFIG_DIR / path).resolve()


def _int(value: Any, key: str, *, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DelegationConfigError(f"{key}: {value!r} is not an integer") from exc
    if number < minimum:
        raise DelegationConfigError(f"{key}: {number} is below {minimum}")
    return number


def _float(value: Any, key: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DelegationConfigError(f"{key}: {value!r} is not a number") from exc
    if number < minimum:
        raise DelegationConfigError(f"{key}: {number} is below {minimum}")
    return number


def _regex(value: Any, key: str) -> str:
    pattern = str(value)
    try:
        re.compile(pattern)
    except re.error as exc:
        raise DelegationConfigError(f"{key}: {pattern!r} is not a valid regular expression ({exc})") from exc
    return pattern


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no"):
        return value.lower() in ("true", "1", "yes")
    raise DelegationConfigError(f"{key}: {value!r} is not a boolean")
