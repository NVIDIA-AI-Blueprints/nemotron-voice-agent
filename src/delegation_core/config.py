# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed settings of the delegation core: pure data, with no provider imported.

:class:`CoreSettings` is the core's own file (the example's ``core.yaml``, or the file ``CORE_CONFIG_PATH`` names),
which a composition root reads once and hands to ``app.build_core``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEndpoint:
    """The backend's model endpoint, resolved from a ``services.yaml`` entry and the session body.

    ``extra_params`` are passed to the client verbatim (``temperature``, ``max_tokens``, ``extra_body``,
    ``stream_options`` ...), so settings written there by ``settings.yaml`` take effect unchanged.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ the core's own file

# The outcome texts the IM returns to a conductor for each decision it applied (texts.outcomes).
OUTCOME_KEYS = ("started", "steered", "restarted", "cancel_requested", "ended", "answered", "refused")
# The IM's work tools a task may be offered (texts.work_tools): show, with the speak/display format; cancel_work, when
# delegation.work_tools lists it (it stops the task's own Work).
WORK_TOOL_KEYS = ("show", "cancel_work")


def _keys(where: str, raw: Any, allowed: Iterable[str], required: Iterable[str] = ()) -> Mapping[str, Any]:
    """``raw`` as a mapping whose keys are all in ``allowed`` and include ``required``; ``ValueError`` otherwise."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{where} is missing {', '.join(missing)}")
    return raw


def _integer(where: str, value: Any, *, minimum: int = 1) -> int:
    """A whole number of at least ``minimum``: a fraction or a boolean is refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be a whole number of at least {minimum}")
    return value


def _seconds(where: str, value: Any, *, zero: bool = False, null: bool = False) -> float | None:
    """A duration in seconds: positive (``zero`` allows 0; ``null`` allows ``None``, meaning no limit)."""
    if value is None and null:
        return None
    valid = not isinstance(value, bool) and isinstance(value, int | float) and (value > 0 or (zero and value == 0))
    if not valid:
        raise ValueError(f"{where} must be a {'non-negative' if zero else 'positive'} number of seconds")
    return float(value)


def _text(where: str, value: Any, *, empty: bool = False) -> str:
    """A string, non-empty unless ``empty``."""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{where} must be {'a string' if empty else 'non-empty text'}")
    return value


def _texts(where: str, raw: Any, *, empty: bool = False) -> Mapping[str, str]:
    """A mapping of names to strings."""
    mapping = _keys(where, raw, raw.keys() if isinstance(raw, Mapping) else ())
    return MappingProxyType({str(key): _text(f"{where}.{key}", value, empty=empty) for key, value in mapping.items()})


def _tools(where: str, raw: Any, names: Iterable[str]) -> Mapping[str, Mapping[str, Any]]:
    """Tool schemas by name: each with a description and JSON-schema parameters."""
    mapping = _keys(where, raw, names, names)
    tools = {}
    for name, schema in mapping.items():
        entry = _keys(f"{where}.{name}", schema, ("description", "parameters"), ("description", "parameters"))
        _text(f"{where}.{name}.description", entry["description"])
        if not isinstance(entry["parameters"], Mapping) or entry["parameters"].get("type") != "object":
            raise ValueError(f"{where}.{name}.parameters must be a JSON schema of type object")
        tools[name] = MappingProxyType(deepcopy(dict(entry)))
    return MappingProxyType(tools)


@dataclass(frozen=True)
class DelegationSettings:
    """The IM's own settings (``delegation`` in the core's file)."""

    max_pending_delegations: int = 16  # requests admitted and not yet running; the next one is refused
    client_tool_timeout_seconds: float | None = 60.0  # how long a task waits for the app's outputs; None: no deadline
    stop_timeout_seconds: float = 10.0  # how long the stop door waits for a run to confirm its stop
    close_grace_seconds: float | None = 5.0  # running Works at close; None: an active response may finish
    work_timeout_seconds: float | None = 600.0  # the backstop on one Work, through the stop door; None: off
    result_format: str = "plain"  # a name in texts.result_formats: plain, or speak_display (the envelope)
    # How calls the backend runs itself reach the wire: items (official mcp_call items) or notes (progress notes
    # only); a name the IM's table of call reports holds.
    backend_calls: str = "items"
    display_label: str = "voiceclaw"  # the server label of the show item; the root hands it to the reader too
    progress_note_seconds: float = 1.0  # at most one progress note per delegation this often; 0: no notes
    work_tools: tuple[str, ...] = ()  # optional work tools every task gets: [cancel_work] stops its own Work
    records_max_works: int = 5  # the recently ended Works the work records list, after the open ones
    context_tokens: int = 24_000  # the backend history's budget (fits a 32,768-token local window)
    tool_output_chars: int = 8_000  # a longer tool output enters the backend history cut to its head and tail
    voice_note: str = "v2"  # a name in texts.voice_notes: v2 describes the voice channel

    @classmethod
    def parse(cls, raw: Any) -> DelegationSettings:
        """Validate the ``delegation`` block; unknown keys, fractions where a whole number is due, are refused."""
        block = _keys("delegation", raw, {f.name for f in fields(cls)})
        values: dict[str, Any] = {}
        for key, value in block.items():
            where = f"delegation.{key}"
            if key in ("max_pending_delegations", "context_tokens", "tool_output_chars"):
                values[key] = _integer(where, value)
            elif key == "records_max_works":
                values[key] = _integer(where, value, minimum=0)
            elif key in ("client_tool_timeout_seconds", "close_grace_seconds", "work_timeout_seconds"):
                values[key] = _seconds(where, value, null=True)
            elif key == "stop_timeout_seconds":
                values[key] = _seconds(where, value)
            elif key == "progress_note_seconds":
                values[key] = _seconds(where, value, zero=True)
            elif key == "work_tools":
                if not isinstance(value, list):
                    raise ValueError(f"{where} must be a list of work tool names")
                values[key] = tuple(_text(f"{where}[]", name) for name in value)
            else:
                values[key] = _text(where, value)
        return cls(**values)


@dataclass(frozen=True)
class Texts:
    """Every line the core writes, sends or adds to a model's input (``texts``): the core's texts live in YAML."""

    voice_notes: Mapping[str, str]  # by name: v1 adds nothing; v2 is the voice-channel note
    result_formats: Mapping[str, str]  # by name: the note each result format adds (plain: none)
    handover_line: str  # the older delegation's commentary line when a newer request carries its Work
    cancelled_line: str  # the older delegation's answer when its Work stopped for a newer request, stop confirmed
    unconfirmed_line: str  # the same answer when the stop could not be confirmed
    owed_output: str  # the output a call that never reported gets: the outcome is unknown
    not_run_output: str  # the output of a call refused while newer input waits (the commit gate): it was not run
    stopped_output: str  # the output of a call refused once its work is stopping (a stop, or its own cancel_work)
    clash_output: str  # the output of a call refused because the app and a work tool name the same function
    records_block: str  # what the Works have done, on every Work's input and every steer: holds {records}
    closing_line: str  # after each records block: say where the work stands, then go on
    task_line: str  # a task's brief from its conductor: holds {task}
    conductor_prompt: str  # the instructions of a model conductor (the LLM agent's, the harness's main session)
    outcomes: Mapping[str, str]  # the outcome text the IM returns for each decision it applied (OUTCOME_KEYS)
    control_tools: Mapping[str, Mapping[str, Any]]  # the conductor's four tools (backend/control.py)
    work_tools: Mapping[str, Mapping[str, Any]]  # the IM's work tools (WORK_TOOL_KEYS)

    @classmethod
    def parse(cls, raw: Any) -> Texts:
        """Validate the ``texts`` block: every key present, every line text, the templates with their fields."""
        names = [f.name for f in fields(cls)]
        block = _keys("texts", raw, names, names)
        lines = {
            key: _text(f"texts.{key}", block[key])
            for key in names
            if key not in ("voice_notes", "result_formats", "outcomes", "control_tools", "work_tools")
        }
        for key, field_name in (("records_block", "{records}"), ("task_line", "{task}")):
            if field_name not in lines[key]:
                raise ValueError(f"texts.{key} must hold {field_name}")
        outcomes = _texts("texts.outcomes", block["outcomes"])
        missing = [key for key in OUTCOME_KEYS if key not in outcomes]
        if missing or set(outcomes) - set(OUTCOME_KEYS):
            raise ValueError(f"texts.outcomes must hold exactly {', '.join(OUTCOME_KEYS)}")
        control_tools = block["control_tools"]
        if not isinstance(control_tools, Mapping):
            raise ValueError("texts.control_tools must be a mapping")
        return cls(
            voice_notes=_texts("texts.voice_notes", block["voice_notes"], empty=True),
            result_formats=_texts("texts.result_formats", block["result_formats"], empty=True),
            outcomes=outcomes,
            control_tools=_tools("texts.control_tools", control_tools, control_tools.keys()),
            work_tools=_tools("texts.work_tools", block["work_tools"], WORK_TOOL_KEYS),
            **lines,
        )


@dataclass(frozen=True)
class BackendChoice:
    """Which backend adapter the core runs, and the blocks its source parses (``backend``).

    ``slot`` and ``prompt`` name the default model and instructions by reference; the composition root resolves them
    (the core does not read the example's catalog). Every other key is one adapter's own block, kept as written: its
    source validates it in ``from_settings``, and the core's builder refuses a block that names no adapter it has.
    """

    slot: str
    prompt: str
    adapter: str = "llm"  # a name in backend/adapters.py: llm (default), llm-agent or harness
    one_shot_adapter: str = "llm"  # the adapter that serves stateless requests (a port.OneShotSource)
    tools: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))  # name -> block
    blocks: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))  # adapter -> block

    @classmethod
    def parse(cls, raw: Any) -> BackendChoice:
        """Validate the ``backend`` block's own keys; keep each adapter's block for its source."""
        if not isinstance(raw, Mapping):
            raise ValueError("backend must be a mapping")
        own = ("adapter", "one_shot_adapter", "slot", "prompt", "tools")
        for key in ("slot", "prompt"):
            if key not in raw:
                raise ValueError(f"backend is missing {key}")
        blocks = {}
        for key, value in raw.items():
            if key not in own:
                if not isinstance(value, Mapping):
                    raise ValueError(f"backend.{key} must be a mapping: an adapter's block")
                blocks[str(key)] = MappingProxyType(deepcopy(dict(value)))
        tools = raw.get("tools", {})
        if not isinstance(tools, Mapping) or not all(isinstance(block, Mapping) for block in tools.values()):
            raise ValueError("backend.tools must map each tool source's name to its block")
        return cls(
            slot=_text("backend.slot", raw["slot"]),
            prompt=_text("backend.prompt", raw["prompt"]),
            adapter=_text("backend.adapter", raw.get("adapter", "llm")),
            one_shot_adapter=_text("backend.one_shot_adapter", raw.get("one_shot_adapter", "llm")),
            tools=MappingProxyType(
                {str(name): MappingProxyType(deepcopy(dict(block))) for name, block in tools.items()}
            ),
            blocks=MappingProxyType(blocks),
        )


@dataclass(frozen=True)
class CoreSettings:
    """Everything the core's own file sets: the IM's settings, its texts, and the backend it runs."""

    delegation: DelegationSettings
    texts: Texts
    backend: BackendChoice

    @classmethod
    def parse(cls, data: Any) -> CoreSettings:
        """Validate a parsed file. Names that point into ``texts`` must be there.

        Raises:
            ValueError: An unknown key, a missing one, a value of the wrong kind, or a name ``texts`` does not hold.
        """
        top = _keys("the core's settings", data, ("delegation", "texts", "backend"), ("texts", "backend"))
        settings = cls(
            delegation=DelegationSettings.parse(top.get("delegation", {})),
            texts=Texts.parse(top["texts"]),
            backend=BackendChoice.parse(top["backend"]),
        )
        delegation, texts = settings.delegation, settings.texts
        if delegation.voice_note not in texts.voice_notes:
            raise ValueError(f"delegation.voice_note {delegation.voice_note!r} is not in texts.voice_notes")
        if delegation.result_format not in texts.result_formats:
            raise ValueError(f"delegation.result_format {delegation.result_format!r} is not in texts.result_formats")
        unknown = set(delegation.work_tools) - set(texts.work_tools)
        if unknown:
            raise ValueError(
                f"delegation.work_tools names tools texts.work_tools has not: {', '.join(sorted(unknown))}"
            )
        return settings

    @classmethod
    def load(cls, path: Path) -> CoreSettings:
        """Read and validate the core's file at ``path`` (the root picks it: ``CORE_CONFIG_PATH`` or its default).

        A profile names the file it builds on with ``extends`` (a path relative to the profile's own folder) and sets
        only what differs: a mapping is merged key by key into the one it extends, and any other value replaces it
        (``null`` included, so a profile can turn a limit off).

        Raises:
            ValueError: A file cannot be read or parsed, ``extends`` loops or names no file, or :meth:`parse` refuses
                the merged settings.
        """
        return cls.parse(_profile(Path(path), ()))


def _profile(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    """The settings of the file at ``path``, merged over the file it ``extends`` (``seen``: the files extending it)."""
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"The core's settings {path} extend themselves: {' -> '.join(map(str, (*seen, resolved)))}")
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read the core's settings {path}: {exc}") from exc
    if not isinstance(data, dict) or "extends" not in data:
        return data
    data = dict(data)
    base = data.pop("extends")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"extends in {path} must name a file")
    extended = _profile(resolved.parent / base, (*seen, resolved))
    if not isinstance(extended, dict):
        raise ValueError(f"The core's settings {resolved.parent / base}, which {path} extends, must be a mapping")
    return _merged(extended, data)


def _merged(base: Any, changes: Any) -> Any:
    """``changes`` over ``base``: mappings merge key by key, anything else replaces."""
    if not isinstance(base, Mapping) or not isinstance(changes, Mapping):
        return deepcopy(changes)
    merged = {key: deepcopy(value) for key, value in base.items()}
    for key, value in changes.items():
        merged[key] = _merged(merged[key], value) if key in merged else deepcopy(value)
    return merged
