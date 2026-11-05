# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed settings of the delegation core: pure data, with no provider imported.

:class:`CoreSettings` is the core's own file (the example's ``core.yaml``, or the file ``CORE_CONFIG_PATH`` names) and
the core's texts (the ``agent_prompts.Core`` entries of the example's prompt file), which a composition root reads once
and hands to ``app.build_core``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from examples.frontend_backend_live.delegation_core.profiles import load_profile

# ------------------------------------------------------------------------------------------------ the core's own file

# The outcome texts the IM returns to a conductor for each decision it applied (texts.outcomes).
OUTCOME_KEYS = ("started", "steered", "restarted", "cancel_requested", "ended", "answered", "refused")
# The reason texts.outcomes.refused gives for each refusal (texts.refusal_reasons), with the fields each one is given:
# a place() after its decide() ended, requests none of which waits to be placed, an unknown Work or task, an answer
# while no Work is open, a start that would put more tasks to run than the backend runs at once, and a control call
# that reaches a harness's conductor before a decide() takes its calls.
REFUSAL_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ended": (),
        "unplaced": ("delegations",),
        "no_work": ("work",),
        "no_task": ("task", "work"),
        "no_open_work": (),
        "over_limit": ("max_running",),
        "idle": (),
    }
)
# The templates among the lines of texts, with the fields each one is given and must hold.
TEMPLATE_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "handover_line": ("delegation",),
        "records_block": ("records",),
        "task_line": ("task",),
        "unknown_tool_output": ("name", "offered"),
        "failed_output": ("name", "error"),
        "misrouted_output": ("name",),
        "repeated_call_output": ("output",),
        "cut_output": ("chars",),
        "conductor_prompt": ("max_running",),
    }
)
# The IM's work tools a task may be offered (texts.work_tools): show, with the speak_display format; cancel_work, when
# delegation.work_tools lists it (it stops the task after its current step).
WORK_TOOL_KEYS = ("show", "cancel_work")
# Where the core's texts are written: the message that refuses a key of the core's file at one of their places names it.
TEXTS_SOURCE = "agent_prompts.Core in the example's prompts.yaml"


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


def _flag(where: str, value: Any) -> bool:
    """``true`` or ``false``: any other value is refused."""
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be true or false")
    return value


def _text(where: str, value: Any, *, empty: bool = False) -> str:
    """A string, non-empty unless ``empty``."""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{where} must be {'a string' if empty else 'non-empty text'}")
    return value


def _texts(where: str, raw: Any, *, empty: bool = False) -> Mapping[str, str]:
    """A mapping of names to strings."""
    mapping = _keys(where, raw, raw.keys() if isinstance(raw, Mapping) else ())
    return MappingProxyType({str(key): _text(f"{where}.{key}", value, empty=empty) for key, value in mapping.items()})


def _template(where: str, text: str, names: Iterable[str]) -> str:
    """``text`` as a template that holds each field of ``names`` and names no other."""
    for name in names:
        if f"{{{name}}}" not in text:
            raise ValueError(f"{where} must hold {{{name}}}")
    try:
        text.format(**dict.fromkeys(names, ""))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"{where} names a field it is not given: {exc}") from None
    return text


def _tools(where: str, raw: Any, names: Iterable[str]) -> Mapping[str, Mapping[str, Any]]:
    """Tool schemas by name: each with a description and JSON-schema parameters, as a mapping or as its YAML text."""
    mapping = _keys(where, raw, names, names)
    tools = {}
    for name, schema in mapping.items():
        if isinstance(schema, str):
            try:
                schema = yaml.safe_load(schema)
            except yaml.YAMLError as exc:
                raise ValueError(f"{where}.{name} is not a schema in YAML: {exc}") from None
        entry = _keys(f"{where}.{name}", schema, ("description", "parameters"), ("description", "parameters"))
        _text(f"{where}.{name}.description", entry["description"])
        if not isinstance(entry["parameters"], Mapping) or entry["parameters"].get("type") != "object":
            raise ValueError(f"{where}.{name}.parameters must be a JSON schema of type object")
        tools[name] = MappingProxyType(deepcopy(dict(entry)))
    return MappingProxyType(tools)


def _placed(texts: Any) -> dict[str, Any]:
    """The core's texts as blocks: each text at the place its dotted name gives (``texts.outcomes.started``).

    Raises:
        ValueError: ``texts`` is not a mapping, or two of its names give one place.
    """
    if not isinstance(texts, Mapping):
        raise ValueError("the core's texts must map each text's place to the text")
    blocks: dict[str, Any] = {}
    for name, text in texts.items():
        *path, key = str(name).split(".")
        block = blocks
        for part in path:
            block = block.setdefault(part, {})
            if not isinstance(block, dict):
                break
        if not isinstance(block, dict) or key in block:
            raise ValueError(f"the core's text {name} names a place that another text holds")
        block[key] = text
    return blocks


def _joined(where: str, block: Any, texts: Any) -> dict[str, Any]:
    """The core file's ``block`` at ``where``, with the core's ``texts`` at their places, which the file may not set.

    Raises:
        ValueError: The file sets a place of a text, or of a block of texts; the message names it and where the core's
            texts are.
    """
    if not isinstance(block, Mapping) or not isinstance(texts, Mapping):
        raise ValueError(f"{where} comes from {TEXTS_SOURCE}")
    joined = deepcopy(dict(block))
    for key, text in texts.items():
        joined[key] = _joined(f"{where}.{key}", block[key], text) if key in block else deepcopy(text)
    return joined


@dataclass(frozen=True)
class DelegationSettings:
    """The IM's own settings (``delegation`` in the core's file)."""

    max_pending_delegations: int = 16  # requests admitted and not yet running; the next one is refused
    client_tool_timeout_seconds: float | None = 60.0  # how long a task waits for the app's outputs; None: no deadline
    stop_timeout_seconds: float = 10.0  # how long the stop door waits for a run to confirm its stop
    close_grace_seconds: float | None = 5.0  # how long running Works may go on at close; None: until they end
    work_timeout_seconds: float | None = 600.0  # the backstop on one Work, through the stop door; None: off
    result_format: str = "plain"  # a name in texts.result_formats: plain, or speak_display (the envelope)
    # How calls the backend runs itself reach the wire: items (official mcp_call items) or notes (progress notes
    # only); a name the IM's table of call reports holds.
    backend_calls: str = "items"
    display_label: str = "delegation_core"  # the server label of the show item; the root hands it to the frontend too
    progress_note_seconds: float = 1.0  # at most one progress note per delegation this often; 0: no notes
    work_tools: tuple[str, ...] = ()  # optional work tools every task gets: cancel_work stops the task after its step
    # An answer a task gives while a request waits for the conductor's decision waits for that decision: steered into
    # the task, it is progress and the task runs one more round with the request; placed elsewhere, it is written.
    hold_answers_while_deciding: bool = False
    records_max_works: int = 5  # the recently ended Works the work records list, after the open ones
    context_tokens: int = 24_000  # the backend history's budget (fits a 32,768-token local window)
    tool_output_chars: int = 8_000  # a longer tool output enters the backend history cut to its head and tail
    caller_note: str = "default"  # a name in texts.caller_notes: the note on how to work with the caller
    voice_note: str = "default"  # a name in texts.voice_notes: the note that describes the voice channel

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
            elif key == "hold_answers_while_deciding":
                values[key] = _flag(where, value)
            elif key == "work_tools":
                if not isinstance(value, list):
                    raise ValueError(f"{where} must be a list of work tool names")
                values[key] = tuple(_text(f"{where}[]", name) for name in value)
            else:
                values[key] = _text(where, value)
        return cls(**values)


@dataclass(frozen=True)
class Texts:
    """The lines the IM and the backend write or add to a model's input: the core's texts under ``texts``.

    The texts of a backend block, such as a harness's prompt templates (``backend.harness.prompts``) or what an app
    output step adds (``backend.app_output_steps``), join that block instead (:meth:`BackendChoice.parse`). An error's
    message (an exception's, a timed-out call's) comes from the code that reports it.
    """

    # By name: what the IM adds to every task's instructions about working with the caller, before the voice note.
    caller_notes: Mapping[str, str]
    voice_notes: Mapping[str, str]  # by name: what the IM adds to every task's instructions about the voice channel
    result_formats: Mapping[str, str]  # by name: the note each result format adds (plain: none)
    # The older delegation's commentary line when a newer request carries its Work: holds {delegation}, the newer
    # request's delegation id.
    handover_line: str
    cancelled_line: str  # the older delegation's answer when its Work stopped for a newer request, stop confirmed
    unconfirmed_line: str  # the same answer when the stop could not be confirmed
    owed_output: str  # the output a call that never reported gets: the outcome is unknown
    not_run_output: str  # the output of a call refused while newer input waits (the commit gate): it was not run
    stopped_output: str  # the output of a call refused once its work is stopping (a stop, or its own cancel_work)
    clash_output: str  # the output of a call refused because the app and a work tool name the same function
    # The error output of a call to a name no tool offers: holds {name} and {offered}, and says how to ask the caller to
    # do a step that is theirs.
    unknown_tool_output: str
    failed_output: str  # the error output of a call that raised: holds {name} and {error}
    misrouted_output: str  # the error output of an app function the IM routes to a harness's agent: holds {name}
    # The output of an app call that repeats the newest call that ran, which its task made and the app answered, with
    # no request since: holds {output}, that answer, which the IM gives instead of sending the call again.
    repeated_call_output: str
    cut_output: str  # between the head and the tail of a tool output cut to its size cap: holds {chars}, those left out
    none: str  # how an empty list of names reads in these texts: no tool offered, no request named
    records_block: str  # what the Works have done, on every Work's input and every steer: holds {records}
    closing_line: str  # after each records block: say where the work stands, then go on
    task_line: str  # a task's brief from its conductor: holds {task}
    round_cap_note: str  # added to the instructions of the backend LLM's round after max_tool_rounds: answer now
    # The instructions of a model conductor (the LLM agent's, the harness's conductor session): holds {max_running},
    # the tasks its backend runs at once.
    conductor_prompt: str
    outcomes: Mapping[str, str]  # the outcome text the IM returns for each decision it applied (OUTCOME_KEYS)
    refusal_reasons: Mapping[str, str]  # the {reason} of outcomes.refused for each refusal (REFUSAL_FIELDS)
    control_tools: Mapping[str, Mapping[str, Any]]  # the conductor's four tools (backend/control.py)
    work_tools: Mapping[str, Mapping[str, Any]]  # the IM's work tools (WORK_TOOL_KEYS)

    @classmethod
    def parse(cls, raw: Any) -> Texts:
        """Validate the ``texts`` block: every key present, every line text, the templates with their fields."""
        names = [f.name for f in fields(cls)]
        block = _keys("texts", raw, names, names)
        mappings = (
            "caller_notes",
            "voice_notes",
            "result_formats",
            "outcomes",
            "refusal_reasons",
            "control_tools",
            "work_tools",
        )
        lines = {key: _text(f"texts.{key}", block[key]) for key in names if key not in mappings}
        for key, template_fields in TEMPLATE_FIELDS.items():
            _template(f"texts.{key}", lines[key], template_fields)
        outcomes = _texts("texts.outcomes", block["outcomes"])
        missing = [key for key in OUTCOME_KEYS if key not in outcomes]
        if missing or set(outcomes) - set(OUTCOME_KEYS):
            raise ValueError(f"texts.outcomes must hold exactly {', '.join(OUTCOME_KEYS)}")
        _template("texts.outcomes.refused", outcomes["refused"], ("reason",))
        reasons = _texts("texts.refusal_reasons", block["refusal_reasons"])
        if set(reasons) != set(REFUSAL_FIELDS):
            raise ValueError(f"texts.refusal_reasons must hold exactly {', '.join(REFUSAL_FIELDS)}")
        for key, template_fields in REFUSAL_FIELDS.items():
            _template(f"texts.refusal_reasons.{key}", reasons[key], template_fields)
        control_tools = block["control_tools"]
        if not isinstance(control_tools, Mapping):
            raise ValueError("texts.control_tools must be a mapping")
        return cls(
            caller_notes=_texts("texts.caller_notes", block["caller_notes"], empty=True),
            voice_notes=_texts("texts.voice_notes", block["voice_notes"], empty=True),
            result_formats=_texts("texts.result_formats", block["result_formats"], empty=True),
            outcomes=outcomes,
            refusal_reasons=reasons,
            control_tools=_tools("texts.control_tools", control_tools, control_tools.keys()),
            work_tools=_tools("texts.work_tools", block["work_tools"], WORK_TOOL_KEYS),
            **lines,
        )

    def refusal(self, reason: str, **values: str) -> str:
        """The outcome of a refused decision: ``outcomes.refused`` with ``refusal_reasons[reason]`` filled in."""
        return self.outcomes["refused"].format(reason=self.refusal_reasons[reason].format(**values))


@dataclass(frozen=True)
class BackendChoice:
    """Which backend adapter the core runs, and the blocks its source parses (``backend``).

    ``slot`` and ``prompt`` name the default model and instructions by reference; the composition root resolves them
    (the core does not read the example's catalog). ``tools`` and ``app_output_steps`` name the server tool sources and
    the steps the app's outputs take, each with its block. Every other key is one adapter's own block, kept as written
    with its texts: its source validates it in ``from_settings``, and the core's builder refuses a block that names no
    adapter it has.
    """

    slot: str
    prompt: str
    adapter: str = "llm"  # a name in backend/adapters.py: llm (default), llm-agent or harness
    one_shot_adapter: str = "llm"  # the adapter that serves stateless requests (a port.OneShotSource)
    tools: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))  # name -> block
    # The steps the app's outputs take before the backend reads them: name -> block, in order.
    app_output_steps: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))
    blocks: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))  # adapter -> block

    @classmethod
    def parse(cls, raw: Any, texts: Mapping[str, Any] | None = None) -> BackendChoice:
        """Validate the ``backend`` block's own keys; keep each adapter's block for its source.

        ``texts`` are the texts of its blocks, at their places (``harness.prompts``, ``app_output_steps.shape_hints``).
        Each joins its block, an adapter's or an app output step's, when the settings have that block, so a block left
        out needs no change to the texts; a block that sets a place of a text itself is refused.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("backend must be a mapping")
        texts = texts or {}
        own = ("adapter", "one_shot_adapter", "slot", "prompt", "tools", "app_output_steps")
        for key in ("slot", "prompt"):
            if key not in raw:
                raise ValueError(f"backend is missing {key}")
        blocks = {}
        for key, value in raw.items():
            if key not in own:
                if not isinstance(value, Mapping):
                    raise ValueError(f"backend.{key} must be a mapping: an adapter's block")
                blocks[str(key)] = MappingProxyType(_joined(f"backend.{key}", value, texts.get(key, {})))
        tools = raw.get("tools", {})
        if not isinstance(tools, Mapping) or not all(isinstance(block, Mapping) for block in tools.values()):
            raise ValueError("backend.tools must map each tool source's name to its block")
        steps = raw.get("app_output_steps", {})
        if not isinstance(steps, Mapping) or not all(isinstance(block, Mapping) for block in steps.values()):
            raise ValueError("backend.app_output_steps must map each step's name to its block")
        step_texts = texts.get("app_output_steps", {})
        step_blocks = {
            str(name): MappingProxyType(_joined(f"backend.app_output_steps.{name}", block, step_texts.get(name, {})))
            for name, block in steps.items()
        }
        return cls(
            slot=_text("backend.slot", raw["slot"]),
            prompt=_text("backend.prompt", raw["prompt"]),
            adapter=_text("backend.adapter", raw.get("adapter", "llm")),
            one_shot_adapter=_text("backend.one_shot_adapter", raw.get("one_shot_adapter", "llm")),
            tools=MappingProxyType(
                {str(name): MappingProxyType(deepcopy(dict(block))) for name, block in tools.items()}
            ),
            app_output_steps=MappingProxyType(step_blocks),
            blocks=MappingProxyType(blocks),
        )


@dataclass(frozen=True)
class CoreSettings:
    """Everything the core's own file and its texts set: the IM's settings, its texts, and the backend it runs."""

    delegation: DelegationSettings
    texts: Texts
    backend: BackendChoice

    @classmethod
    def parse(cls, data: Any, texts: Mapping[str, Any]) -> CoreSettings:
        """Validate a parsed file and the core's texts as blocks. Names that point into ``texts`` must be there.

        ``texts`` holds the core's own texts under ``texts``, and the texts of backend blocks under ``backend``, each at
        its place in the settings (``backend.harness.prompts``): they join their blocks (:meth:`BackendChoice.parse`).

        Raises:
            ValueError: An unknown key, a missing one, a value of the wrong kind, a name ``texts`` does not hold, or a
                place of the texts that the file sets too (``texts``, or one in a backend block).
        """
        if isinstance(data, Mapping) and "texts" in data:
            raise ValueError(f"the core's settings has unknown keys: texts, which comes from {TEXTS_SOURCE}")
        top = _keys("the core's settings", data, ("delegation", "backend"), ("backend",))
        words = _keys("the core's texts", texts, ("texts", "backend"), ("texts",))
        settings = cls(
            delegation=DelegationSettings.parse(top.get("delegation", {})),
            texts=Texts.parse(words["texts"]),
            backend=BackendChoice.parse(top["backend"], words.get("backend", {})),
        )
        delegation, texts = settings.delegation, settings.texts
        if delegation.caller_note not in texts.caller_notes:
            raise ValueError(f"delegation.caller_note {delegation.caller_note!r} is not in texts.caller_notes")
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
    def load(cls, path: Path, texts: Mapping[str, str]) -> CoreSettings:
        """Read and validate the core's file at ``path`` (the root picks it: ``CORE_CONFIG_PATH`` or its default).

        A profile names the file it builds on with ``extends`` (a path relative to the profile's own folder) and sets
        only what differs: a mapping is merged key by key into the one it extends, and any other value replaces it
        (``null`` included, so a profile can turn a limit off).

        ``texts`` are the core's texts, each named by its place in the settings (``texts.handover_line``,
        ``backend.harness.prompts.user``): the root reads them from the example's prompt file (``agent_prompts.Core``),
        and the file may not set their places.

        Raises:
            ValueError: A file cannot be read or parsed, ``extends`` loops or names no file, two texts name one place,
                or :meth:`parse` refuses the merged settings.
        """
        return cls.parse(load_profile(Path(path)), _placed(texts))
