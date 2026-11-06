# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed settings of the delegation core: pydantic models, with no provider imported.

:class:`CoreSettings` is the core's own settings (the ``core`` block of the example's settings in
``examples_registry.yaml``) and the core's texts (the ``agent_prompts.Core`` entries of the example's prompt file),
which a composition root reads once and hands to ``app.build_core``. Each model holds every default, and an unknown key
or a value of the wrong kind is refused. The field types here are the ones the backend adapters' blocks use too
(``backend/*/settings.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Annotated, Any, TypedDict

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
    with_config,
)

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
# The IM's work tools a task may be offered (texts.work_tools): show, with the speak_display format; cancel_work, which
# every task of a conversation gets (it stops the task after its current step).
WORK_TOOL_KEYS = ("show", "cancel_work")
# Where the core's texts are written: the message that refuses a key of the core's settings at one of their places names
# it.
TEXTS_SOURCE = "agent_prompts.Core in the example's prompts.yaml"


# ------------------------------------------------------------------------------------------------ the field types
class Settings(BaseModel):
    """Base of the core's settings models: frozen, and an unknown key is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must be non-empty text")
    return text


# Non-empty text (not only spaces).
Text = Annotated[StrictStr, AfterValidator(_not_blank)]
# A whole number of at least 1, and one of at least 0: a fraction or a boolean is refused.
WholeNumber = Annotated[int, Field(strict=True, ge=1)]
Count = Annotated[int, Field(strict=True, ge=0)]
# A duration in seconds: positive, or (``NonNegativeSeconds``) 0 too. A whole number is a number of seconds as well.
Seconds = Annotated[float, Field(strict=True, gt=0)]
NonNegativeSeconds = Annotated[float, Field(strict=True, ge=0)]


def template(text: str, names: Iterable[str], key: str) -> str:
    """``text``, the template at ``key``, which holds each field of ``names`` and names no other.

    Raises:
        ValueError: The text lacks a field, or names one it is not given; the message names ``key``.
    """
    for name in names:
        if f"{{{name}}}" not in text:
            raise ValueError(f"{key} must hold {{{name}}}")
    try:
        text.format(**dict.fromkeys(names, ""))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"{key} names a field it is not given: {exc}") from None
    return text


def _from_yaml(schema: Any) -> Any:
    """A tool's schema written as its YAML text, read; any other value as it is."""
    if not isinstance(schema, str):
        return schema
    try:
        return yaml.safe_load(schema)
    except yaml.YAMLError as exc:
        raise ValueError(f"is not a schema in YAML: {exc}") from None


def _an_object(parameters: dict[str, Any]) -> dict[str, Any]:
    if parameters.get("type") != "object":
        raise ValueError("must be a JSON schema of type object")
    return parameters


@with_config(ConfigDict(extra="forbid"))
class ToolSchema(TypedDict):
    """One tool's schema in the core's texts, as a mapping or as its YAML text."""

    description: Text
    parameters: Annotated[dict[str, Any], AfterValidator(_an_object)]  # a JSON schema of type object


Tool = Annotated[ToolSchema, BeforeValidator(_from_yaml)]


@with_config(ConfigDict(extra="forbid"))
class Notes(TypedDict):
    """A note in the core's texts: its ``default`` text, which the IM adds to every task's instructions."""

    default: StrictStr


# ------------------------------------------------------------------------------------------------ the core's settings
class DelegationSettings(Settings):
    """The IM's own settings (``delegation`` in the core's settings)."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="delegation")

    max_pending_delegations: WholeNumber = 16  # requests admitted and not yet running; the next one is refused
    client_tool_timeout_seconds: Seconds | None = 60.0  # how long a task waits for the app's outputs; None: no deadline
    stop_timeout_seconds: Seconds = 10.0  # how long the stop door waits for a run to confirm its stop
    close_grace_seconds: Seconds | None = 5.0  # how long running Works may go on at close; None: until they end
    work_timeout_seconds: Seconds | None = 600.0  # the backstop on one Work, through the stop door; None: off
    result_format: Text = "plain"  # a name in texts.result_formats: plain, or speak_display (the envelope)
    display_label: Text = "delegation_core"  # the server label of the show item; the root hands it to the frontend too
    progress_note_seconds: NonNegativeSeconds = 1.0  # at most one progress note per delegation this often; 0: no notes
    records_max_works: Count = 5  # the recently ended Works the work records list, after the open ones
    context_tokens: WholeNumber = 24_000  # the backend history's budget (fits a 32,768-token local window)
    tool_output_chars: WholeNumber = 8_000  # a longer tool output enters the backend history cut to its head and tail


class Texts(Settings):
    """The lines the IM and the backend write or add to a model's input: the core's texts under ``texts``.

    The texts of a backend block, such as a harness's prompt templates (``backend.harness.prompts``) or what an app
    output step adds (``backend.app_output_steps``), join that block instead (:meth:`CoreSettings.parse`). An error's
    message (an exception's, a timed-out call's) comes from the code that reports it. Every key is required.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="texts")

    # What the IM adds to every task's instructions about working with the caller, before the voice note.
    caller_notes: Notes
    voice_notes: Notes  # what the IM adds to every task's instructions about the voice channel
    result_formats: dict[str, StrictStr]  # by name: the note each result format adds (plain: none)
    # The older delegation's commentary line when a newer request carries its Work: holds {delegation}, the newer
    # request's delegation id.
    handover_line: Text
    cancelled_line: Text  # the older delegation's answer when its Work stopped for a newer request, stop confirmed
    unconfirmed_line: Text  # the same answer when the stop could not be confirmed
    owed_output: Text  # the output a call that never reported gets: the outcome is unknown
    not_run_output: Text  # the output of a call refused while newer input waits (the commit gate): it was not run
    stopped_output: Text  # the output of a call refused once its work is stopping (a stop, or its own cancel_work)
    clash_output: Text  # the output of a call refused because the app and a work tool name the same function
    # The error output of a call to a name no tool offers: holds {name} and {offered}, and says how to ask the caller to
    # do a step that is theirs.
    unknown_tool_output: Text
    failed_output: Text  # the error output of a call that raised: holds {name} and {error}
    misrouted_output: Text  # the error output of an app function the IM routes to a harness's agent: holds {name}
    # The output of an app call that repeats the newest call that ran, which its task made and the app answered, with
    # no request since: holds {output}, that answer, which the IM gives instead of sending the call again.
    repeated_call_output: Text
    # Between the head and the tail of a tool output cut to its size cap: holds {chars}, those left out.
    cut_output: Text
    none: Text  # how an empty list of names reads in these texts: no tool offered, no request named
    records_block: Text  # what the Works have done, on every Work's input and every steer: holds {records}
    closing_line: Text  # after each records block: say where the work stands, then go on
    task_line: Text  # a task's brief from its conductor: holds {task}
    round_cap_note: Text  # added to the instructions of the backend LLM's round after max_tool_rounds: answer now
    # The instructions of a model conductor (the LLM agent's, the harness's conductor session): holds {max_running},
    # the tasks its backend runs at once.
    conductor_prompt: Text
    outcomes: dict[str, Text]  # the outcome text the IM returns for each decision it applied (OUTCOME_KEYS)
    refusal_reasons: dict[str, Text]  # the {reason} of outcomes.refused for each refusal (REFUSAL_FIELDS)
    control_tools: dict[str, Tool]  # the conductor's four tools (backend/control.py)
    work_tools: dict[str, Tool]  # the IM's work tools (WORK_TOOL_KEYS)

    @field_validator(*TEMPLATE_FIELDS)
    @classmethod
    def holds_its_fields(cls, text: str, info: ValidationInfo) -> str:
        """A template holds each field it is given (``TEMPLATE_FIELDS``) and names no other."""
        return template(text, TEMPLATE_FIELDS[info.field_name], info.field_name)

    @field_validator("outcomes")
    @classmethod
    def every_outcome(cls, outcomes: dict[str, str]) -> dict[str, str]:
        """Exactly the outcomes ``OUTCOME_KEYS`` names; ``refused`` holds {reason}."""
        if set(outcomes) != set(OUTCOME_KEYS):
            raise ValueError(f"outcomes must hold exactly {', '.join(OUTCOME_KEYS)}")
        template(outcomes["refused"], ("reason",), "outcomes.refused")
        return outcomes

    @field_validator("refusal_reasons")
    @classmethod
    def every_reason(cls, reasons: dict[str, str]) -> dict[str, str]:
        """Exactly the reasons ``REFUSAL_FIELDS`` names, each holding its fields."""
        if set(reasons) != set(REFUSAL_FIELDS):
            raise ValueError(f"refusal_reasons must hold exactly {', '.join(REFUSAL_FIELDS)}")
        for key, names in REFUSAL_FIELDS.items():
            template(reasons[key], names, f"refusal_reasons.{key}")
        return reasons

    @field_validator("work_tools")
    @classmethod
    def every_work_tool(cls, tools: dict[str, ToolSchema]) -> dict[str, ToolSchema]:
        """Exactly the work tools ``WORK_TOOL_KEYS`` names."""
        if set(tools) != set(WORK_TOOL_KEYS):
            raise ValueError(f"work_tools must hold exactly {', '.join(WORK_TOOL_KEYS)}")
        return tools

    def refusal(self, reason: str, **values: str) -> str:
        """The outcome of a refused decision: ``outcomes.refused`` with ``refusal_reasons[reason]`` filled in."""
        return self.outcomes["refused"].format(reason=self.refusal_reasons[reason].format(**values))


class BackendChoice(Settings):
    """Which backend adapter the core runs, and the blocks its source parses (``backend``).

    ``slot`` and ``prompt`` name the default model and instructions by reference; the composition root resolves them
    (the core does not read the example's catalog). ``tools`` and ``app_output_steps`` name the server tool sources and
    the steps the app's outputs take, each with its block. Every other key is one adapter's own block (``blocks``),
    kept as written with its texts: its source validates it in ``from_settings``, and the core's builder refuses a
    block that names no adapter it has.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend")

    slot: Text
    prompt: Text
    adapter: Text = "llm"  # a name in backend/adapters.py: llm (default), llm-agent or harness
    one_shot_adapter: Text = "llm"  # the adapter that serves stateless requests (a port.OneShotSource)
    tools: dict[str, dict[str, Any]] = {}  # name -> block
    # The steps the app's outputs take before the backend reads them: name -> block, in order.
    app_output_steps: dict[str, dict[str, Any]] = {}
    blocks: dict[str, dict[str, Any]] = {}  # adapter -> block: every key that is not one of the above

    @model_validator(mode="before")
    @classmethod
    def adapter_blocks(cls, raw: Any) -> Any:
        """Every key that is not one of the choice's own is one adapter's block."""
        if not isinstance(raw, Mapping):
            return raw
        own = set(cls.model_fields) - {"blocks"}
        blocks = {str(key): block for key, block in raw.items() if key not in own}
        return {**{key: value for key, value in raw.items() if key in own}, "blocks": blocks}


class CoreSettings(Settings):
    """Everything the core's own settings and its texts set: the IM's settings, its texts, and the backend it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="the core's settings")

    delegation: DelegationSettings = Field(default_factory=DelegationSettings)
    texts: Texts
    backend: BackendChoice

    @model_validator(mode="after")
    def names_its_texts(self) -> CoreSettings:
        """The result format the IM's settings name is one of the core's texts."""
        delegation, texts = self.delegation, self.texts
        if delegation.result_format not in texts.result_formats:
            raise ValueError(f"delegation.result_format {delegation.result_format!r} is not in texts.result_formats")
        return self

    @classmethod
    def parse(cls, data: Any, texts: Mapping[str, Any]) -> CoreSettings:
        """The core's settings ``data`` with its ``texts``, each text named by its place in the settings.

        ``texts`` names each text by its place (``texts.handover_line``, ``backend.harness.prompts.user``), as the
        example's prompt file words them, or holds them as blocks at their places. The core's own texts are
        :class:`Texts`; the texts of a backend block join that block, an adapter's or an app output step's, when the
        settings have it, so a block left out needs no change to the texts.

        Raises:
            ValueError: An unknown key, a missing one, a value of the wrong kind, a name ``texts`` does not hold, or a
                place of the texts that the settings set too (``texts``, or one in a backend block); the message names
                where the core's texts are.
        """
        if not isinstance(data, Mapping):
            raise ValueError("the core's settings must be a mapping")
        if "texts" in data:
            raise ValueError(f"the core's settings has unknown keys: texts, which comes from {TEXTS_SOURCE}")
        words = _placed(texts)
        unknown = set(words) - {"texts", "backend"}
        if unknown:
            raise ValueError(f"the core's texts name no place of its settings: {', '.join(sorted(unknown))}")
        values = {**data, **({"texts": words["texts"]} if "texts" in words else {})}
        if isinstance(data.get("backend"), Mapping):
            values["backend"] = _with_texts(data["backend"], words.get("backend", {}))
        return cls.model_validate(values)


# ------------------------------------------------------------------------------------------------ the texts' places
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


def _with_texts(backend: Mapping[str, Any], texts: Mapping[str, Any]) -> dict[str, Any]:
    """``backend`` with the texts of its blocks at their places: each adapter's block and each app output step's.

    A block that is not a mapping is left as it is, for :class:`BackendChoice` to refuse.
    """
    own = set(BackendChoice.model_fields) - {"blocks"}
    joined = dict(backend)
    for key, block in backend.items():
        if key == "app_output_steps" and isinstance(block, Mapping):
            steps = texts.get("app_output_steps", {})
            joined[key] = {
                name: _joined(f"backend.app_output_steps.{name}", step, steps.get(name, {}))
                if isinstance(step, Mapping)
                else step
                for name, step in block.items()
            }
        elif key not in own and isinstance(block, Mapping):
            joined[key] = _joined(f"backend.{key}", block, texts.get(key, {}))
    return joined


def _joined(where: str, block: Any, texts: Any) -> dict[str, Any]:
    """The settings' ``block`` at ``where``, with the core's ``texts`` at their places, which the settings may not set.

    Raises:
        ValueError: The settings set a place of a text, or of a block of texts; the message names it and where the
            core's texts are.
    """
    if not isinstance(block, Mapping) or not isinstance(texts, Mapping):
        raise ValueError(f"{where} comes from {TEXTS_SOURCE}")
    joined = deepcopy(dict(block))
    for key, text in texts.items():
        joined[key] = _joined(f"{where}.{key}", block[key], text) if key in block else deepcopy(text)
    return joined
