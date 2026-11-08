# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core's settings and texts as pydantic models, with every default; no provider is imported."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Annotated, Any, Literal

from loguru import logger
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from examples.shared.strict_settings import Seconds, Text, Tool, ToolSchema, WholeNumber, placed, with_blocks
from examples.shared.strict_settings import StrictSettings as Settings

# Each outcome text the IM returns to a conductor (texts.outcomes), with the fields it must hold.
OUTCOME_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "started": ("work", "tasks"),
        "steered": ("work", "tasks"),
        "restarted": ("work", "new"),
        "cancel_requested": ("work", "tasks"),
        "ended": ("work",),
        "answered": ("delegation",),
        "refused": ("reason",),
    }
)
# Each reason texts.outcomes.refused gives for a refused decision (texts.refusal_reasons), with the fields it must hold.
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
# The templates among the texts, with the fields each one must hold.
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
        "unheard_question_output": ("question",),
        "questions_block": ("questions",),
    }
)
# The core's texts added after a prompt file was written: a file may lack them, and the feature that needs each is off.
OPTIONAL_TEXTS: Mapping[str, str] = MappingProxyType(
    {
        "unheard_write_output": "the write gate",
        "unheard_undeclared_output": "the write gate's hold of undeclared tools",
        "unheard_question_output": "the write gate's open-question hold",
        "questions_block": "the questions block",
    }
)
WORK_TOOL_KEYS = ("show", "cancel_work")
TEXTS_SOURCE = "agent_prompts.Core in the example's prompts.yaml"

Count = Annotated[int, Field(strict=True, ge=0)]
NonNegativeSeconds = Annotated[float, Field(strict=True, ge=0)]


# What str.format raises for a field a template is not given, or one it reads past (``{name.x}``, ``{name[x]}``).
FORMAT_ERRORS = (KeyError, IndexError, AttributeError, TypeError, ValueError)


def template(text: str, names: Iterable[str], key: str) -> str:
    """Return the template ``text`` at ``key``; ``ValueError`` unless it holds each field of ``names`` and no other."""
    for name in names:
        if f"{{{name}}}" not in text:
            raise ValueError(f"{key} must hold {{{name}}}")
    try:
        text.format(**dict.fromkeys(names, ""))
    except FORMAT_ERRORS as exc:
        raise ValueError(f"{key} names a field it is not given: {exc}") from None
    return text


class WriteGateSettings(Settings):
    """The write gate (``delegation.write_gate``).

    An app call that can change data waits until the caller has heard the backend's answers in full and spoken after
    them, and no earlier question of the call went unheard.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="delegation.write_gate")

    enabled: StrictBool = False
    # Experimental, decided by the A/B: what part of the newest answer, left unheard, holds a change back: any part
    # (answer), or a sentence that asks (question).
    unheard: Literal["answer", "question"] = "answer"
    # A tool neither the app nor the operator declares: hold it on the gate's path, as MCP's default for a missing
    # readOnlyHint is "may change data" (hold), or let it run (run).
    undeclared: Literal["run", "hold"] = "hold"
    # A write the app declares easily undone (destructiveHint false).
    reversible: Literal["run", "hold"] = "run"


class ToolKindSettings(Settings):
    """What the operator declares about the app's tools (``delegation.tool_kinds``), for tools that declare nothing."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="delegation.tool_kinds")

    # Names or fnmatch patterns of tools that change nothing.
    read_tools: tuple[Text, ...] = ()
    # Names or fnmatch patterns of tools that may change data.
    write_tools: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def no_entry_in_both(self) -> ToolKindSettings:
        """Check that an entry is in one list only."""
        both = sorted(set(self.read_tools) & set(self.write_tools))
        if both:
            raise ValueError(f"delegation.tool_kinds lists {', '.join(both)} in both read_tools and write_tools")
        return self


class QuestionsSettings(Settings):
    """The questions block (``delegation.questions``): the backend's newest questions on later requests."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="delegation.questions")

    # Questions listed, newest last, after the records block, each with how often it was asked and heard in full; 0: no
    # block.
    shown: Count = 3


class DelegationSettings(Settings):
    """The IM's own settings (``delegation`` in the core's settings)."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="delegation")

    # Requests admitted and not yet running; the next one is refused.
    max_pending_delegations: WholeNumber = 16
    # How long the app's outputs may take, from the end of the response with their calls; None: no deadline.
    client_tool_timeout_seconds: Seconds | None = 60.0
    # How long the stop door waits for a run to confirm its stop.
    stop_timeout_seconds: Seconds = 10.0
    # How long running Works may go on at close; None: until they end.
    close_grace_seconds: Seconds | None = 5.0
    # The backstop on one Work, through the stop door; None: off.
    work_timeout_seconds: Seconds | None = 600.0
    # A name in texts.result_formats: plain, or speak_display (a show item, then the answer to speak).
    result_format: Text = "plain"
    # The server label of the show item; the root hands it to the frontend too.
    display_label: Text = "delegation_core"
    # At most one progress note per delegation this often; 0: no notes.
    progress_note_seconds: NonNegativeSeconds = 1.0
    # The recently ended Works the work records list, before the open ones.
    records_max_works: Count = 5
    # The backend history's budget in tokens (fits a 32,768-token local window).
    context_tokens: WholeNumber = 24_000
    # A longer tool output enters the backend history cut to its head and tail.
    tool_output_chars: WholeNumber = 8_000
    write_gate: WriteGateSettings = Field(default_factory=WriteGateSettings)
    tool_kinds: ToolKindSettings = Field(default_factory=ToolKindSettings)
    questions: QuestionsSettings = Field(default_factory=QuestionsSettings)


class Texts(Settings):
    """The lines the IM and the backend write or add to a model's input; prompts.yaml describes each one.

    Every key is required but the ones added after a prompt file was written (``OPTIONAL_TEXTS``): a file that lacks one
    loads, the feature that needs it is off, and one ``prompts.missing_text`` warning names the key and the feature.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="texts")

    caller_note: StrictStr
    voice_note: StrictStr
    result_formats: dict[str, StrictStr]
    handover_line: Text
    cancelled_line: Text
    unconfirmed_line: Text
    owed_output: Text
    not_run_output: Text
    stopped_output: Text
    clash_output: Text
    unlisted_output: Text
    unknown_tool_output: Text
    failed_output: Text
    misrouted_output: Text
    repeated_call_output: Text
    cut_output: Text
    none: Text
    records_block: Text
    closing_line: Text
    task_line: Text
    round_cap_note: Text
    # The output of an app call that can change data, refused by the write gate while the caller has not heard the
    # newest answer (also for a second attempt of a held declared write); without it, the write gate stays off.
    unheard_write_output: Text | None = None
    # The same for a tool nobody declared: call it again if it only looks something up, else describe the change again;
    # without it, undeclared tools run.
    unheard_undeclared_output: Text | None = None
    # The same while a question asked earlier in the call was never heard: holds {question}; without it, that reason
    # is off.
    unheard_question_output: Text | None = None
    # The backend's newest questions, after the records block: holds {questions}, one JSON line per question; without
    # it, no block is shown.
    questions_block: Text | None = None
    conductor_prompt: Text
    outcomes: dict[str, Text]
    refusal_reasons: dict[str, Text]
    control_tools: dict[str, Tool]
    work_tools: dict[str, Tool]

    @field_validator(*TEMPLATE_FIELDS)
    @classmethod
    def holds_its_fields(cls, text: str | None, info: ValidationInfo) -> str | None:
        """Check that a template holds each field it is given (``TEMPLATE_FIELDS``) and names no other.

        An optional template may be missing.
        """
        return text if text is None else template(text, TEMPLATE_FIELDS[info.field_name], info.field_name)

    @model_validator(mode="after")
    def warns_of_missing_texts(self) -> Texts:
        """Warn once (``prompts.missing_text``) for each optional text the core's texts lack."""
        for key, feature in OPTIONAL_TEXTS.items():
            if getattr(self, key) is None:
                logger.warning(f"prompts.missing_text key=texts.{key} feature={feature}: it is off")
        return self

    @field_validator("outcomes")
    @classmethod
    def every_outcome(cls, outcomes: dict[str, str]) -> dict[str, str]:
        """Check that the outcomes are exactly those ``OUTCOME_FIELDS`` names, each holding its fields."""
        if set(outcomes) != set(OUTCOME_FIELDS):
            raise ValueError(f"outcomes must hold exactly {', '.join(OUTCOME_FIELDS)}")
        for key, names in OUTCOME_FIELDS.items():
            template(outcomes[key], names, f"outcomes.{key}")
        return outcomes

    @field_validator("refusal_reasons")
    @classmethod
    def every_reason(cls, reasons: dict[str, str]) -> dict[str, str]:
        """Check that the reasons are exactly those ``REFUSAL_FIELDS`` names, each holding its fields."""
        if set(reasons) != set(REFUSAL_FIELDS):
            raise ValueError(f"refusal_reasons must hold exactly {', '.join(REFUSAL_FIELDS)}")
        for key, names in REFUSAL_FIELDS.items():
            template(reasons[key], names, f"refusal_reasons.{key}")
        return reasons

    @field_validator("work_tools")
    @classmethod
    def every_work_tool(cls, tools: dict[str, ToolSchema]) -> dict[str, ToolSchema]:
        """Check that the work tools are exactly those ``WORK_TOOL_KEYS`` names."""
        if set(tools) != set(WORK_TOOL_KEYS):
            raise ValueError(f"work_tools must hold exactly {', '.join(WORK_TOOL_KEYS)}")
        return tools

    def refusal(self, reason: str, **values: str) -> str:
        """Return the outcome of a refused decision: ``outcomes.refused`` with ``refusal_reasons[reason]`` filled in."""
        return self.outcomes["refused"].format(reason=self.refusal_reasons[reason].format(**values))


class BackendChoice(Settings):
    """Which backend adapter the core runs, and the blocks its source parses (``backend``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend")

    # The default model's services slot and the backend prompt's key, which the composition root resolves.
    slot: Text
    prompt: Text
    adapter: Text = "llm"
    one_shot_adapter: Text = "llm"
    tools: dict[str, dict[str, Any]] = {}
    app_output_steps: dict[str, dict[str, Any]] = {}
    # Each adapter's own block, kept as written for its source to validate.
    blocks: dict[str, dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def adapter_blocks(cls, raw: Any) -> Any:
        """Gather every key that is not one of the choice's own as one adapter's block."""
        if not isinstance(raw, Mapping):
            return raw
        return with_blocks(raw, set(cls.model_fields) - {"blocks"})


class CoreSettings(Settings):
    """Everything the core's own settings and its texts set: the IM's settings, its texts, and the backend it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="the core's settings")

    delegation: DelegationSettings = Field(default_factory=DelegationSettings)
    texts: Texts
    backend: BackendChoice

    @model_validator(mode="after")
    def names_its_texts(self) -> CoreSettings:
        """Check that the result format the IM's settings name is one of the core's texts."""
        delegation, texts = self.delegation, self.texts
        if delegation.result_format not in texts.result_formats:
            raise ValueError(f"delegation.result_format {delegation.result_format!r} is not in texts.result_formats")
        return self

    @classmethod
    def parse(cls, data: Any, texts: Mapping[str, Any]) -> CoreSettings:
        """Return the core's settings ``data`` with its ``texts``, each text named by its place in the settings.

        A backend block's texts join it only when the settings have the block. A ``ValueError`` names a bad key or
        value, or a place of the texts that the settings set too.
        """
        if not isinstance(data, Mapping):
            raise ValueError("the core's settings must be a mapping")
        if "texts" in data:
            raise ValueError(f"the core's settings has unknown keys: texts, which comes from {TEXTS_SOURCE}")
        if not isinstance(texts, Mapping):
            raise ValueError("the core's texts must map each text's place to the text")
        words = placed(texts, "the core's text")
        unknown = set(words) - {"texts", "backend"}
        if unknown:
            raise ValueError(f"the core's texts name no place of its settings: {', '.join(sorted(unknown))}")
        values = {**data, **({"texts": words["texts"]} if "texts" in words else {})}
        if isinstance(data.get("backend"), Mapping):
            values["backend"] = _with_texts(data["backend"], words.get("backend", {}))
        return cls.model_validate(values)


def _with_texts(backend: Mapping[str, Any], texts: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``backend`` with the texts of each adapter's and app output step's block at their places."""
    if not isinstance(texts, Mapping):
        raise ValueError(f"backend comes from {TEXTS_SOURCE}")
    own = set(BackendChoice.model_fields) - {"blocks"}
    joined = dict(backend)
    for key, block in backend.items():
        if key == "app_output_steps" and isinstance(block, Mapping):
            steps = texts.get("app_output_steps", {})
            if not isinstance(steps, Mapping):
                raise ValueError(f"backend.app_output_steps comes from {TEXTS_SOURCE}")
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
    """Return the settings' ``block`` at ``where`` with the core's ``texts`` at their places, which it may not set."""
    if not isinstance(block, Mapping) or not isinstance(texts, Mapping):
        raise ValueError(f"{where} comes from {TEXTS_SOURCE}")
    joined = deepcopy(dict(block))
    for key, text in texts.items():
        joined[key] = _joined(f"{where}.{key}", block[key], text) if key in block else deepcopy(text)
    return joined
