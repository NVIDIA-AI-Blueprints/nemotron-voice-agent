# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reader of the example's ``prompts.yaml``: the talker prompts and every other text a model reads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

from pydantic import StrictStr, ValidationError

from examples.shared.strict_settings import StrictSettings, placed
from utils import load_yaml_file, resolve_prompt_catalog_path


class FixedLines(StrictSettings):
    """The voice's fixed lines (``agent_prompts.Lines``), said word for word but ``intro``, which the writer words.

    An empty line says nothing.
    """

    apology: StrictStr
    unclear: StrictStr
    lost: StrictStr
    refusal: StrictStr
    app_wait_line: StrictStr
    holding: StrictStr
    intro: StrictStr
    greeting: StrictStr


class WriterNotes(StrictSettings):
    """The notes added to an update for the writer, each only when it applies."""

    told: StrictStr
    newer: StrictStr


class CallHints(StrictSettings):
    """The description of ``speech`` in each call's JSON schema, by the call's role."""

    talker: StrictStr
    answer: StrictStr
    progress: StrictStr


class RoutingTools(StrictSettings):
    """The tool-calling talker's routing tools as YAML texts; ``None`` when the prompt file has no such entry."""

    delegate: StrictStr | None = None
    wait: StrictStr | None = None


class WorkTexts(StrictSettings):
    """The words of one open delegation's line in the work note."""

    item: StrictStr
    running: StrictStr
    ended: StrictStr
    ended_call: StrictStr
    steps: StrictStr
    waiting: StrictStr
    update: StrictStr


class TranscriptText(StrictSettings):
    """The words of a delegation's input: the transcript since the last request, one labeled line per entry."""

    header: StrictStr
    omitted: StrictStr
    caller: StrictStr
    agent: StrictStr
    context: StrictStr
    instruction: StrictStr
    earlier: StrictStr
    unheard: StrictStr


class PromptSet(StrictSettings):
    """The prompt set: each field is the ``agent_prompts.Cascade`` entry of its name; a ``None`` one may be left out.

    A dotted entry name (``work_texts.item``) is a field of a nested group.
    """

    routing: StrictStr
    commentary: StrictStr
    answer: StrictStr
    progress: StrictStr
    work_in_progress: StrictStr
    no_work_in_progress: StrictStr
    answer_ready_label: StrictStr
    typed_request_label: StrictStr
    work_texts: WorkTexts
    update_label: StrictStr
    result_label: StrictStr
    context_label: StrictStr
    cut_mark: StrictStr
    transcript: TranscriptText
    writer_notes: WriterNotes
    call_hints: CallHints
    frame_before: StrictStr
    frame_after: StrictStr
    writer_frame_before: StrictStr
    writer_frame_after: StrictStr
    caller_label: StrictStr
    runtime_label: StrictStr
    decision_cue: StrictStr
    newest_turn: StrictStr
    json_rules: StrictStr
    tool_frame_before: StrictStr | None = None
    tool_routing: StrictStr | None = None
    tool_rules: StrictStr | None = None
    tools: RoutingTools = RoutingTools()
    empty_turn_note: StrictStr

    def writer_framed(self, instructions: str) -> str:
        """Return the client's instructions as the writer reads them: word for word, inside the writer frame."""
        return self.writer_frame_before + instructions + self.writer_frame_after

    def missing(self, entries: Iterable[str]) -> list[str]:
        """Return the entries among ``entries`` (dotted: ``tools.delegate``) that the prompt file lacks."""
        return [entry for entry in entries if reduce(getattr, entry.split("."), self) is None]


@dataclass(frozen=True)
class Prompts:
    """One read of the example's prompt file: the prompt set, the lines and the core's texts, and the named prompts."""

    path: Path
    agent_prompts: Mapping[str, Any]
    cascade: PromptSet
    lines: FixedLines
    core: Mapping[str, str]

    def text(self, group: str, name: str) -> str:
        """Return entry ``name`` of ``agent_prompts.<group>``, stripped; raise ``KeyError`` if missing or blank."""
        entries = self.agent_prompts.get(group)
        entries = entries if isinstance(entries, Mapping) else {}
        entry = entries.get(name)
        content = entry.get("content") if isinstance(entry, Mapping) else None
        if not isinstance(content, str):
            known = ", ".join(map(str, entries)) or "none"
            raise KeyError(f"agent_prompts.{group}.{name} was not found in {self.path}; known: {known}")
        if not content.strip():
            raise KeyError(f"agent_prompts.{group}.{name} in {self.path} has no text")
        return content.strip()


def _texts(agent_prompts: Mapping[str, Any], group: str, path: Path) -> dict[str, str]:
    """Return each entry's ``content`` in ``agent_prompts.<group>`` by its name, as written."""
    entries = agent_prompts.get(group)
    if not isinstance(entries, Mapping):
        raise KeyError(f"agent_prompts.{group} was not found in {path}")
    texts = {}
    for name, entry in entries.items():
        content = entry.get("content") if isinstance(entry, Mapping) else None
        if not isinstance(content, str):
            raise ValueError(f"agent_prompts.{group}.{name} in {path} needs a content")
        texts[str(name)] = content
    return texts


def _group[T: StrictSettings](model: type[T], groups: Mapping[str, Any], group: str, path: Path) -> T:
    """Validate ``agent_prompts.<group>`` as ``model``, a dotted entry name (``work_texts.item``) as a nested field."""
    where = f"agent_prompts.{group} in {path}"
    nested = placed(_texts(groups, group, path), f"{where}: the entry")
    try:
        return model.model_validate(nested)
    except ValidationError as error:
        raise ValueError(f"{where}: {error}") from None


def load_prompts() -> Prompts:
    """Read the example's prompt file: ``PROMPT_FILE_PATH``, or the ``prompts.yaml`` beside this module."""
    path = resolve_prompt_catalog_path(__file__)
    groups = load_yaml_file(path, required=True).get("agent_prompts")
    groups = groups if isinstance(groups, Mapping) else {}
    cascade, lines = _group(PromptSet, groups, "Cascade", path), _group(FixedLines, groups, "Lines", path)
    return Prompts(path, groups, cascade, lines, _texts(groups, "Core", path))
