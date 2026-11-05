# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: Pipecat's ``NvidiaLLMService`` with the voice's structured decision, as generic builds its reply.

Each run of the talker in the pipeline is one call (:class:`TalkerCall`), which the session's voice chooses and acts
on (:class:`Calls`, ``examples.frontend_backend_live.voice.Voice``): a routing decision on the caller's newest turns,
a relay that words what the backend sent, a progress line about open work, or a line said with no model call (a
lead-in, a fixed line, the resume of a cut line). :class:`DecisionReplyMixin` follows the generic
example's ``examples.generic.silent_reply.AddressedReplyMixin``:

* ``build_chat_completion_params`` sets the call's request: its messages, in the layout the talker was measured on
  (:class:`Layout`), and a strict ``json_schema`` decision from the call's spec (``models.decision.decision_schema``),
  at the level the model's services entry declares;
* ``run_inference`` stays out of band: a one-shot inference sends what its arguments give, and never the pipeline
  call's format. A call the voice runs out of band (:meth:`DecisionReplyMixin.run_call`: a relay worded early, the
  normalizer) passes its messages as the context and the system instruction, and its decision schema as
  ``response_schema``; it is not streamed;
* ``get_chat_completions`` streams the decision's ``speech`` field through
  ``examples.shared.json_stream.JsonStringFieldStreamer``, as generic streams its ``reply`` field. As the omni
  Speaker lets its ``response`` through only once ``turn_action`` has decoded
  (``examples.omni_assistant_subagents.subagents.speaker.agent``), only a routing ``speak`` streams, and only once
  ``action`` has decoded: a ``delegate``'s speech is held as its lead-in, a ``wait`` says nothing, and the voice's
  other lines are held for its checks. When the stream ends, the voice reads the whole decision and acts on it, as the
  Speaker does; the words it lets out then go on as the response's last text, after the developer messages of the
  items their line says, so the context takes those right before the line.

A call that fails raises through Pipecat's own error path; the voice's fallback runs from the pipeline's error event,
and the talker stays usable (:class:`TalkerChat`). Each call is bounded by ``talker_attempt_timeout_seconds``, from its
request to the end of its decision, and its request and reply are logged with their timing.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService

from examples.frontend_backend_live.models.decision import (
    UNUSABLE,
    CallSpec,
    TalkerDecision,
    decision_from_output_text,
    decision_schema,
    unusable_decision,
)
from examples.shared.json_stream import JsonStringFieldStreamer

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import ModelEndpoint, ReliabilityConfig
    from examples.frontend_backend_live.prompts import PromptSet

# The metadata key under which the talker numbers each response's start and end frames, so whatever reads them after
# the output knows which response they belong to.
RESPONSE_KEY = "talker_response"
# A call's output limit when the services entry gives none.
DEFAULT_MAX_OUTPUT_TOKENS = 512

_out_of_band = contextvars.ContextVar("talker_out_of_band", default=False)
_current_call: contextvars.ContextVar[TalkerCall | None] = contextvars.ContextVar("talker_call", default=None)


class ResponseFormat(Protocol):
    """How a call asks the model for its decision, at the level the model's services entry declares."""

    name: str  # the level's name: the services entry's response_format

    def request(self, spec: CallSpec) -> dict | None:
        """The ``response_format`` a call in the pipeline sends, or ``None`` for none."""

    def schema(self, spec: CallSpec) -> dict | None:
        """The ``response_schema`` a call run out of band passes to ``run_inference``, or ``None`` for none."""


class JsonSchema:
    """``json_schema``: the strict decision schema the call's spec gives."""

    name = "json_schema"

    def request(self, spec: CallSpec) -> dict:
        """The strict ``voice_decision`` schema: the spec's actions, and its hint for ``speech``."""
        return {
            "type": "json_schema",
            "json_schema": {"name": "voice_decision", "strict": True, "schema": self.schema(spec)},
        }

    def schema(self, spec: CallSpec) -> dict:
        """The decision schema: the spec's actions, and its hint for ``speech``."""
        return decision_schema(spec)


class NoFormat:
    """``none``: no ``response_format``; the decision rules alone ask for the object."""

    name = "none"

    def request(self, spec: CallSpec) -> None:
        """None."""

    def schema(self, spec: CallSpec) -> None:
        """None."""


# The response_format levels, by the name a services entry's ``response_format`` gives. Nothing steps down at run time.
RESPONSE_FORMATS: dict[str, ResponseFormat] = {each.name: each for each in (JsonSchema(), NoFormat())}


@dataclass(eq=False)
class TalkerCall:
    """One run of the talker: what it asks the model for, and what the voice keeps to act on its decision."""

    spec: CallSpec | None = None  # the decision's spec; None: a line said with no model call
    messages: list[dict] = field(default_factory=list)  # the request's messages
    text: str = ""  # a line said with no model call
    streams: bool = False  # a ``speak``'s words go to speech as they arrive: a routing call's
    format: ResponseFormat = field(default_factory=JsonSchema)  # the services entry's level
    deadline: float = 0.0  # the loop time by which the decision must be read
    plan: Any = None  # what the voice keeps to act on the decision
    cut: bool = False  # a result arrived: the call says nothing if it has said nothing yet
    said: str = ""  # the speech already streamed to speech
    started: float = field(default_factory=time.monotonic)
    first_token_ms: float | None = None

    @property
    def role(self) -> str:
        """The call's name in the logs: its spec's role."""
        return self.spec.role if self.spec else "line"

    def response_format(self) -> dict | None:
        """The call's ``response_format``, at the level of the model's services entry."""
        return self.format.request(self.spec)

    def response_schema(self) -> dict | None:
        """The call's decision schema for ``run_inference``, at the level of the model's services entry."""
        return self.format.schema(self.spec)


class Calls(Protocol):
    """What the talker asks of the session's voice: which call each run is, and what happens when it ends."""

    async def next_call(self, context: LLMContext) -> TalkerCall | None:
        """The call this run makes, or ``None`` when there is nothing to decide or to say."""

    def speaking(self, call: TalkerCall, text: str) -> None:
        """A routing ``speak``'s words went on to speech."""

    def finished(self, call: TalkerCall, decision: TalkerDecision | None, held: str) -> str:
        """The call ended with ``decision`` (``None`` for a line said with no model call): the words to let out."""

    def written(self) -> list[dict]:
        """The messages that go into the context right before the words let out: the items their line says."""

    def dropped(self, call: TalkerCall) -> None:
        """A result arrived before the call said anything: it ends with nothing said."""

    def cancelled(self, call: TalkerCall) -> None:
        """An interruption cancelled the call."""

    def failed(self, call: TalkerCall, error: BaseException) -> None:
        """The call failed; Pipecat reports the error, and the voice's fallback runs from the error event."""

    def response_started(self) -> int:
        """The talker starts a response: its number."""

    def response_ended(self) -> int:
        """The talker ends the response in progress: its number."""

    def interrupted(self) -> None:
        """An interruption passed the talker: the responses before it are the ones it drops."""

    def caller_started(self) -> None:
        """The caller started speaking."""

    def caller_stopped(self) -> None:
        """The caller stopped speaking."""


class DecisionReplyMixin:
    """Ask each run of the talker for the voice's decision; stream a routing ``speak``, hold the rest for the voice.

    Built as ``examples.generic.silent_reply.AddressedReplyMixin``; see the module docstring.
    """

    calls: Calls  # the session's voice, bound by the frontend's build

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Send the call's request and decision format; one-shot inference keeps its own (generic's rule)."""
        params = super().build_chat_completion_params(params_from_context)
        call = None if _out_of_band.get() else _current_call.get()
        if call is not None:
            params["messages"] = call.messages
            response_format = call.response_format()
            if response_format is not None:
                params["response_format"] = response_format
            # Each call's exact messages, for reading a decision back from the logs.
            logger.debug(f"{call.role} request: {json.dumps(call.messages, ensure_ascii=False)}")
        return params

    async def run_inference(self, *args, **kwargs) -> str | None:
        """Run one-shot inference without the decision format, as generic does."""
        token = _out_of_band.set(True)
        try:
            return await super().run_inference(*args, **kwargs)
        finally:
            _out_of_band.reset(token)

    async def run_call(self, call: TalkerCall) -> str | None:
        """Run ``call`` out of band: its messages as the context and system instruction, its decision schema."""
        system, *history = call.messages
        # Each call's exact messages, for reading a decision back from the logs, as the pipeline call logs them.
        logger.debug(f"{call.role} request: {json.dumps(call.messages, ensure_ascii=False)}")
        reply = await self.run_inference(
            LLMContext(history), system_instruction=system["content"], response_schema=call.response_schema()
        )
        duration_ms = round((time.monotonic() - call.started) * 1000, 3)
        logger.debug(f"{call.role} reply (one-shot, {duration_ms} ms in all): {json.dumps(reply, ensure_ascii=False)}")
        return reply

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Run the call the voice chooses for this run: no request, a line with no model call, or the decision."""
        call = await self.calls.next_call(context)
        if call is None:
            return _chunks("")
        if call.spec is None:
            return self._line(self.calls.finished(call, None, ""))
        token = _current_call.set(call)
        try:
            async with asyncio.timeout_at(call.deadline):
                stream = await super().get_chat_completions(context)
        except asyncio.CancelledError:
            self.calls.cancelled(call)
            raise
        except Exception as exc:
            self.calls.failed(call, exc)
            raise
        finally:
            _current_call.reset(token)
        return self._decision_chunks(stream, call)

    async def _line(self, text: str) -> AsyncIterator[ChatCompletionChunk]:
        """A line said with no model call: what goes into the context with it, then its one chunk."""
        if text:
            await self._write_with_line()
            yield _text_chunk(text)

    async def _write_with_line(self) -> None:
        """Push the messages the voice writes with the line it lets out, so the context takes them right before it."""
        if messages := self.calls.written():
            await self.push_frame(LLMMessagesAppendFrame(messages=messages))

    async def process_frame(self, frame, direction: FrameDirection):
        """Tell the voice when the caller starts and stops speaking, and when an interruption passes the talker."""
        if isinstance(frame, InterruptionFrame):
            self.calls.interrupted()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.calls.caller_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.calls.caller_stopped()
        await super().process_frame(frame, direction)

    async def push_frame(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Number each response's start and end frames, so what reads them after the output knows the response."""
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                frame.metadata[RESPONSE_KEY] = self.calls.response_started()
            elif isinstance(frame, LLMFullResponseEndFrame):
                frame.metadata[RESPONSE_KEY] = self.calls.response_ended()
        await super().push_frame(frame, direction)

    async def _decision_chunks(
        self, stream: AsyncIterator[ChatCompletionChunk], call: TalkerCall
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Let a routing ``speak``'s words through as they decode; read the whole decision when the stream ends."""
        action, speech = JsonStringFieldStreamer("action"), JsonStringFieldStreamer("speech")
        raw = verb = held = ""
        finish = None
        chunks = stream.__aiter__()
        try:
            while True:
                try:
                    async with asyncio.timeout_at(call.deadline):
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
                choice = chunk.choices[0] if chunk.choices else None
                finish = (choice.finish_reason if choice is not None else None) or finish
                delta = choice.delta if choice is not None else None
                if delta is None or not delta.content:
                    yield chunk
                    continue
                if call.cut and not call.said:
                    self.calls.dropped(call)
                    return
                if call.first_token_ms is None:
                    call.first_token_ms = round((time.monotonic() - call.started) * 1000, 3)
                raw += delta.content
                if not action.done:
                    verb += action.feed(delta.content)
                text = speech.feed(delta.content)
                if call.streams and action.done and verb == "speak":
                    text, held = held + text, ""
                    if text:
                        call.said += text
                        self.calls.speaking(call, text)
                else:
                    held, text = held + text, ""
                delta.content = text or None
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            self.calls.cancelled(call)
            raise
        except Exception as exc:
            self.calls.failed(call, exc)
            raise
        finally:
            close = getattr(chunks, "aclose", None)
            if close is not None:
                await close()
        duration_ms = round((time.monotonic() - call.started) * 1000, 3)
        metadata = {
            "first_token_ms": call.first_token_ms,
            "duration_ms": duration_ms,
            "response_format": call.format.name,
        }
        # Each call's reply, and when its first token and its end came, for reading a decision and its timing back.
        logger.debug(
            f"{call.role} reply ({call.first_token_ms} ms to the first token, {duration_ms} ms in all): "
            f"{json.dumps(raw, ensure_ascii=False)}"
        )
        if finish == "length":
            decision = unusable_decision(metadata, "incomplete_output")
        else:
            decision = decision_from_output_text(raw, metadata)
        if text := self.calls.finished(call, decision, held):
            await self._write_with_line()
            yield _text_chunk(text)


class TalkerChat(DecisionReplyMixin, NvidiaLLMService):
    """``NvidiaLLMService`` that asks each run for the voice's decision (:class:`DecisionReplyMixin`).

    Its client takes ``client_timeout_seconds`` and ``provider_retries`` (the SDK's own retries); temperature is 0
    unless the services entry sets one, and each call's output limit is the entry's ``max_tokens``. A failed call never
    costs the talker its use: the voice's fallback answers it, as the voice always has, and the call goes on.
    """

    def __init__(self, *, reliability: ReliabilityConfig, response_format: ResponseFormat, client=None, **kwargs):
        """Create the service, asking at ``response_format``'s level.

        ``client`` replaces the model client the service makes (tests, probes).
        """
        self.reliability, self.response_format, self._injected_client = reliability, response_format, client
        super().__init__(**kwargs)
        self.add_event_handler("on_error", _stay_usable)

    @classmethod
    def from_config(
        cls, endpoint: ModelEndpoint, reliability: ReliabilityConfig, response_format: ResponseFormat, client=None
    ) -> TalkerChat:
        """The talker on a services entry's model endpoint, with the frontend's reliability settings.

        ``response_format`` is the level of :data:`RESPONSE_FORMATS` that the entry's ``response_format`` names.
        """
        extra = dict(endpoint.extra_params)
        settings = cls.Settings(
            model=endpoint.model,
            temperature=0.0,
            max_tokens=int(extra.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS),
            extra=extra,  # the entry's own sampling and reasoning settings win, as they always did
        )
        return cls(
            reliability=reliability,
            response_format=response_format,
            client=client,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            settings=settings,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """The model client, with the call's timeout and the SDK's retries."""
        if self._injected_client is not None:
            return self._injected_client
        client = super().create_client(api_key=api_key, base_url=base_url, **kwargs)
        return client.with_options(
            timeout=self.reliability.client_timeout_seconds, max_retries=self.reliability.provider_retries
        )


async def _stay_usable(talker: TalkerChat, error) -> None:
    """A failed call is answered by the voice's fallback: the talker keeps its use, so the call never ends on it."""
    await talker.set_usable(True)


# The talker services by the name ``frontend.cascade.provider`` gives.
LLM_SERVICES: dict[str, type[TalkerChat]] = {"chat-completions": TalkerChat}


async def _chunks(text: str) -> AsyncIterator[ChatCompletionChunk]:
    """A response with no request: ``text``'s one chunk, or nothing."""
    if text:
        yield _text_chunk(text)


def _text_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "talker-line",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


# ---------------------------------------------------------------------------------------------- the request layout
@dataclass(frozen=True)
class PromptPart:
    """One part of a call's instructions: where it comes from, its text, and what joins it to the part before it."""

    source: str
    text: str
    sep: str = ""


class Layout:
    """The request each call of the talker sends, in the layout it was measured on.

    The routing call reads the prompt set's frame, the instructions and the routing rules, then the conversation with
    the caller label in front of a caller turn right after reference data, then the runtime data as the last user
    entry. The relay and the progress line read the writer's frame and rules, the conversation as it is, then the
    update. The decision rules (``json_rules``) follow every call's instructions, and messages of one role in a row are
    joined, as the chat-completions talker always sent them.
    """

    def __init__(self, prompts: PromptSet, json_rules: str):
        """Lay requests out in ``prompts``' words, the decision rules ``json_rules`` after each call's instructions."""
        self.prompts, self.json_rules = prompts, json_rules

    def chat_messages(self, instructions: str, history: Sequence[dict]) -> list[dict]:
        """The request's messages: the instructions and the decision rules, then the history, roles in a row joined."""
        system = [instructions + "\n" + self.json_rules]
        messages: list[dict] = []
        for entry in history:
            role, content = entry["role"], entry["content"]
            if role in ("developer", "system"):
                system.append(content)
            elif messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})
        return [{"role": "system", "content": "\n\n".join(system)}, *messages]

    def routing_parts(self, instructions: Sequence[PromptPart]) -> list[PromptPart]:
        """The routing call's instructions, part by part in the order they are sent, each with its source."""
        prompts = self.prompts
        return [
            PromptPart("agent_prompts.Cascade.frame_before", prompts.frame_before),
            *instructions,
            PromptPart("agent_prompts.Cascade.frame_after", prompts.frame_after),
            PromptPart("agent_prompts.Cascade.routing", prompts.routing, "\n"),
        ]

    def runtime_data(self, work_line: str = "", turn: str = "") -> str:
        """The routing call's last user entry: the label, the work line (if any), the newest turn (if any), the cue."""
        prompts = self.prompts
        newest = prompts.newest_turn.format(turn=turn) + "\n" if turn else ""
        return prompts.runtime_label + "\n" + (f"{work_line}\n\n" if work_line else "") + newest + prompts.decision_cue

    def routing(self, instructions: str, history: list[dict], work_line: str, turn: str) -> list[dict]:
        """The routing call's request: ``history`` is the labeled conversation."""
        parts = self.routing_parts([PromptPart("instructions", instructions)])
        text = "".join(part.sep + part.text for part in parts)
        # After the caller's newest turn, a line break sets the runtime data apart: the two entries are joined.
        joined = bool(history) and history[-1]["role"] == "user"
        runtime = {"role": "user", "content": ("\n" if joined else "") + self.runtime_data(work_line, turn)}
        return self.chat_messages(text, [*history, runtime])

    def note(self, told: str | None, newer_requests: Sequence[str] = ()) -> str:
        """The notes an update carries: what the caller already heard of it, and newer requests still open."""
        notes = self.prompts.writer_notes
        text = notes.told.format(told=told) if told else ""
        if newer_requests:
            text += notes.newer.format(requests="; ".join(f'"{request}"' for request in newer_requests))
        return text

    def relay(self, instructions: str, history: list[dict], content: str, note: str) -> list[dict]:
        """The relay's request: the writer's frame and rules, the conversation, then the update to word."""
        prompts = self.prompts
        return self.chat_messages(
            prompts.writer_framed(instructions) + "\n" + prompts.commentary + prompts.answer,
            [*history, {"role": "user", "content": prompts.update_label + content + note}],
        )

    def progress(self, instructions: str, history: list[dict], update: str, note: str) -> list[dict]:
        """The progress line's request: the writer's frame and progress rules, the conversation, the work's update."""
        prompts = self.prompts
        return self.chat_messages(
            prompts.writer_framed(instructions) + "\n" + prompts.progress,
            [*history, {"role": "user", "content": prompts.update_label + update + note}],
        )


# ----------------------------------------------------------------------------------------------- the decisions
def floored(decision: TalkerDecision, *, work_open: bool, holding: str) -> TalkerDecision:
    """The routing decision the voice carries out: a turn that reached the talker never ends with nothing done.

    ``delegate``, ``speak`` with words, and ``wait`` stand; a ``speak`` with nothing to say becomes a ``wait``. Output
    the parser could not use counts as a failure (:func:`after_failure`).
    """
    action = decision.action
    if decision.diagnostics.get("recovery") in UNUSABLE:
        return after_failure(decision.diagnostics, work_open=work_open, holding=holding)
    if action == "delegate" or (action == "speak" and decision.speech.strip()):
        return decision
    if action == "wait":
        return TalkerDecision("wait", "", decision.diagnostics)
    return TalkerDecision("wait", "", {**decision.diagnostics, "recovery": "empty_speak"})


def after_failure(diagnostics: dict, *, work_open: bool, holding: str) -> TalkerDecision:
    """What a turn becomes when the talker's decision failed or said nothing.

    While work is open, the holding line, word for word: a delegation would join the work in progress, and could
    restart it, for a turn nobody decided. Otherwise a delegation: no tool has run at the talker stage, so delegating
    can never repeat an action.
    """
    if work_open and holding:
        held = {**diagnostics, "recovery": "holding", "failed": diagnostics.get("recovery")}
        return TalkerDecision("speak", holding, held)
    return TalkerDecision("delegate", "", diagnostics)


def plain_speech(text: str) -> str:
    """The backend's words without markup: lines become sentences, bullets and emphasis go; numbers and ids stay.

    A list marker goes only when a space follows it, so "12.50 is your total" and "-5 degrees" keep their numbers, and
    stacked markers ("## 1.", "- -") go together. Underscores stay, so an id such as Q91_T is said as written.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:(?:[-*•]+|\d+[.)]|#+)\s+)+", "", line)
        line = re.sub(r"[*`]+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines).replace(":.", ":")).strip()


def settled(text: str) -> str:
    """``text`` cleaned for speech until cleaning changes nothing.

    A cleaning can reveal a list marker (one that emphasis hid, or a line that is only a number), which the next one
    drops. After the first cleaning, each further one only drops characters, so the loop ends.
    """
    while (cleaned := plain_speech(text)) != text:
        text = cleaned
    return text


def relay_line(decision: TalkerDecision, content: str, history: Sequence[dict]) -> str | None:
    """The relay's worded line, or ``None`` when the update must be said as written.

    Not worded: any action but ``speak`` (unusable output parses as ``delegate``), no words, the update given back as
    written or cleaned for speech, or a line the voice already said in this call.
    """
    line = decision.speech
    said = {settled(entry["content"]) for entry in history if entry["role"] == "assistant"}
    if decision.action != "speak" or not line.strip() or settled(line) == settled(content) or settled(line) in said:
        return None
    return line


def progress_line(decision: TalkerDecision, latest: str | None) -> str:
    """The progress line, or nothing.

    A call that answers anything but ``speak``, carries a ``recovery`` or says the backend's newest progress line
    ``latest`` as written says nothing: a progress line holds no fact to keep.
    """
    line = decision.speech.strip()
    if (
        decision.action != "speak"
        or decision.diagnostics.get("recovery")
        or (line and latest and settled(line) == settled(latest))
    ):
        return ""
    return line
