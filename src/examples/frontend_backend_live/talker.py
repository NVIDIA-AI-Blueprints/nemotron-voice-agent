# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: Pipecat's ``NvidiaLLMService`` with the voice's structured decision, as generic builds its reply.

Each run of the talker in the pipeline is one call (:class:`TalkerCall`), which the session's voice chooses and acts
on (:class:`Calls`, ``examples.frontend_backend_live.voice.Voice``): a routing decision on the caller's newest turns,
a relay that words what the backend sent, a progress line about open work, or a line said with no model call (a
lead-in, a fixed line, a line said once more). :class:`DecisionReplyMixin` follows the generic
example's ``examples.generic.silent_reply.AddressedReplyMixin``:

* ``build_chat_completion_params`` keeps the messages Pipecat builds from the talker's context and adjusts them only
  as the request goes out, as generic joins back-to-back speaker lines there and multilingual adds its reminder to the
  last user message on a copy: the call's own system text first, the caller label, reference data as user messages,
  back-to-back messages of one role joined, and the call's runtime data or update last, in the layout the talker was
  measured on (:class:`Request`). It adds the call's format's fields: a strict ``json_schema`` decision from the
  call's spec (``models.decision.decision_schema``), at the level the model's services entry declares, or the routing
  tools (:class:`ToolCalls`);
* ``run_inference`` stays out of band: a one-shot inference sends what its arguments give, and never a call's request
  or format. A call the voice runs out of band (:meth:`DecisionReplyMixin.run_call`: a relay worded early) passes the
  context the voice gives it and the call's system text as the system instruction, and its decision schema as
  ``response_schema``; its messages are adjusted as in the pipeline, and it is not streamed;
* ``get_chat_completions`` streams the decision's ``speech`` field through
  ``examples.shared.json_stream.JsonStringFieldStreamer``, as generic streams its ``reply`` field. As the omni
  Speaker lets its ``response`` through only once ``turn_action`` has decoded
  (``examples.omni_assistant_subagents.subagents.speaker.agent``), only a routing ``speak`` streams, and only once
  ``action`` has decoded: a ``delegate``'s speech is held as its lead-in, a ``wait`` says nothing, and the voice's
  other lines are held for its checks. When the stream ends, the voice reads the whole decision and acts on it, as the
  Speaker does; the words it lets out then go on as the response's last text, after the developer messages of the
  items their line says, so the context takes those right before the line.

The talker's style is its model's: the cascade's ``talker_style`` names its service in :data:`LLM_SERVICES` by the
model's services key. :class:`TalkerChat` makes every decision in JSON. :class:`TalkerTools` makes its routing decision
with Pipecat's function calling, as the Frontend/Backend Agent's talker calls ``call_backend``
(``examples.frontend_backend_agent.pipeline``): its routing call offers the routing tools (:class:`ToolCalls`), whose
handlers it registers with ``register_function`` (``examples.frontend_backend_live.tool_handlers``), and its other
calls keep the JSON decision. Each call's format reads its reply (:class:`Reply`): a routing call's plain text is a
``speak`` that streams as it arrives, and a call's fragments are held to the reply's end, as JSON speech is held until
its action decodes; once the voice has acted on the decision, the routing tool's call that made it goes to Pipecat,
which runs its handler and writes the call and its result into the talker's context.

A call that fails raises through Pipecat's own error path; the voice's fallback runs from the pipeline's error event,
and the talker stays usable (:class:`TalkerChat`). Each call is bounded by ``talker_attempt_timeout_seconds``, from its
request to the end of its decision, and its request and reply are logged with their timing.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import yaml
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
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
from pipecat.utils.text.alnum_utils import alnum_only

from examples.frontend_backend_live.models.decision import (
    TOOL_DECISIONS,
    UNUSABLE,
    CallSpec,
    TalkerDecision,
    decision_from_output_text,
    decision_schema,
    unusable_decision,
)
from examples.frontend_backend_live.tool_handlers import build_handlers
from examples.shared.json_stream import JsonStringFieldStreamer

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    from examples.frontend_backend_live.config_manager.schema import ModelEndpoint, ReliabilityConfig
    from examples.frontend_backend_live.prompts import PromptSet

# The metadata key under which the talker numbers each response's start and end frames, so whatever reads them after
# the output knows which response they belong to.
RESPONSE_KEY = "talker_response"
# A call's output limit when the services entry gives none.
DEFAULT_MAX_OUTPUT_TOKENS = 512

_out_of_band = contextvars.ContextVar("talker_out_of_band", default=False)
_current_call: contextvars.ContextVar[TalkerCall | None] = contextvars.ContextVar("talker_call", default=None)


class Reply(Protocol):
    """One streamed reply as its call reads it: the words that go to speech as it arrives, and its decision."""

    held: str  # the words it read and did not let out: a lead-in, a line held for the voice's checks
    raw: Any  # the reply as it came, for the log

    def feed(self, delta: ChoiceDelta) -> str:
        """Read one delta of the reply, which keeps only what Pipecat should read of it: the words to let out now."""

    def decision(self, metadata: dict) -> TalkerDecision:
        """The reply's decision once its stream has ended, with ``metadata`` in its diagnostics."""

    def handed(self) -> list[ChatCompletionChunk]:
        """What goes to Pipecat once the voice has acted on the decision: the routing tool's call that made it."""


class ResponseFormat(Protocol):
    """How a call asks the model for its decision, and how its reply becomes one.

    Every call asks at the level the model's services entry declares (:data:`RESPONSE_FORMATS`), except the routing
    call of the tool-calling talker, which offers the routing tools (:class:`ToolCalls`).
    """

    name: str  # the format's name: the services entry's response_format, or tools

    def request_fields(self, spec: CallSpec) -> dict:
        """The fields a call in the pipeline adds to its request after its messages."""

    def schema(self, spec: CallSpec) -> dict | None:
        """The ``response_schema`` a call run out of band passes to ``run_inference``, or ``None`` for none."""

    def rules(self, prompts: PromptSet) -> PromptPart:
        """The rules after a routing call's instructions, with their source."""

    def reply(self, streams: bool) -> Reply:
        """A reader for one reply; ``streams``: a ``speak``'s words go to speech as they arrive."""


class JsonDecision:
    """The JSON decision: the decision rules (``json_rules``) ask for one object, read as it streams."""

    def rules(self, prompts: PromptSet) -> PromptPart:
        """The decision rules: one JSON object with action and speech."""
        return PromptPart("agent_prompts.Cascade.json_rules", prompts.json_rules)

    def reply(self, streams: bool) -> JsonReply:
        """A reader of the JSON decision."""
        return JsonReply(streams)


class JsonSchema(JsonDecision):
    """``json_schema``: the strict decision schema the call's spec gives."""

    name = "json_schema"

    def request_fields(self, spec: CallSpec) -> dict:
        """The strict ``voice_decision`` schema as the ``response_format``: the spec's actions, its hint for speech."""
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "voice_decision", "strict": True, "schema": self.schema(spec)},
            }
        }

    def schema(self, spec: CallSpec) -> dict:
        """The decision schema: the spec's actions, and its hint for ``speech``."""
        return decision_schema(spec)


class NoFormat(JsonDecision):
    """``none``: no ``response_format``; the decision rules alone ask for the object."""

    name = "none"

    def request_fields(self, spec: CallSpec) -> dict:
        """Nothing."""
        return {}

    def schema(self, spec: CallSpec) -> None:
        """None."""


# The response_format levels, by the name a services entry's ``response_format`` gives. Nothing steps down at run time.
RESPONSE_FORMATS: dict[str, ResponseFormat] = {each.name: each for each in (JsonSchema(), NoFormat())}


class ToolCalls:
    """``tools``: the routing tools, with ``tool_choice`` "auto", so a reply in text with no call stays possible.

    Only the tool-calling talker's routing call asks this way (:class:`TalkerTools`). The tools go on its request alone,
    never on the context, which every call of the talker reads.
    """

    name = "tools"

    def __init__(self, tools: list[dict]):
        """Offer ``tools``: the routing tools as Pipecat's adapter converts them for the request."""
        self.tools = tools

    def request_fields(self, spec: CallSpec) -> dict:
        """The routing tools, and ``tool_choice`` "auto"."""
        return {"tools": self.tools, "tool_choice": "auto"}

    def schema(self, spec: CallSpec) -> None:
        """None: a routing call never runs out of band."""

    def rules(self, prompts: PromptSet) -> PromptPart:
        """The tool-calling talker's rules."""
        return PromptPart("agent_prompts.Cascade.tool_rules", prompts.tool_rules)

    def reply(self, streams: bool) -> ToolReply:
        """A reader of the reply's text and calls."""
        return ToolReply(streams)


def routing_tools(prompts: PromptSet) -> ToolsSchema:
    """The routing tools of ``prompts``: an OpenAI function entry for each name of ``TOOL_DECISIONS``, in its order.

    Each text (``tools.<name>``) is YAML, as the delegation core's tool texts are: a description, and the parameters as
    a JSON schema of type object. The entries are custom OpenAI tools, as the generic example's ``tools.yaml`` and the
    Frontend/Backend Agent's ``call_backend`` give theirs.

    Raises:
        ValueError: A text is not such a schema; the message names its entry.
    """
    texts, entries = asdict(prompts.tools), []
    for name in TOOL_DECISIONS:
        where = f"agent_prompts.Cascade.tools.{name}"
        try:
            schema = yaml.safe_load(texts[name])
        except yaml.YAMLError as exc:
            raise ValueError(f"{where} is not a schema in YAML: {exc}") from None
        if not _tool_schema(schema):
            raise ValueError(f"{where} must hold exactly a description and parameters, a JSON schema of type object")
        entries.append({"type": "function", "function": {"name": name, **schema}})
    return ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: entries})


def _tool_schema(schema: Any) -> bool:
    """Whether ``schema`` is a tool's text: a description, and parameters that are a JSON schema of type object."""
    return (
        isinstance(schema, Mapping)
        and set(schema) == {"description", "parameters"}
        and isinstance(schema["description"], str)
        and bool(schema["description"].strip())
        and isinstance(schema["parameters"], Mapping)
        and schema["parameters"].get("type") == "object"
    )


class JsonReply:
    """The JSON decision as it streams: a routing ``speak``'s words go out once its action has decoded."""

    def __init__(self, streams: bool):
        """Read the ``action`` and ``speech`` fields as they decode; ``streams``: a ``speak`` goes out as it does."""
        self.streams = streams
        self.action, self.speech = JsonStringFieldStreamer("action"), JsonStringFieldStreamer("speech")
        self.raw = self.verb = self.held = ""

    def feed(self, delta: ChoiceDelta) -> str:
        """A ``speak``'s words once its action has decoded, when the call streams; any other speech is held."""
        content = delta.content or ""
        self.raw += content
        if not self.action.done:
            self.verb += self.action.feed(content)
        text = self.speech.feed(content)
        if self.streams and self.action.done and self.verb == "speak":
            text, self.held = self.held + text, ""
            return text
        self.held += text
        return ""

    def decision(self, metadata: dict) -> TalkerDecision:
        """The reply's first JSON object (``models.decision.decision_from_output_text``)."""
        return decision_from_output_text(self.raw, metadata)

    def handed(self) -> list[ChatCompletionChunk]:
        """Nothing: a JSON decision makes no call."""
        return []


@dataclass
class ToolCall:
    """One call a reply makes, as its fragments arrive: its id, the tool's name, and its arguments as text."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def parsed(self) -> dict | None:
        """Its arguments: the JSON object they hold (none at all: an empty one), or ``None`` when they hold none."""
        try:
            arguments = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return None
        return arguments if isinstance(arguments, dict) else None

    def chunk(self) -> ChatCompletionChunk:
        """The call whole, in one chunk of its own, as Pipecat reads a call from a stream."""
        function = {"name": self.name, "arguments": self.arguments}
        call = {"index": 0, "id": self.id, "type": "function", "function": function}
        return ChatCompletionChunk.model_validate(
            {
                "id": "talker-call",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
            }
        )


class ToolReply:
    """A reply of the tool-calling talker's routing call: its text is a ``speak``, and a routing tool's call the move.

    The text streams from its first word as it arrives: white space before that word is never said. Each call's
    fragments are held by their index until the reply ends, so a call a result cuts short runs nothing. Then the first
    name of ``TOOL_DECISIONS`` that a call with usable arguments made is the move, and that call alone goes to Pipecat
    (:meth:`handed`); a reply with no such call is a ``speak`` with its words, or unusable. As stock Pipecat does, it
    says no text that shares a chunk with a call.
    """

    def __init__(self, streams: bool):
        """Read the reply's text and calls; ``streams``: its words go to speech as they arrive."""
        self.streams = streams
        self.calls: dict[int, ToolCall] = {}  # each call so far, by its index
        self.text = ""  # the text from its first word on
        self.held = ""  # the words not let out: all of them when the call does not stream
        self.made: ToolCall | None = None  # the call that made the move

    @property
    def raw(self) -> dict:
        """The reply as it came, for the log: its text, then each call's name and arguments."""
        calls = [{"name": call.name, "arguments": call.arguments} for call in self._in_order()]
        return {"content": self.text, "tool_calls": calls}

    def feed(self, delta: ChoiceDelta) -> str:
        """Hold the delta's call fragments, and say no text beside them; else its text from the reply's first word."""
        if delta.tool_calls:
            for fragment in delta.tool_calls:
                call = self.calls.setdefault(fragment.index, ToolCall())
                call.id = call.id or fragment.id or ""
                if fragment.function is not None:
                    call.name += fragment.function.name or ""
                    call.arguments += fragment.function.arguments or ""
            delta.tool_calls = None
            return ""
        text = delta.content if self.text else delta.content.lstrip()
        self.text += text
        if self.streams:
            return text
        self.held += text
        return ""

    def decision(self, metadata: dict) -> TalkerDecision:
        """The move of the first name in ``TOOL_DECISIONS`` that a call with usable arguments made, by index.

        ``TOOL_DECISIONS`` lists ``delegate`` first, so a reply that calls it delegates, whatever else it calls. With no
        such call, the words are a ``speak``; with none either, the reply is unusable: a call it could not use
        (``invalid_decision``: no name, a tool it does not offer, arguments that hold no object) or nothing at all.
        The diagnostics list every call's name, in order.
        """
        calls = self._in_order()
        diagnostics = {**metadata, "tool_calls": [call.name for call in calls]}
        for name, decide in TOOL_DECISIONS.items():
            for call in calls:
                arguments = call.parsed() if call.name == name else None
                if arguments is not None:
                    call.id = call.id or f"call_{uuid.uuid4().hex}"
                    self.made = call
                    return replace(decide(arguments, diagnostics), call_id=call.id)
        if self.text:
            return TalkerDecision("speak", self.text, diagnostics)
        return unusable_decision(diagnostics, "invalid_decision" if calls else "empty_output")

    def handed(self) -> list[ChatCompletionChunk]:
        """The call that made the move, whole, for Pipecat to run its handler; nothing when no call made it."""
        return [self.made.chunk()] if self.made is not None else []

    def _in_order(self) -> list[ToolCall]:
        return [self.calls[index] for index in sorted(self.calls)]


@dataclass(eq=False)
class TalkerCall:
    """One run of the talker: what it asks the model for, and what the voice keeps to act on its decision."""

    spec: CallSpec | None = None  # the decision's spec; None: a line said with no model call
    request: Request | None = None  # what the call does to Pipecat's messages as its request goes out
    text: str = ""  # a line said with no model call
    streams: bool = False  # a ``speak``'s words go to speech as they arrive: a routing call's
    format: ResponseFormat = field(default_factory=JsonSchema)  # the services entry's level, or the routing tools
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

    def request_fields(self) -> dict:
        """The fields the call's request adds after its messages: its format's ``response_format``, or its tools."""
        return self.format.request_fields(self.spec)

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

    async def hand_off(self, call_id: str) -> bool | None:
        """Whether the delegation part accepted the turn the ``delegate`` call ``call_id`` delegated, once it answered.

        The call's handler asks; ``None``: the call handed nothing over.
        """

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

    calls: Calls  # the session's voice, bound by the frontend's build (:meth:`bind`)
    response_format: ResponseFormat  # the services entry's level, which every call but its own routing call asks at

    def bind(self, calls: Calls) -> None:
        """Ask ``calls``, the session's voice, which call each run makes, and tell it how each one ends."""
        self.calls = calls

    def routing_format(self, prompts: PromptSet) -> ResponseFormat:
        """How a routing call asks for its decision: in JSON, at the services entry's level, as every other call."""
        return self.response_format

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Adjust Pipecat's messages as the call's request goes out, with its format's fields in the pipeline.

        One-shot inference with no call keeps its own request (generic's rule); a call run out of band takes its
        decision schema from ``run_inference``'s ``response_schema``.
        """
        params = super().build_chat_completion_params(params_from_context)
        call = _current_call.get()
        if call is not None:
            params["messages"] = call.request.messages(params["messages"])
            if not _out_of_band.get():
                params.update(call.request_fields())
            # Each call's exact messages, for reading a decision back from the logs.
            logger.debug(f"{call.role} request: {json.dumps(params['messages'], ensure_ascii=False)}")
        return params

    async def run_inference(self, *args, **kwargs) -> str | None:
        """Run one-shot inference with its own request and no decision format, as generic does."""
        return await self._one_shot(None, *args, **kwargs)

    async def run_call(self, call: TalkerCall, context: LLMContext) -> str | None:
        """Run ``call`` out of band on ``context``: its system text as the system instruction, its decision schema."""
        reply = await self._one_shot(
            call, context, system_instruction=call.request.system, response_schema=call.response_schema()
        )
        duration_ms = round((time.monotonic() - call.started) * 1000, 3)
        logger.debug(f"{call.role} reply (one-shot, {duration_ms} ms in all): {json.dumps(reply, ensure_ascii=False)}")
        return reply

    async def _one_shot(self, call: TalkerCall | None, *args, **kwargs) -> str | None:
        """Pipecat's one-shot inference, out of band, adjusted as ``call``'s request (none: as its arguments give)."""
        out_of_band, current = _out_of_band.set(True), _current_call.set(call)
        try:
            return await super().run_inference(*args, **kwargs)
        finally:
            _current_call.reset(current)
            _out_of_band.reset(out_of_band)

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
        """Let a routing ``speak``'s words through as they arrive; read the whole decision when the stream ends.

        The call's format reads each delta (:class:`Reply`): only the words it lets out stay in the chunk Pipecat reads.
        Once the voice has acted on the decision, what the reply hands Pipecat follows: the routing tool's call that
        made it, whose handler Pipecat then runs.
        """
        reply = call.format.reply(call.streams)
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
                if delta is None or not (delta.content or delta.tool_calls):
                    yield chunk
                    continue
                if call.cut and not call.said:
                    self.calls.dropped(call)
                    return
                if call.first_token_ms is None:
                    call.first_token_ms = round((time.monotonic() - call.started) * 1000, 3)
                text = reply.feed(delta)
                if text:
                    call.said += text
                    self.calls.speaking(call, text)
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
            f"{json.dumps(reply.raw, ensure_ascii=False)}"
        )
        if finish == "length":
            decision = unusable_decision(metadata, "incomplete_output")
        else:
            decision = reply.decision(metadata)
        text = self.calls.finished(call, decision, reply.held)
        for handed in reply.handed():
            yield handed
        if text:
            await self._write_with_line()
            yield _text_chunk(text)


class TalkerChat(DecisionReplyMixin, NvidiaLLMService):
    """``NvidiaLLMService`` that asks each run for the voice's decision (:class:`DecisionReplyMixin`).

    Its client is Pipecat's, with the SDK's timeout and ``provider_retries`` (the SDK's own retries); temperature is 0
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
        """The model client Pipecat makes, as generic's, with ``provider_retries`` as the SDK's retries.

        The talker's attempt deadline (``talker_attempt_timeout_seconds``) bounds each call.
        """
        if self._injected_client is not None:
            return self._injected_client
        client = super().create_client(api_key=api_key, base_url=base_url, **kwargs)
        return client.with_options(max_retries=self.reliability.provider_retries)


class TalkerTools(TalkerChat):
    """``TalkerChat`` whose routing call makes its move with Pipecat's function calling (:class:`ToolCalls`).

    Built as the Frontend/Backend Agent's talker with its ``call_backend`` (``examples.frontend_backend_agent``): the
    routing tools' handlers (``examples.frontend_backend_live.tool_handlers``) are registered with ``register_function``
    and ``cancel_on_interruption=False``. Its plain text is a ``speak``; a ``delegate`` call hands the turn over, its
    ``lead_in`` said once the delegation part has accepted it, after the part's delay; a ``wait`` call says nothing. The
    relay and the progress line keep the JSON decision, at the services entry's level. It needs an endpoint that streams
    the calls in ``tool_calls``.
    """

    def bind(self, calls: Calls) -> None:
        """Ask ``calls``, the session's voice, about each run, and register a handler for each routing tool."""
        super().bind(calls)
        for name, handler in build_handlers(calls).items():
            self.register_function(name, handler, cancel_on_interruption=False)

    def routing_format(self, prompts: PromptSet) -> ToolCalls:
        """The routing tools of ``prompts``, as Pipecat's adapter converts a context's tools for the request.

        Raises:
            ValueError: A tool's text is not a schema (:func:`routing_tools`); the message names its entry.
        """
        return ToolCalls(self.get_llm_adapter().from_standard_tools(routing_tools(prompts)))


async def _stay_usable(talker: TalkerChat, error) -> None:
    """A failed call is answered by the voice's fallback: the talker keeps its use, so the call never ends on it."""
    await talker.set_usable(True)


# The talker's services by its style, which the cascade's settings name by the model's services key (``talker_style``):
# ``json`` makes every decision in JSON, ``tools`` the routing decision with function calling. Nothing else picks one.
LLM_SERVICES: dict[str, type[TalkerChat]] = {"json": TalkerChat, "tools": TalkerTools}


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
# The fields of a tool call and of its result in a context message, as Pipecat writes them.
CALL_FIELDS = ("tool_calls", "tool_call_id")


def message_text(message: dict) -> str:
    """A context message's text, its parts joined when the content is a list."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


@dataclass(frozen=True)
class Request:
    """What a call does to the messages Pipecat builds from the talker's context as its request goes out.

    The talker's context holds the caller's turns (``user``), the lines that played (``assistant``), reference data
    (``developer``: backend results, the application's context and typed questions), and the tool-calling talker's
    routing calls and their results (an ``assistant`` message with ``tool_calls``, a ``tool`` message), as Pipecat
    writes them. In order: the call's own system text goes first, in place of any system message Pipecat put there;
    ``label`` goes in front of a caller turn right after reference data; reference data reads as the caller's side
    (``user``); messages of one role in a row are joined with a line break; and ``tail`` joins the last message after
    ``joiner`` when that is a user message, else follows it as one. A request that keeps the ``calls`` (the routing
    call's, which may offer the routing tools) keeps each call and result as Pipecat wrote it, a call joined to the
    line right before it; any other request offers no tools and leaves them out.
    """

    system: str  # the call's instructions and the decision rules
    tail: str  # the routing call's runtime data, or the update the writer words
    joiner: str = "\n"  # what joins the tail to a last user message
    label: str = ""  # the caller label; empty: no label
    calls: bool = False  # the routing call's: the talker's tool calls and their results stay in it

    def messages(self, messages: Sequence[dict]) -> list[dict]:
        """The request's messages from Pipecat's ``messages``, which it leaves as they are."""
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]
        joined: list[dict] = []
        previous = None
        for message in messages:
            role, content = message.get("role"), message_text(message)
            if role == "tool" or message.get("tool_calls"):
                previous = role
                if self.calls:
                    _add_call(joined, message)
                continue
            if self.label and role == "user" and previous == "developer":
                content = self.label + content
            previous, role = role, "user" if role == "developer" else role
            if joined and joined[-1]["role"] == role and not _is_call(joined[-1]):
                joined[-1]["content"] += "\n" + content
            else:
                joined.append({"role": role, "content": content})
        if joined and joined[-1]["role"] == "user":
            joined[-1]["content"] += self.joiner + self.tail
        else:
            joined.append({"role": "user", "content": self.tail})
        return [{"role": "system", "content": self.system}, *joined]


def _is_call(message: dict) -> bool:
    """Whether ``message`` is a tool call or its result, as Pipecat writes them."""
    return any(field in message for field in CALL_FIELDS)


def _add_call(joined: list[dict], message: dict) -> None:
    """Add ``message``, a tool call or its result, as Pipecat wrote it; a call joins the line right before it."""
    if message.get("tool_calls") and joined and joined[-1]["role"] == "assistant" and not _is_call(joined[-1]):
        joined[-1]["tool_calls"] = message["tool_calls"]
    else:
        joined.append({key: message[key] for key in ("role", "content", *CALL_FIELDS) if key in message})


@dataclass(frozen=True)
class PromptPart:
    """One part of a call's instructions: where it comes from, its text, and what joins it to the part before it."""

    source: str
    text: str
    sep: str = ""


class Layout:
    """The request each call of the talker sends, in the layout it was measured on (:class:`Request`).

    The routing call reads the prompt set's frame, the instructions and the routing rules, then the conversation with
    the caller label in front of a caller turn right after reference data, then the runtime data, a blank line after
    the conversation's last user message. The relay and the progress line read the writer's frame and rules, the
    conversation as it is, then the update. The decision rules (``json_rules``) follow every call's instructions, but
    the routing call's, which reads its format's rules (``routing_rules``): the tool-calling talker's routing call
    reads ``tool_rules`` in their place.
    """

    def __init__(self, prompts: PromptSet, json_rules: str):
        """Lay requests out in ``prompts``' words, the decision rules ``json_rules`` after each call's instructions."""
        self.prompts, self.json_rules = prompts, json_rules
        # The rules after the routing call's instructions: its format's, which the voice sets as it binds the talker.
        self.routing_rules = PromptPart("agent_prompts.Cascade.json_rules", json_rules)

    def routing_parts(self, instructions: Sequence[PromptPart]) -> list[PromptPart]:
        """The routing call's instructions, part by part in the order they are sent, each with its source."""
        prompts = self.prompts
        return [
            PromptPart("agent_prompts.Cascade.frame_before", prompts.frame_before),
            *instructions,
            PromptPart("agent_prompts.Cascade.frame_after", prompts.frame_after),
            PromptPart("agent_prompts.Cascade.routing", prompts.routing, "\n"),
        ]

    def runtime_data(self, work_line: str = "", turn: str = "", note: str = "") -> str:
        """The routing call's last user entry: the label, the work line, the newest turn and the note (if any), the cue.

        ``note`` is Pipecat's note on a caller turn with no words, after the newest turn's place.
        """
        prompts = self.prompts
        newest = prompts.newest_turn.format(turn=turn) + "\n" if turn else ""
        noted = note + "\n" if note else ""
        work = f"{work_line}\n\n" if work_line else ""
        return prompts.runtime_label + "\n" + work + newest + noted + prompts.decision_cue

    def routing(self, instructions: str, work_line: str, turn: str, note: str = "") -> Request:
        """The routing call's request: it keeps the talker's tool calls and their results."""
        parts = self.routing_parts([PromptPart("instructions", instructions)])
        text = "".join(part.sep + part.text for part in parts)
        runtime = self.runtime_data(work_line, turn, note)
        return Request(text + "\n" + self.routing_rules.text, runtime, "\n\n", self.prompts.caller_label, calls=True)

    def note(self, told: str | None, newer_requests: Sequence[str] = ()) -> str:
        """The notes an update carries: what the caller already heard of it, and newer requests still open."""
        notes = self.prompts.writer_notes
        text = notes.told.format(told=told) if told else ""
        if newer_requests:
            text += notes.newer.format(requests="; ".join(f'"{request}"' for request in newer_requests))
        return text

    def relay(self, instructions: str, content: str, note: str) -> Request:
        """The relay's request: the writer's frame and rules, the conversation, then the update to word."""
        prompts = self.prompts
        rules = prompts.writer_framed(instructions) + "\n" + prompts.commentary + prompts.answer
        return Request(rules + "\n" + self.json_rules, prompts.update_label + content + note)

    def progress(self, instructions: str, update: str, note: str) -> Request:
        """The progress line's request: the writer's frame and progress rules, the conversation, the work's update."""
        prompts = self.prompts
        rules = prompts.writer_framed(instructions) + "\n" + prompts.progress
        return Request(rules + "\n" + self.json_rules, prompts.update_label + update + note)


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


def relay_line(decision: TalkerDecision, content: str, history: Sequence[dict]) -> str | None:
    """The relay's worded line, or ``None`` when the update must be said as written.

    Not worded: any action but ``speak`` (unusable output parses as ``delegate``), no words, the update given back, or
    a line the voice already said in this call. Lines compare by their letters and digits (Pipecat's ``alnum_only``),
    so markup, punctuation, spacing, case and accents never tell them apart.
    """
    line = decision.speech
    said = {alnum_only(message_text(entry)) for entry in history if entry["role"] == "assistant"}
    if decision.action != "speak" or not line.strip() or alnum_only(line) in {alnum_only(content), *said}:
        return None
    return line


def progress_line(decision: TalkerDecision, latest: str | None) -> str:
    """The progress line, or nothing.

    A call that answers anything but ``speak``, carries a ``recovery`` or gives back the backend's newest progress line
    ``latest`` (the same letters and digits, by Pipecat's ``alnum_only``) says nothing: a progress line holds no fact
    to keep.
    """
    line = decision.speech.strip()
    if (
        decision.action != "speak"
        or decision.diagnostics.get("recovery")
        or (line and latest and alnum_only(line) == alnum_only(latest))
    ):
        return ""
    return line
