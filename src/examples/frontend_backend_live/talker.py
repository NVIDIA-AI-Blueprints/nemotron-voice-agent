# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: Pipecat's ``NvidiaLLMService`` with the voice's structured decision."""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

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
from pipecat.processors.aggregators.llm_context import LLMContext, standard_message_text
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.utils.text.alnum_utils import alnum_only
from pydantic import TypeAdapter, ValidationError

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
from examples.shared.strict_settings import Tool

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    from examples.frontend_backend_live.config_manager.schema import ModelEndpoint, ReliabilityConfig
    from examples.frontend_backend_live.prompts import PromptSet

RESPONSE_KEY = "talker_response"
DEFAULT_MAX_OUTPUT_TOKENS = 512

_out_of_band = contextvars.ContextVar("talker_out_of_band", default=False)
_current_call: contextvars.ContextVar[TalkerCall | None] = contextvars.ContextVar("talker_call", default=None)


class Reply(Protocol):
    """One streamed reply as its call reads it: the words it lets out or ``held``, its ``raw`` form, its decision."""

    held: str
    raw: Any

    def feed(self, delta: ChoiceDelta) -> str:
        """Read one delta, leaving in it only what Pipecat should read, its calls included; return the words to say."""

    def decision(self, metadata: dict, finish: str | None) -> TalkerDecision:
        """Return the reply's decision once its stream ended with ``finish``, with ``metadata`` in its diagnostics."""


class ResponseFormat(Protocol):
    """How a call asks the model for its decision, and how its reply becomes one."""

    name: str

    def request_fields(self, spec: CallSpec) -> dict:
        """Return the fields a call in the pipeline adds to its request after its messages."""

    def schema(self, spec: CallSpec) -> dict | None:
        """Return the ``response_schema`` a call run out of band passes to ``run_inference``, or ``None``."""

    def routing_texts(self, prompts: PromptSet) -> RoutingTexts:
        """Return the texts a routing call frames its instructions in, with their sources."""

    def reply(self, streams: bool) -> Reply:
        """Return a reader for one reply; ``streams``: a ``speak``'s words go to speech as they arrive."""


class JsonDecision:
    """The JSON decision: the decision rules (``json_rules``) ask for one object, read as it streams."""

    entries = ("frame_before", "routing", "json_rules")

    def routing_texts(self, prompts: PromptSet) -> RoutingTexts:
        """Return the texts its :attr:`entries` name."""
        return RoutingTexts.named(prompts, *self.entries)

    def reply(self, streams: bool) -> JsonReply:
        """Return a reader of the JSON decision."""
        return JsonReply(streams)


class JsonSchema(JsonDecision):
    """``json_schema``: the strict decision schema the call's spec gives."""

    name = "json_schema"

    def request_fields(self, spec: CallSpec) -> dict:
        """Return the strict ``voice_decision`` schema as the ``response_format``."""
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "voice_decision", "strict": True, "schema": self.schema(spec)},
            }
        }

    def schema(self, spec: CallSpec) -> dict:
        """Return the decision schema: the spec's actions, and its hint for ``speech``."""
        return decision_schema(spec)


class NoFormat(JsonDecision):
    """``none``: no ``response_format``; the decision rules alone ask for the object."""

    name = "none"

    def request_fields(self, spec: CallSpec) -> dict:
        """Return no fields."""
        return {}

    def schema(self, spec: CallSpec) -> None:
        """Return ``None``: no schema."""


RESPONSE_FORMATS: dict[str, ResponseFormat] = {each.name: each for each in (JsonSchema(), NoFormat())}


class ToolCalls:
    """``tools``: the routing tools with ``tool_choice`` "auto", so a reply in text with no call stays possible."""

    name = "tools"
    entries = ("tool_frame_before", "tool_routing", "tool_rules")

    def __init__(self, tools: list[dict]):
        """Offer ``tools``: the routing tools as Pipecat's adapter converts them for the request."""
        self.tools = tools

    def request_fields(self, spec: CallSpec) -> dict:
        """Return the routing tools and ``tool_choice`` "auto", on this request alone: every call reads the context."""
        return {"tools": self.tools, "tool_choice": "auto"}

    def schema(self, spec: CallSpec) -> None:
        """Return ``None``: a routing call never runs out of band."""

    def routing_texts(self, prompts: PromptSet) -> RoutingTexts:
        """Return the texts its :attr:`entries` name."""
        return RoutingTexts.named(prompts, *self.entries)

    def reply(self, streams: bool) -> ToolReply:
        """Return a reader of the reply's text and calls."""
        return ToolReply(streams)


TOOL_TEXT = TypeAdapter(Tool)


def routing_tools(prompts: PromptSet) -> ToolsSchema:
    """Return the routing tools of ``prompts``: an OpenAI function for each name of ``TOOL_DECISIONS``, in order."""
    texts, entries = prompts.tools.model_dump(), []
    for name in TOOL_DECISIONS:
        try:
            schema = TOOL_TEXT.validate_python(texts[name])
        except ValidationError as exc:
            raise ValueError(f"agent_prompts.Cascade.tools.{name} is not a tool's schema: {exc}") from None
        entries.append({"type": "function", "function": {"name": name, **schema}})
    return ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: entries})


class JsonReply:
    """The JSON decision as it streams: a routing ``speak``'s words go out once its action has decoded."""

    def __init__(self, streams: bool):
        """Read the ``action`` and ``speech`` fields as they decode; ``streams``: a ``speak`` goes out as it does."""
        self.streams = streams
        self.action, self.speech = JsonStringFieldStreamer("action"), JsonStringFieldStreamer("speech")
        self.raw = self.verb = self.held = ""

    def feed(self, delta: ChoiceDelta) -> str:
        """Return a ``speak``'s words once its action has decoded, when the call streams; hold any other speech."""
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

    def decision(self, metadata: dict, finish: str | None) -> TalkerDecision:
        """Return the reply's first JSON object as a decision; a reply the token limit cut is ``incomplete_output``."""
        if finish == "length":
            return unusable_decision(metadata, "incomplete_output")
        return decision_from_output_text(self.raw, metadata)


@dataclass
class ToolCall:
    """One call a reply makes, as its fragments arrive: its id, the tool's name, and its arguments as text."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolReply:
    """A reply of the tool-calling talker's routing call: its text is a ``speak``, a routing tool's call the move."""

    def __init__(self, streams: bool):
        """Read the reply's text and calls; ``streams``: its words go to speech as they arrive."""
        self.streams = streams
        self.calls: dict[int, ToolCall] = {}
        self.text = ""
        self.held = ""

    @property
    def raw(self) -> dict:
        """The reply as it came, for the log: its text, then each call's name and arguments."""
        calls = [{"name": call.name, "arguments": call.arguments} for call in self._in_order()]
        return {"content": self.text, "tool_calls": calls}

    def feed(self, delta: ChoiceDelta) -> str:
        """Note the delta's call fragments and say no text beside them; else return its text from the first word on.

        Each fragment carrying part of a call's name carries the call's id, so Pipecat runs the call under the id
        the voice hands the turn over under.
        """
        if delta.tool_calls:
            for fragment in delta.tool_calls:
                call = self.calls.setdefault(fragment.index, ToolCall())
                call.id = call.id or fragment.id or ""
                if fragment.function is not None and fragment.function.name:
                    call.id = call.id or f"call_{uuid.uuid4().hex}"
                    fragment.id = call.id
                if fragment.function is not None:
                    call.name += fragment.function.name or ""
                    call.arguments += fragment.function.arguments or ""
            return ""
        text = delta.content if self.text else delta.content.lstrip()
        self.text += text
        if self.streams:
            return text
        self.held += text
        return ""

    def decision(self, metadata: dict, finish: str | None) -> TalkerDecision:
        """Return the move of the first ``TOOL_DECISIONS`` name a call made, else a ``speak`` unless cut or empty."""
        calls = self._in_order()
        diagnostics = {**metadata, "tool_calls": [call.name for call in calls]}
        for name, decide in TOOL_DECISIONS.items():
            for call in calls:
                if call.name != name:
                    continue
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    # Pipecat runs no handler for a call whose arguments hold no JSON.
                    continue
                return replace(decide(arguments, diagnostics), call_id=call.id)
        if finish == "length":
            return unusable_decision(diagnostics, "incomplete_output")
        if self.text:
            return TalkerDecision("speak", self.text, diagnostics)
        return unusable_decision(diagnostics, "invalid_decision" if calls else "empty_output")

    def _in_order(self) -> list[ToolCall]:
        return [self.calls[index] for index in sorted(self.calls)]


@dataclass(eq=False)
class TalkerCall:
    """One run of the talker: what it asks the model for, and what the voice keeps to act on its decision."""

    spec: CallSpec | None = None
    request: Request | None = None
    text: str = ""
    streams: bool = False
    format: ResponseFormat = field(default_factory=JsonSchema)
    deadline: float = 0.0
    plan: Any = None
    cut: bool = False
    said: str = ""
    started: float = field(default_factory=time.monotonic)
    first_token_ms: float | None = None

    @property
    def role(self) -> str:
        """The call's name in the logs: its spec's role."""
        return self.spec.role if self.spec else "line"

    def request_fields(self) -> dict:
        """Return the fields the call's request adds after its messages: its format's ``response_format``, or tools."""
        return self.format.request_fields(self.spec)

    def response_schema(self) -> dict | None:
        """Return the call's decision schema for ``run_inference``, at the level of the model's services entry."""
        return self.format.schema(self.spec)


class Calls(Protocol):
    """What the talker asks of the session's voice: which call each run is, and what happens when it ends."""

    async def next_call(self, context: LLMContext) -> TalkerCall | None:
        """Return the call this run makes, or ``None`` when there is nothing to decide or say."""

    def speaking(self, call: TalkerCall, text: str) -> None:
        """Note that a routing ``speak``'s words went on to speech."""

    def finished(self, call: TalkerCall, decision: TalkerDecision | None, held: str) -> str:
        """Act on the call's ``decision`` (``None`` for a line with no model call); return the words to let out."""

    def written(self) -> list[dict]:
        """Return the messages that go into the context right before the words let out: the items their line says."""

    def dropped(self, call: TalkerCall) -> None:
        """End the call with nothing said: a result arrived before it said anything."""

    def cancelled(self, call: TalkerCall) -> None:
        """Note that an interruption cancelled the call."""

    def failed(self, call: TalkerCall, error: BaseException) -> None:
        """Note that the call failed; Pipecat reports the error, and the voice's fallback runs from its error event."""

    async def hand_off(self, call_id: str) -> bool | None:
        """Hand over the turn the ``delegate`` call ``call_id`` decided; return whether the part accepted it.

        The call's handler asks, so the hand-off runs in its task; ``None``: the call hands nothing over.
        """

    def response_started(self) -> int:
        """Return the number of the response the talker starts."""

    def response_ended(self) -> int:
        """Return the number of the response the talker ends."""

    def interrupted(self) -> None:
        """Note an interruption that passed the talker: the responses started before it are dropped."""

    def caller_started(self) -> None:
        """Note that the caller started speaking."""

    def caller_stopped(self) -> None:
        """Note that the caller stopped speaking."""


def _bare(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    """Return a copy of ``chunk`` with no words, calls or usage, which Pipecat reads as the reply's start alone."""
    bare = chunk.model_copy(deep=True)
    bare.usage = None
    bare.choices[0].delta.content = None
    bare.choices[0].delta.tool_calls = None
    return bare


class DecisionReplyMixin:
    """Ask each talker run for the voice's decision; stream a routing ``speak`` and hold the rest for the voice."""

    calls: Calls
    response_format: ResponseFormat

    def bind(self, calls: Calls) -> None:
        """Ask ``calls``, the session's voice, which call each run makes, and tell it how each one ends."""
        self.calls = calls

    def routing_format(self, prompts: PromptSet) -> ResponseFormat:
        """Return how a routing call asks for its decision: in JSON, at the services entry's level."""
        return self.response_format

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Adjust Pipecat's messages as the call's request goes out, adding its format's fields in the pipeline."""
        params = super().build_chat_completion_params(params_from_context)
        call = _current_call.get()
        if call is not None:
            params["messages"] = call.request.messages(params["messages"])
            if not _out_of_band.get():
                params.update(call.request_fields())
            logger.debug(f"{call.role} request: {json.dumps(params['messages'], ensure_ascii=False)}")
        return params

    async def run_inference(self, *args, **kwargs) -> str | None:
        """Run one-shot inference with its own request and no decision format."""
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
        """Run Pipecat's one-shot inference out of band, adjusted as ``call``'s request, or as its arguments give."""
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
            return self._line("")
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
        """Yield a line said with no model call, after what goes into the context with it."""
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
        """Let a routing ``speak``'s words through as they arrive, and read the whole decision when the stream ends.

        Chunks with calls are held until the voice has acted, so a reply a result cuts short hands Pipecat no call;
        Pipecat runs calls only after the stream anyway. A reply that opens with calls yields a bare copy of its first
        chunk at once, so its time to first byte stops at the reply's start.
        """
        reply = call.format.reply(call.streams)
        finish, calls, opened = None, [], False
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
                    opened = opened or choice is not None
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
                if delta.tool_calls:
                    calls.append(chunk)
                    if not opened:
                        opened = True
                        yield _bare(chunk)
                    continue
                opened = True
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
        logger.debug(
            f"{call.role} reply ({call.first_token_ms} ms to the first token, {duration_ms} ms in all): "
            f"{json.dumps(reply.raw, ensure_ascii=False)}"
        )
        text = self.calls.finished(call, reply.decision(metadata, finish), reply.held)
        for held in calls:
            yield held
        async for line in self._line(text):
            yield line


class TalkerChat(DecisionReplyMixin, NvidiaLLMService):
    """``NvidiaLLMService`` that asks each run for the voice's decision (:class:`DecisionReplyMixin`)."""

    def __init__(self, *, reliability: ReliabilityConfig, response_format: ResponseFormat, **kwargs):
        """Create the service, asking at ``response_format``'s level."""
        self.reliability, self.response_format = reliability, response_format
        super().__init__(**kwargs)
        self.add_event_handler("on_error", _stay_usable)

    @classmethod
    def from_config(
        cls, endpoint: ModelEndpoint, reliability: ReliabilityConfig, response_format: ResponseFormat
    ) -> TalkerChat:
        """Return the talker on a services entry's model endpoint, with the frontend's reliability settings."""
        extra = dict(endpoint.extra_params)
        settings = cls.Settings(
            model=endpoint.model,
            temperature=0.0,
            max_tokens=int(extra.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS),
            # The entry's own sampling and reasoning settings win.
            extra=extra,
        )
        return cls(
            reliability=reliability,
            response_format=response_format,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            settings=settings,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Return the model client Pipecat makes, with ``provider_retries`` as the SDK's retries."""
        client = super().create_client(api_key=api_key, base_url=base_url, **kwargs)
        return client.with_options(max_retries=self.reliability.provider_retries)


class TalkerTools(TalkerChat):
    """``TalkerChat`` whose routing call moves by function calling; its endpoint must stream ``tool_calls``."""

    def bind(self, calls: Calls) -> None:
        """Ask ``calls`` about each run, and register a handler for each routing tool."""
        super().bind(calls)
        for name, handler in build_handlers(calls).items():
            # Not cancelled on interruption: a cancelled hand-off releases its entry even if the part took the request.
            self.register_function(name, handler, cancel_on_interruption=False)

    def routing_format(self, prompts: PromptSet) -> ToolCalls:
        """Return the routing tools of ``prompts``, as Pipecat's adapter converts them for the request."""
        missing = prompts.missing([*ToolCalls.entries, *(f"tools.{name}" for name in TOOL_DECISIONS)])
        if missing:
            raise ValueError(f"agent_prompts.Cascade has no {', '.join(missing)}: the tools talker style reads them")
        return ToolCalls(self.get_llm_adapter().from_standard_tools(routing_tools(prompts)))


async def _stay_usable(talker: TalkerChat, error) -> None:
    """Keep the talker usable after a failed call, which the voice's fallback answers."""
    await talker.set_usable(True)


LLM_SERVICES: dict[str, type[TalkerChat]] = {"json": TalkerChat, "tools": TalkerTools}


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


CALL_FIELDS = ("tool_calls", "tool_call_id")


@dataclass(frozen=True)
class Request:
    """What a call does to Pipecat's messages as its request goes out: its ``system`` text first, ``tail`` last."""

    system: str
    tail: str
    joiner: str = "\n"
    label: str = ""
    calls: bool = False

    def messages(self, messages: Sequence[dict]) -> list[dict]:
        """Return the request's messages from Pipecat's ``messages``, which it leaves as they are."""
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]
        joined: list[dict] = []
        previous = None
        for message in messages:
            role, content = message.get("role"), standard_message_text(message)
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
    """Return whether ``message`` is a tool call or its result, as Pipecat writes them."""
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


@dataclass(frozen=True)
class RoutingTexts:
    """The prompt set's texts that frame a routing call's instructions, each with its source: its format's own."""

    frame_before: PromptPart
    routing: PromptPart
    rules: PromptPart

    @classmethod
    def named(cls, prompts: PromptSet, frame_before: str, routing: str, rules: str) -> RoutingTexts:
        """Return the texts of ``prompts`` that these ``agent_prompts.Cascade`` entries name."""

        def part(name: str, sep: str = "") -> PromptPart:
            return PromptPart(f"agent_prompts.Cascade.{name}", getattr(prompts, name), sep)

        return cls(part(frame_before), part(routing, "\n"), part(rules))


class Layout:
    """The request each call of the talker sends (:class:`Request`)."""

    def __init__(self, prompts: PromptSet, json_rules: str):
        """Lay requests out in ``prompts``' words, the decision rules ``json_rules`` after each call's instructions."""
        self.prompts, self.json_rules = prompts, json_rules
        self.routing_texts = RoutingTexts.named(prompts, *JsonDecision.entries)

    def routing_parts(self, instructions: Sequence[PromptPart]) -> list[PromptPart]:
        """Return the routing call's instructions part by part, in the order they are sent, each with its source."""
        texts = self.routing_texts
        frame_after = PromptPart("agent_prompts.Cascade.frame_after", self.prompts.frame_after)
        return [texts.frame_before, *instructions, frame_after, texts.routing]

    def runtime_data(self, work_line: str = "", turn: str = "", note: str = "") -> str:
        """Return the routing call's last user entry: its label, work line, newest turn, ``note`` and cue."""
        prompts = self.prompts
        newest = prompts.newest_turn.format(turn=turn) + "\n" if turn else ""
        noted = note + "\n" if note else ""
        work = f"{work_line}\n\n" if work_line else ""
        return prompts.runtime_label + "\n" + work + newest + noted + prompts.decision_cue

    def routing(self, instructions: str, work_line: str, turn: str, note: str = "") -> Request:
        """Return the routing call's request, which keeps the talker's tool calls and their results."""
        parts = self.routing_parts([PromptPart("instructions", instructions)])
        text = "".join(part.sep + part.text for part in parts)
        runtime = self.runtime_data(work_line, turn, note)
        rules = self.routing_texts.rules.text
        return Request(text + "\n" + rules, runtime, "\n\n", self.prompts.caller_label, calls=True)

    def note(self, told: str | None, newer_requests: Sequence[str] = ()) -> str:
        """Return the notes an update carries: what the caller already heard of it, and newer requests still open."""
        notes = self.prompts.writer_notes
        text = notes.told.format(told=told) if told else ""
        if newer_requests:
            text += notes.newer.format(requests="; ".join(f'"{request}"' for request in newer_requests))
        return text

    def relay(self, instructions: str, content: str, note: str) -> Request:
        """Return the relay's request: the writer's frame and rules, the conversation, then the update to word."""
        prompts = self.prompts
        rules = prompts.writer_framed(instructions) + "\n" + prompts.commentary + prompts.answer
        return Request(rules + "\n" + self.json_rules, prompts.update_label + content + note)

    def progress(self, instructions: str, update: str, note: str) -> Request:
        """Return the progress line's request: the writer's frame and progress rules, then the work's update."""
        prompts = self.prompts
        rules = prompts.writer_framed(instructions) + "\n" + prompts.progress
        return Request(rules + "\n" + self.json_rules, prompts.update_label + update + note)


def floored(decision: TalkerDecision, *, work_open: bool, holding: str) -> TalkerDecision:
    """Return the routing decision the voice carries out: an empty ``speak`` waits, and unusable output is a failure."""
    action = decision.action
    if decision.diagnostics.get("recovery") in UNUSABLE:
        return after_failure(decision.diagnostics, work_open=work_open, holding=holding)
    if action == "delegate" or (action == "speak" and decision.speech.strip()):
        return decision
    if action == "wait":
        return TalkerDecision("wait", "", decision.diagnostics)
    return TalkerDecision("wait", "", {**decision.diagnostics, "recovery": "empty_speak"})


def after_failure(diagnostics: dict, *, work_open: bool, holding: str) -> TalkerDecision:
    """Return a failed turn's decision: the holding line while work is open, else a delegation with no lead-in.

    A delegation would join the open work, and could restart it, for a turn nobody decided; with none open, no tool
    has run at the talker stage, so delegating never repeats an action.
    """
    if work_open and holding:
        held = {**diagnostics, "recovery": "holding", "failed": diagnostics.get("recovery")}
        return TalkerDecision("speak", holding, held)
    return TalkerDecision("delegate", "", diagnostics)


def relay_line(decision: TalkerDecision, content: str, history: Sequence[dict]) -> str | None:
    """Return the relay's worded line, or ``None`` when the update must be said as written.

    Only a ``speak`` with words that are neither the update nor a line said already is worded; lines compare by
    ``alnum_only``, so markup, punctuation, case and accents never tell them apart.
    """
    line = decision.speech
    said = {alnum_only(standard_message_text(entry)) for entry in history if entry["role"] == "assistant"}
    if decision.action != "speak" or not line.strip() or alnum_only(line) in {alnum_only(content), *said}:
        return None
    return line


def progress_line(decision: TalkerDecision, latest: str | None, history: Sequence[dict]) -> str:
    """Return the progress line, or nothing when the call does not ``speak``, carries a ``recovery`` or repeats."""
    line = decision.speech.strip()
    said = {alnum_only(standard_message_text(entry)) for entry in history if entry["role"] == "assistant"}
    if (
        decision.action != "speak"
        or decision.diagnostics.get("recovery")
        or (line and latest and alnum_only(line) == alnum_only(latest))
        or (line and alnum_only(line) in said)
    ):
        return ""
    return line
