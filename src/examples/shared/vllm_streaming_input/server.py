# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``vllm serve`` plus a text StreamingInput websocket at ``/v1/streaming-session``.

vLLM's ``/v1/realtime`` feeds StreamingInput from audio only. This wrapper
mounts the text counterpart on the same engine and app, next to the stock
OpenAI routes. Events follow ``/v1/realtime``. The client side, and the event
list, is ``examples.shared.nvidia_streaming_llm``.

Each ``session.update`` or ``input_text_buffer.append`` renders the chat
request with vLLM's own template and appends only the new prompt tokens to the
turn's engine request. ``input_text_buffer.commit`` then generates the answer
through vLLM's chat completions stream, so reasoning and tool parsing are
unchanged, and ends the request. A render that no longer extends what the
engine holds starts a new request. Once an answer starts a tool call, a standby
request prefills the conversation alongside it and then appends that answer, so
the follow-up turn appends only the tool result.

Usage is that of ``vllm serve``::

    python -m examples.shared.vllm_streaming_input.server MODEL [VLLM_SERVE_ARGS...]
"""

import asyncio
import contextlib
import json
import sys
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.cli.main import main as vllm_main
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.launchers.api_server import entry
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

logger = init_logger(f"vllm.{__name__}")
router = APIRouter()


def _prefill_params(request: ChatCompletionRequest) -> SamplingParams:
    """Prefill a chunk with one throwaway token, which vLLM drops when the next chunk arrives.

    vLLM detokenizes the whole engine request as its first chunk says, so the
    chunk keeps the special tokens the tool parser asked the render to keep.
    """
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=request.skip_special_tokens,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
    )


def _add_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
    """Accumulate a chat completion chunk's content and tool calls into an assistant message."""
    if delta.get("content"):
        message["content"] = (message.get("content") or "") + delta["content"]
    for call in delta.get("tool_calls") or []:
        calls = message.setdefault("tool_calls", [])
        while len(calls) <= call["index"]:
            calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        entry, function = calls[call["index"]], call.get("function") or {}
        entry["id"] = call.get("id") or entry["id"]
        entry["function"]["name"] += function.get("name") or ""
        entry["function"]["arguments"] += function.get("arguments") or ""


@dataclass
class _Answer:
    response_id: str
    request: ChatCompletionRequest
    conversation: list
    params: SamplingParams
    prompt_token_ids: list[int]
    outputs: asyncio.Queue[RequestOutput | None] = field(default_factory=asyncio.Queue)


class _EngineRequest:
    """One StreamingInput engine request, fed prompt chunks until an answer closes it.

    The answer must be the last chunk: a resumable request whose input stream
    stays open can keep decoding past EOS, so an answer is never resumed.
    """

    def __init__(self, engine) -> None:
        """Bind the request to the shared engine."""
        self.engine = engine
        self.fed: list[int] = []
        self._pending: list[int] = []
        self._next_answer: _Answer | None = None
        self._answer: _Answer | None = None
        self._prefill_params: SamplingParams | None = None
        self._wake = asyncio.Event()
        self._chunk_done = asyncio.Event()
        self._closed = False
        self._request_id = f"stream-{uuid4()}"
        self._task: asyncio.Task | None = None

    def extends(self, token_ids: list[int]) -> bool:
        """Return whether ``token_ids`` continue what this request holds and can still be appended."""
        return not self._closed and token_ids[: len(self.fed)] == self.fed

    def feed(self, token_ids: list[int], prefill_params: SamplingParams, answer: _Answer | None = None) -> None:
        """Queue the tokens beyond ``fed``, which ``token_ids`` must extend, and the answer if any."""
        delta = token_ids[len(self.fed) :]
        if not delta and answer is None:
            return
        self.fed += delta
        self._pending += delta
        self._next_answer = answer
        self._prefill_params = prefill_params
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        self._wake.set()

    async def stop(self) -> None:
        """Cancel the request and abort it on the engine."""
        task, self._task = self._task, None
        self._closed = True
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            with contextlib.suppress(Exception):
                await self.engine.abort(self._request_id)

    async def _inputs(self):
        """Yield one chunk at a time, merging appends that arrive while one runs."""
        while True:
            await self._wake.wait()
            self._wake.clear()
            token_ids, self._pending = self._pending, []
            answer, self._next_answer = self._next_answer, None
            if not token_ids and answer is None:
                continue
            self._answer = answer
            self._chunk_done.clear()
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=token_ids),
                sampling_params=answer.params if answer else self._prefill_params,
            )
            if answer is not None:
                self._closed = True
                return
            await self._chunk_done.wait()

    async def _run(self) -> None:
        try:
            async for output in self.engine.generate(self._inputs(), self._prefill_params, self._request_id):
                answer = self._answer
                if answer is None:
                    self._chunk_done.set()
                    continue
                answer.outputs.put_nowait(output)
                if output.outputs[0].finish_reason is not None:
                    self._end_answer()
        finally:
            self._end_answer()

    def _end_answer(self) -> None:
        answer, self._answer = self._answer, None
        if answer is not None:
            answer.outputs.put_nowait(None)
        self._chunk_done.set()


class StreamingSession:
    """One caller's text StreamingInput session on the shared engine."""

    def __init__(self, websocket: WebSocket) -> None:
        """Bind the session to the app's engine and chat completions handler."""
        self.websocket = websocket
        self.engine = websocket.app.state.engine_client
        self.chat = websocket.app.state.openai_serving_chat
        self.request: dict[str, Any] = {}
        self.text = ""
        self._prefill_params: SamplingParams | None = None
        self._engine: _EngineRequest | None = None
        self._standby: _EngineRequest | None = None
        self._response_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def serve(self) -> None:
        """Handle events until the client disconnects."""
        await self.websocket.accept()
        await self._send("session.created", id=f"sess-{uuid4()}")
        try:
            while True:
                event = await self.websocket.receive_json()
                try:
                    await self._handle(event)
                except Exception as e:
                    logger.exception("Streaming session event failed")
                    await self._send("error", error=str(e), code="processing_error")
        except WebSocketDisconnect:
            pass
        finally:
            await self._cancel_response()
            await self._stop_engine()
            await self._stop_standby()

    async def _handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "session.update":
            self.request = dict(event.get("request") or {})
            await self._prefill()
        elif event_type == "input_text_buffer.append":
            self.text += str(event.get("text", ""))
            await self._prefill()
        elif event_type == "input_text_buffer.clear":
            self.text = ""
        elif event_type == "input_text_buffer.commit":
            await self._commit(str(event.get("response_id") or uuid4()))
        elif event_type == "response.cancel":
            await self._cancel_response()
        else:
            await self._send("error", error=f"Unknown event type: {event_type}", code="unknown_event")

    def _user_text(self, *, final: bool) -> str:
        """Return the user text to render.

        Before the commit only whole words are rendered. ASR can still extend
        the last word, which would change its tokens and restart the request.
        """
        return self.text if final else self.text[: self.text.rfind(" ") + 1].rstrip()

    def _chat_request(self, *, final: bool) -> ChatCompletionRequest:
        """Render the conversation with the user text so far."""
        text = self._user_text(final=final)
        payload = {**self.request, "stream": True, "stream_options": {"include_usage": True}}
        payload["messages"] = list(payload.get("messages", []))
        if text:
            payload["messages"].append({"role": "user", "content": text})
        if not final:
            payload.update(add_generation_prompt=False, continue_final_message=True)
        return ChatCompletionRequest(**payload)

    async def _render(self, request: ChatCompletionRequest) -> tuple[list, list[int]]:
        result = await self.chat.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            raise ValueError(result.error.message)
        self._prefill_params = _prefill_params(request)
        conversation, engine_inputs = result
        return conversation, list(self.chat._extract_prompt_components(engine_inputs[0]).token_ids or [])

    async def _prefill(self) -> None:
        if not self._user_text(final=False):
            return
        _, token_ids = await self._render(self._chat_request(final=False))
        await self._append(token_ids)

    async def _commit(self, response_id: str) -> None:
        request = self._chat_request(final=True)
        conversation, token_ids = await self._render(request)
        answer = _Answer(response_id, request, conversation, self._answer_params(request, len(token_ids)), token_ids)
        await self._append(token_ids, answer)
        self.text = ""
        self._response_task = asyncio.create_task(self._respond(answer))

    async def _append(self, token_ids: list[int], answer: _Answer | None = None) -> None:
        """Queue new input, which supersedes an answer still streaming."""
        await self._cancel_response()
        await self._feed(token_ids, answer)

    async def _feed(self, token_ids: list[int], answer: _Answer | None = None) -> None:
        """Queue the prompt tokens beyond what the engine request already holds.

        A prompt that no longer extends the engine's starts a new request.
        """
        if self._engine is not None and not self._engine.extends(token_ids):
            await self._stop_engine()
        if self._engine is None:
            self._engine = _EngineRequest(self.engine)
        self._engine.feed(token_ids, self._prefill_params, answer)

    def _answer_params(self, request: ChatCompletionRequest, prompt_len: int) -> SamplingParams:
        max_tokens = get_max_tokens(
            self.chat.model_config.max_model_len,
            request.max_completion_tokens or request.max_tokens,
            prompt_len,
            self.chat.default_sampling_params,
            self.chat.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        return request.to_sampling_params(max_tokens, self.chat.default_sampling_params)

    async def _stop_engine(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            await engine.stop()

    def _start_standby(self, answer: _Answer) -> None:
        """Prefill the answer's prompt on a second request while the answer still streams."""
        if self._standby is None:
            self._standby = _EngineRequest(self.engine)
            self._standby.feed(answer.prompt_token_ids, self._prefill_params)

    async def _render_with_answer(self, answer: _Answer, message: dict[str, Any]) -> list[int]:
        """Render the conversation ending with the answer as the client sends it back.

        The template drops tokens the model generates around a tool call, such
        as a stray ``</think>`` after the spoken text, so the generated tokens
        do not prefix the follow-up turn's prompt.
        """
        messages = [*answer.request.messages, message]
        request = answer.request.model_copy(update={"messages": messages, "add_generation_prompt": False})
        _, token_ids = await self._render(request)
        return token_ids

    async def _stop_standby(self) -> None:
        standby, self._standby = self._standby, None
        if standby is not None:
            await standby.stop()

    async def _respond(self, answer: _Answer) -> None:
        """Stream the answer through vLLM's chat completions output path."""

        async def outputs():
            while (output := await answer.outputs.get()) is not None:
                yield output

        request_id = f"chatcmpl-{answer.response_id}"
        stream = self.chat.chat_completion_stream_generator(
            answer.request,
            outputs(),
            request_id,
            self.chat.models.model_name(),
            answer.conversation,
            self.chat.renderer.tokenizer,
            RequestResponseMetadata(request_id=request_id),
            chat_template_kwargs=self.chat._effective_chat_template_kwargs(answer.request),
        )
        called_tool = False
        message: dict[str, Any] = {"role": "assistant", "content": None}
        try:
            async for event in stream:
                data = event.removeprefix("data: ").strip()
                if data and data != "[DONE]":
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    for choice in choices:
                        _add_delta(message, choice.get("delta") or {})
                    if message.get("tool_calls"):
                        self._start_standby(answer)
                    called_tool |= any(c.get("finish_reason") == "tool_calls" for c in choices)
                    await self._send("response.delta", response_id=answer.response_id, chunk=chunk)
            if called_tool:
                self._start_standby(answer)
                standby, self._standby = self._standby, None
                token_ids = await self._render_with_answer(answer, message)
                if not standby.extends(token_ids):
                    await standby.stop()
                    standby = _EngineRequest(self.engine)
                standby.feed(token_ids, self._prefill_params)
                finished, self._engine = self._engine, standby
                if finished is not None:
                    await finished.stop()
            await self._send("response.done", response_id=answer.response_id)
        except Exception as e:
            logger.exception("Streaming response failed")
            await self._send("error", error=str(e), code="processing_error")
        finally:
            await self._stop_standby()

    async def _cancel_response(self) -> None:
        """Stop an answer still streaming, along with the engine request generating it."""
        task, self._response_task = self._response_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await self._stop_engine()

    async def _send(self, event_type: str, **data: Any) -> None:
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.websocket.send_json({"type": event_type, **data})


@router.websocket("/v1/streaming-session")
async def streaming_session(websocket: WebSocket) -> None:
    """Serve one caller's text StreamingInput session."""
    await StreamingSession(websocket).serve()


def _build_app_with_streaming_session(build_app):
    def build(*args, **kwargs) -> FastAPI:
        app = build_app(*args, **kwargs)
        app.include_router(router)
        return app

    return build


def main() -> None:
    """Run ``vllm serve`` with the streaming session route mounted."""
    entry.build_app = _build_app_with_streaming_session(entry.build_app)
    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
