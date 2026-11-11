# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``vllm serve`` plus OpenAI Realtime text sessions at ``/v1/realtime``.

vLLM's own ``/v1/realtime`` transcribes audio only. This wrapper serves text
sessions there instead, on the same engine and app as the stock OpenAI routes,
with the events NVIDIA Dynamo's realtime text workers serve (refer to
``examples.shared.vllm_streaming_input.realtime``). The client side is
``examples.shared.nvidia_streaming_llm``.

Each ``input_text.append`` renders the conversation and the user text so far
with vLLM's own template and appends only the new prompt tokens to the turn's
engine request. ``response.create`` then generates the answer through vLLM's
chat completions stream, so tool parsing and structured output are unchanged,
and ends the request. Reasoning text is not streamed, as realtime has no event
for it. A render that no longer extends what the engine holds starts a new
request. Once an answer starts a tool call, a standby request prefills the
conversation alongside it and then appends that answer, so the follow-up turn
appends only the tool result.

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

from examples.shared.vllm_streaming_input import realtime
from examples.shared.vllm_streaming_input.structured_output import keep_structured_output

keep_structured_output()

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


class RealtimeSession:
    """One caller's OpenAI Realtime text session on the shared engine."""

    def __init__(self, websocket: WebSocket) -> None:
        """Bind the session to the app's engine and chat completions handler."""
        self.websocket = websocket
        self.engine = websocket.app.state.engine_client
        self.chat = websocket.app.state.openai_serving_chat
        self.session = realtime.new_session(self.chat.models.model_name())
        self.items: list[dict[str, Any]] = []
        self.text = ""
        self._prefill_params: SamplingParams | None = None
        self._engine: _EngineRequest | None = None
        self._standby: _EngineRequest | None = None
        self._response: realtime.ResponseEvents | None = None
        self._response_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def serve(self) -> None:
        """Handle events until the client disconnects."""
        await self.websocket.accept()
        await self._send(realtime.event("session.created", session=self.session))
        try:
            while True:
                event = await self.websocket.receive_json()
                try:
                    await self._handle(event if isinstance(event, dict) else {})
                except ValueError as e:
                    await self._send(realtime.error_event("invalid_request", str(e), event))
                except Exception as e:
                    logger.exception("Realtime session event failed")
                    await self._send(realtime.error_event("processing_error", str(e), event))
        except WebSocketDisconnect:
            pass
        finally:
            await self._cancel_response(announce=False)
            await self._stop_engine()
            await self._stop_standby()

    async def _handle(self, event: dict[str, Any]) -> None:
        if self._response is None and self._response_task is not None:
            task, self._response_task = self._response_task, None
            with contextlib.suppress(Exception):
                await task
        event_type = event.get("type")
        running = self._response is not None
        if event_type in ("conversation.item.create", "conversation.item.delete", "input_text.append") and running:
            await self._send(realtime.error_event("response_in_progress", "a response is running", event))
        elif (
            event_type
            in (
                "session.update",
                "session.chat_completion_options.update",
                "conversation.item.create",
                "conversation.item.delete",
            )
            and self.text
        ):
            await self._send(realtime.error_event("input_buffer_active", "the input text buffer is active", event))
        elif event_type in ("session.update", "session.chat_completion_options.update"):
            update = (
                event.get("session")
                if event_type == "session.update"
                else {"chat_completion_options": event.get("chat_completion_options")}
            )
            if not isinstance(update, dict):
                raise ValueError("session must be an object")
            session = {**self.session, **update}
            realtime.validate_session(session)
            self.session = session
            await self._send(realtime.event("session.updated", session=session))
        elif event_type == "conversation.item.create":
            if event.get("previous_item_id") is not None:
                raise ValueError("only appending conversation items is supported")
            item = realtime.normalize_item(event.get("item"))
            if any(existing["id"] == item["id"] for existing in self.items):
                raise ValueError(f"conversation item {item['id']!r} already exists")
            await self._add_items([item])
        elif event_type == "conversation.item.delete":
            index = next((i for i, item in enumerate(self.items) if item["id"] == event.get("item_id")), None)
            if index is None:
                raise ValueError(f"conversation item {event.get('item_id')!r} not found")
            del self.items[index]
            await self._send(realtime.event("conversation.item.deleted", item_id=event["item_id"]))
        elif event_type == "input_text.append":
            if not isinstance(event.get("text"), str) or not event["text"]:
                raise ValueError("text must be a non-empty string")
            self.text += event["text"]
            await self._prefill()
        elif event_type == "input_text.clear":
            self.text = ""
        elif event_type == "input_text.commit":
            if not self.text:
                raise ValueError("input text buffer is empty")
            item, self.text = realtime.user_item(self.text), ""
            await self._add_items([item])
        elif event_type == "response.create":
            if running:
                await self._send(realtime.error_event("response_in_progress", "a response is already running", event))
            elif self.text:
                await self._send(realtime.error_event("input_not_committed", "commit the input text first", event))
            elif event.get("response"):
                raise ValueError("response options are not supported")
            else:
                await self._create_response()
        elif event_type == "response.cancel":
            if not running:
                await self._send(realtime.error_event("no_active_response", "there is no active response", event))
            elif event.get("response_id") not in (None, self._response.response_id):
                await self._send(realtime.error_event("response_not_found", "that response is not active", event))
            else:
                await self._cancel_response(announce=True)
        else:
            await self._send(realtime.error_event("unsupported_event", f"unsupported event type: {event_type}", event))

    async def _add_items(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            previous_item_id = self.items[-1]["id"] if self.items else None
            self.items.append(item)
            for event_type in ("conversation.item.added", "conversation.item.done"):
                await self._send(realtime.event(event_type, previous_item_id=previous_item_id, item=item))

    def _user_text(self) -> str:
        """Return the user text to prefill.

        Only whole words are rendered. ASR can still extend the last word,
        which would change its tokens and restart the request.
        """
        return self.text[: self.text.rfind(" ") + 1].rstrip()

    def _chat_request(self, *, user_text: str = "", add_generation_prompt: bool = True) -> ChatCompletionRequest:
        """Render the session's conversation, followed by the user text so far if any."""
        messages = realtime.chat_messages(self.items, self.session["instructions"])
        if user_text:
            messages.append({"role": "user", "content": user_text})
        payload = {
            **realtime.chat_request_fields(self.session),
            "model": self.session["model"],
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if user_text:
            payload.update(add_generation_prompt=False, continue_final_message=True)
        elif not add_generation_prompt:
            payload["add_generation_prompt"] = False
        return ChatCompletionRequest(**payload)

    async def _render(self, request: ChatCompletionRequest) -> tuple[list, list[int]]:
        result = await self.chat.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            raise ValueError(result.error.message)
        self._prefill_params = _prefill_params(request)
        conversation, engine_inputs = result
        return conversation, list(self.chat._extract_prompt_components(engine_inputs[0]).token_ids or [])

    async def _prefill(self) -> None:
        if not (text := self._user_text()):
            return
        _, token_ids = await self._render(self._chat_request(user_text=text))
        await self._feed(token_ids)

    async def _create_response(self) -> None:
        if not self.items and not self.session["instructions"]:
            raise ValueError("response requires conversation input or instructions")
        request = self._chat_request()
        conversation, token_ids = await self._render(request)
        answer = _Answer(str(uuid4()), request, conversation, self._answer_params(request, len(token_ids)), token_ids)
        await self._feed(token_ids, answer)
        self._response = realtime.ResponseEvents(self.items[-1]["id"] if self.items else None)
        for event in self._response.started():
            await self._send(event)
        self._response_task = asyncio.create_task(self._respond(answer, self._response))

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

    async def _stop_standby(self) -> None:
        standby, self._standby = self._standby, None
        if standby is not None:
            await standby.stop()

    async def _respond(self, answer: _Answer, response: realtime.ResponseEvents) -> None:
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
        status, details = "completed", None
        try:
            async for chunk_event in stream:
                data = chunk_event.removeprefix("data: ").strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"].get("message", "chat generation failed"))
                for event in response.chunk(chunk):
                    if event["type"] == "response.output_item.added" and event["item"]["type"] == "function_call":
                        self._start_standby(answer)
                    await self._send(event)
            if response.finish_reason == "length":
                status, details = "incomplete", {"type": "incomplete", "reason": "max_output_tokens"}
        except Exception:
            logger.exception("Realtime response failed")
            status = "failed"
            details = {"type": "failed", "error": {"type": "server_error", "code": "generation_error"}}
        called_tool = status == "completed" and response.finish_reason == "tool_calls"
        await self._finish_response(response, status, details, tool_answer=answer if called_tool else None)
        await self._stop_standby()

    async def _finish_response(
        self,
        response: realtime.ResponseEvents,
        status: str,
        details: dict[str, Any] | None,
        *,
        tool_answer: _Answer | None = None,
    ) -> None:
        """Add the response's items to the conversation, then send its closing events.

        After a tool call, the standby request takes over with the
        conversation rendered up to the answer, so the follow-up turn appends
        only the tool result.
        """
        self._response = None
        events, items = response.finished(status, details)
        self.items.extend(items)
        if tool_answer is not None:
            try:
                await self._hand_off_to_standby(tool_answer)
            except Exception:
                logger.exception("Prefilling the tool call follow-up failed")
        for event in events:
            await self._send(event)

    async def _hand_off_to_standby(self, answer: _Answer) -> None:
        self._start_standby(answer)
        _, token_ids = await self._render(self._chat_request(add_generation_prompt=False))
        standby, self._standby = self._standby, None
        if not standby.extends(token_ids):
            await standby.stop()
            standby = _EngineRequest(self.engine)
        standby.feed(token_ids, self._prefill_params)
        finished, self._engine = self._engine, standby
        if finished is not None:
            await finished.stop()

    async def _cancel_response(self, *, announce: bool) -> None:
        """Stop an answer still streaming, along with the engine request generating it.

        An announced cancel adds the partial answer to the conversation and
        closes the response, as Dynamo does.
        """
        task, self._response_task = self._response_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await self._stop_engine()
            await self._stop_standby()
        if self._response is None:
            return
        if announce:
            await self._finish_response(
                self._response, "cancelled", {"type": "cancelled", "reason": "client_cancelled"}
            )
        else:
            self._response = None

    async def _send(self, event: dict[str, Any]) -> None:
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.websocket.send_json(event)


@router.websocket("/v1/realtime")
async def realtime_session(websocket: WebSocket) -> None:
    """Serve one caller's realtime text session."""
    await RealtimeSession(websocket).serve()


def _build_app_with_realtime_sessions(build_app):
    def build(*args, **kwargs) -> FastAPI:
        app = build_app(*args, **kwargs)
        if any(getattr(route, "path", None) == "/v1/realtime" for route in app.routes):
            raise RuntimeError("vLLM already serves /v1/realtime for this model, so text sessions cannot be mounted")
        app.include_router(router)
        return app

    return build


def main() -> None:
    """Run ``vllm serve`` with the realtime text session route mounted."""
    entry.build_app = _build_app_with_realtime_sessions(entry.build_app)
    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
