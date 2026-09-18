# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve native vLLM StreamingInput sessions for the Pipecat example."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

from examples.generic.streaming.transcript import (
    StableTranscriptCommitter,
    TranscriptCommit,
    extract_token_ids,
)
from examples.generic.streaming.vllm_patch import verify_installed_patch

logger = logging.getLogger(__name__)


def _history(payload: Any) -> list[dict[str, str]]:
    """Keep only nonempty user and assistant messages from client history."""
    if not isinstance(payload, list):
        return []
    messages: list[dict[str, str]] = []
    for message in payload:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            messages.append({"role": str(message["role"]), "content": " ".join(content.split())})
    return messages


@dataclass(slots=True)
class InputChunk:
    """One source-token append and its short decode window."""

    command_id: str
    is_final: bool
    token_ids: list[int]
    transcript: TranscriptCommit
    sampling_params: SamplingParams
    expected_tokens: int
    is_prefill: bool = False
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    sampled_tokens: int = 0
    text_parts: list[str] = field(default_factory=list)
    ttft_ms: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class NativeStreamingSession:
    """Keep one append-only AsyncLLM request alive for one user turn."""

    def __init__(
        self,
        *,
        engine: AsyncLLM,
        websocket: WebSocket,
        session_id: str,
        system_prompt: str,
        history: list[dict[str, str]],
        hold_words: int,
        final_max_tokens: int,
    ) -> None:
        """Initialize turn state without starting generation."""
        self.engine = engine
        self.websocket = websocket
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.history = history
        self.committer = StableTranscriptCommitter(hold_words=hold_words)
        self.final_max_tokens = final_max_tokens
        self.tokenizer = engine.get_tokenizer()
        self.anchor_id = str(uuid4())
        self.request_id = f"stream-{self.anchor_id}"
        self.input_token_ids: list[int] = []
        self._queue: asyncio.Queue[InputChunk | None] = asyncio.Queue()
        self._generation: asyncio.Task[None] | None = None
        self._active: InputChunk | None = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    async def send(self, event_type: str, **data: Any) -> None:
        """Send one ordered event to the Pipecat-side client."""
        async with self._send_lock:
            await self.websocket.send_json(
                {
                    "type": event_type,
                    "session_id": self.session_id,
                    "anchor_id": self.anchor_id,
                    **data,
                }
            )

    def _prompt_ids(self, transcript: str, *, is_final: bool) -> list[int]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": transcript},
        ]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": is_final,
            "enable_thinking": False,
        }
        if not is_final:
            kwargs["continue_final_message"] = True
        return extract_token_ids(self.tokenizer.apply_chat_template(messages, **kwargs))

    @staticmethod
    def _partial_params() -> SamplingParams:
        # vLLM retains max_tokens - 1 outputs between continuations. One token
        # updates source KV state while retaining no generated partial output.
        return SamplingParams(
            max_tokens=1,
            min_tokens=1,
            temperature=0.0,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )

    def _final_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.final_max_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )

    async def prefill(self, command_id: str) -> None:
        """Queue the stable system/history prefix before user text arrives."""
        if self.input_token_ids or self._generation is not None:
            return
        token_ids = self._prompt_ids("", is_final=False)
        if not token_ids:
            return
        chunk = InputChunk(
            command_id=command_id,
            is_final=False,
            token_ids=token_ids,
            transcript=TranscriptCommit(
                observed="",
                committed="",
                text_delta="",
                reset_required=False,
            ),
            sampling_params=self._partial_params(),
            expected_tokens=1,
            is_prefill=True,
        )
        self.input_token_ids.extend(token_ids)
        await self._queue.put(chunk)
        self._generation = asyncio.create_task(self._run_generation())

    async def _restart(self, reason: str) -> None:
        previous_request_id = self.request_id
        if self._generation is not None and not self._generation.done():
            self._generation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._generation
        with contextlib.suppress(Exception):
            await self.engine.abort(previous_request_id)
        previous_anchor = self.anchor_id
        self.anchor_id = str(uuid4())
        self.request_id = f"stream-{self.anchor_id}"
        self.input_token_ids = []
        self._queue = asyncio.Queue()
        self._generation = None
        self._active = None
        await self.send("session.reset", reason=reason, previous_anchor_id=previous_anchor)

    async def observe(self, command_id: str, transcript: str, *, is_final: bool) -> None:
        """Append the stable token delta from one cumulative ASR update."""
        commit = self.committer.observe(transcript, is_final=is_final)
        if commit.reset_required:
            await self._restart("committed-asr-revision")

        candidate = self._prompt_ids(commit.committed, is_final=is_final)
        if self.input_token_ids and candidate[: len(self.input_token_ids)] != self.input_token_ids:
            await self._restart("token-prefix-mismatch")
        token_ids = candidate[len(self.input_token_ids) :]
        if not token_ids and not is_final:
            return

        params = self._final_params() if is_final else self._partial_params()
        chunk = InputChunk(
            command_id=command_id,
            is_final=is_final,
            token_ids=token_ids,
            transcript=commit,
            sampling_params=params,
            expected_tokens=self.final_max_tokens if is_final else 1,
        )
        self.input_token_ids.extend(token_ids)
        await self._queue.put(chunk)
        await self.send(
            "input.committed",
            command_id=command_id,
            transcript=commit.committed,
            token_delta_count=len(token_ids),
            cumulative_input_tokens=len(self.input_token_ids),
            is_final=is_final,
        )
        if self._generation is None:
            self._generation = asyncio.create_task(self._run_generation())

    async def _inputs(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            self._active = chunk
            chunk.started_at = time.monotonic()
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=chunk.token_ids),
                sampling_params=chunk.sampling_params,
            )
            await chunk.done.wait()
            if chunk.is_final:
                return

    async def _finish(self, chunk: InputChunk) -> None:
        if chunk.done.is_set():
            return
        try:
            if chunk.is_final:
                response = "".join(chunk.text_parts).strip()
                await self.send(
                    "response.done",
                    command_id=chunk.command_id,
                    response=response,
                    ttft_ms=chunk.ttft_ms,
                    prompt_tokens=len(self.input_token_ids),
                    completion_tokens=chunk.sampled_tokens,
                )
            elif chunk.is_prefill:
                await self.send(
                    "session.prefilled",
                    command_id=chunk.command_id,
                    prefix_tokens=len(chunk.token_ids),
                    ttft_ms=chunk.ttft_ms,
                )
        finally:
            # A client can close immediately after receiving response.done.
            # Always release the input generator even if that races this send.
            self._active = None
            chunk.done.set()

    async def _run_generation(self) -> None:
        try:
            async for output in self.engine.generate(
                prompt=self._inputs(),
                sampling_params=self._partial_params(),
                request_id=self.request_id,
            ):
                chunk = self._active
                if chunk is None:
                    continue
                finish_reason = None
                for completion in output.outputs:
                    tokens = list(completion.token_ids)
                    if tokens and chunk.ttft_ms is None:
                        chunk.ttft_ms = round((time.monotonic() - (chunk.started_at or chunk.queued_at)) * 1000, 3)
                        if chunk.is_final:
                            await self.send(
                                "response.first_token",
                                command_id=chunk.command_id,
                                ttft_ms=chunk.ttft_ms,
                            )
                    chunk.sampled_tokens += len(tokens)
                    if chunk.is_final and completion.text:
                        chunk.text_parts.append(completion.text)
                        await self.send(
                            "response.delta",
                            command_id=chunk.command_id,
                            delta=completion.text,
                        )
                    finish_reason = completion.finish_reason or finish_reason
                if finish_reason is not None or chunk.sampled_tokens >= chunk.expected_tokens:
                    await self._finish(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Streaming generation failed")
            with contextlib.suppress(Exception):
                await self.send("session.error", message=str(exc))

    async def close(self) -> None:
        """Cancel the active request and close this turn."""
        if self._closed:
            return
        self._closed = True
        if self._generation is not None and not self._generation.done():
            self._generation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._generation
        with contextlib.suppress(Exception):
            await self.engine.abort(self.request_id)


def create_app(engine: AsyncLLM) -> FastAPI:
    """Create the health and streaming-session endpoints."""
    app = FastAPI(title="Generic vLLM StreamingInput")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Report model server readiness."""
        return {"status": "healthy", "streaming_input": True}

    @app.websocket("/v1/streaming-session")
    async def streaming_session(websocket: WebSocket) -> None:
        """Serve one turn-scoped native StreamingInput session."""
        await websocket.accept()
        session: NativeStreamingSession | None = None
        try:
            while True:
                message = await websocket.receive_json()
                event_type = message.get("type")
                command_id = str(message.get("command_id") or uuid4())
                if event_type == "session.start":
                    if session is not None:
                        await session.close()
                    session = NativeStreamingSession(
                        engine=engine,
                        websocket=websocket,
                        session_id=str(message.get("session_id") or uuid4()),
                        system_prompt=str(message.get("system_prompt") or "You are a helpful assistant."),
                        history=_history(message.get("history")),
                        hold_words=max(0, int(message.get("hold_words", 0))),
                        final_max_tokens=max(1, int(message.get("final_max_tokens", 384))),
                    )
                    await session.prefill(command_id)
                    await session.send("session.started", command_id=command_id)
                elif session is None:
                    await websocket.send_json({"type": "session.error", "message": "session.start is required"})
                elif event_type == "input.append":
                    await session.observe(command_id, str(message.get("transcript", "")), is_final=False)
                elif event_type == "input.finalize":
                    await session.observe(command_id, str(message.get("transcript", "")), is_final=True)
                elif event_type == "session.close":
                    await session.close()
                    return
                else:
                    await session.send("session.error", command_id=command_id, message=f"Unknown event: {event_type}")
        except WebSocketDisconnect:
            if session is not None:
                await session.close()
        except Exception as exc:
            logger.exception("Streaming session failed")
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.send("session.error", message=str(exc))
                await session.close()

    return app


def build_parser() -> argparse.ArgumentParser:
    """Build vLLM engine arguments plus the streaming listener options."""
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    AsyncEngineArgs.add_cli_args(parser)
    return parser


def main() -> None:
    """Start the native vLLM model server."""
    args = build_parser().parse_args()
    logger.info(verify_installed_patch())
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs.from_cli_args(args))
    app = create_app(engine)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
