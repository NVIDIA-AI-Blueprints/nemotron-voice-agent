#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Generate time-synchronous Full-Duplex-Bench audio through OpenAI Realtime."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import ssl
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import resampy
import soundfile as sf
import websockets

SAMPLE_RATE = 16000
CHUNK_MS = 32
INPUT_TAIL_SILENCE = 2.0
RECV_POLL_TIMEOUT = 1.0
POST_INPUT_RESPONSE_TIMEOUT = 30.0
POST_AUDIO_IDLE_TIMEOUT = 3.0
SEND_TIMEOUT = 30.0


def parse_output_audio(event: Mapping[str, Any]) -> bytes:
    """Decode an audio delta from supported OpenAI Realtime event variants."""
    if event.get("type") not in {"response.audio.delta", "response.output_audio.delta"}:
        return b""
    delta = event.get("delta")
    if not isinstance(delta, str) or not delta:
        return b""
    try:
        return base64.b64decode(delta, validate=True)
    except (ValueError, TypeError):
        return b""


class RealtimeSocket:
    """Minimal authenticated OpenAI Realtime WebSocket session."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str,
        auth_scheme: str,
        connect_timeout: float,
        verify_tls: bool,
    ) -> None:
        """Initialize a disconnected Realtime session."""
        self.url = url
        self.api_key = api_key
        self.auth_scheme = auth_scheme
        self.connect_timeout = connect_timeout
        self.verify_tls = verify_tls
        self.ws: Any = None

    async def __aenter__(self) -> RealtimeSocket:
        """Connect and return this session."""
        headers = {"Authorization": f"{self.auth_scheme} {self.api_key}"} if self.api_key else {}
        kwargs: dict[str, Any] = {
            "additional_headers": headers,
            "max_size": None,
            "open_timeout": self.connect_timeout,
        }
        if self.url.startswith("wss://") and not self.verify_tls:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            kwargs["ssl"] = ssl_context
        self.ws = await websockets.connect(self.url, **kwargs)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the WebSocket when leaving the session."""
        del exc_type, exc, traceback
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
        self.ws = None

    async def send_event(self, payload: Mapping[str, Any]) -> None:
        """Serialize and send one Realtime JSON event."""
        if self.ws is None:
            raise RuntimeError("Realtime socket is not connected")
        await self.ws.send(json.dumps(dict(payload)))

    async def send_pcm(self, pcm: bytes) -> None:
        """Append one signed PCM16 audio chunk."""
        await self.send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def receive_event(self, timeout: float) -> dict[str, Any]:
        """Receive and validate one Realtime event."""
        if self.ws is None:
            raise RuntimeError("Realtime socket is not connected")
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            return {"type": "_binary", "data": raw}
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise RuntimeError(f"Expected a JSON object, received {type(event).__name__}")
        if event.get("type") == "error":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"Realtime API error: {message or event}")
        return event

    async def configure(self, input_sample_rate: int, timeout: float) -> None:
        """Configure PCM input and wait for acknowledgement."""
        await self.send_event(
            {
                "type": "session.update",
                "session": {
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": input_sample_rate,
                            }
                        }
                    }
                },
            }
        )
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for session.updated")
            event = await self.receive_event(remaining)
            if event.get("type") == "session.updated":
                return


class InferenceClient:
    """Stream benchmark audio and write aligned Realtime response audio."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str,
        auth_scheme: str,
        output_sample_rate: int,
        connect_timeout: float,
        verify_tls: bool,
        input_tail_silence: float,
        post_input_response_timeout: float,
        post_audio_idle_timeout: float,
        preserve_late_output: bool,
    ) -> None:
        """Configure transport, audio, and output-window behavior."""
        self.url = url
        self.api_key = api_key
        self.auth_scheme = auth_scheme
        self.output_sample_rate = output_sample_rate
        self.connect_timeout = connect_timeout
        self.verify_tls = verify_tls
        self.input_tail_silence = max(0.0, input_tail_silence)
        self.post_input_response_timeout = max(0.0, post_input_response_timeout)
        self.post_audio_idle_timeout = max(0.0, post_audio_idle_timeout)
        self.preserve_late_output = preserve_late_output

    @staticmethod
    def preprocess_audio(path: Path) -> tuple[np.ndarray, float]:
        """Load audio as mono 16 kHz signed PCM samples."""
        audio, sample_rate = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio, -1.0, 1.0) * 32767.0
            audio = audio.astype(np.int16)
        return audio, len(audio) / SAMPLE_RATE

    async def send_audio(self, socket: RealtimeSocket, audio: np.ndarray) -> None:
        """Stream input and tail silence at wall-clock speed."""
        chunk_samples = round(SAMPLE_RATE * CHUNK_MS / 1000)
        chunk_duration = chunk_samples / SAMPLE_RATE
        silence = np.zeros(chunk_samples, dtype=np.int16)
        next_send_time = asyncio.get_running_loop().time()

        for offset in range(0, len(audio), chunk_samples):
            await asyncio.sleep(max(0.0, next_send_time - asyncio.get_running_loop().time()))
            chunk = audio[offset : offset + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            await asyncio.wait_for(socket.send_pcm(chunk.tobytes()), timeout=SEND_TIMEOUT)
            next_send_time += chunk_duration

        silence_chunks = round(self.input_tail_silence / chunk_duration)
        for _ in range(silence_chunks):
            await asyncio.sleep(max(0.0, next_send_time - asyncio.get_running_loop().time()))
            await asyncio.wait_for(socket.send_pcm(silence.tobytes()), timeout=SEND_TIMEOUT)
            next_send_time += chunk_duration

    def decode_audio(self, event: Mapping[str, Any]) -> np.ndarray | None:
        """Decode and resample one output-audio event."""
        encoded = parse_output_audio(event)
        if not encoded:
            return None
        chunk = np.frombuffer(encoded, dtype=np.int16)
        if self.output_sample_rate == SAMPLE_RATE:
            return chunk.copy()
        normalized = chunk.astype(np.float32) / 32768.0
        resampled = resampy.resample(normalized, self.output_sample_rate, SAMPLE_RATE)
        return np.clip(resampled * 32767.0, -32768, 32767).astype(np.int16)

    async def receive_audio(
        self,
        socket: RealtimeSocket,
        start_time: float,
        send_task: asyncio.Task[None],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Collect output chunks until the post-input idle deadline."""
        chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        send_done_at: float | None = None
        last_post_send_audio_at: float | None = None

        while True:
            try:
                event = await socket.receive_event(RECV_POLL_TIMEOUT)
                chunk = self.decode_audio(event)
                now = time.monotonic()
                if send_task.done() and send_done_at is None:
                    send_task.result()
                    send_done_at = now
                if chunk is not None and len(chunk):
                    previous = chunks[-1] if chunks else None
                    current_time = now - start_time
                    chunks.append(chunk)
                    if send_task.done():
                        last_post_send_audio_at = now
                    if not timestamps:
                        timestamps.append(current_time)
                    else:
                        assert previous is not None
                        expected = timestamps[-1] + len(previous) / SAMPLE_RATE
                        timestamps.append(expected if abs(current_time - timestamps[-1]) < 0.05 else current_time)
            except TimeoutError:
                now = time.monotonic()
            except websockets.exceptions.ConnectionClosed:
                break

            if not send_task.done():
                continue
            send_task.result()
            send_done_at = send_done_at or now
            if last_post_send_audio_at is None:
                if now - send_done_at >= self.post_input_response_timeout:
                    break
            elif now - last_post_send_audio_at >= self.post_audio_idle_timeout:
                break

        return chunks, timestamps

    def assemble_output(
        self,
        chunks: list[np.ndarray],
        timestamps: list[float],
        input_duration: float,
    ) -> np.ndarray:
        """Place response chunks on the input timeline and enforce its duration."""
        target_samples = round(input_duration * SAMPLE_RATE)
        if self.preserve_late_output and chunks:
            target_samples = max(
                target_samples,
                max(
                    round(timestamp * SAMPLE_RATE) + len(chunk)
                    for chunk, timestamp in zip(chunks, timestamps, strict=True)
                ),
            )
        output = np.zeros(target_samples, dtype=np.int16)
        next_expected_time: float | None = None
        for chunk, timestamp in zip(chunks, timestamps, strict=True):
            start = round(timestamp * SAMPLE_RATE)
            chunk_duration = len(chunk) / SAMPLE_RATE
            if next_expected_time is not None and timestamp - next_expected_time <= chunk_duration * 1.5:
                start = round(next_expected_time * SAMPLE_RATE)
            if start >= target_samples:
                continue
            end = min(target_samples, start + len(chunk))
            if start < 0:
                chunk = chunk[-start:]
                start = 0
            output[start:end] = chunk[: end - start]
            next_expected_time = end / SAMPLE_RATE
        return output

    async def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process one WAV using an isolated Realtime session."""
        audio, duration = self.preprocess_audio(input_path)
        socket = RealtimeSocket(
            self.url,
            api_key=self.api_key,
            auth_scheme=self.auth_scheme,
            connect_timeout=self.connect_timeout,
            verify_tls=self.verify_tls,
        )
        async with socket:
            await socket.configure(SAMPLE_RATE, self.connect_timeout)
            send_task = asyncio.create_task(self.send_audio(socket, audio))
            try:
                chunks, timestamps = await self.receive_audio(
                    socket,
                    time.monotonic(),
                    send_task,
                )
                await send_task
            finally:
                if not send_task.done():
                    send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await send_task

        output = self.assemble_output(chunks, timestamps, duration)
        sf.write(output_path, output, SAMPLE_RATE, subtype="PCM_16")

    async def process_directory(
        self,
        input_dir: Path,
        retry_samples: set[int] | None,
        concurrency: int,
    ) -> int:
        """Process numeric benchmark sample directories with bounded concurrency."""
        sample_dirs = sorted(
            (path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        if retry_samples is not None:
            sample_dirs = [path for path in sample_dirs if int(path.name) in retry_samples]

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def process_sample(sample_dir: Path) -> int:
            async with semaphore:
                processed = 0
                failed = 0
                for input_name, output_name in (
                    ("input.wav", "output.wav"),
                    ("clean_input.wav", "clean_output.wav"),
                ):
                    input_path = sample_dir / input_name
                    if not input_path.exists():
                        continue
                    print(f"Processing {sample_dir.name}/{input_name}...")
                    try:
                        await self.process_file(input_path, sample_dir / output_name)
                    except Exception as exc:
                        print(f"Error processing {sample_dir.name}/{input_name}: {exc}")
                        failed += 1
                    processed += 1
                if processed == 0:
                    print(f"Warning: no input WAV found in {sample_dir}")
                    failed += 1
                return failed

        results = await asyncio.gather(*(process_sample(path) for path in sample_dirs))
        return sum(results)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--realtime-ws-url", required=True)
    parser.add_argument("--api-key-env", default="REALTIME_API_KEY")
    parser.add_argument("--auth-scheme", default="Bearer")
    parser.add_argument("--output-sample-rate", type=int, default=24000)
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--input-tail-silence", type=float, default=INPUT_TAIL_SILENCE)
    parser.add_argument("--post-input-response-timeout", type=float, default=POST_INPUT_RESPONSE_TIMEOUT)
    parser.add_argument("--post-audio-idle-timeout", type=float, default=POST_AUDIO_IDLE_TIMEOUT)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retry-samples", "--retry_samples", dest="retry_samples", nargs="+", type=int)
    parser.add_argument("--insecure-skip-verify", action="store_true")
    parser.add_argument(
        "--preserve-late-output",
        action="store_true",
        help="Write audio after input EOF. Use only for the v1.0 user-interruption workflow.",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the command-line client."""
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Set the Realtime API key in {args.api_key_env}")
    client = InferenceClient(
        args.realtime_ws_url,
        api_key=api_key,
        auth_scheme=args.auth_scheme,
        output_sample_rate=args.output_sample_rate,
        connect_timeout=args.connect_timeout,
        verify_tls=not args.insecure_skip_verify,
        input_tail_silence=args.input_tail_silence,
        post_input_response_timeout=args.post_input_response_timeout,
        post_audio_idle_timeout=args.post_audio_idle_timeout,
        preserve_late_output=args.preserve_late_output,
    )
    failed_count = await client.process_directory(
        args.input_dir,
        set(args.retry_samples) if args.retry_samples else None,
        args.concurrency,
    )
    if failed_count:
        raise SystemExit(f"Inference failed for {failed_count} sample/audio file(s)")


if __name__ == "__main__":
    asyncio.run(main())
