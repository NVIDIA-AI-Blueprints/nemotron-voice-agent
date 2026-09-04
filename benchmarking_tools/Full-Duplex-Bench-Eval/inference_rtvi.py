# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Full-Duplex Voice Agent Inference Client for Benchmark version v1, v1.5.

Connects to the Nemotron Voice Agent WebSocket API: registers a minimal session via
POST /api/session-config, then streams audio on /api/ws with protobuf frames.

Configure the voice agent (``.env``, ``services.yaml``, etc.) before starting the server;
this client does not override pipeline settings.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import resampy
import soundfile as sf
import websockets
from pipecat.frames.protobufs import frames_pb2

# Audio processing constants
SAMPLE_RATE = 16000  # Target sample rate in Hz
CHUNK_MS = 32  # Chunk duration in milliseconds
SILENCE_DUR = 2.0  # Silence duration after input audio (seconds) for end-of-utterance detection
RECV_POLL_TIMEOUT = 1.0
POST_INPUT_RESPONSE_TIMEOUT = 30.0
POST_AUDIO_IDLE_TIMEOUT = 3.0
SEND_TIMEOUT = 30.0
BOT_INTRO_FIRST_FRAME_TIMEOUT = 30.0
BOT_INTRO_IDLE_TIMEOUT = 1.5

# Example selection and ASR/LLM/TTS come from server configuration.
MINIMAL_SESSION_BODY: dict[str, str] = {}

DEFAULT_HTTP_PORT = 7860


def _ssl_context_insecure() -> ssl.SSLContext:
    """TLS context for local servers using self-signed certificates (dev/benchmark only)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def parse_server_url(url: str, *, insecure_skip_verify: bool = False) -> tuple[str, str, ssl.SSLContext | None]:
    """Parse ``http(s)://host[:port]`` into HTTP base URL, WebSocket origin, and optional SSL context.

    If the URL omits a port, ``7860`` is used (same default as ``src/server.py``).
    """
    p = urllib.parse.urlsplit(url.strip())
    scheme = (p.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("--server-url must use http:// or https:// (e.g. http://127.0.0.1:7860)")
    use_tls = scheme == "https"
    host = p.hostname
    if not host:
        raise ValueError("--server-url must include a host")
    port = p.port if p.port is not None else DEFAULT_HTTP_PORT

    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]:{port}"
    else:
        netloc = f"{host}:{port}"

    http_scheme = "https" if use_tls else "http"
    http_base = urllib.parse.urlunsplit((http_scheme, netloc, "", "", ""))

    ws_scheme = "wss" if use_tls else "ws"
    ws_origin = urllib.parse.urlunsplit((ws_scheme, netloc, "", "", ""))

    ssl_ctx = _ssl_context_insecure() if (use_tls and insecure_skip_verify) else None
    return http_base, ws_origin, ssl_ctx


def request_session_id(
    http_base: str,
    *,
    ssl_context: ssl.SSLContext | None,
    timeout_sec: float = 60.0,
) -> str:
    """POST /api/session-config and return session_id."""
    url = f"{http_base.rstrip('/')}/api/session-config"
    data = json.dumps(MINIMAL_SESSION_BODY).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=timeout_sec) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"session-config failed: HTTP {e.code} {err_body}") from e
    session_id = payload.get("session_id")
    if not session_id:
        raise RuntimeError("session-config response missing session_id")
    return session_id


class InferenceClient:
    """Client for the Nemotron Voice Agent WebSocket server."""

    def __init__(
        self,
        http_base: str,
        ws_origin: str,
        ssl_context: ssl.SSLContext | None,
        *,
        input_tail_silence: float = SILENCE_DUR,
        post_input_response_timeout: float = POST_INPUT_RESPONSE_TIMEOUT,
        post_audio_idle_timeout: float = POST_AUDIO_IDLE_TIMEOUT,
        preserve_late_output: bool = False,
        send_client_ready: bool = True,
        drain_bot_intro: bool = False,
        bot_intro_first_frame_timeout: float = BOT_INTRO_FIRST_FRAME_TIMEOUT,
        bot_intro_idle_timeout: float = BOT_INTRO_IDLE_TIMEOUT,
    ):
        """Initialize with parsed ``--server-url`` components."""
        self.http_base = http_base.rstrip("/")
        self.ws_origin = ws_origin.rstrip("/")
        self._ssl_context = ssl_context
        self._input_tail_silence = max(0.0, input_tail_silence)
        self._post_input_response_timeout = max(0.0, post_input_response_timeout)
        self._post_audio_idle_timeout = max(0.0, post_audio_idle_timeout)
        self._preserve_late_output = preserve_late_output
        self._send_client_ready = send_client_ready
        self._drain_bot_intro = drain_bot_intro
        self._bot_intro_first_frame_timeout = max(0.0, bot_intro_first_frame_timeout)
        self._bot_intro_idle_timeout = max(0.0, bot_intro_idle_timeout)

    @staticmethod
    def _parse_frame(response: Any) -> frames_pb2.Frame | None:
        """Parse a binary protobuf frame, ignoring unrelated messages."""
        if not isinstance(response, bytes | bytearray):
            return None
        try:
            return frames_pb2.Frame.FromString(response)
        except Exception:
            return None

    async def _send_client_ready_frame(self, websocket: Any) -> None:
        """Send the RTVI ready signal required to start the pipeline."""
        payload = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": "full-duplex-bench-client-ready",
            "data": {"version": "0.1.0", "about": {"name": "full-duplex-bench"}},
        }
        frame = frames_pb2.Frame(message=frames_pb2.MessageFrame(data=json.dumps(payload)))
        await asyncio.wait_for(websocket.send(frame.SerializeToString()), timeout=SEND_TIMEOUT)

    async def _drain_bot_intro_audio(self, websocket: Any) -> tuple[int, float]:
        """Discard an optional welcome turn before benchmark input starts."""
        received_audio = False
        chunk_count = 0
        duration = 0.0
        while True:
            timeout = self._bot_intro_idle_timeout if received_audio else self._bot_intro_first_frame_timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except (TimeoutError, websockets.exceptions.ConnectionClosed):
                return chunk_count, duration
            frame = self._parse_frame(response)
            if frame is None or frame.WhichOneof("frame") != "audio" or not frame.audio.audio:
                continue
            received_audio = True
            chunk_count += 1
            bytes_per_sample = 2 * max(1, frame.audio.num_channels)
            duration += len(frame.audio.audio) / bytes_per_sample / max(1, frame.audio.sample_rate)

    def _websocket_url(self, session_id: str) -> str:
        q = urllib.parse.urlencode({"session_id": session_id})
        return f"{self.ws_origin}/api/ws?{q}"

    def preprocess_audio(self, audio_path: str) -> tuple[np.ndarray, float]:
        """Preprocess audio to 16 kHz mono int16."""
        audio, sample_rate = sf.read(audio_path)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sample_rate != SAMPLE_RATE:
            audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)

        if audio.dtype != np.int16:
            if audio.dtype in (np.float32, np.float64):
                audio = np.clip(audio, -1.0, 1.0)
                audio = (audio * 32767).astype(np.int16)
            elif audio.dtype == np.uint8:
                audio = ((audio.astype(np.int16) - 128) * 256).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        duration = len(audio) / SAMPLE_RATE
        return audio, duration

    async def send_audio_stream(self, websocket: Any, audio: np.ndarray) -> None:
        """Stream preprocessed audio in real-time chunks."""
        chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
        chunk_duration = CHUNK_MS / 1000.0

        silence = np.zeros(chunk_samples, dtype=np.int16).tobytes()
        next_send_time = time.time()
        silence_start: float | None = None

        total_samples = len(audio)
        current_idx = 0

        while True:
            await asyncio.sleep(max(0, next_send_time - time.time()))

            if current_idx < total_samples:
                end_idx = min(current_idx + chunk_samples, total_samples)
                chunk = audio[current_idx:end_idx]

                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                frame = frames_pb2.Frame(
                    audio=frames_pb2.AudioRawFrame(audio=chunk.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)
                )
                await asyncio.wait_for(websocket.send(frame.SerializeToString()), timeout=SEND_TIMEOUT)

                current_idx = end_idx
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self._input_tail_silence:
                    break

                frame = frames_pb2.Frame(
                    audio=frames_pb2.AudioRawFrame(audio=silence, sample_rate=SAMPLE_RATE, num_channels=1)
                )
                await asyncio.wait_for(websocket.send(frame.SerializeToString()), timeout=SEND_TIMEOUT)

            next_send_time += chunk_duration

    async def receive_audio_stream(
        self, websocket: Any, start_time: float, send_task: asyncio.Task
    ) -> tuple[list[np.ndarray], list[float]]:
        """Receive output audio until idle timeout after send completes."""
        output_chunks: list[np.ndarray] = []
        chunk_times: list[float] = []
        send_done_at: float | None = None
        last_post_send_audio_at: float | None = None

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=RECV_POLL_TIMEOUT)

                frame = self._parse_frame(response)
                if frame is not None and frame.WhichOneof("frame") == "audio":
                    audio_data = frame.audio.audio
                    if not audio_data:
                        continue

                    chunk = np.frombuffer(audio_data, dtype=np.int16)
                    previous_chunk = output_chunks[-1] if output_chunks else None
                    current_time = time.time() - start_time
                    output_chunks.append(chunk)
                    now = time.time()
                    if send_task.done():
                        send_task.result()
                        send_done_at = send_done_at or now
                        last_post_send_audio_at = now

                    if not chunk_times:
                        chunk_times.append(current_time)
                    else:
                        assert previous_chunk is not None
                        expected_time = chunk_times[-1] + len(previous_chunk) / SAMPLE_RATE
                        if abs(current_time - chunk_times[-1]) < 0.05:
                            chunk_times.append(expected_time)
                        else:
                            chunk_times.append(current_time)

            except TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                break

            if not send_task.done():
                continue
            send_task.result()
            now = time.time()
            send_done_at = send_done_at or now
            if last_post_send_audio_at is None:
                if now - send_done_at >= self._post_input_response_timeout:
                    break
            elif now - last_post_send_audio_at >= self._post_audio_idle_timeout:
                break

        return output_chunks, chunk_times

    @staticmethod
    async def _settle_send_task(send_task: asyncio.Task) -> None:
        """Cancel and consume a sender task during connection cleanup."""
        if not send_task.done():
            send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await send_task

    def assemble_and_trim_output(
        self, output_chunks: list[np.ndarray], chunk_times: list[float], target_duration: float
    ) -> np.ndarray:
        """Assemble chunks on a time axis and trim to input duration."""
        target_samples = int(round(target_duration * SAMPLE_RATE))
        if not output_chunks:
            return np.zeros(target_samples, dtype=np.int16)
        if self._preserve_late_output:
            target_samples = max(
                target_samples,
                max(
                    int(timestamp * SAMPLE_RATE) + len(chunk)
                    for chunk, timestamp in zip(output_chunks, chunk_times, strict=True)
                ),
            )
        output = np.zeros(target_samples, dtype=np.int16)

        next_expected_time: float | None = None

        for chunk, timestamp in zip(output_chunks, chunk_times, strict=True):
            if len(chunk) == 0:
                continue

            chunk_duration = len(chunk) / SAMPLE_RATE
            start_sample = int(timestamp * SAMPLE_RATE)
            end_sample = start_sample + len(chunk)

            if next_expected_time is not None:
                time_gap = timestamp - next_expected_time
                if time_gap <= chunk_duration * 1.5:
                    start_sample = int(next_expected_time * SAMPLE_RATE)
                    end_sample = start_sample + len(chunk)

            if start_sample >= target_samples:
                break

            end_sample = min(end_sample, target_samples)
            chunk_to_write = chunk[: end_sample - start_sample]

            output[start_sample:end_sample] = chunk_to_write

            next_expected_time = start_sample / SAMPLE_RATE + len(chunk_to_write) / SAMPLE_RATE

        return output

    async def process_single_file(self, input_path: str, output_path: str) -> None:
        """Run one file: new session per call (server consumes session on WebSocket connect)."""
        input_audio, input_duration = self.preprocess_audio(input_path)

        session_id = request_session_id(self.http_base, ssl_context=self._ssl_context)
        ws_url = self._websocket_url(session_id)

        ssl_kw = {"ssl": self._ssl_context} if self._ssl_context else {}
        async with websockets.connect(ws_url, **ssl_kw) as websocket:
            if self._send_client_ready:
                await self._send_client_ready_frame(websocket)
            if self._drain_bot_intro:
                intro_chunks, intro_duration = await self._drain_bot_intro_audio(websocket)
                print(f"Drained bot intro: chunks={intro_chunks} duration={intro_duration:.2f}s")
            send_task = asyncio.create_task(self.send_audio_stream(websocket, input_audio))
            try:
                start_time = time.time()
                output_chunks, chunk_times = await self.receive_audio_stream(websocket, start_time, send_task)
                await send_task
            finally:
                await self._settle_send_task(send_task)

        output_audio = self.assemble_and_trim_output(output_chunks, chunk_times, input_duration)
        sf.write(output_path, output_audio, SAMPLE_RATE)

    async def process_directory(self, input_dir: str, retry_samples: list[int] | None = None) -> int:
        """Process numeric sample subdirectories (input.wav / clean_input.wav)."""
        if retry_samples:
            sample_ids = retry_samples
        else:
            sample_ids = sorted(
                int(name)
                for name in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit()
            )

        failed_count = 0
        for sample_id in sample_ids:
            sample_dir = os.path.join(input_dir, str(sample_id))
            file_pairs = [("input.wav", "output.wav"), ("clean_input.wav", "clean_output.wav")]

            processed_count = 0
            for input_filename, output_filename in file_pairs:
                input_path = os.path.join(sample_dir, input_filename)
                output_path = os.path.join(sample_dir, output_filename)

                if not os.path.exists(input_path):
                    continue

                print(f"Processing sample {sample_id}/{input_filename}...")
                try:
                    await self.process_single_file(input_path, output_path)
                    print(f"Successfully processed sample {sample_id}/{input_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing sample {sample_id}/{input_filename}: {e}")
                    failed_count += 1

            if processed_count == 0:
                print(f"Warning: Skipped sample {sample_id}")
                failed_count += 1
        return failed_count


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Full-Duplex-Bench inference client for Nemotron Voice Agent (WebSocket mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example: python inference_rtvi.py --input-dir /path/to/samples --server-url http://127.0.0.1:7860"),
    )

    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=str,
        required=True,
        help="Directory containing numeric sample subfolders with audio files",
    )

    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Server base URL with http:// or https:// (default port if omitted: 7860). Example: http://127.0.0.1:7860",
    )

    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        help="Disable TLS certificate verification for https:// server URLs. Use only for local self-signed certs.",
    )

    parser.add_argument(
        "--retry-samples",
        "--retry_samples",
        dest="retry_samples",
        nargs="+",
        type=int,
        help="Only process these sample IDs",
    )
    parser.add_argument("--input-tail-silence", type=float, default=SILENCE_DUR)
    parser.add_argument("--post-input-response-timeout", type=float, default=POST_INPUT_RESPONSE_TIMEOUT)
    parser.add_argument("--post-audio-idle-timeout", type=float, default=POST_AUDIO_IDLE_TIMEOUT)
    parser.add_argument("--skip-client-ready", action="store_true")
    parser.add_argument(
        "--drain-bot-intro",
        action="store_true",
        help="Discard an enabled server welcome turn before streaming benchmark input.",
    )
    parser.add_argument("--bot-intro-first-frame-timeout", type=float, default=BOT_INTRO_FIRST_FRAME_TIMEOUT)
    parser.add_argument("--bot-intro-idle-timeout", type=float, default=BOT_INTRO_IDLE_TIMEOUT)
    parser.add_argument(
        "--preserve-late-output",
        action="store_true",
        help="Write audio after input EOF. Use only for the v1.0 user-interruption workflow.",
    )

    return parser.parse_args()


async def main() -> None:
    """CLI entry point."""
    args = parse_arguments()

    try:
        http_base, ws_origin, ssl_context = parse_server_url(
            args.server_url,
            insecure_skip_verify=args.insecure_skip_verify,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}") from e

    client = InferenceClient(
        http_base,
        ws_origin,
        ssl_context,
        input_tail_silence=args.input_tail_silence,
        post_input_response_timeout=args.post_input_response_timeout,
        post_audio_idle_timeout=args.post_audio_idle_timeout,
        preserve_late_output=args.preserve_late_output,
        send_client_ready=not args.skip_client_ready,
        drain_bot_intro=args.drain_bot_intro,
        bot_intro_first_frame_timeout=args.bot_intro_first_frame_timeout,
        bot_intro_idle_timeout=args.bot_intro_idle_timeout,
    )

    print(f"Server: {http_base} (WebSocket {ws_origin}/api/ws)")
    print(f"Processing directory: {args.input_dir}")

    failed_count = await client.process_directory(input_dir=args.input_dir, retry_samples=args.retry_samples)
    if failed_count:
        raise SystemExit(f"Inference failed for {failed_count} sample/audio file(s)")

    print("Processing complete!")


if __name__ == "__main__":
    asyncio.run(main())
