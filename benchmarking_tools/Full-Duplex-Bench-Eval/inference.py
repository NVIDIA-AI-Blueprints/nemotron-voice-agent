# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-Duplex Voice Agent Inference Client for Benchmark version v1, v1.5.

Connects to the Nemotron Voice Agent WebSocket API: registers a minimal session via
POST /api/session-config, then streams audio on /api/ws with protobuf frames.

Configure the voice agent (``.env``, ``services.yaml``, etc.) before starting the server;
this client does not override pipeline settings.
"""

from __future__ import annotations

import argparse
import asyncio
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
RECV_TIMEOUT = 5.0  # Timeout for receiving responses after input ends (seconds)

# Minimal body so the server selects cascaded mode; ASR/LLM/TTS come from server config.
MINIMAL_SESSION_BODY: dict[str, str] = {"pipeline_mode": "cascaded"}

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

    def __init__(self, http_base: str, ws_origin: str, ssl_context: ssl.SSLContext | None):
        """Initialize with parsed ``--server-url`` components."""
        self.http_base = http_base.rstrip("/")
        self.ws_origin = ws_origin.rstrip("/")
        self._ssl_context = ssl_context

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
                await websocket.send(frame.SerializeToString())

                current_idx = end_idx
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DUR:
                    break

                frame = frames_pb2.Frame(
                    audio=frames_pb2.AudioRawFrame(audio=silence, sample_rate=SAMPLE_RATE, num_channels=1)
                )
                await websocket.send(frame.SerializeToString())

            next_send_time += chunk_duration

    async def receive_audio_stream(
        self, websocket: Any, start_time: float, send_task: asyncio.Task
    ) -> tuple[list[np.ndarray], list[float]]:
        """Receive output audio until idle timeout after send completes."""
        output_chunks: list[np.ndarray] = []
        chunk_times: list[float] = []

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=RECV_TIMEOUT)

                frame = frames_pb2.Frame.FromString(response)
                if frame.WhichOneof("frame") == "audio":
                    audio_data = frame.audio.audio
                    if not audio_data:
                        continue

                    chunk = np.frombuffer(audio_data, dtype=np.int16)

                    current_time = time.time() - start_time
                    output_chunks.append(chunk)

                    if not chunk_times:
                        chunk_times.append(current_time)
                    else:
                        expected_time = chunk_times[-1] + len(output_chunks[-1]) / SAMPLE_RATE
                        if abs(current_time - chunk_times[-1]) < 0.05:
                            chunk_times.append(expected_time)
                        else:
                            chunk_times.append(current_time)

            except TimeoutError:
                if send_task.done():
                    break
                continue
            except websockets.exceptions.ConnectionClosed:
                break

        return output_chunks, chunk_times

    def assemble_and_trim_output(
        self, output_chunks: list[np.ndarray], chunk_times: list[float], target_duration: float
    ) -> np.ndarray:
        """Assemble chunks on a time axis and trim to input duration."""
        if not output_chunks:
            return np.array([], dtype=np.int16)

        target_samples = int(target_duration * SAMPLE_RATE)
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
            send_task = asyncio.create_task(self.send_audio_stream(websocket, input_audio))
            start_time = time.time()
            output_chunks, chunk_times = await self.receive_audio_stream(websocket, start_time, send_task)
            await send_task

        output_audio = self.assemble_and_trim_output(output_chunks, chunk_times, input_duration)
        sf.write(output_path, output_audio, SAMPLE_RATE)

    async def process_directory(self, input_dir: str, retry_samples: list[int] | None = None) -> None:
        """Process numeric sample subdirectories (input.wav / clean_input.wav)."""
        if retry_samples:
            sample_ids = retry_samples
        else:
            sample_ids = sorted(
                int(name)
                for name in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit()
            )

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

                await asyncio.sleep(1)

            if processed_count == 0:
                print(f"Warning: Skipped sample {sample_id}")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Full-Duplex-Bench inference client for Nemotron Voice Agent (WebSocket mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python inference.py --input_dir /path/to/samples --server-url http://127.0.0.1:7860",
    )

    parser.add_argument(
        "--input_dir",
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

    parser.add_argument("--retry_samples", nargs="+", type=int, help="Only process these sample IDs")

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

    client = InferenceClient(http_base, ws_origin, ssl_context)

    print(f"Server: {http_base} (WebSocket {ws_origin}/api/ws)")
    print(f"Processing directory: {args.input_dir}")

    await client.process_directory(input_dir=args.input_dir, retry_samples=args.retry_samples)

    print("Processing complete!")


if __name__ == "__main__":
    asyncio.run(main())
