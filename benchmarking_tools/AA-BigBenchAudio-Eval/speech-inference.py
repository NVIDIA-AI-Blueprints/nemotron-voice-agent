#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speech inference and preprocess for Big Bench Audio. See README for prerequisites and usage."""

import asyncio
import contextlib
import json
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

import numpy as np
import resampy
import soundfile as sf
import websockets
from pipecat.frames.protobufs import frames_pb2

# 16 kHz mono WAV is required by the Nemotron voice agent WebSocket transport. Adjust SAMPLE_RATE if your stack differs.
SAMPLE_RATE, CHUNK_MS = 16000, 32
# After we finish sending input, allow a short initial delay for the first output frame to arrive.
POST_SEND_INITIAL_WAIT_SEC = 5.0
# Once output has started, tolerate output gaps up to ~2s (measured since last received frame).
POST_SEND_GAP_TOLERANCE_SEC = 2.0
# How long to wait for a single websocket recv() before checking termination conditions.
RECV_TIMEOUT_SEC = 2.0
# Bump websocket timeouts to tolerate slower connections.
OPEN_TIMEOUT_SEC = 5.0

# POST /api/session-config body so the server runs cascaded mode; ASR/LLM/TTS come from server config.
MINIMAL_SESSION_BODY: dict[str, str] = {"pipeline_mode": "cascaded"}
DEFAULT_HTTP_PORT = 7860


def _ssl_context_insecure() -> ssl.SSLContext:
    """TLS context for self-signed certificates (local benchmarking only)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def parse_server_url(url: str, *, insecure_skip_verify: bool = False) -> tuple[str, str, ssl.SSLContext | None]:
    """Parse ``--server-url`` into HTTP base (for session POST), ws/wss origin, and SSL context if HTTPS."""
    p = urllib.parse.urlsplit(url.strip())
    scheme = (p.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("--server-url must use http:// or https:// (e.g. http://127.0.0.1:7860)")
    use_tls = scheme == "https"
    host = p.hostname
    if not host:
        raise ValueError("--server-url must include a host")
    port = p.port if p.port is not None else DEFAULT_HTTP_PORT  # same default as ``src/server.py``

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


class BenchmarkClient:
    """WebSocket client for streaming WAV to the voice agent and writing ``output.wav`` per sample.

    Each sample uses ``POST /api/session-config`` then ``/api/ws?session_id=...`` (see Nemotron ``src/server.py``).
    """

    def __init__(self, http_base: str, ws_origin: str, ssl_context: ssl.SSLContext | None):
        """Store parsed ``--server-url`` components (HTTP base, ws/wss origin, optional TLS for HTTPS)."""
        self.http_base = http_base.rstrip("/")
        self.ws_origin = ws_origin.rstrip("/")
        self._ssl_context = ssl_context

    def _websocket_url(self, session_id: str) -> str:
        q = urllib.parse.urlencode({"session_id": session_id})
        return f"{self.ws_origin}/api/ws?{q}"

    async def process_conversation(
        self,
        websocket: Any,
        input_wav_path: str,
        output_dir: str,
        output_filename: str = "output.wav",
    ):
        """Stream ``input_wav_path`` over the websocket, collect responses, write ``output_dir``/``output_filename``.

        Send and receive run in parallel (non-blocking). Output chunk times use ``start_time`` so input and output
        share a timeline. Exit rules (gap tolerance, initial wait) apply only after the full input file has been
        sent (including the silence tail), so we capture the main response. Incoming audio is raw int16 PCM from
        the Nemotron serializer (no WAV header).
        """
        output_chunks: list[np.ndarray] = []
        chunk_times: list[float] = []
        stop_silence_event = asyncio.Event()
        input_send_complete_event = asyncio.Event()
        send_task = asyncio.create_task(
            self.send_audio_file(websocket, input_wav_path, stop_silence_event, input_send_complete_event)
        )
        sender_done_at = None
        last_frame_at = None
        received_any_frame = False
        start_time = time.time()

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=RECV_TIMEOUT_SEC)
                frame = frames_pb2.Frame.FromString(response)
                if frame.WhichOneof("frame") == "audio":
                    audio = frame.audio.audio
                    if not audio:
                        continue
                    chunk = np.frombuffer(audio, dtype=np.int16)
                    if len(chunk) == 0:
                        continue

                    last_frame_at = time.time()
                    received_any_frame = True
                    output_chunks.append(chunk)
                    curr_time = time.time() - start_time  # same time base as session for input/output sync
                    # Use current time if this is the first chunk or if time gap is significant,
                    # otherwise use sequential timing from previous chunk
                    if not chunk_times or abs(curr_time - chunk_times[-1]) >= 0.05:
                        chunk_times.append(curr_time)
                    else:
                        chunk_times.append(chunk_times[-1] + len(chunk) / SAMPLE_RATE)
            except TimeoutError:
                current_time = time.time()

                # Apply exit rules only after the full input file has been sent (so we don't
                # treat intermediate response as "done" while we're still sending a long file).
                if not input_send_complete_event.is_set():
                    continue

                # Input send complete: apply gap tolerance or initial wait.
                if received_any_frame:
                    if last_frame_at and (current_time - last_frame_at) >= POST_SEND_GAP_TOLERANCE_SEC:
                        stop_silence_event.set()
                        break
                    continue

                # No output yet: wait POST_SEND_INITIAL_WAIT_SEC after input send complete.
                if sender_done_at is None:
                    sender_done_at = current_time
                elif (current_time - sender_done_at) >= POST_SEND_INITIAL_WAIT_SEC:
                    stop_silence_event.set()
                    break
                continue
            except websockets.exceptions.ConnectionClosed:
                stop_silence_event.set()
                break

        # Stop sending silence; send_task will exit when it checks the event or when caller closes websocket.
        stop_silence_event.set()
        if not send_task.done():
            send_task.cancel()

        try:
            if output_chunks and len(output_chunks) > 0:
                # Calculate duration based on actual received chunks.
                # Use the latest timestamp + duration of last chunk, or total audio duration, whichever is larger.
                if len(chunk_times) != len(output_chunks):
                    # Safety check: ensure arrays match
                    min_len = min(len(chunk_times), len(output_chunks))
                    chunk_times = chunk_times[:min_len]
                    output_chunks = output_chunks[:min_len]

                duration = max(
                    chunk_times[-1] + len(output_chunks[-1]) / SAMPLE_RATE,
                    sum(len(c) / SAMPLE_RATE for c in output_chunks),
                )
                output = np.zeros(int(duration * SAMPLE_RATE), dtype=np.int16)

                for chunk, t in zip(output_chunks, chunk_times, strict=True):
                    if not len(chunk):
                        continue
                    start = max(0, int(t * SAMPLE_RATE))
                    if start >= len(output):
                        print(
                            f"warning: dropping output chunk for {output_filename}: "
                            f"start={start} end={start + len(chunk)} output_len={len(output)} chunk_size={len(chunk)}"
                        )
                        continue
                    end = min(start + len(chunk), len(output))

                    if np.any(output[start:end] != 0):
                        max_iterations = max(1, (len(output) - start) // max(1, len(chunk)) + 1)
                        iterations = 0
                        while start < len(output) and np.any(output[start : start + len(chunk)] != 0):
                            start += len(chunk)
                            iterations += 1
                            if iterations >= max_iterations:
                                break
                        if start >= len(output):
                            print(
                                f"warning: unable to place output chunk for {output_filename}: "
                                f"start={start} end={start + len(chunk)} "
                                f"output_len={len(output)} chunk_size={len(chunk)}"
                            )
                            continue
                        if iterations >= max_iterations and np.any(output[start : start + len(chunk)] != 0):
                            print(
                                f"warning: overlap placement guard hit for {output_filename}: "
                                f"start={start} output_len={len(output)} chunk_size={len(chunk)}"
                            )
                            continue
                        end = min(start + len(chunk), len(output))

                    output[start:end] = chunk[: end - start]

                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, output, SAMPLE_RATE)
            else:
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, np.array([], dtype=np.int16), SAMPLE_RATE)
        except Exception as e:
            print(f"Error writing output file for {output_filename}: {e}")
            raise

    async def process_directory(self, input_dir, batch_size: int = 1, start=None, end=None, samples=None):
        """Process sample dirs: stream ``input.wav``, write ``output.wav``.

        Optional filters: ``start``, ``end``, or ``samples``.
        """
        sample_ids = []
        for name in os.listdir(input_dir):
            if not os.path.isdir(os.path.join(input_dir, name)):
                continue
            try:
                sample_ids.append(int(name))
            except ValueError:
                print(f"warning: skipping non-numeric subdir: {name!r}")
        sample_ids.sort()

        # Filter by specific samples list if provided
        if samples is not None:
            sample_ids = [i for i in sample_ids if i in samples]
        # Otherwise filter by start/end range if specified
        elif start is not None and end is not None:
            sample_ids = [i for i in sample_ids if start <= i <= end]
        elif start is not None:
            sample_ids = [i for i in sample_ids if i >= start]
        elif end is not None:
            sample_ids = [i for i in sample_ids if i <= end]

        semaphore = asyncio.Semaphore(max(1, batch_size))

        async def process_single(sample_id: str):
            sample_dir = os.path.join(input_dir, sample_id)
            input_wav = os.path.join(sample_dir, "input.wav")
            output_filename = "output.wav"

            if not os.path.exists(input_wav):
                return

            # New session per sample (server consumes session on WebSocket connect).
            print(f"processing idx= {sample_id} -> {self.http_base} (session + {self.ws_origin}/api/ws)")
            websocket = None
            try:
                session_id = request_session_id(self.http_base, ssl_context=self._ssl_context)
                ws_url = self._websocket_url(session_id)
                ssl_kw = {"ssl": self._ssl_context} if self._ssl_context else {}
                websocket = await websockets.connect(ws_url, open_timeout=OPEN_TIMEOUT_SEC, **ssl_kw)
                await self.process_conversation(websocket, input_wav, sample_dir, output_filename)
                print(f"successfully processed idx: {sample_id}")
            except Exception as e:
                print(f"unsuccessful idx: {sample_id} — {e}")
            finally:
                # Close websocket with very short timeout to avoid blocking
                if websocket:
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(websocket.close(), timeout=0.1)
                await asyncio.sleep(1)

        async def guarded(sample_id: str):
            async with semaphore:
                await process_single(sample_id)

        tasks = [asyncio.create_task(guarded(str(sample_id))) for sample_id in sample_ids]
        if tasks:
            await asyncio.gather(*tasks)

    async def send_audio_file(self, websocket, file_path, stop_silence_event, input_send_complete_event: asyncio.Event):
        """Stream ``file_path`` WAV to websocket; set ``input_send_complete_event`` when file + silence tail is done."""
        if not os.path.exists(file_path):
            print(f"Input audio file not found: {file_path}")
            return

        try:
            with wave.open(file_path, "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                frame_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

                # Calculate chunk size based on target parameters
                chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
                chunk_dur = CHUNK_MS / 1000

                # Calculate original chunk size for reading from file
                original_chunk_samples = int(frame_rate * CHUNK_MS / 1000)

                silence = np.zeros(chunk_samples, dtype=np.int16).tobytes()
                next_time = time.time()
                # Only signal "input send over" after we've been sending silence this long
                # (so we don't apply exit rules while server is still receiving/processing the stream).
                silence_phase_min_sec = 2.0
                silence_send_start = None

                # Stream the audio file chunk by chunk
                while True:
                    try:
                        await asyncio.sleep(max(0, next_time - time.time()))
                    except asyncio.CancelledError:
                        break

                    if stop_silence_event.is_set():
                        break

                    chunk_bytes = wav_file.readframes(original_chunk_samples)
                    if not chunk_bytes:
                        if silence_send_start is None:
                            silence_send_start = time.time()
                        if (time.time() - silence_send_start) >= silence_phase_min_sec:
                            input_send_complete_event.set()
                        if stop_silence_event.is_set():
                            break
                        try:
                            await websocket.send(
                                frames_pb2.Frame(
                                    audio=frames_pb2.AudioRawFrame(
                                        audio=silence, sample_rate=SAMPLE_RATE, num_channels=1
                                    )
                                ).SerializeToString()
                            )
                        except (
                            websockets.exceptions.ConnectionClosed,
                            websockets.exceptions.ConnectionClosedOK,
                            Exception,
                        ):
                            break
                    else:
                        # Convert bytes to numpy array
                        if sample_width == 1:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
                        elif sample_width == 2:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                        elif sample_width == 3:
                            raw = np.frombuffer(chunk_bytes, dtype=np.uint8)
                            usable = (len(raw) // 3) * 3
                            raw = raw[:usable].reshape(-1, 3).astype(np.int32)
                            samples = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
                            sign_mask = 1 << 23
                            samples = np.where(samples & sign_mask, samples - (1 << 24), samples)
                            chunk = samples.astype(np.float32) / 8388607.0
                        elif sample_width == 4:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int32).astype(np.float32) / 2147483647.0
                        else:
                            print(
                                f"warning: unsupported sample_width={sample_width} in {file_path}, "
                                f"falling back to int16 "
                                f"(chunk_bytes len={len(chunk_bytes)}). Supported: 1, 2, 3, 4 bytes."
                            )
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0

                        # Handle multi-channel by averaging
                        if n_channels > 1:
                            chunk = chunk.reshape(-1, n_channels).mean(axis=1)

                        # Resample if necessary
                        if frame_rate != SAMPLE_RATE:
                            chunk = resampy.resample(chunk, frame_rate, SAMPLE_RATE)

                        # Ensure correct chunk size
                        if len(chunk) < chunk_samples:
                            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                        elif len(chunk) > chunk_samples:
                            chunk = chunk[:chunk_samples]

                        # Convert back to int16 and send
                        chunk_int16 = (chunk * 32767).astype(np.int16)
                        try:
                            await websocket.send(
                                frames_pb2.Frame(
                                    audio=frames_pb2.AudioRawFrame(
                                        audio=chunk_int16.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1
                                    )
                                ).SerializeToString()
                            )
                        except (
                            websockets.exceptions.ConnectionClosed,
                            websockets.exceptions.ConnectionClosedOK,
                            Exception,
                        ):
                            break

                    next_time += chunk_dur

        except wave.Error as e:
            print(f"Failed to read WAV file {file_path}: {e}")
            return
        except Exception as e:
            print(f"Error in send_audio_file: {e}")
            return


def validate_input_dir(input_dir: str) -> str:
    """Resolve and validate input directory; raise ValueError if missing or not a directory."""
    resolved = os.path.realpath(os.path.expanduser(str(input_dir)))
    if not os.path.exists(resolved):
        raise ValueError(f"Input directory does not exist: {resolved}")
    if not os.path.isdir(resolved):
        raise ValueError(f"Input path is not a directory: {resolved}")
    return resolved


def preprocess_bigbench_audio(input_root: str):
    """Convert input.mp3 → input.wav (16 kHz mono). See README for format notes."""
    sample_ids = []
    for name in os.listdir(input_root):
        if not os.path.isdir(os.path.join(input_root, name)):
            continue
        try:
            sample_ids.append(int(name))
        except ValueError:
            print(f"warning: skipping non-numeric subdir: {name!r}")
    sample_ids.sort()

    for sample_id in map(str, sample_ids):
        sample_dir = os.path.join(input_root, sample_id)
        mp3_path = os.path.join(sample_dir, "input.mp3")
        wav_path = os.path.join(sample_dir, "input.wav")
        if not os.path.exists(mp3_path):
            continue
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    mp3_path,
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    wav_path,
                ],
                check=True,
            )
        except Exception as e:
            print(f"Warning: preprocessing failed for id={sample_id}: {e}")
            continue


async def main():
    """Parse args, optionally preprocess MP3→WAV, then stream WAVs to the voice agent when ``--inference`` is set."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Speech inference for Big Bench Audio: preprocess MP3→WAV and/or stream WAVs to the voice agent."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing per-sample folders with input.mp3 or input.wav"
    )
    parser.add_argument(
        "--server-url",
        default="",
        help="Voice agent base URL: http:// or https:// (default port if omitted: 7860). Required with --inference.",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        help="Disable TLS certificate verification for https:// server URLs. Use only for local self-signed certs.",
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Convert input.mp3 to input.wav (16 kHz mono) under input_dir"
    )
    parser.add_argument("--start", type=int, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, help="End index (inclusive)")
    parser.add_argument("--retry_samples", type=str, help="Comma-separated list of specific sample IDs to retry")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to stream in parallel during inference (default: 1)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Stream prepared WAVs to the voice agent (disabled by default)",
    )
    args = parser.parse_args()
    input_root = validate_input_dir(args.input_dir)

    if args.preprocess:
        print(f"Preprocessing MP3s to WAV under: {input_root}")
        preprocess_bigbench_audio(input_root)

    if not args.inference:
        print("Inference skipped (pass --inference to enable streaming).")
        return

    if not (args.server_url or "").strip():
        parser.error("--server-url is required with --inference (e.g. http://127.0.0.1:7860)")

    try:
        http_base, ws_origin, ssl_context = parse_server_url(
            args.server_url,
            insecure_skip_verify=args.insecure_skip_verify,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}") from e

    samples_list = None
    if args.retry_samples:
        try:
            samples_list = [int(s.strip()) for s in args.retry_samples.split(",")]
        except ValueError:
            parser.error("--retry_samples must contain only comma-separated integers; got malformed ID")

    client = BenchmarkClient(http_base, ws_origin, ssl_context)
    await client.process_directory(
        input_root,
        batch_size=args.batch_size,
        start=args.start,
        end=args.end,
        samples=samples_list,
    )


if __name__ == "__main__":
    asyncio.run(main())
