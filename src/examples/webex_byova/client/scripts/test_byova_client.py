# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Smoke-test client for exercising the Cisco Webex BYOVA adapter locally."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import time
import wave
from pathlib import Path

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from cisco_webex_byova_adapter.generated import (
    byova_common_pb2,
    voicevirtualagent_pb2,
    voicevirtualagent_pb2_grpc,
)

TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 32
ENCODING_CHOICES = {
    "linear16": voicevirtualagent_pb2.VoiceInput.LINEAR16_FORMAT,
    "mulaw": voicevirtualagent_pb2.VoiceInput.MULAW_FORMAT,
    "alaw": voicevirtualagent_pb2.VoiceInput.ALAW_FORMAT,
    "unspecified": voicevirtualagent_pb2.VoiceInput.UNSPECIFIED_FORMAT,
}


def _silence_chunk(encoding: int, chunk_size: int) -> bytes:
    if encoding in {
        voicevirtualagent_pb2.VoiceInput.MULAW_FORMAT,
        voicevirtualagent_pb2.VoiceInput.UNSPECIFIED_FORMAT,
    }:
        return b"\xff" * chunk_size
    if encoding == voicevirtualagent_pb2.VoiceInput.ALAW_FORMAT:
        return b"\xd5" * chunk_size
    return b"\x00" * chunk_size


def load_caller_audio(path: Path, encoding: str, sample_rate: int | None) -> tuple[bytes, int, int]:
    """Load caller audio for the smoke client.

    Returns ``(payload_bytes, wire_sample_rate, wire_encoding)``.
    For mulaw/alaw, returns companded bytes at 8 kHz by default.
    """
    wire_encoding = ENCODING_CHOICES[encoding]
    suffix = path.suffix.lower()

    if encoding in {"mulaw", "alaw", "unspecified"}:
        if suffix == ".bin":
            payload = path.read_bytes()
            return payload, 8000, wire_encoding

        pcm, src_rate = _read_wav_pcm16(path, target_sample_rate=None)
        if src_rate != 8000:
            pcm, _ = audioop.ratecv(pcm, 2, 1, src_rate, 8000, None)
        if encoding in {"mulaw", "unspecified"}:
            payload = audioop.lin2ulaw(pcm, 2)
        else:
            payload = audioop.lin2alaw(pcm, 2)
        return payload, 8000, wire_encoding

    pcm, src_rate = _read_wav_pcm16(path)
    wire_rate = sample_rate or src_rate
    if wire_rate != src_rate:
        pcm, _ = audioop.ratecv(pcm, 2, 1, src_rate, wire_rate, None)
    return pcm, wire_rate, wire_encoding


def _read_wav_pcm16(path: Path, target_sample_rate: int | None = TARGET_SAMPLE_RATE) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav files are supported, got width={sample_width}")

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)

    if target_sample_rate and sample_rate != target_sample_rate:
        frames, _ = audioop.ratecv(frames, sample_width, 1, sample_rate, target_sample_rate, None)
        sample_rate = target_sample_rate

    return frames, sample_rate


async def _send_request(call, request: voicevirtualagent_pb2.VoiceVARequest) -> None:
    print({"sending": request.WhichOneof("voice_va_input_type") or "none"})
    await call.write(request)


async def send_audio_turn(call, args, timings: dict, greeting_done: asyncio.Event) -> None:
    """Send a greeting-start event followed by one caller audio turn."""
    try:
        audio_bytes, sample_rate, wire_encoding = load_caller_audio(
            Path(args.audio_file),
            args.caller_encoding,
            args.caller_sample_rate,
        )
        print(
            {
                "caller_audio": {
                    "file": args.audio_file,
                    "encoding": args.caller_encoding,
                    "sample_rate_hz": sample_rate,
                    "bytes": len(audio_bytes),
                }
            }
        )
        timings["t0_start"] = time.monotonic()
        await _send_request(
            call,
            voicevirtualagent_pb2.VoiceVARequest(
                conversation_id=args.conversation_id,
                customer_org_id=args.customer_org_id,
                virtual_agent_id=args.virtual_agent_id,
                vendor_specific_config=args.vendor_specific_config,
                allow_partial_responses=True,
                event_input=byova_common_pb2.EventInput(
                    event_type=byova_common_pb2.EventInput.SESSION_START,
                    name="session_start",
                ),
            ),
        )
        # Wait for the bot to finish its intro greeting (first FINAL on the
        # response stream) before pretending to be a caller. This matches a
        # real caller who would let the bot finish greeting first, and gives
        # us a clean reference point for the user-turn TTFB measurement.
        if args.skip_greeting_wait:
            timings["t_greeting_done"] = time.monotonic()
        else:
            print({"waiting_for_greeting_done": True})
            try:
                await asyncio.wait_for(greeting_done.wait(), timeout=args.greeting_timeout_secs)
                print({"greeting_done": True})
            except TimeoutError:
                print({"greeting_wait_timeout": True})
            timings["t_greeting_done"] = time.monotonic()
        bytes_per_sample = (
            1
            if wire_encoding
            in {
                voicevirtualagent_pb2.VoiceInput.MULAW_FORMAT,
                voicevirtualagent_pb2.VoiceInput.ALAW_FORMAT,
                voicevirtualagent_pb2.VoiceInput.UNSPECIFIED_FORMAT,
            }
            else 2
        )
        chunk_size = max(1, int(sample_rate * (CHUNK_MS / 1000)) * bytes_per_sample)
        for offset in range(0, len(audio_bytes), chunk_size):
            now = Timestamp()
            now.GetCurrentTime()
            await _send_request(
                call,
                voicevirtualagent_pb2.VoiceVARequest(
                    conversation_id=args.conversation_id,
                    customer_org_id=args.customer_org_id,
                    virtual_agent_id=args.virtual_agent_id,
                    allow_partial_responses=True,
                    audio_input=voicevirtualagent_pb2.VoiceInput(
                        caller_audio=audio_bytes[offset : offset + chunk_size],
                        encoding=wire_encoding,
                        sample_rate_hertz=sample_rate,
                        audio_timestamp=now,
                        language_code="en-US",
                        is_single_utterance=False,
                    ),
                ),
            )
            await asyncio.sleep(CHUNK_MS / 1000)

        silence_chunk = _silence_chunk(wire_encoding, chunk_size)
        # Streaming ASR requires a trailing-silence window to finalize
        # transcripts reliably. Send it explicitly before closing writes.
        silence_chunks = max(1, int(args.post_silence_ms / CHUNK_MS))
        print({"post_silence_ms": args.post_silence_ms, "post_silence_chunks": silence_chunks})
        for idx in range(silence_chunks):
            silence_ts = Timestamp()
            silence_ts.GetCurrentTime()
            await _send_request(
                call,
                voicevirtualagent_pb2.VoiceVARequest(
                    conversation_id=args.conversation_id,
                    customer_org_id=args.customer_org_id,
                    virtual_agent_id=args.virtual_agent_id,
                    allow_partial_responses=True,
                    audio_input=voicevirtualagent_pb2.VoiceInput(
                        caller_audio=silence_chunk,
                        encoding=wire_encoding,
                        sample_rate_hertz=sample_rate,
                        audio_timestamp=silence_ts,
                        language_code="en-US",
                        is_single_utterance=(idx == silence_chunks - 1),
                    ),
                ),
            )
            await asyncio.sleep(CHUNK_MS / 1000)
        # "Caller has stopped speaking": this is the reference point for the
        # client-side latency measurement. We close the write side of the
        # bidi stream immediately, which matches what Webex Universal
        # Harness actually does — one gRPC stream per caller turn, write
        # side closed promptly when the caller stops. The adapter then
        # drains the bot's reply over the still-open read side.
        timings["t_user_audio_end"] = time.monotonic()
        print({"done_writing": True})
        await call.done_writing()
    except Exception as exc:
        print({"sender_error": repr(exc)})
        raise


async def receive_responses(call, timings: dict, greeting_done: asyncio.Event) -> None:
    """Read adapter responses and capture simple client-side timing markers."""
    audio_chunks = 0
    audio_bytes = 0
    final_count = 0
    async for response in call:
        now = time.monotonic()
        if response.prompts:
            for prompt in response.prompts:
                if prompt.audio_content:
                    if "t_first_bot_audio" not in timings:
                        timings["t_first_bot_audio"] = now
                    timings["t_last_bot_audio"] = now
                    if greeting_done.is_set():
                        timings.setdefault("t_first_bot_audio_response", now)
                        timings["t_last_bot_audio_response"] = now
                    else:
                        timings.setdefault("t_first_bot_audio_greeting", now)
                    audio_chunks += 1
                    audio_bytes += len(prompt.audio_content)
        if response.response_type == voicevirtualagent_pb2.VoiceVAResponse.FINAL:
            final_count += 1
            if final_count == 1:
                timings["t_greeting_final"] = now
                greeting_done.set()
            else:
                timings.setdefault("t_response_final", now)
                timings.setdefault("t_final", now)
        event_names = [event.event_type for event in response.output_events]
        transcript = response.session_transcript.text if response.HasField("session_transcript") else ""
        print(
            {
                "response_type": response.response_type,
                "prompt_count": len(response.prompts),
                "audio_chunks": audio_chunks,
                "audio_bytes": audio_bytes,
                "events": event_names,
                "transcript": transcript,
            }
        )


async def main() -> None:
    """Run the local BYOVA smoke test client."""
    parser = argparse.ArgumentParser(description="Smoke-test the BYoVA adapter against a running Nemotron backend.")
    parser.add_argument("--grpc-target", default="127.0.0.1:50061")
    parser.add_argument("--audio-file", required=True)
    parser.add_argument(
        "--caller-encoding",
        choices=sorted(ENCODING_CHOICES),
        default="linear16",
        help="Wire format sent to adapter (Webex typically uses mulaw or unspecified at 8 kHz).",
    )
    parser.add_argument(
        "--caller-sample-rate",
        type=int,
        default=0,
        help="Wire sample rate in Hz (default: 16000 for linear16, 8000 for mulaw/alaw).",
    )
    parser.add_argument("--conversation-id", default=f"test-{int(time.time())}")
    parser.add_argument("--customer-org-id", default="local-test-org")
    parser.add_argument("--virtual-agent-id", default="nemotron-generic")
    parser.add_argument("--vendor-specific-config", default="{}")
    parser.add_argument("--response-wait-secs", type=float, default=8.0)
    parser.add_argument(
        "--post-silence-ms",
        type=int,
        default=1000,
        help="Trailing silence appended after caller audio so ASR can finalize.",
    )
    parser.add_argument(
        "--greeting-timeout-secs",
        type=float,
        default=20.0,
        help="Max seconds to wait for the bot's intro greeting (FIRST FINAL) before starting user audio.",
    )
    parser.add_argument(
        "--skip-greeting-wait",
        action="store_true",
        help="Start sending user audio immediately, without waiting for the greeting to end.",
    )
    parser.add_argument("--tls", action="store_true", help="Use a TLS gRPC channel.")
    parser.add_argument("--tls-ca", default="", help="Optional CA certificate (PEM) to validate the server cert.")
    parser.add_argument(
        "--tls-server-name",
        default="",
        help="Override the SNI / authority for TLS (useful when target is an IP).",
    )
    args = parser.parse_args()
    if not args.caller_sample_rate:
        args.caller_sample_rate = (
            8000 if args.caller_encoding in {"mulaw", "alaw", "unspecified"} else TARGET_SAMPLE_RATE
        )

    if args.tls:
        if args.tls_ca:
            with open(args.tls_ca, "rb") as f:
                root_certificates = f.read()
        else:
            root_certificates = None
        channel_credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
        options: list[tuple[str, str]] = []
        if args.tls_server_name:
            options.append(("grpc.ssl_target_name_override", args.tls_server_name))
        channel_ctx = grpc.aio.secure_channel(args.grpc_target, channel_credentials, options=options or None)
    else:
        channel_ctx = grpc.aio.insecure_channel(args.grpc_target)

    timings: dict[str, float] = {}
    greeting_done = asyncio.Event()
    async with channel_ctx as channel:
        stub = voicevirtualagent_pb2_grpc.VoiceVirtualAgentStub(channel)
        call = stub.ProcessCallerInput()
        sender = asyncio.create_task(send_audio_turn(call, args, timings, greeting_done))
        try:
            await asyncio.wait_for(
                receive_responses(call, timings, greeting_done),
                timeout=args.greeting_timeout_secs + args.response_wait_secs + 30,
            )
        except grpc.aio.AioRpcError as exc:
            print({"grpc_code": exc.code().name, "grpc_details": exc.details()})
        except asyncio.CancelledError:
            print({"grpc_cancelled": True})
        except TimeoutError:
            print({"receiver_timeout": True})
        finally:
            await sender
            _print_timings(timings)


def _print_timings(timings: dict) -> None:
    def delta_ms(start: str, end: str) -> str:
        if start in timings and end in timings:
            return f"{(timings[end] - timings[start]) * 1000:8.1f} ms"
        return "       -- "

    print("\n=== client-side E2E latency ===")
    print(
        "  SESSION_START -> greeting FINAL             :",
        delta_ms("t0_start", "t_greeting_final"),
        "   <-- bot intro time (cold start, can be high)",
    )
    print(
        "  greeting end -> caller stops speaking       :",
        delta_ms("t_greeting_done", "t_user_audio_end"),
        "   <-- smoke client paces audio at real time",
    )
    print(
        "  caller stops speaking -> bot response audio :",
        delta_ms("t_user_audio_end", "t_first_bot_audio_response"),
        "   <-- USER-TURN TTFB (the number you want)",
    )
    print(
        "  bot response audio: first chunk -> last     :",
        delta_ms("t_first_bot_audio_response", "t_last_bot_audio_response"),
    )
    print(
        "  caller stops -> response FINAL              :",
        delta_ms("t_user_audio_end", "t_response_final"),
    )
    print(
        "  full turn (SESSION_START -> response FINAL) :",
        delta_ms("t0_start", "t_response_final"),
    )


if __name__ == "__main__":
    asyncio.run(main())
