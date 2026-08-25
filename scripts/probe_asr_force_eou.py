#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Prove Nemotron ASR ``force_eou`` is distinct from 400 ms silence endpointing.

Streams speech with ``stop_history=4500``, then 1.5 s of silence. A final
transcript in that window means ASR is still using a short stop_history.
Then sends ``runtime_config.force_eou=true``; a final shortly after means
force_eou is what flushed the utterance.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

import riva.client
import riva.client.proto.riva_asr_pb2 as rasr

SAMPLE_RATE = 16000
CHUNK_MS = 80
SILENCE_WAIT_SECS = 1.5
FORCE_EOU_WAIT_SECS = 2.0
STOP_HISTORY_MS = 4500
ASR_FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"


def _chunk_bytes() -> int:
    return int(SAMPLE_RATE * CHUNK_MS / 1000) * 2


def _auth(*, server: str, use_ssl: bool, function_id: str, api_key: str) -> riva.client.Auth:
    metadata = []
    if function_id:
        metadata.append(["function-id", function_id])
    if api_key and api_key != "not-needed":
        metadata.append(["authorization", f"Bearer {api_key}"])
    return riva.client.Auth(None, use_ssl, server, metadata)


def _synthesize_speech(api_key: str, text: str) -> bytes:
    auth = _auth(
        server="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        function_id=TTS_FUNCTION_ID,
        api_key=api_key,
    )
    tts = riva.client.SpeechSynthesisService(auth)
    response = tts.synthesize(
        text,
        voice_name=TTS_VOICE,
        language_code="en-US",
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=SAMPLE_RATE,
    )
    audio = getattr(response, "audio", b"")
    if not audio:
        raise RuntimeError("TTS returned empty audio")
    return audio


def _iter_requests(audio_q: queue.Queue, config: rasr.StreamingRecognitionConfig):
    yield rasr.StreamingRecognizeRequest(streaming_config=config)
    while True:
        item = audio_q.get()
        if item is None:
            return
        chunk, runtime_config = item
        yield rasr.StreamingRecognizeRequest(
            audio_content=chunk,
            runtime_config=runtime_config or {},
        )


def main() -> int:
    """Run the silence-then-force_eou probe and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.getenv("ASR_SERVER", "grpc.nvcf.nvidia.com:443"))
    parser.add_argument("--function-id", default=os.getenv("ASR_FUNCTION_ID", ASR_FUNCTION_ID))
    parser.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY", ""))
    parser.add_argument("--insecure", action="store_true", help="Disable TLS (local NIM).")
    parser.add_argument("--text", default="Hello, how are you doing today?")
    args = parser.parse_args()

    if args.insecure:
        use_ssl = False
    elif "nvcf" in args.server or args.server.endswith(":443"):
        use_ssl = True
    else:
        use_ssl = False
    asr_function_id = args.function_id if use_ssl else ""

    print(f"Synthesizing probe utterance via Magpie TTS: {args.text!r}")
    speech = _synthesize_speech(args.api_key, args.text)
    print(f"Speech bytes={len(speech)} ({len(speech) / (SAMPLE_RATE * 2):.2f}s)")

    asr_auth = _auth(
        server=args.server,
        use_ssl=use_ssl,
        function_id=asr_function_id,
        api_key=args.api_key if use_ssl else "",
    )
    asr = riva.client.ASRService(asr_auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=1,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=True,
        ),
        interim_results=True,
    )
    riva.client.add_endpoint_parameters_to_config(config, -1, -1.0, STOP_HISTORY_MS, -1, -1.0, -1.0)

    audio_q: queue.Queue = queue.Queue()
    finals: list[tuple[float, str]] = []
    interims: list[tuple[float, str]] = []
    t0 = time.monotonic()

    def on_responses():
        try:
            for response in asr.stub.StreamingRecognize(
                _iter_requests(audio_q, config),
                metadata=asr.auth.get_auth_metadata(),
            ):
                now = time.monotonic() - t0
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript
                    if result.is_final:
                        finals.append((now, text))
                        print(f"  [{now:6.3f}s] FINAL  {text!r}")
                    elif text:
                        interims.append((now, text))
                        print(f"  [{now:6.3f}s] interim {text!r}")
        except Exception as exc:
            print(f"ASR stream error: {exc}", file=sys.stderr)

    reader = threading.Thread(target=on_responses, daemon=True)
    reader.start()
    time.sleep(0.2)

    chunk_size = _chunk_bytes()
    print(f"ASR server={args.server} ssl={use_ssl} stop_history={STOP_HISTORY_MS}ms")
    print("Streaming speech ...")
    for offset in range(0, len(speech), chunk_size):
        audio_q.put((speech[offset : offset + chunk_size], {}))
        time.sleep(CHUNK_MS / 1000)
    speech_end = time.monotonic() - t0
    print(f"  [{speech_end:6.3f}s] last speech byte sent")

    silence = b"\x00" * chunk_size
    silence_deadline = time.monotonic() + SILENCE_WAIT_SECS
    print(f"Streaming {SILENCE_WAIT_SECS:.1f}s silence (no force_eou) ...")
    while time.monotonic() < silence_deadline:
        audio_q.put((silence, {}))
        time.sleep(CHUNK_MS / 1000)
    silence_end = time.monotonic() - t0
    finals_during_silence = [(t, text) for t, text in finals if t > speech_end]
    print(f"  [{silence_end:6.3f}s] silence window closed; finals in window={len(finals_during_silence)}")

    if finals_during_silence:
        print(
            "FAIL: ASR finalized during 1.5s silence with stop_history=4500ms. "
            "That is short endpointing (~400ms), not force_eou."
        )
        audio_q.put(None)
        return 2

    force_at = time.monotonic() - t0
    print(f"  [{force_at:6.3f}s] sending force_eou")
    audio_q.put((silence, {"force_eou": "true"}))
    force_deadline = time.monotonic() + FORCE_EOU_WAIT_SECS
    while time.monotonic() < force_deadline:
        audio_q.put((silence, {}))
        time.sleep(CHUNK_MS / 1000)
        if any(t > force_at for t, _ in finals):
            break
    audio_q.put(None)
    reader.join(timeout=3)

    finals_after_force = [(t, text) for t, text in finals if t > force_at]
    if not finals_after_force:
        print("FAIL: no FINAL after force_eou. The flag was ignored or the stream failed.")
        return 3

    flush_ms = (finals_after_force[0][0] - force_at) * 1000
    print(
        f"PASS: no FINAL in {SILENCE_WAIT_SECS:.1f}s silence, then force_eou flushed "
        f"{finals_after_force[0][1]!r} in {flush_ms:.0f}ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
