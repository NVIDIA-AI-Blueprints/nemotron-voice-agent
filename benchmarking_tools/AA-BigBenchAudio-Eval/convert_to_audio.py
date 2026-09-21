# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert text files to WAV using Riva TTS. See README for RIVA_TTS_* env vars."""

import os
import sys
import threading
import wave
from pathlib import Path

import riva.client

# Set via environment (see README). Example: RIVA_TTS_SERVER=localhost:50153
RIVA_SERVER = os.environ.get("RIVA_TTS_SERVER", "localhost:50153")
USE_SSL = os.environ.get("RIVA_TTS_USE_SSL", "false").lower() == "true"
SSL_CERT_PATH = os.environ.get("RIVA_TTS_SSL_CERT") or None
VOICE_NAME = os.environ.get("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Sofia")
LANGUAGE_CODE = os.environ.get("RIVA_TTS_LANGUAGE_CODE", "en-US")
SAMPLE_RATE_HZ = 16000
TTS_TIMEOUT_SECONDS = 10.0

INPUT_DIR = Path(os.environ.get("CONVERT_TO_AUDIO_INPUT_DIR", "."))
OUTPUT_DIR = Path(os.environ.get("CONVERT_TO_AUDIO_OUTPUT_DIR", "./audio"))

AUTH = riva.client.Auth(ssl_cert=SSL_CERT_PATH, use_ssl=USE_SSL, server=RIVA_SERVER)
TTS = riva.client.SpeechSynthesisService(AUTH)


def synthesize_text_to_wav(text: str, output_wav: Path) -> None:
    """Synthesize given text to a 16 kHz linear PCM WAV file using Riva TTS."""
    nchannels = 1
    sampwidth_bytes = 2  # 16-bit linear PCM

    result: list[object | None] = [None]
    error_msg: list[BaseException | None] = [None]

    def worker() -> None:
        try:
            result[0] = TTS.synthesize(
                text=text,
                voice=VOICE_NAME,
                language_code=LANGUAGE_CODE,
                sample_rate_hz=SAMPLE_RATE_HZ,
                audio_prompt_file=None,
                quality=20,
                custom_dictionary={},
            )
        except BaseException as exc:
            error_msg[0] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=TTS_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise TimeoutError(f"TTS synthesis timed out after {TTS_TIMEOUT_SECONDS:.0f}s")

    if error_msg[0] is not None:
        raise error_msg[0]

    resp = result[0]
    if resp is None:
        raise RuntimeError("TTS synthesis returned no response")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth_bytes)
        wf.setframerate(SAMPLE_RATE_HZ)
        wf.writeframes(resp.audio)


def main() -> None:
    """Convert all .txt under INPUT_DIR to WAV in OUTPUT_DIR."""
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    txt_files = sorted([p for p in INPUT_DIR.glob("**/*.txt") if p.is_file()])
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    print(f"Found {len(txt_files)} text files. Generating audio to {OUTPUT_DIR}...")
    for txt_path in txt_files:
        rel = txt_path.relative_to(INPUT_DIR)
        out_wav = OUTPUT_DIR.joinpath(rel).with_suffix(".wav")
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"Skipping empty file: {txt_path}")
                continue
            synthesize_text_to_wav(text=text, output_wav=out_wav)
            print(f"OK: {txt_path} -> {out_wav}")
        except Exception as exc:
            print(f"FAILED: {txt_path}: {exc}")


if __name__ == "__main__":
    main()
