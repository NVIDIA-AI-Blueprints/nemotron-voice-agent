# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Record committed output intervals without observing packet arrival times."""

import json
import wave
from pathlib import Path

from .models import RATE, AudioSource, SessionResult


def render_pcm(result: SessionResult) -> bytes:
    """Render silence and committed PCM spans, rejecting invalid overlaps."""
    if result.rendered_samples < 0:
        raise ValueError("Rendered duration cannot be negative")
    output = bytearray(result.rendered_samples * 2)
    previous_end = 0
    for span in sorted(result.spans, key=lambda item: item.start_sample):
        if span.start_sample < previous_end or len(span.pcm) % 2 or span.end_sample > result.rendered_samples:
            raise ValueError("Committed audio spans must be nonoverlapping PCM16 inside the output extent")
        output[span.start_sample * 2 : span.end_sample * 2] = span.pcm
        previous_end = span.end_sample
    return bytes(output)


def write_wav(path: Path, pcm: bytes, sample_rate: int = RATE) -> None:
    """Write canonical mono PCM16 and preserve every final partial chunk."""
    if len(pcm) % 2 or sample_rate <= 0:
        raise ValueError("WAV data must be PCM16 with a valid sample rate")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def write_source(directory: Path, source: AudioSource) -> None:
    """Persist sealed ingress before provider execution, including silence."""
    write_wav(Path(directory) / "source.wav", source.pcm, source.sample_rate)


def write_session(directory: Path, source: AudioSource, result: SessionResult) -> dict:
    """Save source, full aligned output, event stream, and portable manifest."""
    if result.rendered_samples < source.sample_count:
        raise ValueError("Session output must cover the complete input horizon")
    pcm = render_pcm(result)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_source(directory, source)
    write_wav(directory / "full_output.wav", pcm)
    with (directory / "events.jsonl").open("w", encoding="utf-8") as output:
        for event in result.events:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "sample_rate": RATE,
        "channels": 1,
        "sample_width": 2,
        "input_samples": source.sample_count,
        "rendered_samples": result.rendered_samples,
        "source": "source.wav",
        "output": "full_output.wav",
        "events": "events.jsonl",
        "messages": list(result.messages),
        "metrics": result.metrics,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
