# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record committed output intervals without observing packet arrival times."""

import json
import os
import uuid
import wave
from contextlib import contextmanager, suppress
from pathlib import Path

from .models import RATE, AudioSource, SessionResult


def create_private_directory(directory: Path, *, exist_ok: bool = True) -> None:
    """Create a session directory without following a final symlink or changing umask."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=exist_ok)
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


@contextmanager
def _private_output(path: Path, *, binary: bool = False):
    """Publish one complete private artifact atomically, replacing rather than following links."""
    path = Path(path)
    create_private_directory(path.parent)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = None
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"}))
        descriptor = None
        with output:
            yield output
        os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            if created:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=parent)
        finally:
            os.close(parent)


def write_private_text(path: Path, text: str) -> None:
    """Write an immutable UTF-8 artifact or atomically replace an earlier version."""
    with _private_output(path) as output:
        output.write(text)


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
    with _private_output(path, binary=True) as artifact, wave.open(artifact, "wb") as output:
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
    create_private_directory(directory)
    write_source(directory, source)
    write_wav(directory / "full_output.wav", pcm)
    with _private_output(directory / "events.jsonl") as output:
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
    write_private_text(directory / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
