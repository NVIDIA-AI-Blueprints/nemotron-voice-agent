# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ASR/TTS endpoints from an example's service catalog or inline YAML.

The catalogs are the same files the official voice example reads
(``src/examples/frontend_backend_agent/services.{local,cloud}.yaml``), so one
entry behaves identically in both stacks. Authentication mirrors Pipecat 1.7's
``NvidiaSTTService`` / ``NvidiaTTSService``: SSL exactly when the server is an
NVCF host, ``function-id`` plus ``Bearer`` metadata for NVCF, nothing required
for a local nemo-speech server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from utils import is_nvcf

API_KEY_ENV = "NVIDIA_API_KEY"


@dataclass(frozen=True, slots=True)
class SpeechEndpoint:
    """One resolved ASR or TTS endpoint."""

    kind: str
    server: str
    model: str = ""
    function_id: str = ""
    language_code: str = "en-US"
    voice_id: str = ""
    use_ssl: bool = False
    source: str = ""
    api_key: str = field(default="", repr=False)

    @property
    def is_nvcf(self) -> bool:
        """Whether this endpoint is an NVIDIA Cloud Functions host."""
        return is_nvcf(self.server)

    def metadata(self) -> list[list[str]]:
        """The gRPC metadata pairs, exactly as Pipecat builds them."""
        pairs: list[list[str]] = []
        if self.function_id:
            pairs.append(["function-id", self.function_id])
        if self.api_key:
            pairs.append(["authorization", f"Bearer {self.api_key}"])
        return pairs

    def describe(self) -> str:
        """Log-safe one-line description (the API key is never included)."""
        key_state = "set" if self.api_key else "unset"
        return (
            f"{self.kind}: server={self.server} ssl={self.use_ssl} model={self.model or '-'} "
            f"voice={self.voice_id or '-'} function_id={self.function_id or '-'} api_key={key_state} "
            f"source={self.source}"
        )


def _catalog_file(example_dir: Path, services: str) -> Path:
    name = "services.local.yaml" if services == "local" else "services.cloud.yaml"
    return example_dir / name


def load_catalog_entry(kind: str, *, example_dir: Path, services: str, platform: str, key: str) -> dict[str, Any]:
    """Return the raw catalog entry for ``kind``/``key``, raising a config error when absent."""
    path = _catalog_file(example_dir, services)
    if not path.is_file():
        raise VoiceConfigError(f"{kind}.catalog: service catalog not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if services == "local":
        if not platform:
            raise VoiceConfigError(f"{kind}.catalog.platform is required for services: local ({path})")
        section = data.get(platform)
        if not isinstance(section, dict):
            raise VoiceConfigError(f"{kind}.catalog.platform {platform!r} is not a section of {path}")
    else:
        if platform:
            raise VoiceConfigError(f"{kind}.catalog.platform must be empty for services: cloud (got {platform!r})")
        section = data
    entries = section.get(kind)
    if not isinstance(entries, dict) or key not in entries:
        available = sorted(entries) if isinstance(entries, dict) else []
        raise VoiceConfigError(f"{kind}.catalog.key {key!r} not found in {path}; available: {available}")
    entry = entries[key]
    if not isinstance(entry, dict):
        raise VoiceConfigError(f"{kind}.catalog entry {key!r} in {path} is not a mapping")
    return entry


def build_endpoint(
    kind: str,
    entry: dict[str, Any],
    *,
    source: str,
    server_override: str = "",
    use_ssl: bool | None = None,
    environ: dict[str, str] | None = None,
) -> SpeechEndpoint:
    """Build and validate one endpoint from a catalog or inline entry."""
    env = os.environ if environ is None else environ
    server = (server_override or str(entry.get("server") or "")).strip()
    if not server:
        raise VoiceConfigError(f"{kind}: no server configured ({source})")
    nvcf = is_nvcf(server)
    function_id = str(entry.get("function_id") or "").strip()
    api_key = str(env.get(API_KEY_ENV, "") or "").strip()
    if nvcf:
        if not function_id:
            raise VoiceConfigError(f"{kind}: NVCF server {server} needs a non-empty function_id ({source})")
        if not api_key:
            raise VoiceConfigError(f"{kind}: NVCF server {server} needs {API_KEY_ENV} in the environment ({source})")
    return SpeechEndpoint(
        kind=kind,
        server=server,
        model=str(entry.get("model") or ""),
        function_id=function_id,
        language_code=str(entry.get("language_code") or "en-US"),
        voice_id=str(entry.get("voice_id") or ""),
        use_ssl=bool(use_ssl) or nvcf,
        source=source,
        api_key=api_key,
    )
