# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow compatibility fix for vLLM resumable-session token budgets."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution
from pathlib import Path

SUPPORTED_VLLM_VERSIONS = {"0.27.1", "0.28.0"}

_UPDATE_ANCHOR = (
    "        session.sampling_params = update.sampling_params\n"
    "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
)
_PATCHED_UPDATE = (
    "        session.sampling_params = update.sampling_params\n"
    "        # Compatibility for vLLM streaming-input issue #4385.\n"
    "        session.max_tokens = update.max_tokens\n"
    "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
)
_PATCH_MARKER = "session.max_tokens = update.max_tokens"


def patch_scheduler_source(source: str) -> tuple[str, bool]:
    """Return scheduler source with the exact guarded compatibility fix."""
    if _PATCH_MARKER in source:
        return source, False
    if source.count(_UPDATE_ANCHOR) != 1:
        raise RuntimeError("Unsupported vLLM scheduler source; max_tokens patch anchor changed")
    return source.replace(_UPDATE_ANCHOR, _PATCHED_UPDATE), True


def _installed_paths() -> tuple[str, Path, Path]:
    package = distribution("vllm")
    root = Path(package.locate_file(""))
    return (
        package.version,
        root / "vllm/v1/core/sched/scheduler.py",
        root / "vllm/v1/request.py",
    )


def apply_installed_patch() -> str:
    """Patch only the pinned, structurally verified vLLM installation."""
    version, scheduler_path, request_path = _installed_paths()
    scheduler_source = scheduler_path.read_text(encoding="utf-8")
    patched_source, changed = patch_scheduler_source(scheduler_source)
    if not changed:
        return f"vLLM {version}: per-update max_tokens support already present"
    if version not in SUPPORTED_VLLM_VERSIONS:
        raise RuntimeError(
            f"Refusing to patch unsupported vLLM {version}; expected one of {sorted(SUPPORTED_VLLM_VERSIONS)}"
        )
    request_source = request_path.read_text(encoding="utf-8")
    if "class StreamingUpdate:" not in request_source or "max_tokens: int" not in request_source:
        raise RuntimeError("Unsupported vLLM StreamingUpdate structure; max_tokens field missing")
    scheduler_path.write_text(patched_source, encoding="utf-8")
    return f"vLLM {version}: installed per-update max_tokens compatibility patch"


def verify_installed_patch() -> str:
    """Fail fast if the spawned EngineCore would miss the compatibility fix."""
    version, scheduler_path, _ = _installed_paths()
    source = scheduler_path.read_text(encoding="utf-8")
    if _PATCH_MARKER not in source:
        raise RuntimeError("vLLM StreamingInput per-update max_tokens support is missing from the runtime image")
    return f"vLLM {version}: per-update max_tokens verified"


def main() -> None:
    """Apply or verify the installed vLLM compatibility patch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(apply_installed_patch() if args.apply else verify_installed_patch())


if __name__ == "__main__":
    main()
