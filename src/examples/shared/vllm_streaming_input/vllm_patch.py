# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make vLLM honour each StreamingInput chunk's ``max_tokens``.

vLLM's scheduler updates a resumed session's sampling params but keeps the
first chunk's ``max_tokens``, so an answer after a one-token prefill chunk is
cut to one token. The fix is one line in the installed scheduler, written
before the engine process imports it.
TODO: Remove once vLLM copies ``update.max_tokens`` in ``_update_request_as_session``.
"""

from importlib.metadata import distribution
from pathlib import Path

SUPPORTED_VLLM_VERSIONS = {"0.29.0"}

_UPDATE_ANCHOR = (
    "        session.sampling_params = update.sampling_params\n"
    "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
)
_PATCH_LINE = "        session.max_tokens = update.max_tokens\n"


def patch_scheduler_source(source: str) -> tuple[str, bool]:
    """Return the scheduler source with the fix, and whether it changed."""
    if _PATCH_LINE in source:
        return source, False
    if source.count(_UPDATE_ANCHOR) != 1:
        raise RuntimeError("Unsupported vLLM scheduler source; the max_tokens patch anchor changed")
    head, tail = _UPDATE_ANCHOR.split("\n", 1)
    return source.replace(_UPDATE_ANCHOR, f"{head}\n{_PATCH_LINE}{tail}"), True


def apply_installed_patch() -> str:
    """Patch the installed vLLM scheduler in place."""
    package = distribution("vllm")
    scheduler_path = Path(package.locate_file("vllm/v1/core/sched/scheduler.py"))
    source, changed = patch_scheduler_source(scheduler_path.read_text(encoding="utf-8"))
    if not changed:
        return f"vLLM {package.version}: StreamingInput max_tokens fix already present"
    if package.version not in SUPPORTED_VLLM_VERSIONS:
        raise RuntimeError(f"Refusing to patch vLLM {package.version}; supported: {sorted(SUPPORTED_VLLM_VERSIONS)}")
    scheduler_path.write_text(source, encoding="utf-8")
    return f"vLLM {package.version}: applied the StreamingInput max_tokens fix"
