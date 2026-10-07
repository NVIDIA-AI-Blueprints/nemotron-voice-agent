# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional frontend prompt note that goes with the transcript hook."""

from __future__ import annotations

from dataclasses import replace

from examples.frontend_backend_verdict.text.config import Config
from examples.frontend_backend_verdict.text.prompts import load_catalog


def with_frontend_note(config: Config, note_key: str) -> Config:
    """``config`` with the catalog entry ``note_key`` appended to the frontend prompt.

    Applied to the per-session config (after the session's instructions), so a
    policy already appended to the frontend prompt stays before the note. A
    no-op without a key or without a frontend.
    """
    if not note_key or not config.frontend_enabled:
        return config
    catalog = load_catalog(config.prompts_path, config.prompts.inline)
    key = config.frontend.prompt_key
    inline = dict(config.prompts.inline)
    inline[key] = f"{catalog.get(key).rstrip()}\n\n{catalog.get(note_key).strip()}"
    return replace(config, prompts=replace(config.prompts, inline=inline))
