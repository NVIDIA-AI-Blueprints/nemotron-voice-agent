# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example's ``prompts.yaml``: the static prompts a role starts from, and the lines said word for word."""

from __future__ import annotations

from pathlib import Path

from utils import load_yaml_file

CHANNEL_PROMPT_KEY = "channel"
# The catalog entry that holds the lines the voice says word for word, by name. It has no content, so it is no prompt.
LINES_KEY = "lines"
# This example's own prompts, never a ``--prompt-file`` the server was started with: the live engines need the
# ``talker``, ``thinker`` and ``channel`` entries that only this catalog has.
CATALOG_PATH = Path(__file__).resolve().parents[1] / "prompts.yaml"


def catalog_prompt(key: str) -> str:
    """Return the text of catalog entry ``key``; raise ``KeyError`` when it is missing or empty."""
    entry = load_yaml_file(CATALOG_PATH).get(key)
    content = str(entry.get("content") or "").strip() if isinstance(entry, dict) else ""
    if not content:
        raise KeyError(f"Prompt {key!r} was not found in the Frontend/Backend Live prompts.yaml")
    return content


def role_instructions(key: str) -> str:
    """Return a role's static instructions: its catalog prompt followed by the shared channel context."""
    return f"{catalog_prompt(key)}\n\n{catalog_prompt(CHANNEL_PROMPT_KEY)}"


def catalog_line(name: str) -> str:
    """Return the line ``name`` that the voice says word for word; an empty line means the voice says none.

    Raises:
        KeyError: The catalog's ``lines`` has no line of that name.
    """
    lines = load_yaml_file(CATALOG_PATH).get(LINES_KEY)
    if not isinstance(lines, dict) or name not in lines:
        raise KeyError(f"Line {name!r} was not found in the Frontend/Backend Live prompts.yaml")
    return str(lines[name] or "")
