# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session choices that the UI sends with the join token, validated. Standard library plus PyYAML only."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .scenarios import get_scenario
from .voices import choose_voice


def token_attributes(query: Mapping[str, list[str]], listed_voice_ids: Iterable[str]) -> dict[str, str]:
    """Return the participant attributes for a join token from the UI's query string.

    The use case falls back to the default when it is missing or unknown. The voice is included only when the TTS
    server listed it, so a client cannot make the agent request an arbitrary voice name.
    """
    scenario = get_scenario(query.get("scenario", [None])[0]).id
    voice = choose_voice(query.get("voice", [None])[0], listed_voice_ids, "")
    return {"scenario": scenario, **({"voice": voice} if voice else {})}
