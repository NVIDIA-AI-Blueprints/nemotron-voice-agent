# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The voice names a client may send; the frontend adapter decides which voice it speaks with.

``LIVE_VOICES`` are the Live API's built-in voices, and ``TTS_VOICES`` OpenAI's text-to-speech voices.
"""

LIVE_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "beacon",
        "bossa",
        "brise",
        "cedar",
        "cinder",
        "coral",
        "delta",
        "echo",
        "flitz",
        "gleam",
        "harema",
        "juni",
        "marin",
        "meridian",
        "nira",
        "noeul",
        "nuri",
        "quartz",
        "ripple",
        "sage",
        "shida",
        "shimmer",
        "sillage",
        "stone",
        "tempo",
        "verse",
        "vesper",
        "willow",
    }
)
TTS_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)
