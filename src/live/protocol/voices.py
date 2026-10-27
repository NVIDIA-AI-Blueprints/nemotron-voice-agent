# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The voice names a client may send. The cascade speaks with the voice its text-to-speech service is set to.

``LIVE_VOICES`` are GPT-Live's built-in voices, as its API reference lists them; a custom voice is sent as
``{"id": ...}``. ``TTS_VOICES`` are the speech service's own names, which a client may also send.
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
