# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascaded voice agent's Pipecat speech pipeline, which both hosts build around the talker.

The voice agent is generic's chain with the talker as Pipecat's LLM service (``examples.frontend_backend_live.talker``
and ``examples.frontend_backend_live.voice``), and these modules around it. Two hosts build it
(``examples.frontend_backend_live.frontend``): the cascade adapter (``adapter/``), which translates
between the Live protocol and the voice agent, and the repository UI bot (``pipeline.py``), in its own RTVI pipeline.

* ``services``: building speech-to-text and text-to-speech, and the TTS modes, which keep each line's text behind its
  audio.
* ``turns``: how the caller's turn starts and ends.
* ``preroll``: the reply-audio pre-roll, which both hosts place before their audio output.
* ``silent_lines``: ``SilentLines``, which reads after the output what the assistant aggregator cannot tell of each of
  the talker's lines, such as whether any of its audio played, and keeps the text of a sentence nobody heard out of the
  talker's context.
* ``task_cards``: ``RtviTaskCards``, where the UI bot shows the display part of each delegation.
"""
