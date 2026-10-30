# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascaded voice agent's Pipecat speech pipeline, which the hosts of the cascade build from these modules.

The voice agent is the cascade (``cascade/``, Pipecat-free) and this pipeline, which runs the cascade in Pipecat and
keeps its records to what the caller heard. Two hosts build it: the cascade adapter (``adapter/``), which translates
between the Live protocol and the voice agent, and the repository UI bot (``pipeline.py``), in its own RTVI pipeline.

* ``processor``: ``LiveTalkerProcessor``, which takes the LLM service's place and binds the cascade to the pipeline.
* ``services``: building speech-to-text and text-to-speech, and the TTS modes, which keep each line's text behind its
  audio.
* ``turns``: the turn ends, which say how the caller's turn ends and when the speech-to-text finalizes what it heard.
* ``playout``: the playout tracker, which reports what each line really played, and ``SpeechPlayout``, which wires it
  to the cascade of each host.
* ``preroll``: the reply-audio pre-roll, which the cascade adapter places before a live session's audio output.
* ``task_cards``: ``RtviTaskCards``, where the UI bot shows the display part of each delegation.
"""
