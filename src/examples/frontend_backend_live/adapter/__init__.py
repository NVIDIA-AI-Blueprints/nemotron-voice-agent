# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend adapter's Pipecat modules, between a live session and its pipeline.

* ``adapter``: the cascade adapter, which serves a live session (``live.ports``) with the cascade, and its settings.
* ``contract``: what a frontend adapter class offers the composition root (types only).
* ``registry``: ``FRONTEND_ADAPTERS``, the adapters a deployment chooses by name.
* ``readiness``: whether the services an adapter class needs accept connections.
* ``bridge``: the input gate, the event observer and the WebSocket audio output.
* ``playout``: the playout tracker, which reports what each line really played.
* ``pipeline_host``: hosting the Pipecat pipeline of a live session.
* ``services``: building speech-to-text and text-to-speech, and the TTS modes.
* ``preroll``: the reply-audio pre-roll.
* ``task_cards``: ``RtviTaskCards``, where the UI bot shows the display part of each delegation.
"""
