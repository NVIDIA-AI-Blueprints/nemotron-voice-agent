# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live session protocol (``/v1/live/sessions``), with one package per behavior.

The packages:

* ``protocol``: the strict startup schema, the voices, and the rejection error.
* ``events``: building events, who may see them, delivering them to each connection, and the media timeline.
* ``session``: one session's state and life cycle, and a module for each kind of client command
  (``audio_input``, ``updates``, ``appends``, ``tool_calling``).
* ``gateway``: settings, security, the session registry, and the WebRTC and WebSocket routes.
* ``media``: wire audio codecs, the WebRTC connection, and the Pipecat processors.
* ``engine_contract``: what the protocol needs from the voice pipeline behind a session.
* ``mount``: adds the routes (and ``/v1/live/info``) to the server's FastAPI app.
"""
