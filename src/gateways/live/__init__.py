# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live session protocol (``/v1/live/sessions``), with one package per behavior.

The packages and modules:

* ``protocol``: the strict startup schema, the voices, and the rejection error.
* ``events``: building events, who may see them, delivering them to each connection, and the media timeline.
* ``ports``: the types between a session and the parts behind it: the frontend adapter, the delegation part and the
  session views they get.
* ``session``: one session's state and life cycle, and a module for each kind of client command
  (``audio_input``, ``updates``, ``appends``, ``tool_calling``).
* ``server_delegation``: a ``responses`` session's pass-through to the delegation core over the Responses API.
* ``client_delegation``: a client-mode session's delegation, which announces each delegation to the client app.
* ``gateway``: settings, security, the session registry, and the WebRTC and WebSocket routes.
* ``media``: wire audio codecs and the WebRTC connection.
* ``extras``: ``GET /v1/live/info`` and ``POST /v1/live/delegate``, beside the protocol.
* ``mount``: adds the routes and the extras to the server's FastAPI app.
"""
