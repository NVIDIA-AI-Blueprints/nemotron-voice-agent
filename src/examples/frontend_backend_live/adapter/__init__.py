# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend adapter's Live integration, between a live session and the cascaded voice agent.

The voice agent is the frontend's Pipecat pipeline (``examples.frontend_backend_live.frontend``) and the voice that acts
on its talker's decisions (``examples.frontend_backend_live.voice``). These modules only translate between the Live
protocol and it: they host its pipeline in a live session and wire the session's delegation part.

* ``adapter``: the cascade adapter, which serves a live session (``gateways.live.ports``) with the cascade, and its
  settings.
* ``contract``: what a frontend adapter class offers the composition root (types only).
* ``registry``: ``FRONTEND_ADAPTERS``, the adapters a deployment chooses by name.
* ``readiness``: whether the services an adapter class needs accept connections.
* ``bridge``: the input gate, the event observer, the output transcript and a primary WebSocket's silence between
  lines.
* ``serializer``: a primary WebSocket's audio events and frames, on Pipecat's WebSocket transport.
* ``pipeline_host``: hosting the Pipecat pipeline of a live session.
* ``connection``: a WebRTC session's peer connection, on Pipecat's SmallWebRTC, which the gateway answers the
  client's offer with and Pipecat's transport carries the audio on.
"""
