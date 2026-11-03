# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend adapter's Live integration, between a live session and the cascaded voice agent.

The voice agent is the cascade (``cascade/``) and its Pipecat speech pipeline (``media/``). These modules only
translate between the Live protocol and it: they host its pipeline in a live session and wire the session's delegation
part.

* ``adapter``: the cascade adapter, which serves a live session (``live.ports``) with the cascade, and its settings.
* ``contract``: what a frontend adapter class offers the composition root (types only).
* ``registry``: ``FRONTEND_ADAPTERS``, the adapters a deployment chooses by name.
* ``readiness``: whether the services an adapter class needs accept connections.
* ``bridge``: the input gate, the event observer and the WebSocket audio output.
* ``pipeline_host``: hosting the Pipecat pipeline of a live session.
"""
