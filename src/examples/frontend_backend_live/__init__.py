# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live: the frontend adapter, a Pipecat pipeline that talks to the caller and delegates the work.

The example sits behind the live gateway (``gateways.live``). In ``responses`` mode, what the voice delegates goes
through the session's server delegation (``gateways.live.server_delegation``) to the delegation core
(``delegation_core``); in client mode the session's client delegation (``gateways.live.client_delegation``) announces
each delegation and the client app does the work. ``README.md`` beside this file describes the example.

Two composition roots build each call:

* ``factory.py``: the hooks of the registry's ``live:`` block. ``create_frontend_adapter`` builds each session's parts:
  its frontend adapter, of the class ``frontend.adapter`` names, and a WebRTC session's peer connection; ``core`` gives
  the client of the process's one delegation core, built on first use from the core's settings; ``describe`` describes
  the example for ``GET /v1/live/info``.
* ``pipeline.py``: the repository UI bot, with the same frontend build, a delegation core of its own (the core's
  settings with ``ui_bot.core`` merged over them), and the task cards.

Both read the example's settings once: the ``config`` block of its ``examples_registry.yaml`` entry.

Packages, one per responsibility:

* ``adapter``: the cascade adapter (a ``gateways.live.ports.FrontendAdapter``), its class contract and table, and its
  Live integration: the pipeline host, the bridge between a live session and its pipeline, and a WebRTC session's
  peer connection.
* ``frontend.py``: the one build of the frontend's pipeline for both hosts.
* ``talker.py``, ``voice.py``, ``spoken.py``, ``handoffs.py`` and ``tool_handlers.py``: the talker as Pipecat's LLM
  service with the voice's structured decision, the voice that chooses each of its calls and acts on them, the lines it
  said, the hand-off of a delegated turn with its lead-in, and the handlers of the tool-calling talker's routing tools.
* ``delegations.py``, ``worknote.py``, ``heard_log.py`` and ``display.py``: the reader of the official events, the
  work note, the heard log and the display sink that shows nothing, which the voice uses.
* ``media``: the speech services and TTS modes, the caller's turns, the edge that reads each line's playout after the
  output, the reply pre-roll, and the UI bot's task cards.
* ``models``: the talker's decision contract, and the example's catalog and endpoint resolution.
* ``prompts.py``: the reader of the example's ``prompts.yaml``, which holds the talker prompts, the backend prompts,
  the channel context, the prompt set, the lines and the delegation core's texts.
* ``config_manager``: the validation of the example's settings.
* ``delegation_core``: the delegation core, a package of its own that only the two composition roots build; it has its
  own settings (the ``core`` block of the example's settings), tests and ``README.md``.
"""
