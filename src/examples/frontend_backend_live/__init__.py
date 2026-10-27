# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live: the frontend adapter hosting the cascade, which talks to the caller and delegates the work.

The example sits behind the live gateway (``live``). In ``responses`` mode, what the cascade delegates goes through the
session's server delegation (``live.server_delegation``) to the delegation core (``delegation_core``); in client mode
the session's client delegation (``live.client_delegation``) announces each delegation and the client app does the
work. ``docs/design/architecture.md`` describes the design.

Two composition roots build each call:

* ``factory.py``: the hooks of the registry's ``live:`` block. ``create_frontend_adapter`` builds each session's
  frontend adapter, of the class ``frontend.adapter`` names; ``core`` gives the client of the process's one delegation
  core, built on first use from ``core.yaml``; ``describe`` describes the example for ``GET /v1/live/info``.
* ``pipeline.py``: the repository UI bot, with the same cascade parts, a delegation core of its own from
  ``core.ui-bot.yaml``, and the task cards.

Packages, one per responsibility:

* ``adapter``: the cascade adapter (a ``live.ports.FrontendAdapter``), its class contract and table, and its Pipecat
  modules: the pipeline host, the bridge between a live session and its pipeline, the playout tracker, the speech
  services, the reply pre-roll, and the UI bot's task cards.
* ``cascade``: the cascade's parts for one call: caller turns, the routing talker, the writer, the floor, the work note,
  the heard log, the talker's memory, the inbox of results, the reader of the official events, and the Pipecat
  processor that runs them.
* ``models``: the ``Frontend`` base class and its providers, the decision parser, the example's catalog and endpoint
  resolution, and retries with backoff.
* ``prompts``: the prompt sets and the reader of the example's ``prompts.yaml``.
* ``config_manager``: ``config.yaml`` loading and validation.
* ``parked``: code that nothing imports, runs or tests.
"""
