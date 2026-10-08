# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live: a frontend that talks to the caller and a backend that does the work.

Packages, one per responsibility:

* ``engine``: what every live engine shares (session settings, delegation, the pipeline host) and the factory
  the live server calls.
* ``cascade``: the core cascaded pipeline: speech in, the routing talker, commentary, and speech out.
* ``realtime``: a remote realtime model as the frontend, in place of the cascade.
* ``delegation``: handing work to the backend one task at a time, and relaying what happens to the client.
* ``tool_calling``: running the backend's function calls: by the client, or by the server (the cafe tools).
* ``models``: the ``Frontend`` and ``Backend`` base classes and their registered implementations.
* ``prompts``: prompt sets and the static prompts (plain text only).
* ``config_manager``: ``config.yaml`` loading and validation.
* ``common``: ids, history conventions, word helpers, and retries with backoff.
"""
