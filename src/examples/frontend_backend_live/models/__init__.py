# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker's decision contract and the example's catalog.

* ``decision``: the ``{action, speech}`` decision, its parser, and the spec of the call that asks for it.
* ``endpoints``: the example's catalog and each model's endpoint, resolved from ``services.yaml`` and the session body.

The talker itself is Pipecat's LLM service (``examples.frontend_backend_live.talker``). The backend role (``Backend``,
``BACKENDS``) is in the delegation core (``examples.frontend_backend_live.delegation_core.backend.providers``).
"""
