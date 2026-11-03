# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gateway's server delegation: a thin pass-through between a live session and the delegation core.

* ``passthrough``: :class:`ServerDelegation`, one per ``responses`` session (a client-mode session gets
  ``live.client_delegation`` instead). It keeps the session's conversation, the open streams, the call routes and the
  client's pending input; the core judges every result it passes on.
* ``sender``: its ordered sender, which sends each request in the order it was made, off the command lock.
* ``responses_client``: the port it reaches the delegation core through, the official Responses API (types only).
"""

from live.server_delegation.passthrough import ServerDelegation
from live.server_delegation.responses_client import ResponsesClient, ResponseStream

__all__ = ["ResponseStream", "ResponsesClient", "ServerDelegation"]
