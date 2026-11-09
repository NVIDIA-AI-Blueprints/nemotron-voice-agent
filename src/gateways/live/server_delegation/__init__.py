# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation: a pass-through between a live session and the delegation core."""

from gateways.live.server_delegation.passthrough import ServerDelegation
from gateways.live.server_delegation.responses_client import ResponsesClient, ResponseStream

__all__ = ["ResponseStream", "ResponsesClient", "ServerDelegation"]
