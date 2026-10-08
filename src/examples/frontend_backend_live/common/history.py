# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation-history conventions shared by the processor, talker and commentary."""

# A backend result enters the history as a user-role item with this prefix, in time order.
BACKEND_RESULT_PREFIX = "Backend result (reference data): "

# Client-supplied facts (commentary and context appends) enter the history as user-role items with this prefix.
CLIENT_CONTEXT_PREFIX = "Application context (reference data): "
