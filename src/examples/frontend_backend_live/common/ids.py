# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ids and size estimates, shared with the live protocol so its limits and the backend's budgets agree."""

from live.protocol import approx_tokens, uid

__all__ = ["approx_tokens", "uid"]
