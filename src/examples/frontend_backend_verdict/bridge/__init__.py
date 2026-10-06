# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat hosting for the copied voice prototype.

Everything that decides behaviour (wire events, session, turn taking, barge-in,
frontend verdict, agent, normalization, speech adapters) is the prototype's code in
``text/`` and ``voice/``. This package only runs that session inside a Pipecat
pipeline on the socket that the Realtime gateway authenticated and routed.
"""
