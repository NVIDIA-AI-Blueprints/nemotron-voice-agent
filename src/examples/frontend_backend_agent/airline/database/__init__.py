# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Booking server — SQLite-backed airline backend behind an HTTP API.

Runs as its own process so the voice-agent talks to it via HTTP.  Same
service layer can move behind an MCP transport later with only the
framework swap; the :class:`BookingAPI` + DB schema stay identical.
"""
