# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Demo Air demo desk: tool schemas and an in-memory tool source (``backend.tools: {airline: {}}``).

Demo Air is a made-up airline that flies a nonstop route between every pair of twelve U.S. cities (``airports.py``,
``schedule.py``). Everything is sample data: flights come from a fixed timetable, fares are demo fares, and bookings
live in the session's memory only. No network is used and nothing is charged. Any other place is answered with the
closest city that the airline serves.
"""
