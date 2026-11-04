# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend side of the delegation core: the backend adapters, their model providers, and the server's tools.

``port`` is the interface between the interaction manager and a backend adapter, and ``adapters`` names the adapters a
core can be built with (``backend.adapter``).
"""
