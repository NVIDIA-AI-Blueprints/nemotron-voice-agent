# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core: the work a voice frontend hands off, placed by a conductor and run on a backend adapter.

It imports nothing from the live gateway, the example frontends or Pipecat; ``docs/design/architecture.md`` describes
the design and ``README.md`` beside this file is the package reference.
"""
