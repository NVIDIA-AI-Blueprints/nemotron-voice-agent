# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A test-only Realtime head for the FDA core tests (no Pipecat, no wall clock).

These modules were FDA's τ³ adapter (``adapters/tau``, removed from the example in
``shared-stack-plan.md`` step 1). The core tests drive the delegation turn manager through it,
because it ends turns on the input-audio clock with an energy VAD and stub speech, so a test
gives the same result on every run. It is not shipped and is not an FDA runtime.
"""
