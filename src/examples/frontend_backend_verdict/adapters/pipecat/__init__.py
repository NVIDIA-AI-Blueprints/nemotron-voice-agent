# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat adapter: connects core/ to the client and to shared Pipecat VAD, STT and TTS.

Ported from the Frontend Delegation Hermes example's Pipecat head
(``frontend_delegation_hermes/adapters/pipecat`` at ``a0ef50c``) and re-pointed at FBV's
``core/`` (plan ``misc/frontend-backend-verdict/shared-stack-plan.md`` §3.1). It has no
ASR, TTS or VAD implementation of its own.
"""
