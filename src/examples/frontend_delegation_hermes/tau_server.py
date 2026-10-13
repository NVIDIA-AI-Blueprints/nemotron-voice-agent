# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""τ³ entry point: the custom OpenAI Realtime head, used only for τ³ benchmarking.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
        --config src/examples/frontend_delegation_hermes/config/profiles/realtime_eval.yaml

``--stub-speech`` replaces Riva ASR/TTS; ``--stub-backend`` runs an in-process gateway
with a fake backend agent and the rule-based frontend (no GPU, no LLM, no Hermes).
This module never imports the Pipecat head.
"""

from examples.frontend_delegation_hermes.adapters.tau.app import main

if __name__ == "__main__":
    main()
