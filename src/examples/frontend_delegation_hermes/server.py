# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Default entry point: the Pipecat adapter (WebSocket + OpenAI Realtime, no WebRTC or RTVI).

Demo (browser page at ``/``, demo tools in the server, greeting)::

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.server \
        --config src/examples/frontend_delegation_hermes/config/profiles/pipecat_demo.yaml --tls

Evaluation by a τ² client (client-executed tools, external policy, 0.8 s VAD silence in audio time)::

    FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml PYTHONPATH=src uv run python -m \
        examples.frontend_delegation_hermes.server \
        --config src/examples/frontend_delegation_hermes/config/profiles/pipecat_realtime_eval.yaml

The τ³ benchmark adapter is ``tau_server.py``.
"""

from examples.frontend_delegation_hermes.adapters.pipecat.app import main

if __name__ == "__main__":
    main()
