# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build from the repository root:
#   docker build -f src/examples/generic_assistant_livekit/Dockerfile -t generic-assistant-livekit .
# One image runs both the agent worker and the web UI server; Compose picks the command.

# Base image - OSRB approved Ubuntu base (same as docker/Dockerfile)
FROM nvcr.io/nvidia/base/ubuntu:noble-20260610

LABEL maintainer="NVIDIA"
LABEL description="Nemotron Voice Agent - LiveKit generic assistant"

ENV PYTHONUNBUFFERED=1

# git is required for the temporary livekit-plugins-nvidia git dependency (see pyproject.toml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3.12 \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

RUN mkdir -p /app/legal
COPY LICENSE /app/legal/LICENSE
COPY THIRD_PARTY_NOTICES.txt /app/legal/THIRD_PARTY_NOTICES.txt

# Dependencies come from the example's own pyproject.toml and uv.lock, not the root ones.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY src/examples/generic_assistant_livekit/pyproject.toml src/examples/generic_assistant_livekit/uv.lock /opt/livekit-project/
RUN uv sync --project /opt/livekit-project --frozen --no-install-project --python python3.12

COPY src/examples/__init__.py src/examples/__init__.py
COPY src/examples/generic_assistant_livekit/ src/examples/generic_assistant_livekit/
COPY services.yaml ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

CMD ["python", "-m", "examples.generic_assistant_livekit.agent", "start"]
