# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build from the repository root:
#   docker build -f docker/livekit/Dockerfile -t generic-assistant-livekit .
# One image runs both the agent worker and the web UI server; Compose picks the command.

# Base image - OSRB approved Ubuntu base (same as docker/pipecat/Dockerfile)
FROM nvcr.io/nvidia/base/ubuntu:noble-20260610

LABEL maintainer="NVIDIA"
LABEL description="Nemotron Voice Agent - LiveKit generic assistant"

ENV PYTHONUNBUFFERED=1

# git is required for the temporary livekit-plugins-nvidia git dependency (see the root pyproject.toml)
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

# The image installs only the LiveKit group of the root project.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --python python3.12 --no-default-groups --group livekit --no-install-project

COPY src/ src/
RUN uv sync --frozen --python python3.12 --no-default-groups --group livekit
COPY clients/livekit/ clients/livekit/
COPY services.yaml examples_registry.yaml ./

CMD ["nva-livekit", "start"]
