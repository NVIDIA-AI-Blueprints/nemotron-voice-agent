# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Base image - OSRB approved Ubuntu base (Bug 4960044)
FROM nvcr.io/nvidia/base/ubuntu:noble-20251013

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Nemotron Voice Agent"
LABEL version="1.0"

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# System dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    curl \
    libglib2.0-0 \
    python3.12 \
    python3.12-venv \
    gpgv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# App directory setup
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY config ./config
COPY nvidia-pipecat ./nvidia-pipecat

# Create legal directory and copy license files
RUN mkdir -p /app/legal
COPY LICENSE /app/legal/LICENSE
COPY third_party_oss_license.txt /app/legal/third_party_oss_license.txt

# Download the sources of all Ubuntu packages within the container for legal compliance
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils dpkg-dev && \
    mkdir -p /app/legal/source && \
    # Enable deb-src repositories for source package downloads
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i -E 's/^(Types:.*deb)(\s*)$/\1 deb-src\2/; s/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources; \
    fi && \
    apt-get update && \
    cd /app/legal/source && \
    # Download sources for all installed packages
    dpkg -l | grep '^ii' | awk '{print $2}' | cut -d: -f1 | xargs -r apt-get source --download-only 2>/dev/null || true && \
    # Remove temporary packages
    apt-get remove -y xz-utils dpkg-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Dependencies
# Note: Python package source code is available via uv's cache and can be
# extracted using: uv pip show <package> --files
# For full source compliance, see /app/legal/third_party_oss_license.txt
RUN uv venv --python python3.12 && . .venv/bin/activate && uv sync --frozen

# Port configuration
EXPOSE 7860

# Start command
CMD ["uv", "run", "src/pipeline.py"]
