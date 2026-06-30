# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Base image - OSRB approved Ubuntu base (Bug 4960044)
FROM nvcr.io/nvidia/base/ubuntu:noble-20251013 AS client-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /client

COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM nvcr.io/nvidia/base/ubuntu:noble-20251013

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Nemotron Voice Agent"
LABEL version="2.0"

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    python3.12 \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

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

COPY pyproject.toml uv.lock ./

# Dependencies
# Note: Python package source code is available via uv's cache and can be
# extracted using: uv pip show <package> --files
# For full source compliance, see /app/legal/third_party_oss_license.txt
RUN uv venv --python python3.12 && . .venv/bin/activate && uv sync --frozen

COPY src/ src/
COPY examples_registry.yaml examples_registry.yaml
COPY benchmarking_tools/scaling-perf/ benchmarking_tools/scaling-perf/
COPY --from=client-builder /client/dist/ client/dist/

EXPOSE 7860

CMD ["uv", "run", "python", "src/server.py", "--host", "0.0.0.0"]
