# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Build stage - OSRB approved NVIDIA Ubuntu base (Bug 4960044)
FROM nvcr.io/nvidia/base/ubuntu:noble-20251013 AS builder

# Image metadata
LABEL stage="builder"
LABEL description="UI build environment"

# Install Node.js 24.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy webrtc_ui package files and install dependencies
COPY frontend/webrtc_ui/package*.json ./
RUN npm ci --frozen-lockfile && \
    npm cache clean --force

# Copy webrtc_ui source code and build
COPY frontend/webrtc_ui/ .
RUN npm run build

# Production stage - NVIDIA distroless Python
FROM nvcr.io/nvidia/distroless/python:3.13-v3.1.3 AS production

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Voice Agent UI static server"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Copy both UI static files
COPY --from=builder /app/dist ./webrtc_static
COPY frontend/websocket_ui/static ./websocket_static

# Copy node_modules and package files to legal directory for security scanning and OSS compliance
COPY --from=builder /app/node_modules ./legal/node_modules
COPY --from=builder /app/package*.json ./legal/
COPY LICENSE ./legal/
COPY third_party_oss_license.txt ./legal/

# Copy Python start script for handling TRANSPORT env variable
COPY frontend/start-server.py ./

# Port configuration
EXPOSE 8000

# Start command using Python's built-in http.server
CMD ["python", "start-server.py"]
