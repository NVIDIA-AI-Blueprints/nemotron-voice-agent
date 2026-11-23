# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vLLM with the NVIDIA Dynamo nightly whose vLLM workers serve realtime text
# sessions with function tools, held-request prefill, and chat completions on
# one engine. Move to the Dynamo release image once a release includes them.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}

ARG DYNAMO_VERSION=1.7.0.dev20261110
RUN pip install --no-cache-dir "ai-dynamo==${DYNAMO_VERSION}"
