# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo vLLM runtime with function tools, held-request prefill, reasoning
# effort, and chat completions for realtime text workers. Drop the patch
# once Dynamo ships it.
ARG DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
FROM ${DYNAMO_VLLM_IMAGE}

COPY realtime-text-tools.patch /tmp/realtime-text-tools.patch
USER root
RUN dynamo_dir="$(python3 -c 'import dynamo.vllm, os; print(os.path.dirname(os.path.dirname(dynamo.vllm.__file__)))')" \
    && grep -q '^diff --git' /tmp/realtime-text-tools.patch \
    && patch -p1 -d "${dynamo_dir}" < /tmp/realtime-text-tools.patch \
    && rm /tmp/realtime-text-tools.patch
USER 1000
