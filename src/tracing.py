# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry tracing setup for Nemotron Voice Agent.

Reads ENABLE_TRACING, OTEL_EXPORTER_OTLP_ENDPOINT, and OTEL_CONSOLE_EXPORT
from the environment and initialises the Pipecat tracing SDK once at import
time.  The cascaded pipelines import ``IS_TRACING_ENABLED`` from
here and pass it to their ``PipelineTask``.
"""

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

IS_TRACING_ENABLED: bool = os.getenv("ENABLE_TRACING", "").lower() == "true"


def _init_tracing() -> None:
    """Bootstrap OpenTelemetry with the configured OTLP exporter."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from pipecat.utils.tracing.setup import setup_tracing

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            exporter = OTLPSpanExporterHTTP(endpoint=endpoint)
        else:
            exporter = OTLPSpanExporterGRPC(endpoint=endpoint, insecure=True)

        setup_tracing(
            service_name="nemotron-voice-agent",
            exporter=exporter,
            console_export=os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true",
        )
        logger.info(f"OpenTelemetry tracing initialized (endpoint={endpoint})")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry tracing: {e} — continuing without tracing")


if IS_TRACING_ENABLED:
    _init_tracing()
