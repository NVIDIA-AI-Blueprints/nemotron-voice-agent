# Enable OpenTelemetry Tracing

OpenTelemetry tracing provides observability for the cascaded voice pipelines, allowing you to monitor latency, debug issues, and analyze conversation flows. The steps below show how to enable tracing with [Phoenix](https://arize.com/docs/phoenix/self-hosting).

## Steps

1. Enable tracing in `.env` **before** starting the stack, so the app picks it up:

    ```env
    ENABLE_TRACING=true
    OTEL_CONSOLE_EXPORT=false
    OTEL_EXPORTER_OTLP_ENDPOINT=phoenix:4317
    ```

2. Start the stack with the `tracing` overlay added to your recipe profile. This brings up the `phoenix` collector alongside the pipeline app. For example, Generic Cascaded on a workstation:

    ```bash
    docker compose --profile generic-assistant/workstation --profile tracing up -d
    ```

    If the stack is already running, the same command recreates the app container with the new settings, so no separate restart is needed. The `phoenix` service (defined in `docker/docker-compose.phoenix.yaml` and included by the root compose file) exposes:
    - **Port 6006**: Phoenix UI
    - **Port 4317**: OTLP gRPC collector

3. Open the Phoenix UI.

    ```
    http://localhost:6006
    ```

    For remote access replace `localhost` with your server's IP address.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRACING` | `false` | Enable OpenTelemetry tracing for both pipelines |
| `OTEL_CONSOLE_EXPORT` | `false` | Also print traces to stdout (useful for local debugging) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `localhost:4317` | OTLP collector endpoint |

**Endpoint format:**
- **gRPC** (port 4317): `host:port` (e.g. `phoenix:4317` or `localhost:4317`)
- **HTTP** (port 4318 or custom): `http://host:port` (e.g. `http://phoenix:4318`)

## Trace Structure

```
Conversation
├── turn
│   ├── stt          (ASR: user speech to text)
│   ├── llm          (LLM: generate response)
│   └── tts          (TTS: synthesise audio)
└── turn ...
```

## What You Can See in Phoenix

- Distributed traces across every pipeline component
- Per-turn latency breakdown (STT, LLM, TTS)
- Token usage for LLM calls
- Character counts for TTS
- Time-to-first-byte (TTFB) for each service

## Alternative Backends

Any OTLP-compatible backend works (Jaeger, Grafana Tempo, Langfuse, Datadog, etc.). Point `OTEL_EXPORTER_OTLP_ENDPOINT` at your collector and the traces will flow there instead of Phoenix.

See the [Pipecat OpenTelemetry docs](https://docs.pipecat.ai/server/utilities/opentelemetry) for additional exporter options.
