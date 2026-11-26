# Enable OpenTelemetry Tracing

OpenTelemetry tracing provides observability for the cascaded voice pipelines, allowing you to monitor latency, debug issues, and analyze conversation flows. The steps below show how to enable tracing with [Phoenix](https://arize.com/docs/phoenix/self-hosting). This guide applies to the Pipecat examples (`nva-pipecat`).

## Start and Verify Phoenix

The [Phoenix Compose service](../../docker/sidecars/docker-compose.phoenix.yaml) publishes the UI (port `6006`) and the OTLP gRPC collector (port `4317`) on all host interfaces. Traces can contain conversation data. Before starting the overlay, block direct access to both ports from untrusted networks. For remote UI access, use a trusted network or an authenticated HTTPS reverse proxy.

1. Enable tracing in `.env` **before** starting the stack, so the app picks it up:

    ```env
    ENABLE_TRACING=true
    OTEL_CONSOLE_EXPORT=false
    OTEL_EXPORTER_OTLP_ENDPOINT=phoenix:4317
    ```

2. Set `recipe` to your Pipecat recipe profile. If the app is already running, use the profile you started it with. This example uses the Generic Assistant Server recipe.

    ```bash
    recipe=generic-assistant/server
    docker compose --profile "$recipe" --profile tracing up -d
    ```

    Compose starts the Phoenix collector and applies the tracing settings to the selected app.

3. Open the Phoenix UI.

    ```
    http://localhost:6006
    ```

4. Run a voice turn in the Pipecat UI.
5. Look for a new turn trace in Phoenix.
6. Save any traces you need before stopping the stack. The bundled Compose service has no persistent volume configured for Phoenix. In the same terminal, stop the recipe and its tracing overlay.

    ```bash
    docker compose --profile "$recipe" --profile tracing down
    ```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRACING` | `false` | Enable tracing for instrumented Pipecat pipelines |
| `OTEL_CONSOLE_EXPORT` | `false` | Also print traces to stdout (useful for local debugging) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `localhost:4317` | OTLP collector endpoint |

**Endpoint format:**
- **gRPC**: `host:port` (for example, `phoenix:4317`). The app uses an insecure gRPC channel.
- **HTTP**: the full OTLP traces URL, including the path (for example, `https://collector.example.com/v1/traces`).

## Trace Structure

```text
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

OTLP-compatible backends include Jaeger, Grafana Tempo, Langfuse, and Datadog. Point `OTEL_EXPORTER_OTLP_ENDPOINT` at your collector. Traces flow there instead of Phoenix.

See the [Pipecat OpenTelemetry docs](https://docs.pipecat.ai/server/utilities/opentelemetry) for additional exporter options.
