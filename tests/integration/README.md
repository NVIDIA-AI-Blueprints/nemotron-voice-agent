# Integration and Live Tests

These tests talk to a **running** voice-agent server. They are **not** part of
CI (`pytest tests/unit`).

## OpenAI Realtime Compatibility

`test_realtime_openai_sdk_compat.py` is an opt-in live suite for the OpenAI
Realtime-compatible WebSocket subset. It covers:

- OpenAI Python SDK audio conversations.
- Text responses, session updates, response overrides, and isolated input.
- Rejection of unsupported session configuration.
- Manual turn control and automatic VAD negotiation.
- Client-owned function output and follow-up response ordering.

Start a local server without TLS:

```bash
PIPELINE_TLS=false uv run python src/server.py --host 127.0.0.1 --port 7860
```

Run the live suite from another shell:

```bash
OPENAI_REALTIME_WS_BASE=ws://127.0.0.1:7860/v1 \
  RUN_REALTIME_COMPAT=1 \
  uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v -s
```

`ENABLE_WELCOME_MESSAGE` controls RTVI and does not change the Realtime
handshake. The manual-configuration test skips when the selected deployment is
not an eligible cascaded ASR pipeline. If the server sets `REALTIME_API_KEY`,
export the accepted master key or client secret as
`OPENAI_REALTIME_API_KEY` for the test process.

The live suite uses signed 24 kHz PCM16 and verifies manual-mode negotiation
rather than a complete manual audio turn. It does not cover every media format,
timeout, conversation mutation, MCP path, or long-session boundary. Run the
unit suite for deterministic protocol coverage. Use the [Realtime test
guidance](../../docs/how-to/use-realtime-gateway.md#collect-metrics-and-run-tests)
and [scaling client](../../benchmarking_tools/scaling-perf/README.md) for
deployed and concurrent qualification.
