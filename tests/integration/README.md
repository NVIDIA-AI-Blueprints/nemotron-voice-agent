# Integration / live tests

These tests talk to a **running** voice-agent server. They are **not** part of
CI (`pytest tests/unit`).

## OpenAI Realtime compatibility

[`test_realtime_openai_sdk_compat.py`](test_realtime_openai_sdk_compat.py) against `WS /v1/realtime`:

1. OpenAI Python SDK multi-turn (GA-shaped session fields)
2. Mapped fields — instructions, Magpie voice, temperature,
   Nemotron welcome gate, soft unknown-voice fallback, post-handoff / `response.create` rejects
3. Client-owned tool round trip — completed function-call response, stock-SDK
   `function_call_output`, explicit `response.create`, and spoken result
4. Rejects — text-only modalities

```bash
# Plain HTTP (recommended for local integration):
# Option A — set in .env then recreate:
#   PIPELINE_TLS=false
# Option B — one-shot compose override:
printf '%s\n' 'services:' '  generic-assistant:' '    environment:' '      PIPELINE_TLS: "false"' > /tmp/nva-tls-off.yml
PIPELINE_TLS=false docker compose -f docker-compose.yml -f /tmp/nva-tls-off.yml \
  --profile generic-assistant up -d --force-recreate generic-assistant

OPENAI_REALTIME_WS_BASE=ws://127.0.0.1:7860/v1 RUN_REALTIME_COMPAT=1 \
  uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v -s

# TLS docker default (self-signed; browsers often fail wss without trusting the cert):
RUN_REALTIME_COMPAT=1 OPENAI_REALTIME_WS_BASE=wss://127.0.0.1:7860/v1 \
  uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v -s
```

Set `REALTIME_COMPAT_WAIT_INTRO=0` only when the test server has
`ENABLE_WELCOME_MESSAGE=false`.
