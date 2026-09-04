# Full-Duplex-Bench eval

Batch client for [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1, v1.5: streams WAVs to the voice agent over WebSocket and writes reply audio. Configure the server (`.env` and the selected example's `services.cloud.yaml` / `services.local.yaml`), start it, then point this tool at it with `--server-url`.

## Install

This tool reuses the repo's root environment — no separate venv required.
Dependencies live in the `benchmark` group of the root `pyproject.toml`.
From the **repository root**:

```bash
uv sync --group benchmark
```

## Server

From the repo root (see [Configuration Guide](../../docs/02-configuration-guide.md) for `.env` and example-local service catalogs):

```bash
PIPELINE_TLS=false uv run python src/server.py
```

Defaults to `http://localhost:7860`. Set `PIPELINE_TLS=true` or unset it to use HTTPS on the same port.

## RTVI client

`--server-url` uses `http://` or `https://` (not `ws://`). Omit the port to use `7860`.

```bash
cd benchmarking_tools/Full-Duplex-Bench-Eval
uv run python inference_rtvi.py --input-dir /path/to/samples --server-url http://127.0.0.1:7860
```

HTTPS uses normal certificate verification by default. For local self-signed certs, add `--insecure-skip-verify`, for example:

```bash
uv run python inference_rtvi.py --input-dir /path/to/samples --server-url https://127.0.0.1:7860 --insecure-skip-verify
```

## OpenAI Realtime client

Set the API key and pass the provider's WebSocket endpoint:

```bash
export REALTIME_API_KEY=...
uv run python inference_realtime.py \
  --input-dir /path/to/samples \
  --realtime-ws-url wss://example.com/realtime
```

Use `--api-key-env` or `--auth-scheme` when the provider expects different authentication.
Both clients support `--retry-samples 1 5 10`.

## Dataset

Numeric subfolders under `--input_dir`: `input.wav` → `output.wav`, `clean_input.wav` → `clean_output.wav` when present.

## Reference

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) · [Nemotron Voice Agent](../../README.md)
