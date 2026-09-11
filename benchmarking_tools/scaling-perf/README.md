# Scaling Perf

Load-test the Nemotron Voice Agent with synthetic clients. Each "client" is a
Python process that connects to the running server over WebSocket, submits a
WAV utterance or Realtime text item as the user, listens for the bot's
response, and records how long each turn took. You can run a single client (smoke test) or fan out to N
parallel clients (concurrency / scaling test) and produce a sweep across
multiple concurrency levels.

Select the wire protocol explicitly with `--protocol`. The default remains
`rtvi`, so existing benchmark commands that omit this option continue to use
RTVI. New client result files include an explicit `protocol` and canonical
`run_config`. Aggregation rejects historical result files that do not contain
those fields, so do not combine them in one run directory.

| Protocol | Endpoint and behavior |
| --- | --- |
| `rtvi` | Connects to `WS /api/ws`, reads Pipecat Real-Time Voice/Video Inference (RTVI) audio and server metric messages, and does not use the Realtime session flow. |
| `openai-realtime` | Connects to `WS /v1/realtime`, validates the canonical session and conversation handshake, sends nested Realtime session configuration with either 24 kHz PCM audio or user `input_text` items, and records the response, item, error, usage, and tool event graph. |

The Realtime client does not translate or fall back to RTVI. This separation
keeps each result attributable to one protocol and makes protocol failures
visible.

## Layout

| File | Job |
|------|-----|
| `benchmark.py` | Drives **one** client by default. Also produces summaries when invoked with `--aggregate-run-dir` / `--aggregate-suite-dir`. |
| `openai_realtime_client.py` | Implements the raw JSON Realtime client with strict lifecycle validation, audio streaming, and Realtime-specific result capture. |
| `simulate_concurrency.sh` | Spawns N parallel `benchmark.py` workers per concurrency level with shared nominal metric boundaries and cooldowns, then calls `benchmark.py` in aggregate mode. |

## Setup

These scripts reuse the repo's root environment — no separate venv required.
Dependencies are managed via the
root `benchmark` dependency group (shared by every tool under
`benchmarking_tools/`).

1. From the repository root, sync the project (one-time):

   ```bash
   uv sync --group benchmark
   ```

   This host-side sync is still required even when the server runs under
   Docker Compose, because `benchmark.py` and `simulate_concurrency.sh` run
   on the host, not inside the app container.

2. Start the voice-agent server through [Docker
   Compose](../../docs/01-getting-started.md) or `uv run python src/server.py`.
   For RTVI scaling, disable the welcome message so its greeting cannot become
   the measured turn. The `generic-assistant/server-perf` profile does this
   automatically. For a host-native server, run:

   ```bash
   ENABLE_WELCOME_MESSAGE=false uv run python src/server.py
   ```

   `ENABLE_WELCOME_MESSAGE` governs RTVI. Realtime waits for a client-initiated
   turn, and the benchmark rejects a response that precedes its input.
3. Add WAV files to `dataset/` for RTVI or Realtime audio-input runs. Realtime
   text-input runs do not require a dataset. The benchmark cycles through the
   selected WAV files or text prompts as the simulated user's turns. Prepare
   each WAV file so the benchmark can time audio turns correctly:

   - **Record the query or reuse existing audio**, using generic queries or ones that match your specific use case.
   - **Use one continuous utterance per file, with no long internal pauses.** The server runs voice activity detection and turn endpointing on incoming audio. A long mid-file silence can close the turn early. In Realtime automatic mode, the client then stops sending the remaining PCM so it cannot become a second utterance. The benchmark has still exercised only a partial query, so that result is not representative.
   - **Trim all trailing silence from the end** (for example in Audacity). RTVI and Realtime manual mode measure client latency from WAV streaming completion. Realtime automatic mode uses the first received turn boundary when it arrives during streaming, and otherwise uses streaming completion. Clean file boundaries keep both protocol measurements comparable. The scripts insert silence between files automatically, so do not pad the files yourself.
   - **Save as 16 kHz, mono, linear PCM (`int16`) WAV.** This matches the RTVI
     pipeline input. The Realtime client resamples each file to canonical
     24 kHz mono PCM before streaming it.

   If you do not trim trailing silence, RTVI and Realtime manual client-side
   latency are biased by the difference between speech end and file end.
   Realtime automatic mode avoids that bias when it observes the boundary before
   streaming finishes. In RTVI
   mode, server-reported metrics remain reliable because the server measures
   them from actual speech and turn events. Refer to [What It
   Measures](#what-it-measures).

`simulate_concurrency.sh` auto-dispatches `benchmark.py` through `uv run`
when the root `pyproject.toml` is detected, so the commands below work
straight from a fresh `uv sync --group benchmark`.

### Prompt Override for Perf Runs

By default, `generic-assistant/server-perf` uses the same default prompt
as the normal Generic Assistant server profile.

If you want to experiment with custom prompts with different input-token sizes,
point the server at the prompt catalog in this directory and select the prompt
key you want:

```bash
PROMPT_FILE_PATH=/app/benchmarking_tools/scaling-perf/perf_prompts.yaml \
PROMPT_SELECTOR=prompt_200_tokens \
docker compose --profile generic-assistant/server-perf up -d
```

This catalog defaults to `prompt_1000_tokens`. Available prompt entries are
`prompt_200_tokens`, `prompt_1000_tokens`, and `prompt_5000_tokens`.

## Reproducing the Recommended Scaling Setup

The recommended scaling setup uses four Blackwell GPUs with `1 GPU` for ASR,
`1 GPU` for TTS, and `2 GPUs` for the `Nemotron 3.5 Lightning 30B` LLM.

This setup is available as the dedicated Compose recipe
`generic-assistant/server-perf`. It automatically applies the published
scaling configuration:

- Generic Assistant inherits the existing `nemotron-lightning` default from
  [`examples_registry.yaml`](../../examples_registry.yaml)
- `nvidia-llm`: `NIM_MODEL_PROFILE=vllm-nvfp4-tp2-pp1-18.0`, GPUs `2,3`, alias
  `nvidia-llm`
- `nemotron-asr-streaming-english`:
  `NIM_TAGS_SELECTOR=type=en-US,mode=str,batch_size=128`, GPU `0`, alias
  `nemotron-asr-streaming-english`
- `magpie-multilingual-tts-service-perf`:
  `NIM_TAGS_SELECTOR=name=magpie-tts-multilingual,batch_size=64`, GPU `1`, alias
  `magpie-multilingual-tts-service`
- app env: `UVICORN_WORKERS=200`,
  `USE_SILERO_VAD_TURN_DETECTION=true`, `SILERO_VAD_STOP_SECS=0.5`,
  `AUDIO_OUT_10MS_CHUNKS=40`, `ENABLE_WELCOME_MESSAGE=false`

> **Hardware-specific profile:** `vllm-nvfp4-tp2-pp1-18.0` was selected from
> `list-model-profiles` and benchmarked on two NVIDIA RTX PRO 6000 Blackwell
> GPUs. The checked-in pin is an RTX PRO 6000 benchmark baseline, not a portable
> recommendation. Before running this performance recipe on any other hardware,
> including H100, replace the pin by following these steps:
>
> 1. Run `list-model-profiles` with the deployed image on the actual LLM GPUs.
> 1. Benchmark the compatible TP2 profiles for time to first token, inter-token
>    latency, and total throughput per GPU.
> 1. Set `NIM_MODEL_PROFILE` to the winning profile's exact ID or full
>    description. H100 requires its own comparison of the listed FP8 and BF16
>    TP2 profiles.
>
> If portability matters more than predictable benchmark performance, remove
> the pin and let NIM select automatically. Do not use the deprecated NIM 1.x
> `NIM_TAGS_SELECTOR` for an LLM.
>
> See NVIDIA NIM's [model profile selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)
> and [environment variable](https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html)
> documentation.

Deploy it with:

```bash
docker compose --profile generic-assistant/server-perf up -d
```

After the stack is healthy, run the sweep from this directory:

```bash
./simulate_concurrency.sh --clients "1 2 4 8 16"
```

## Run

From this directory:

```bash
# RTVI single client. RTVI is the default.
uv run python3 benchmark.py

# RTVI concurrent run and sweep.
./simulate_concurrency.sh --clients 4
./simulate_concurrency.sh --clients "1 2 4 8 16"

# OpenAI Realtime single client against this gateway.
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-turn-mode automatic \
  --realtime-insecure

# OpenAI Realtime text input and text output. Repeat --realtime-text-input to
# cycle through multiple prompts. This mode does not require dataset WAV files.
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-input-mode text \
  --realtime-text-input "Reply with one short greeting." \
  --realtime-text-input "What is two plus two?" \
  --realtime-output-modality text \
  --realtime-insecure

# Manual Realtime single client on an eligible cascaded ASR pipeline.
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-turn-mode manual \
  --realtime-insecure

# OpenAI Realtime concurrency sweep.
./simulate_concurrency.sh \
  --protocol openai-realtime \
  --realtime-turn-mode automatic \
  --realtime-insecure \
  --clients "1 2 4 8 16"
```

If the gateway sets `REALTIME_API_KEY`, export the key in the benchmark shell
and pass only its environment-variable name. The value can be the
deployment master key or a client secret that remains valid until every worker
opens its WebSocket. Account for the configured worker stagger. The secret's
top-level `expires_at` value does not shorten an established connection's
separate 60-minute maximum:

```bash
read -rsp "Realtime API key: " REALTIME_API_KEY
printf '\n'
export REALTIME_API_KEY

./simulate_concurrency.sh \
  --protocol openai-realtime \
  --host realtime.example.com \
  --port 443 \
  --realtime-turn-mode automatic \
  --realtime-api-key-env REALTIME_API_KEY \
  --clients "1 2 4 8 16"

unset REALTIME_API_KEY
```

This authenticated example uses certificate and hostname verification. Add
`--realtime-ca-file /path/to/ca.pem` when the deployment uses a private
certificate authority. Do not combine credentials with
`--realtime-insecure` outside an explicitly accepted local test.

Realtime input defaults to audio. Text mode requires at least one
`--realtime-text-input` and supports text or audio output. Automatic mode
repeats the server-advertised VAD object. Manual mode sends an explicit commit
and `response.create` and is available only for eligible cascaded ASR
pipelines. The client rejects unsolicited responses before its input turn is
committed.

The shell wrapper accepts `-h`/`--help`. Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--protocol` | `rtvi` | Select `rtvi` or `openai-realtime`. The client never changes protocols after a failure. |
| `--clients "N1 N2 …"` | `1` | One run per concurrency level. Quote the list. |
| `--host` / `--port` | `localhost` / `7860` | Server target. |
| `--test-duration` | `300` | Seconds in the nominal turn-admission interval per level. Eligibility is latched when each turn starts, and an admitted turn may finish after the interval closes. |
| `--client-start-delay` | `1` | Stagger before each client begins connecting (s). With N clients and delay D, the nominal metric start is `now + (N-1)*D`; the final client begins its connection attempt at that time, so session setup can extend into the nominal window. |
| `--cooldown` | `10` | Pause between sweep levels (s) — lets the server settle between bursts. |
| `--turn-response-timeout` | `10` | Seconds to wait for the first expected audio/text output or subsequent response activity. Realtime tool waits use their separate deadline below. |
| `--reverse-barge-in-threshold` | `0.4` | Classify an audio-playback turn below this latency as a *reverse* barge-in and exclude it from average and p95 latency. Realtime text turns are never filtered by this threshold. Used internally and not surfaced in summaries. |
| `--no-save-audio` | audio modes save | Skip writing per-client output WAVs. |
| `--dataset-dir DIR` | `./dataset` | Override input WAV directory. |
| `--output-dir DIR` | this folder | Override result destination. |

Realtime-specific flags are shown in the following table.

| Flag | Default | Description |
| --- | --- | --- |
| `--realtime-scheme` | `wss` | Select `wss` with certificate verification or unencrypted `ws`. The client refuses to send a configured Bearer token over `ws`. |
| `--realtime-path` | `/v1/realtime` | WebSocket route. |
| `--realtime-model` | empty | Optional standard `model` query parameter. Supply an exact visible Realtime profile ID from `examples_registry.yaml` to select its complete trusted pipeline configuration. An unknown or hidden value returns `model_not_available` before `session.created`; an empty value uses the deployment default. |
| `--realtime-voice` | empty | Optional nested `session.audio.output.voice`. Leave it empty to inherit, or supply an exact voice ID from the trusted TTS catalog. Unknown IDs fail without fallback. For Multilingual audio, leave it empty unless the test intentionally qualifies language-specific voice resolution. |
| `--realtime-instructions` | empty | Optional session instructions. |
| `--realtime-input-mode` | `audio` | Select `audio` to stream dataset WAVs or `text` to create user `input_text` items. Text mode does not require a dataset. |
| `--realtime-text-input` | none | Add a nonempty prompt for text input mode. Repeat the flag to cycle prompts across turns. At least one value is required in text mode. |
| `--realtime-output-modality` | `audio` | Request exactly `audio` or `text`. This gateway supports both. Text responses skip TTS readiness, warmup, and synthesis, while the pipeline keeps TTS linked for a later response-local audio override. |
| `--realtime-turn-mode` | `automatic` | Repeat the server-advertised `semantic_vad` or `server_vad` configuration and wait for automatic response creation. `manual` sends explicit commit plus response creation for targets that support it. |
| `--realtime-vad-silence-ms` | `800` | Silence appended after speech in automatic mode. |
| `--realtime-api-key-env` | empty | Name of an environment variable containing a Bearer master key or `ek_` client secret. It is required when this gateway sets `REALTIME_API_KEY`. Neither the value nor the variable name is written to results. |
| `--realtime-ca-file` | empty | Optional Privacy-Enhanced Mail (PEM) certificate-authority bundle for verified `wss`. |
| `--realtime-insecure` | disabled | Explicitly disable certificate and hostname verification for `wss`, typically for a local self-signed certificate. It cannot be combined with `--realtime-ca-file`. |
| `--realtime-client-tools-config` | empty | Strict JSON file containing native Realtime function and/or remote MCP tools. Function declarations require exact-name scripted or HTTP handlers. Without it, the client keeps the server's trusted tool set. |
| `--realtime-tool-timeout` | `120` | Overall deadline for a correlated client-tool round or trusted tool-output phase. Client handlers reserve the early-send margin described below. |
| `--realtime-max-tool-rounds` | `4` | Maximum function-call response rounds admitted in one turn. |
| `--realtime-session-timeout` | `60` | Maximum wait for each required Realtime initialization event after the socket opens. This is separate from the fixed 30-second TCP/WebSocket open timeout. |

`wss` validates the certificate chain and hostname by default. Supply
`--realtime-ca-file` for a private certificate authority. Use
`--realtime-insecure` only when you explicitly accept a local self-signed
certificate. If the server runs with `PIPELINE_TLS=false`, select
`--realtime-scheme ws`; CA and insecure options are invalid for `ws`.

The Realtime client verifies the requested model, voice, instructions, output
modality, VAD mode, and PCM format in the session snapshots.

`--realtime-tool-timeout` bounds one complete tool round. It is independent
of the gateway's `REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS`. Keep the benchmark
deadline at or below the gateway deadline for ordinary runs. One gateway
WebSocket lasts at most 60 minutes, so keep initialization, admitted turns,
and completion time within that bound.

Tool events do not count as first user-visible output. Audio latency stops on
the first audio delta; text latency stops on the first non-whitespace text
delta, or on the terminal response when no visible delta arrives.

### Configure Client Tools

Use `--realtime-client-tools-config` to replace the server's active trusted
tool set with native Realtime function and/or remote MCP declarations. The JSON
root accepts `tools`, `handlers`, optional `tool_choice`, optional
`parallel_tool_calls`, and optional tester-only `mcp_approval`. Handler keys
must match the declared function names exactly; MCP tools never have local
handlers. The headless client does not interpret names or descriptions. A
function handler can use a scripted sequence for deterministic protocol testing
or post the generated arguments to an application-owned HTTP endpoint.
`parallel_tool_calls` defaults to `true`. With `false`, the client fails a turn
if the server publishes more than one call for a response. This configuration
intentionally replaces trusted tools; use another native client to qualify the
gateway's mixed trusted-plus-client support.

#### Use Native MCP Tools

Before the test, add the exact remote server URL to the gateway's `.env`
allowlist:

```dotenv
REALTIME_MCP_ALLOWED_SERVER_URLS=["https://developers.openai.com/mcp"]
```

The gateway host must have outbound HTTPS access to that URL. Restart the
application after changing `.env`.

An MCP entry is sent unchanged in `session.tools`; the Realtime service owns
discovery and execution. `handlers` therefore contains only function tools and
is `{}` for an MCP-only run. Save this configuration as
`remote-mcp-tools.json` in this directory:

```json
{
  "tools": [
    {
      "type": "mcp",
      "server_label": "openai_docs",
      "server_url": "https://developers.openai.com/mcp",
      "allowed_tools": ["search_openai_docs"],
      "require_approval": "always"
    }
  ],
  "handlers": {},
  "tool_choice": "required",
  "parallel_tool_calls": true,
  "mcp_approval": {
    "default": "reject",
    "rules": [
      {
        "server_label": "openai_docs",
        "name": "search_openai_docs",
        "approve": true
      }
    ]
  }
}
```

`mcp_approval` is benchmark policy, not a Realtime session field. Rules match
`server_label` and tool `name` exactly. An unmatched request follows the
configured default. The client completes native discovery, approval, and call
events before it requests the final response. It never sends
`function_call_output` for an MCP call.

Run a short text test from this directory:

```bash
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-input-mode text \
  --realtime-text-input \
    "Search the OpenAI docs for Realtime WebSocket authentication." \
  --realtime-output-modality text \
  --realtime-client-tools-config ./remote-mcp-tools.json \
  --test-duration 1 \
  --realtime-insecure
```

If the gateway sets `REALTIME_API_KEY`, enter and export only that key as shown
in [Run](#run), and add `--realtime-api-key-env REALTIME_API_KEY`. A successful
worker exits with status `0`. In its `result_<id>.json`, confirm that
`protocol_events` contains MCP discovery, approval, call, and terminal events,
`response_status_counts.completed` is nonzero, and `response_transcripts`
contains a nonempty final answer.

#### Use Scripted Handlers

Each worker starts its own scripted step sequence, so concurrent clients do not
share invocation state.

The following configuration exposes one lookup function and returns a fixed
result whenever the model selects it:

```json
{
  "tools": [
    {
      "type": "function",
      "name": "browser_lookup",
      "description": "Look up current information.",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": false
      }
    }
  ],
  "handlers": {
    "browser_lookup": {
      "steps": [
        {
          "expected_arguments": {"query": "weather"},
          "delay_ms": 25,
          "output": {"ok": true, "weather": "sunny"}
        }
      ],
      "repeat_last": true
    }
  },
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

With `tool_choice: "auto"`, the model can answer without a call. Use a named
function choice when the test must exercise the tool lifecycle.

Run one worker with that file:

```bash
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-turn-mode automatic \
  --realtime-insecure \
  --realtime-output-modality text \
  --realtime-client-tools-config ./client-tools.json
```

Every scripted handler needs a nonempty `steps` array. Each step contains one
of `output` or `error` and can set `delay_ms` or exact
`expected_arguments`. The client validates generated arguments against the
function schema before it runs the step. Script failures become correlated
JSON error outputs so the model can recover in the follow-up response.

#### Use HTTP Handlers

Set a handler's `type` to `http` to post generated arguments as JSON. The URL
must use `http` or `https` without embedded credentials or a fragment.
`timeout_seconds` defaults to 60. The endpoint must return a successful JSON
response no larger than 1 MiB. HTTP, timeout, redirect, size, and JSON failures
become structured tool outputs.

The following configuration binds an arbitrary public function name to an
application-owned HTTP endpoint. Save it as `client-tools-http.json` and
replace the example URL with an endpoint that your benchmark host can reach:

```json
{
  "tools": [
    {
      "type": "function",
      "name": "consult_airline_specialist",
      "description": "Delegate an airline request to the configured specialist agent.",
      "parameters": {
        "type": "object",
        "properties": {
          "request": {"type": "string"}
        },
        "required": ["request"],
        "additionalProperties": false
      }
    }
  ],
  "handlers": {
    "consult_airline_specialist": {
      "type": "http",
      "url": "https://tools.example.com/airline/consult",
      "timeout_seconds": 60
    }
  },
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

Start the application endpoint, and use a dataset WAV that contains a matching
request. Run the authoritative headless Realtime client with explicit routing
instructions:

```bash
uv run python3 benchmark.py \
  --protocol openai-realtime \
  --realtime-turn-mode automatic \
  --realtime-insecure \
  --realtime-output-modality text \
  --realtime-instructions \
    'For an airline request, call consult_airline_specialist exactly once with a self-contained request. After its result, answer the user without calling it again.' \
  --realtime-client-tools-config ./client-tools-http.json
```

The gateway receives only the standard function declaration. The benchmark
client maps its exact name to the configured endpoint. Inspect `tool_calls` and
confirm Response A, the original `call_id`, the output acknowledgement, and
Response B. Rename the function in both configuration locations and the
instructions to verify that no `delegate*` naming convention is required.

`Ctrl-C` is graceful. Workers stop, and partial results stay on disk.

## Output Layout

**Where to look first:** open `results.txt` for `simulate_concurrency.sh` runs
(single-level or sweeps). For direct `uv run python3 benchmark.py` runs, check
the client summary line and `result_<id>.json`. Per-client `.log` files are
mainly for debugging specific failures.

Single concurrency level (`--clients 1`, `--clients 4`, etc.):

```
results_<timestamp>/
├── benchmark_summary.json       # rolled-up summary across all clients
├── results.txt                  # one-row summary table (human-readable)
├── results.tsv                  # one-row summary table (tab-separated)
├── results.json                 # one-row summary object list
└── client_<i>_<run_id>/         # i = 1..N; run_id contains UTC nanoseconds and the shell PID
    ├── benchmark_<id>.log       # turn-by-turn log
    ├── result_<id>.json         # per-client metrics, parsed back by aggregation
    ├── process_stdout.log       # worker stdout and stderr
    └── audio_output_<id>.wav    # audio mode only, after bot audio arrives (unless --no-save-audio)
```

Multi-level sweep (`--clients "1 4 16"`):

```
perf_suite_<timestamp>/
├── results.txt                  # column-aligned, human-readable
├── results.tsv                  # tab-separated (spreadsheets / pandas)
├── results.json                 # one object per concurrency level
└── run_<N>_clients/             # one of these per --clients value
    ├── benchmark_summary.json
    └── client_<i>_<run_id>/...
```

Each per-client result records its selected protocol and sanitized
`run_config`. Realtime results also include protocol diagnostics, response
transcripts, terminal status counts, and correlated tool-call records.
High-volume audio events store byte and event counts instead of Base64 payloads.
In captured MCP declarations, `authorization`, header values, and URL query
values become `[REDACTED]`. This redaction affects only `protocol_events`; the
client sends the configured values unchanged over the WebSocket. Other events
can contain instructions, transcripts, arguments, and results, so treat output
files as potentially sensitive.

A worker exits nonzero for protocol or lifecycle failures. Realtime
WebSocket-open, session-initialization, and turn-response timeout errors name
the affected phase and timeout value. Aggregation keeps successful, failed,
and no-response outcomes separate and excludes failed or missing workers from
headline latency and server-metric averages.

## What It Measures

Both protocols measure end-to-end response latency and audio-buffer underruns.
RTVI measures through the first bot audio frame. Realtime measures through the
first audio delta or first non-whitespace text delta. A Realtime tool-only
Response A is activity, not user-visible output, so its latency includes tool
execution and the final Response B.

Realtime results also retain response status, lifecycle duration, tool-call
correlation, available response usage, and recognized
`nvidia.metrics.updated` values. The gateway does not expose RTVI or
`rate_limits.updated` events on the Realtime socket. The client does not
synthesize unavailable metrics.

Collection eligibility is determined when each turn starts. Aggregation keeps
successful, failed, and no-response clients separate and uses finite samples
from successful clients only.

The following server metric keys can appear when the selected pipeline and
turn mode supply them:

| Metric | Meaning |
|--------|---------|
| `llm_ttft` | Time-to-first-token from the LLM |
| `tts_ttfb` | Time-to-first-byte from the TTS |
| `asr_ttfb` | Time-to-first-byte from the ASR |
| `server_e2e` | Server-side end-to-end (user-stop → first bot speech) |
| `vad_smart_turn` | Full user-turn release duration, including VAD silence and turn-release work |
| `smart_turn_inference` | Smart Turn prediction runtime only |
| `llm_processing_time` | LLM end-to-end (request → final token) |
| `llm_tokens_per_sec` | Completion tokens / `llm_processing_time` |

Manual Realtime turns do not normally include Pipecat-native `asr_ttfb` or
`server_e2e` because those measurements start from VAD events. Automatic turns
can include them. RTVI does not currently populate `smart_turn_inference`.
