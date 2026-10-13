# Pipecat Adapter

The Pipecat adapter (`src/examples/frontend_delegation_hermes/server.py`) is the default. It serves OpenAI Realtime (GA) over WebSocket at `/v1/realtime`, `/health`, and a browser page at `/`. It has no WebRTC and no RTVI. Pipecat owns the turn taking:

- Silero VAD with `VADParams` from the settings file;
- Pipecat's default user-turn start strategies;
- `SpeechTimeoutUserTurnStopStrategy` as the stop strategy, with Smart Turn disabled;
- Pipecat interruptions and audio pacing;
- `NvidiaForceEouSTTService` and `NvidiaTTSService` (`per_sentence`) on the catalog speech endpoints.

The FDH spelling hold is off in both Pipecat profiles. The service is `fdh-voice` on host port 8776. It has `restart: "no"`, so it never comes back by itself during a τ³ campaign.

Complete `prerequisites.md` first, then set `F=docker/docker-compose.frontend-delegation-hermes.yaml`.

## Demo

The demo pairs `pipecat_demo.yaml` with `gateway.yaml` and `config/adapters/pipecat.yaml` (Pipecat defaults). It greets you, runs the demo tools in the server, and waits `FDH_BACKEND_DELAY_S` seconds (default 5) before each backend run, so you can try status questions and steering while the backend works. TLS is on by default.

### Start the Demo

```bash
docker compose --env-file .env -f $F up -d --build fdh-voice
sleep 30; curl -sk https://localhost:8776/health; echo
docker logs fdh-voice 2>&1 | grep -E 'backend gateway|warm-up|Pipecat adapter settings|config hash'
```

Expected `/health`:

- `"status": "ok"`, `"prototype": "frontend-delegation-hermes"`, and `"adapter": "pipecat"`;
- `"pipecat"` with `"settings_file"` naming `adapters/pipecat.yaml`, `"vad": {}`, `"tts_synthesis_mode": "per_sentence"`, and `"smart_turn": false`;
- `"backend": {"link": "websocket", ...}` and the profile's `"features"`.

The gateway health is the same as for the τ³ adapter. Check it with `docker exec fdh-voice curl -s 127.0.0.1:8790/health`.

### Use the Browser Page

Open `https://$HOST_IP:8776/`. Use the IP address, not a `*.nvidia.com` hostname, which HSTS forces through a proxy. Accept the self-signed certificate and allow the microphone. From a laptop, you can instead tunnel with `ssh -N -L 8776:localhost:8776 <user>@<host>` and open `https://localhost:8776/`.

The demo tools know order `5512` (shipped) and `5513` (processing). `cancel_order` works on `5513`, and any other number is "not found." The following table lists what to say and what to expect:

| You Say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 s plus the model time (new session) |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both |
| "Okay sure" | No reply (acknowledgement) |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same Hermes session |
| Talk over an answer | The answer stops, and the backend learns what you heard (a delivery note) |

Change the delay with `FDH_PIPECAT_BACKEND_DELAY_S=<seconds>` on the `up` command. A value of `0` disables it. The eval profile never delays.

A check without a browser: `curl -sk https://localhost:8776/` returns the page.

## Eval

The eval mode lets a τ² client evaluate the Pipecat adapter over OpenAI Realtime. It pairs `pipecat_realtime_eval.yaml` (client-executed tools with `tools.executor: wire`, the client's `session.update` instructions as the policy, no greeting) with `gateway.eval.yaml` and `config/adapters/pipecat.eval.yaml`. The eval settings set the Silero `stop_secs` to 2.5. The user-turn stop timeout keeps Pipecat's default, so a turn ends after about 3.1 s of silence. TLS is off.

### Start the Eval Mode

```bash
FDH_PIPECAT_PROFILE=pipecat_realtime_eval.yaml FDH_PIPECAT_GATEWAY_CONFIG=gateway.eval.yaml \
FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml FDH_PIPECAT_TLS=false \
  docker compose --env-file .env -f $F up -d --build fdh-voice
sleep 30; curl -s http://localhost:8776/health; echo
```

`/health` must show `"settings_file"` naming `adapters/pipecat.eval.yaml` and `"vad": {"stop_secs": 2.5}`. Export the four variables instead of prefixing them when later commands in the same shell must keep the eval mode. Running `up -d fdh-voice` without them recreates the demo.

### Point τ² at the Pipecat Adapter

Run this command in the τ² checkout, after its own setup (`uv sync --extra voice --extra dev` and the Inference Hub variables). Model tags must start with `pine-`. Use a `pine-fdh-pipecat-*` tag so the run is never confused with a τ³ arm:

```bash
IHUB=https://inference-api.nvidia.com/v1
RUN=fdh_pipecat_mock_smoke
PINE_REALTIME_BASE_URL=ws://localhost:8776/v1/realtime PINE_API_KEY=unused \
uv run python misc/prototypes/fba_voice_eval/tau2_ihub.py run --domain mock --audio-native \
  --audio-native-provider openai --audio-native-model pine-fdh-pipecat-mock-smoke \
  --speech-complexity regular --user-llm openai/azure/openai/gpt-5.2 \
  --user-llm-args "{\"temperature\": 0.0, \"api_base\": \"$IHUB\"}" --review-model openai/azure/openai/gpt-5.2 \
  --task-split-name base --num-trials 1 --max-concurrency 1 --num-tasks 1 --save-to "$RUN" --verbose-logs
```

Report Pipecat scores separately from the τ³ adapter. The two adapters time turns differently.

### Logs and Report

The Pipecat service writes its own logs, so it never appends to the τ³ logs:

- voice events: `logs/fdh_voice_events.pipecat.jsonl` (`FDH_PIPECAT_EVENT_LOG`);
- gateway events: `logs/fdh_gateway_events.pipecat.jsonl` (`FDH_PIPECAT_GATEWAY_LOG`);
- Hermes workers: `logs/fdh_workers_pipecat/` (`FDH_PIPECAT_WORKER_LOG_DIR`).

The following command converts the voice event log for `fba_voice_metrics.py`:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.report_adapter \
  logs/fdh_voice_events.pipecat.jsonl --out logs/fdh_voice_events.pipecat.legacy.jsonl
```

## Smoke Test

The command-line client `cli/voice_chat.py` has no TLS option, so run its text or audio smoke test against a plain `ws://` server: the eval mode, or the demo started with `FDH_PIPECAT_TLS=false`.

The following commands synthesize three user turns with the local TTS and play them in τ²'s wire format (μ-law, 8 kHz):

```bash
mkdir -p /tmp/fdh-wav && uv run python - <<'PY'
import wave, riva.client
tts = riva.client.SpeechSynthesisService(riva.client.Auth(uri="localhost:50051", use_ssl=False))
for i, text in enumerate(["Can you give me the status of order five five one two?", "Okay, thanks.",
                          "What about order five five one three?"], 1):
    audio = tts.synthesize(text, voice_name="", language_code="en-US", sample_rate_hz=22050).audio
    with wave.open(f"/tmp/fdh-wav/u{i}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(audio)
PY
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.voice_chat \
  --url ws://localhost:8776/v1/realtime --model pine-fdh-pipecat-audio --io wav --format pcmu \
  --in /tmp/fdh-wav/u1.wav --in /tmp/fdh-wav/u2.wav --in /tmp/fdh-wav/u3.wav --out /tmp/fdh-wav/agent.wav \
  --ui plain --tools examples.frontend_delegation_hermes.core.text.demo_tools:TOOLS
```

Pass `--tools` only in eval mode, where the client executes the tools. In the demo, the server runs them, so omit `--tools`. Expect a filler, the order status, nothing for "Okay, thanks," and a filler for order 5513. With the 2.5 s eval silence, each answer starts at least about 3.1 s after the end of speech.

## Known Limits

The Pipecat adapter supports a subset of OpenAI Realtime. The following limits apply:

- **Heard audio is an estimate.** Pipecat's WebSocket output writes audio at twice real time. The adapter advances its playout estimate at real time from the first written chunk, capped at the written amount. A client that pauses playback, such as τ²'s tick loop, hears less than estimated.
- **Speech onset is late.** `audio_start_ms` is the cursor at Pipecat's VAD confirmation minus `start_secs`. Stub measurements put it about 200 ms after the actual onset.
- **Server VAD only.** A client that asks for `turn_detection: null` (manual turns) is accepted with a logged warning: text items work, Pipecat's VAD still ends audio turns, and `input_audio_buffer.commit` is rejected.
- **Catalog voice only.** The TTS voice is the catalog voice. The client's requested voice is ignored with a warning.
- **No spelling hold.** Pipecat decides when a turn ends, so FDH never holds a turn open.
