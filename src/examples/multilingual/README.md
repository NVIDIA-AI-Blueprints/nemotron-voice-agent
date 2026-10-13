# Multilingual - cascaded pipeline example

Multilingual cascaded voice pipeline using Pipecat's built-in NVIDIA services (`NvidiaSTTService` -> `NvidiaLLMService` -> `NvidiaTTSService`). The session is locked to a single language for the whole connection (selected in the UI, default `de-DE`): the ASR, the TTS voice, and the LLM all operate in that one language. The LLM replies with plain spoken text, kept on-language by the fixed-session prompt addon and a per-turn reminder.

The pattern uses dedicated ASR, LLM, and TTS services with a plain-text response, a per-turn language reminder injected at request time only, and a clean chat history that stores just the spoken reply.

![Architecture Diagram](../../../docs/images/arch.png)

## Default Models

The defaults in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following models for each profile:

| Profile | ASR | LLM | TTS |
| --- | --- | --- | --- |
| Cloud | Parakeet 1.1B RNNT Multilingual ASR | Nemotron 3.5 Lightning 30B A3B | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming Multilingual NIM | Nemotron 3.5 Lightning 30B A3B NIM | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron 3.5 ASR Streaming Multilingual 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Magpie TTS Multilingual through NeMo-Speech.cpp |

The registry declares `nemotron-asr-streaming-multilingual` as the ASR default. When that local service is unreachable, the resolver tries another reachable local ASR before falling back to the cloud catalog. The cloud catalog has no Nemotron ASR Streaming Multilingual entry, so it uses Parakeet 1.1B RNNT Multilingual ASR.

## Running the example

This example runs with **Cloud**, **Server** (NIM, recommended for scaling), and universal **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor). The single-gpu profile covers workstations, DGX Spark, and Jetson Thor. Refer to the [Jetson Thor guide](../../../docs/03-jetson-thor.md) when applicable. See the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run every command from the repository root.

1. Preserve any existing `.env` file. Otherwise, copy the template, and then set `NVIDIA_API_KEY` in `.env` for the Cloud or Server profile:

   ```bash
   test -f .env || cp .env.example .env
   ```

   > **Single GPU:** set `HF_TOKEN` in `.env` only. Do not set `NVIDIA_API_KEY` or log in to `nvcr.io`. This recipe serves the LLM with vLLM, which downloads model weights from Hugging Face.

2. Log in to the NVIDIA NGC container registry (Server only. Skip for Cloud and Single GPU):

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Deploy the profile that matches your hardware:

   ```bash
   docker compose --profile multilingual-assistant up -d              # Cloud (no local GPU)
   docker compose --profile multilingual-assistant/server up -d  # Server (NIM, recommended for scaling)

   # One GPU (incl. DGX Spark and Jetson Thor). Download speech weights once, as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile multilingual-assistant/single-gpu up -d
   ```

   | Recipe profile | App service | Sidecars |
   | --- | --- | --- |
   | `multilingual-assistant` | `multilingual-assistant` | none (cloud NVCF) |
   | `multilingual-assistant/server` | `multilingual-assistant-server` | `nvidia-llm`, `nemotron-asr-streaming-multilingual`, `magpie-multilingual-tts-service` |
   | `multilingual-assistant/single-gpu` | `multilingual-assistant-single-gpu` | `nvidia-llm-vllm-lightning`, `nemo-speech-multilingual` |

4. Open the UI at `https://localhost:7860/`. Keep TLS enabled for browser UI testing. `PIPELINE_TLS=false` serves plain HTTP for headless performance and API testing. For plain-HTTP browser testing, see [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile you started with:

   ```bash
   docker compose --profile multilingual-assistant down              # Cloud (no local GPU)
   docker compose --profile multilingual-assistant/server down       # Server
   docker compose --profile multilingual-assistant/single-gpu down   # One GPU (incl. DGX Spark and Jetson Thor)
   ```

To run host-native without Docker, set `selection: multilingual-assistant` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

After deploying, validate the session language with the steps in [Testing](#testing).

## Customization

TTS voices and supported language codes are discovered at runtime by prewarming the configured TTS service. The UI language selector contains only locales supported by the selected ASR, TTS, and built-in LLM. Changing the LLM refreshes that compatible set. The selected session language is injected into the prompt and pins the ASR and the TTS voice for the whole connection. For Magpie and Chatterbox TTS language coverage, see [Configure TTS](../../../docs/how-to/configure-tts.md#supported-languages).

| Path | Role |
| --- | --- |
| `pipeline.py` | pipecat entry point, multilingual mode always on |
| `prompts.yaml` | multilingual prompt catalog (`multilingual_voice_assistant`) |
| `services.cloud.yaml` | cloud service endpoints and defaults |
| `services.local.yaml` | on-prem service endpoints (server / single GPU), registry default `nemotron-asr-streaming-multilingual` |
| `tools.py` | validates client-declared tool schemas into a `ToolsSchema` for the LLM context |
| `tool_handlers.py` | server-side RTVI forwarding handler for those client-executed tools |

### TTS text normalization

Before synthesis, the pipeline rewrites numbers, currency, percentages, units, dates, times, phone numbers, and emails into spoken words for English and French sessions. Other languages pass through unchanged. The transcript and chat history keep the original text. Set `TTS_TEXT_NORMALIZATION=false` in `.env` to turn it off. For details and the `*/single-gpu` grammar limits, refer to [Configure TTS](../../../docs/how-to/configure-tts.md#text-normalization).

### Client-executed tools

Tools demonstrate Pipecat's RTVI client-side function calling: the LLM calls a tool as normal, but the *browser* executes it and returns the result. Unlike a fixed server-side schema, the set of tools is declared **by the client at connect time** (Pattern B: dynamic discovery) rather than hardcoded on the server — different client builds can expose different capabilities without any server change.

| Tool | Why it runs on the client |
| --- | --- |
| `get_client_local_time` | Only the browser knows the user's own wall-clock time and IANA timezone. |
| `evaluate_math_expression` | Arithmetic runs inside a sandboxed Web Worker (`client/src/lib/mathWorkerClient.ts`) instead of as arbitrary code on the server. The worker disables every network-capable API (`fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `importScripts`, `RTCPeerConnection`, `navigator.sendBeacon`) and only accepts a whitelisted character set plus a whitelisted set of `Math` member names, so it can only ever evaluate plain arithmetic. |

Both are defined once, client-side, in `client/src/lib/clientTools.ts` (name, description, JSON-schema parameters, and the handler that implements them). The flow:

1. On connect, the client sends its tool declarations as `requestData.tools` (WebRTC: `webrtcRequestParams.requestData` on `client.connect(...)`, carried through to the `/api/offer` POST body as `request_data`/`requestData`; WebSocket: a `client_tools` query parameter on the `/api/ws` URL, since that transport has no HTTP request to hang `requestData` on).
2. `tools.py`'s `build_client_tools` validates that untrusted, client-supplied JSON (name pattern, description length, parameter shape, dedup, a cap on tool count) and turns it into a `ToolsSchema`. Only the name, description, and parameters reach the LLM. Invalid entries are dropped with a warning rather than reaching the LLM.
3. The LLM decides to call one of the declared tools. `NvidiaLLMService` broadcasts a function-call-in-progress frame before invoking the registered server handler (`tool_handlers.py`, registered per-session for each name the client declared).
4. `RTVIObserverParams.function_call_report_level` is set to `FULL` for each declared tool name, so the RTVI observer turns that frame into an `llm-function-call-in-progress` message carrying the function name and arguments.
5. The client SDK's `registerFunctionCallHandler(...)` (wired up in `client/src/App.tsx` from the same `clientTools.ts` list) reacts to that event, runs the tool locally, and replies with an `llm-function-call-result` message on its own — no custom protocol needed.
6. `RTVIProcessor` turns that reply directly into a result frame that the assistant context aggregator matches by `tool_call_id` and folds into the conversation.

The server-side handler in `tool_handlers.py` never computes a result itself — it only keeps the call open (`cancel_on_interruption=True`, the plain sync tool pattern) and applies its own `CLIENT_FUNCTION_TIMEOUT_SECS` (default 5s) timeout so a client that never answers can't stall the conversation. `cancel_on_interruption=False` (Pipecat's async-tool pattern, used elsewhere for genuinely long-running backend work) was tried and rejected here: it injects the final result as a `role="developer"` message, which Nemotron 3.5 Lightning doesn't recognize — it kept telling the user the tool was "still running" even with the real result sitting in context. Add a new client-executed tool entirely in `client/src/lib/clientTools.ts`; nothing on the server needs to change.

#### Perceivable Tools

A tool declaration accepts an optional boolean field named `perceivable`. Set it to `true` when running the tool is something the user sees or hears by itself, such as a robot that moves its head or plays a sound. Leave it out, or set it to `false`, for tools that only return data, such as a clock or a web search.

```json
{
  "name": "move_head",
  "description": "Turn the robot head left, right, up, or down.",
  "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
  "perceivable": true
}
```

The flag only affects recorded sessions (`MONITORING_ENABLED=true`). For a turn that calls a perceivable tool, the response latency stops when the server sends the call, instead of waiting for the first bot audio. Refer to [Per-Turn Metrics](../../../docs/how-to/enable-conversation-recording.md#per-turn-metrics). The server stores the flag with the session snapshot (`client_tools_perceivable`) and never forwards it to the LLM. A value that is not a boolean is ignored with a warning, and the tool stays available. There is no server-side list of perceivable tools: a client that does not send the flag gets a response latency equal to its voice latency. External clients, such as the Reachy Mini conversation app, send the field in the same `tools` list they already declare at connect time.

If a client-executed tool "does nothing," check two things: that the client actually declared it at connect time (see step 1 above), and the prompt catalog — the tool result must be summarized in natural spoken text, not read back verbatim (raw JSON read aloud by TTS sounds like a broken response).

### How it works

1. The user selects a locale compatible with the active ASR, TTS, and LLM in the UI (default `de-DE`) before connecting. A non-UI client can set the same thing directly by sending `asr_language_code` in the WebRTC offer's `request_data` (e.g. `{"asr_language_code": "fr"}`) — the [Reachy Mini conversation app](https://codefloe.com/fcollonval/reachy_mini_pipecat_app) integration does this since it has no UI of its own. A bare base code like `fr` is expanded against the prewarmed TTS voice catalog to the full locale (`fr-FR`) the TTS engine expects — critical for TTS voice/language selection, which otherwise silently keeps whatever voice was already selected.
2. The ASR and the TTS voice are pinned to that language when the connection starts. They do not change mid-session.
3. The fixed-session prompt addon instructs the LLM to reply only in that language, and the LLM returns plain spoken text (no JSON, labels, or metadata) that flows straight to TTS, the client transcript, and chat history.
4. `PerTurnReminderProcessor` re-states the "reply only in <language>" reminder on each user turn at request time only, so the reminder never pollutes stored history.

A non-UI client can likewise set the system prompt directly by sending `prompt_content` in `request_data` (the same field the UI uses for a custom, non-catalog prompt; e.g. `{"prompt_content": "You are a pirate. Always say Arr."}`). Omit it to keep the catalog default (`multilingual_voice_assistant`), or send `prompt_key` to select a different catalog entry by name instead.

### Memories about the person talking

When session recording is enabled (`MONITORING_ENABLED=true`), the client header shows a **Who's talking** menu. The selected person id is sent as `person_id` in the session config. The pipeline then renders the `person_memory_addon` prompt block with the person's name and up to 30 of their `active` memories. After the session, the `dream` job of the dreamer extracts new memories, which you review in the web UI. For setup, review, and privacy details, refer to [Remember people between conversations](../../../docs/how-to/enable-conversation-recording.md#remember-people-between-conversations).

### Voice identification

A client that computes speaker embeddings can identify who speaks in each turn by sending `speaker-update` messages over RTVI. The pipeline then tags each user message with the speaker for the LLM, attributes the turn, loads the memories of the recognized person, and offers a server-side `enroll_speaker` tool to remember a new voice. This also requires `MONITORING_ENABLED=true`. A client with a camera can also confirm the speaker by face (tier `verified`) and report who is in view with `presence-update` messages; the pipeline then names the people in view to the LLM and greets a recognized person who comes into view while nobody speaks. For details, refer to [Identify Speakers by Voice](../../../docs/how-to/enable-conversation-recording.md#identify-speakers-by-voice) and the [Voice ID protocol](../../../docs/voice-id-protocol.md).

### Switching the multilingual ASR model

**Parakeet 1.1B RNNT Multilingual** offers stronger multilingual recognition quality at higher latency (see [Model Selection Notes](#model-selection-notes)). To run it instead of the default Nemotron ASR Streaming Multilingual on-prem:

1. In [`examples_registry.yaml`](../../../examples_registry.yaml), under `multilingual-assistant`, set `defaults.asr: [parakeet-rnnt]`.
2. Redeploy with the recipe profile plus the Parakeet profile, scaling the Nemotron sidecar off (only one local ASR may bind port `50152`):

   ```bash
   # Server
   docker compose --profile multilingual-assistant/server \
     --profile parakeet-rnnt-asr up -d --scale nemotron-asr-streaming-multilingual=0
   ```

3. Switch back to Nemotron by reversing the registry edit and redeploying the stock on-prem recipe.

## Tips & best practices

### Model Selection Notes

Multilingual behavior depends on the ASR model, the LLM, and the selected TTS voice. Use the notes below when choosing a deployment profile or setting expectations for demo and validation runs.

| Component | Recommendation and trade-offs |
| --- | --- |
| Nemotron ASR Streaming Multilingual | Prefer this model when latency and throughput are the main constraints. It is faster in this pipeline, but recognition quality is currently weaker for a few languages. Since the session language is fixed, its pinned-language recognition is a good fit here. In noisy environments, it can occasionally emit an empty transcript for turns, so the user may need to repeat themselves. A good microphone and reduced background noise help. |
| Parakeet 1.1B RNNT Multilingual | Prefer this model when multilingual recognition quality matters more than raw latency. Hindi and Chinese recognition are generally better than Nemotron ASR in this setup. The trade-off is slower latency and throughput. It can also miss the first word of an utterance in some cases and may produce occasional false transcripts when the microphone is muted or no user speech is intended, so validate turn-start and silence handling for production. |
| Nemotron 3 Super LLM | **Recommended for multilingual.** Supports English, German, Spanish, French, Italian, Japanese, and Chinese. Stays more reliably in the fixed session language and delivers stronger conversation quality across that set, especially where Nemotron 3.5 Lightning is weak (for example Hindi). |
| Nemotron 3.5 Lightning LLM | Useful for lower latency, lower resource usage, and faster local experiments. It supports English, German, Spanish, French, Italian, and Japanese. Conversation quality is weaker in some languages (for example Hindi), and with reasoning disabled it can occasionally slip in foreign words on quantized builds, so the fixed-session prompt addon and the per-turn reminder both enforce a single language. Prefer Nemotron 3 Super when multilingual quality matters. |

### Testing

1. Start the app with the `multilingual-assistant` profile.
2. In Voice Settings, pick the session language (for example German, French, or Spanish), then connect.
3. Speak in the selected language and verify the bot responds in that same language.
4. Verify that:
   - the bot always responds in the selected session language, regardless of the language you speak
   - the transcript shows the clean spoken text
   - changing the language requires disconnecting, selecting a new language, and connecting again

### Troubleshooting

| Issue | Cause | What to check |
|-------|-------|---------------|
| Bot responds in the wrong language | LLM ignored the fixed session language | Confirm the fixed-session prompt addon and per-turn reminder name the selected language. Try the larger Nemotron 3 Super LLM |
| Bot slips in foreign words | Quantized small-model sampling artifacts | Lower the LLM `temperature` in `services.*.yaml`, or use a larger LLM |
| Session language is unavailable or startup is rejected | The selected locale is not supported by the active ASR, TTS, or built-in LLM | Select a locale shown in Voice Settings. For built-in LLM support, see [Configure LLM](../../../docs/how-to/configure-llm.md#multilingual-session-languages). |
| TTS uses the wrong voice or language | Selected session language is not supported by the active TTS service | Check the configured TTS service exposes that language code, or pick a supported language |
| TTS skips or misreads numbers or prices | Pipeline normalization covers only English and French; on `*/single-gpu`, the server grammars also cover only English and French | For other languages, prompt the LLM to write numbers as words. Refer to [Configure TTS](../../../docs/how-to/configure-tts.md#text-normalization) |
| No voices discovered at startup | TTS prewarm failed | For Cloud, confirm `NVIDIA_API_KEY` in `.env`. For Server, also confirm NGC login and TTS sidecar health with `docker compose ps`. For Single-GPU, confirm that the NeMo-Speech.cpp sidecar is healthy and `models/nemo-speech` contains the downloaded weights. |
| Bot does not respond to a turn (no transcript) | Nemotron ASR Multilingual can drop a turn in noisy environments | Speak again, reduce background noise, and use a good microphone. See [Configure ASR](../../../docs/how-to/configure-asr.md#choosing-a-multilingual-asr-model) |
| Weak or awkward replies in some languages (for example Hindi) | Nemotron 3.5 Lightning has weaker conversation quality in a few languages | Use Nemotron 3 Super for better multilingual quality. See [Configure LLM](../../../docs/how-to/configure-llm.md) |
| Port conflict on the ASR sidecar | Parakeet and Nemotron streaming both bind `50152` | Run only one local ASR. When opting into Parakeet, scale the Nemotron sidecar off (`--scale nemotron-asr-streaming-multilingual=0`) |
| The bot does not know who is talking or what it remembers | Monitoring is off, no person is selected, the person is archived, or no memory is `active` yet | Set `MONITORING_ENABLED=true`, pick a person in **Who's talking** before connecting, and approve their `proposed` memories in the review UI. The `dream` job runs only after the session ends and while no session is live |
| Random ASR text while silent | Parakeet RNNT noise sensitivity | Expected with the Parakeet opt-in. The default Nemotron ASR is less prone to this. Otherwise reduce room noise and use a good mic |

For ASR, LLM, and TTS model details and general failure modes, see [Configure ASR](../../../docs/how-to/configure-asr.md), [Configure TTS](../../../docs/how-to/configure-tts.md), [Configure LLM](../../../docs/how-to/configure-llm.md), and the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
