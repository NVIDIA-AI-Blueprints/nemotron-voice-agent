# Browser UI (`default` Profile)

A `default` deployment serves browser UI sessions over WebRTC or the UI's WebSocket transport, with RTVI, from the shared app (`src/server.py`). It also serves OpenAI Realtime sessions, which always use the `eval` settings but pair them with `gateway.yaml`. Do not use a `default` deployment for benchmark runs: use `tau.md`.

Complete `prerequisites.md` first, then set `P=frontend-delegation-hermes-agent/single-gpu`.

## What a Browser Session Runs

A browser session is the former FDH demo on the shared UI transport (`adapters/pipecat/ui_session.py`, `ui_pipeline.py`):

- delegation profile `profiles/pipecat_demo.yaml`: the greeting "Hi! How can I help you today?" (off with `ENABLE_WELCOME_MESSAGE=false`), the demo tools `get_order` and `cancel_order` run in the server, and `FDH_BACKEND_DELAY_S` seconds (5 in Compose) before each backend run, so you can try status questions and steering while the backend works;
- Pipecat settings `adapters/pipecat.yaml`: Pipecat's defaults (Silero VAD `stop_secs` 0.2, `audio_idle_timeout` 1.0 s, `user_speech_timeout` 0.6 s, Smart Turn off);
- gateway configuration `gateway.yaml` in `fdh-hermes`.

The pipeline is `create_transport` → VAD → STT → user turn → `FdhTurnBridge` → TTS → `OutputTagger` → `UiOutputGate` → transport output, with `enable_rtvi=True`. The turn manager, delegation, and backend link are the same objects that a Realtime session uses.

## Start and Check

```bash
docker compose --profile $P up -d --build
curl -fk https://localhost:7860/health                         # {"status":"ok"}
docker compose --profile $P exec fdh-hermes curl -s http://localhost:8790/health; echo
docker compose --profile $P logs fdh-hermes | grep 'FDH gateway'   # config gateway.yaml, source <hash>
```

After the first browser session, the app log shows `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=default)`, `Pipecat adapter settings`, and `browser UI session started (profile=default, adapter=pipecat)`:

```bash
docker compose --profile $P logs frontend-delegation-hermes-agent-single-gpu | grep -E 'ready \(adapter|Pipecat adapter settings|browser UI session'
```

## Use the Browser UI

Open `https://<host>:7860`. Use the host's IP address, not a corporate host name that HSTS forces through a proxy. Accept the self-signed certificate and allow the microphone. From a laptop, you can instead tunnel with `ssh -N -L 7860:localhost:7860 <user>@<host>` and open `https://localhost:7860`.

The recipe pins `EXAMPLE_SELECTION=frontend-delegation-hermes-agent`. The UI controls work as follows:

- **LLM** selects the frontend model and base URL. The backend (Hermes) model stays the gateway's.
- **ASR**, **TTS**, and the voice select the speech endpoints from this example's catalogs. A voice change applies from the next sentence on.
- **Prompt** shows the informational `frontend` prompt of the example's `prompts.yaml`. Edits are not applied. The real frontend prompt is `config/prompts.yaml` (`frontend_system`).
- **Tools** lists `get_order` and `cancel_order` (`tools.yaml`).

The demo tools know order `5512` (shipped) and `5513` (processing). `cancel_order` works on `5513`, and any other number is "not found." The following table lists what to say and what to expect:

| You Say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 s plus the model time (new session) |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both |
| "Okay sure" | No reply (acknowledgement) |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same Hermes session |
| Talk over an answer | The answer stops, and the backend learns what you heard (a delivery note) |

## Limits

- Browser sessions share one cap of 4 sessions (`FDH_MAX_SESSIONS` in Compose). A refused session ends, and the app log shows `browser session refused`. The cap is separate from the Realtime cap.
- An eval deployment (`FDH_PROFILE=eval`) refuses every browser session. The app log shows `runs with FDH_PROFILE=eval`.
- Each new frontend LLM and speech selection builds its own runtime, which warms up once on its first session.
