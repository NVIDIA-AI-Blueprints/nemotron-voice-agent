# Browser UI (`default` Profile)

A `default` deployment serves browser UI sessions over WebRTC or the UI's WebSocket transport, with RTVI, from the shared app (`src/server.py`). It also serves OpenAI Realtime sessions, which always use the `eval` settings but pair them with `gateway.yaml`. Do not use a `default` deployment for benchmark runs: use `benchmarking.md`.

Complete `prerequisites.md` first, then set `P=frontend-delegation-agent/single-gpu`.

## What a Browser Session Runs

A browser session is the former FDA demo on the shared UI transport (`adapters/pipecat/ui_session.py`, `ui_pipeline.py`):

- delegation profile `profiles/pipecat_demo.yaml`: the greeting "Hi! How can I help you today?" (off with `ENABLE_WELCOME_MESSAGE=false`), the demo tools `get_order` and `cancel_order` run in the server, and `FDA_BACKEND_DELAY_S` seconds (5 in Compose) before each backend run, so you can try status questions and steering while the backend works;
- Pipecat settings `adapters/pipecat.yaml`: Pipecat's defaults (Silero VAD `stop_secs` 0.2, `audio_idle_timeout` 1.0 s, `user_speech_timeout` 0.6 s, Smart Turn off);
- gateway configuration `gateway.yaml`, in the Hermes runtime (the OpenShell sandbox by default) or, with the LLM backend, in the voice process.

The pipeline is `create_transport` → VAD → STT → user turn → `FdaTurnBridge` → TTS → `OutputTagger` → `UiOutputGate` → transport output, with `enable_rtvi=True`. The turn manager, delegation, and backend link are the same objects that a Realtime session uses.

## Start and Check

```bash
docker compose --profile $P up -d --build
curl -fk https://localhost:7860/health                         # {"status":"ok"}
docker compose --profile $P exec fda-hermes \
  curl -s -H 'Host: default--fda-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
docker compose --profile $P logs fda-hermes | grep 'gateway config'   # controller: gateway config gateway.yaml
docker logs "$(docker ps --filter ancestor=nemotron-voice-agent-fda:local --format '{{.Names}}' | head -1)" 2>&1 \
  | grep 'FDA gateway'                                               # config gateway.yaml, source <hash>
```

For the plain runtime, add `-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml` to every Compose command. Then check health with `exec fda-hermes curl -s http://localhost:8790/health` and read `logs fda-hermes | grep 'FDA gateway'`.

For the LLM backend, add `-f docker-compose.yml -f docker/docker-compose.fda-llm.yaml` to every Compose command and skip the `fda-hermes` lines (the service does not run):

```bash
L="-f docker-compose.yml -f docker/docker-compose.fda-llm.yaml"
docker compose $L --profile $P up -d --build
curl -fk https://localhost:7860/health                         # {"status":"ok"}
docker compose $L --profile $P exec frontend-delegation-agent-single-gpu \
  printenv FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG      # llm, in_process, gateway.yaml
```

After the first browser session (with the LLM backend, pass `$L` to the command below), the app log shows `frontend-delegation-agent ready (adapter=pipecat, profile=default)`, `Pipecat adapter settings`, and `browser UI session started (profile=default, adapter=pipecat)`:

```bash
docker compose --profile $P logs frontend-delegation-agent-single-gpu | grep -E 'ready \(adapter|Pipecat adapter settings|browser UI session'
```

## Use the Browser UI

Open `https://<host>:7860`. Use the host's IP address, not a corporate host name that HSTS forces through a proxy. Accept the self-signed certificate and allow the microphone. From a laptop, you can instead tunnel with `ssh -N -L 7860:localhost:7860 <user>@<host>` and open `https://localhost:7860`.

The recipe pins `EXAMPLE_SELECTION=frontend-delegation-agent`. The UI controls work as follows:

- **LLM** selects the frontend model and base URL. The backend model (Hermes or the LLM backend) stays the gateway's.
- **ASR**, **TTS**, and the voice select the speech endpoints from this example's catalogs. A voice change applies from the next sentence on.
- **Prompt** shows the informational `frontend` prompt of the example's `prompts.yaml`. Edits are not applied. The real frontend prompt is `config/prompts/<profile>/frontend.yaml` (`frontend_system`); a `default` deployment uses the `generic` profile unless `FDA_PROMPT_PROFILE` names another.
- **Tools** lists `get_order` and `cancel_order` (`tools.yaml`).

The demo tools know order `5512` (shipped) and `5513` (processing). `cancel_order` works on `5513`, and any other number is "not found." The following table lists what to say and what to expect:

| You Say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 s plus the model time (new session) |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both |
| "Okay sure" | No reply (acknowledgement) |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same backend session |
| Talk over an answer | The answer stops, and the backend learns what you heard (a delivery note) |

## Limits

- Browser sessions share one cap of 4 sessions (`FDA_MAX_SESSIONS` in Compose). A refused session ends, and the app log shows `browser session refused`. The cap is separate from the Realtime cap.
- An eval deployment (`FDA_PROFILE=eval`) refuses every browser session. The app log shows `runs with FDA_PROFILE=eval`.
- Each new frontend LLM and speech selection builds its own runtime, which warms up once on its first session.
