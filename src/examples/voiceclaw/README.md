# VoiceClaw

VoiceClaw adds a realtime voice interface to a deployed agent. Clients use one
OpenAI Realtime-compatible WebSocket for audio, text, transcripts, assistant
speech, and application updates. The optional browser UI uses the same
connection.

The current example runs VoiceClaw as an independent service. A NemoClaw
deployment supplies the OpenShell workspace, sandbox, and Fabric agent that
receive delegated work.

## Architecture

```text
browser or native client
    OpenAI Realtime-compatible WebSocket
                    |
                    v
VoiceClaw facade and Interaction Manager
    routing, local request state, speech queue,
    display projection, and playback tracking
          |                              |
          v                              v
realtime frontend                 OpenShell adapter
    bundled NVA cascade               Fabric invocation
    or external Realtime API              |
          |                               v
          v                         deployed agent
    LLM / ASR / TTS
```

The realtime frontend handles greetings and short conversational turns. For
agent work, it creates a complete goal from the relevant conversation. The
Interaction Manager invokes the configured agent, sends rich Markdown to the
UI, and queues concise result material for conversational speech through the
frontend model.

The browser never receives OpenShell credentials and never calls NemoClaw or
the deployed agent directly.

## Requirements

Run all commands from the repository root. You need Docker Compose v2.20 or
newer, a NemoClaw-deployed agent, and the target's applied OpenShell endpoint,
workspace, sandbox, and Fabric agent name.

The default local frontend also requires one supported NVIDIA GPU or DGX Spark
and an `HF_TOKEN` in the repository `.env`. Refer to the
[Getting Started guide](../../../docs/01-getting-started.md) for host and GPU
prerequisites.

## Run the Developer Stack

1. Create the environment file and operator directory:

   ```bash
   test -f src/examples/voiceclaw/.env || \
     cp src/examples/voiceclaw/.env.example src/examples/voiceclaw/.env
   chmod 0600 src/examples/voiceclaw/.env

   voiceclaw_file_gid="$(id -g)"
   install -d -m 0710 src/examples/voiceclaw/operator
   chgrp "$voiceclaw_file_gid" src/examples/voiceclaw/operator
   ```

2. Set the applied NemoClaw target in `src/examples/voiceclaw/.env`:

   ```dotenv
   VOICECLAW_FILE_GID=1000
   VOICECLAW_OPENSHELL_ENDPOINT=http://127.0.0.1:17723
   VOICECLAW_OPENSHELL_WORKSPACE=nc-example
   VOICECLAW_OPENSHELL_SANDBOX=assistant
   VOICECLAW_FABRIC_ADAPTER_ID=nvidia.fabric.openclaw
   VOICECLAW_FABRIC_AGENT=main
   VOICECLAW_NATIVE_AGENT=main
   ```

   Replace `1000` with `id -g`. Use the exact endpoint and workspace returned
   by NemoClaw. The loopback developer configuration requires no OpenShell client secret.

3. Start the local large language model (LLM), automatic speech recognition
   (ASR), and text-to-speech (TTS) services:

   ```bash
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile generic-assistant/single-gpu up -d \
     nvidia-llm-vllm-lightning nemo-speech
   ```

4. Build and start VoiceClaw:

   ```bash
   VOICECLAW_FILE_GID="$(id -g)" docker compose \
     --env-file src/examples/voiceclaw/.env \
     -f src/examples/voiceclaw/docker-compose.yml \
     up --build -d
   ```

5. Open `http://127.0.0.1:7860/`. The WebSocket endpoint is
   `ws://127.0.0.1:7860/v1/realtime?model=nvidia%2Fvoiceclaw`.

The Compose recipe starts the optional UI. Remove `--ui` from its command for
a headless service.

Stop VoiceClaw with:

```bash
VOICECLAW_FILE_GID="$(id -g)" docker compose \
  --env-file src/examples/voiceclaw/.env \
  -f src/examples/voiceclaw/docker-compose.yml down
```

## Configuration

VoiceClaw reads one YAML file. Paths in this table are relative to
`src/examples/voiceclaw/`.

| File | Use |
| --- | --- |
| `src/voiceclaw/resources/voiceclaw.example.yaml` | Local NVA cascade and loopback OpenShell |
| `src/voiceclaw/resources/voiceclaw.container.yaml` | Hosted model endpoints and authenticated OpenShell |

Copy a template to an operator-owned path and set
`VOICECLAW_CONFIG_FILE_HOST` in `.env`. Configure the listener under
`server`, model services under `frontend_profiles`, the deployed target
under `backend_profiles`, and queue and routing bounds under `interaction`.
Use `kind: openai_realtime` to replace the bundled cascade with an external
Realtime endpoint.

For remote OpenShell, use TLS, `client_credentials`, and an operator-owned
credential file or environment reference. Do not store secret values directly
in YAML or `.env`.

For a shared browser listener, enable TLS and ephemeral client authentication.
Issue a short-lived browser secret with:

```bash
uv run --project src/examples/voiceclaw voiceclaw-client-secret \
  --master-key-file /absolute/path/to/voiceclaw-public-master \
  --output /absolute/path/to/voiceclaw-client-secret
```

## Run from the Python Package

Package mode uses an independently supervised Realtime frontend. The image
supervisor starts the bundled NVA cascade.

```bash
python3.12 -m venv .venv-voiceclaw
. .venv-voiceclaw/bin/activate
python -m pip install './src/examples/voiceclaw[server,openshell]'
voiceclaw --config /absolute/path/to/voiceclaw.yaml --check-config
voiceclaw --config /absolute/path/to/voiceclaw.yaml --ui
```

Omit `--ui` for a headless facade.

## Verify the Deployment

`/livez` checks the process. `/readyz` checks the configured agent binding
without invoking the agent.

```bash
curl --fail http://127.0.0.1:7860/livez
curl --fail http://127.0.0.1:7860/readyz
```

Run one end-to-end turn against a fresh target:

```bash
uv run --project src/examples/voiceclaw --extra server --extra openshell \
  python src/examples/voiceclaw/scripts/verify-realtime-delegation.py \
  --url 'ws://127.0.0.1:7860/v1/realtime?model=nvidia%2Fvoiceclaw' \
  --allow-loopback-ws \
  --query 'Compare an AVL tree with an unbalanced binary search tree for a read-heavy workload.' \
  --display-contains AVL
```

The current response-only target is consumed after a successful or ambiguous
invocation. Provision a fresh target before repeating the test.

## Current Limitations

- The OpenShell profile supports one concurrent VoiceClaw session.
- The R1a integration supports one non-durable delegated turn for each fresh
  target.
- Agent results are terminal responses. Backend event streaming, durable
  acceptance, replay, steering, cancellation, and qualified conversation
  continuity are not available yet.
- SQLite stores local request and presentation state, not durable backend work.

Refer to the repository [Troubleshooting guide](../../../docs/06-troubleshooting.md)
for model-service and browser-access problems.
