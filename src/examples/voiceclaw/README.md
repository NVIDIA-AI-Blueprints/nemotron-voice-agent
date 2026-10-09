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

## Install with NemoClaw Container v1

The `nemoclaw-container-v1` image target implements the accepted
`voiceclaw.nemoclaw.container.v1` installer contract. NemoClaw declares and
manages an ordinary application container, delivers protected inputs, and binds
it to an existing OpenShell sandbox. VoiceClaw does not install or manage the
agent.

This profile preserves the upstream OpenShell/Fabric adapter, request codec,
result validation, and fresh-target limits. A narrow installer executor supplies
the protected descriptor and pinned SDK transport. It does not replace the
default image supervisor, developer Compose recipe, or cloud configuration.

Use the following nonsecret environment inputs in the application declaration:

```text
VOICECLAW_RUNTIME_PROFILE=nemoclaw-container-v1
VOICECLAW_INSTALL_CONTRACT=voiceclaw.nemoclaw.container.v1
VOICECLAW_AGENT_CONNECTION_FILE=/var/lib/voiceclaw/config/agent-connection.json
VOICECLAW_SPEECH_CREDENTIAL_FILE=/var/lib/voiceclaw/credentials/speech
VOICECLAW_STATE_PATH=/var/lib/voiceclaw/state/state.db
```

NemoClaw supplies the descriptor and credential files before startup. Do not
place credentials in YAML, Docker environment values, or command arguments.
Refer to the [installer contract](NEMOCLAW_CONTAINER_V1.md) for the descriptor,
image build, ownership, transport, and validation requirements.

The installer profile serves the packaged browser console at `GET /` and
`WS /v1/realtime` on port `18790`. Publish it only on host loopback or the
approved private deployment network. After NemoClaw apply returns, check local
installation readiness on the Linux host:

```bash
curl --fail --max-time 15 http://127.0.0.1:18790/readyz
```

An empty `200` response confirms protected inputs, initialized local components,
and the bundled frontend listener. It does not prove native Fabric health,
provider credentials, model responses, or a complete voice turn. Agent checks
and invocations retain upstream validation; unsupported native health is not
reported as healthy. Provision a fresh agent target for each delegated turn.

The earlier installer revision passed a manual ARM64 Station installation and
local readiness check. That result does not qualify this upstream adaptation;
refer to the [evidence boundaries](NEMOCLAW_CONTAINER_V1.md#validation-and-evidence).

### Open the Installed Browser Console

Use the installed facade directly. The console, browser assets, and Realtime
WebSocket share port `18790`; no separate browser server is required.

1. For a remote installation, forward the Linux host's loopback listener from
   your local computer. Replace `operator@your-linux-host` with your SSH target:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes \
     -L 127.0.0.1:18790:127.0.0.1:18790 operator@your-linux-host
   ```

   Keep the SSH session running while you use the console. The installer must
   publish the application port on the remote host's loopback address first.

2. Open `http://localhost:18790/` locally and select **Connect**. The console
   derives its WebSocket endpoint from this address. Leave **Client secret**
   empty for the private installer profile.
3. Wait for **Session ready**, then select **Start listening** and grant
   microphone access. Follow the negotiated turn behavior shown below the
   composer, or enter a typed request and press **Enter**.
4. Ask for one delegated agent task against a fresh target. Inspect the result
   in **Backend turns** and check audible result playback separately.

Microphone capture requires a secure browser context, such as localhost.
Opening the remote host's plain HTTP address does not provide that context.
If microphone access is unavailable, typed requests remain available.

A valid result can contain `speech: null`. VoiceClaw displays that result
without queuing result speech. Display success alone does not prove hosted
speech-provider access or audible playback. A successful or ambiguous delegated
invocation consumes the target; provision a fresh target before another one.

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

Run a typed delegation check against a fresh target:

```bash
uv run --project src/examples/voiceclaw --extra server --extra openshell \
  python src/examples/voiceclaw/scripts/verify-realtime-delegation.py \
  --url 'ws://127.0.0.1:7860/v1/realtime?model=nvidia%2Fvoiceclaw' \
  --allow-loopback-ws \
  --query 'Compare an AVL tree with an unbalanced binary search tree for a read-heavy workload.' \
  --display-contains AVL
```

### Verify a Recorded Voice Turn

To check recorded voice input, agent delegation, and generated result audio,
prepare a mono WAV file with one continuous spoken request. Use 16-bit
pulse-code modulation (PCM16), a 24 kHz sample rate, and a duration greater
than zero and at most 60 seconds. Avoid long pauses and trim trailing silence;
the verifier rejects early or multiple committed turns.

For example, record this request:

> Ask the agent to compare a balanced search tree with an unbalanced tree for a
> read-heavy workload. Include the word balanced in your answer.

After NemoClaw apply exits, run the following command against the installed
listener on the Linux host or through its loopback SSH forward. Replace the
input and output paths with your recording and a new output filename. The
output directory must already exist.

```bash
uv run --project src/examples/voiceclaw --extra server --extra openshell \
  python src/examples/voiceclaw/scripts/verify-realtime-delegation.py \
  --url 'ws://127.0.0.1:18790/v1/realtime?model=nvidia%2Fvoiceclaw' \
  --allow-loopback-ws \
  --input-audio-wav /absolute/path/agent-question.wav \
  --transcript-contains balanced \
  --display-contains balanced \
  --speech-contains balanced \
  --output-audio-wav /absolute/path/agent-result.wav
```

For the developer stack, use port `7860` in the URL. Audio input and `--query`
are mutually exclusive; the recorded-input check sends no typed fallback.
The verifier sends paced 20 ms audio chunks and follows the session's
negotiated automatic or manual turn behavior. It preserves the cascade's
startup turn-detection mode.

`--transcript-contains` requires a case-insensitive marker in the final input
transcript correlated to the committed audio item. Repeat it to check multiple
markers. `--display-contains` and `--speech-contains` check the result display
and generated result-delivery transcript separately.

`--output-audio-wav` requires a successful result with nonempty,
backend-authored result-delivery audio. A display-only `speech: null` result
does not satisfy that check. After the lifecycle and acknowledgement checks
pass, the verifier creates a new mode `0600` WAV containing only correlated
result audio, bounded to 120 seconds. It does not overwrite an existing file.

The saved WAV records generated result audio. The verifier reports
`generated_not_played` and zero playback; its playback acknowledgements do not
prove audible delivery. Listen to the saved result separately before reporting
that you heard the reply.

The current response-only target is consumed after a successful or ambiguous
invocation. Provision a fresh target before either check, including after a
failed verification following dispatch. Do not replay the recording against
the consumed target.

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
