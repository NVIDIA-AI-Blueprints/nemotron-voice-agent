# VoiceClaw

VoiceClaw adds a realtime voice interface to a deployed agent. Clients use one
OpenAI Realtime-compatible WebSocket for audio, text, transcripts, assistant
speech, and application updates. The optional browser UI uses the same
connection.

The current example runs VoiceClaw as an independent service. A NemoClaw
deployment supplies the OpenShell workspace, sandbox, and Fabric agent that
receive delegated work. VoiceClaw binds to the Fabric runtime that NemoClaw
already provisioned; it does not create or stop that runtime.

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

By default, the realtime frontend model chooses between a conversational reply
and agent delegation, building a standalone goal for delegated work. The
NemoClaw container profile preserves that model-selected routing: greetings and direct
conversation stay in the frontend, while delegated work uses the validated,
standalone goal produced by the frontend. The frontend presents
acknowledgements and result speech. The Interaction Manager invokes the
configured agent and sends its rich Markdown result to the UI.

The browser never receives OpenShell credentials and never calls NemoClaw or
the deployed agent directly.

## Requirements

Run all commands from the repository root. You need Docker Compose v2.20 or
newer, a NemoClaw-deployed agent, and the target's applied OpenShell endpoint,
workspace, sandbox, Fabric agent name, runtime ID, adapter identity, and result
pointer.

The default local frontend also requires one supported NVIDIA GPU or DGX Spark
and an `HF_TOKEN` in the repository `.env`. Refer to the
[Getting Started guide](../../../docs/01-getting-started.md) for host and GPU
prerequisites.

## Deploy as a NemoClaw Container Service

The `nemoclaw-container-v1` image target implements the VoiceClaw side of the
NemoClaw [Container Service Contract v1](https://github.com/NVIDIA/NemoClaw/blob/bf93570b92c08230cee3816f382b93f97cd519e5/docs/design/container-contract.md),
pinned to the revision reviewed for this profile.
NemoClaw deploys it as an ordinary `kind: container` service with literal
environment values. There is no application-specific installer, generated
connection descriptor, or credential file. VoiceClaw does not install or manage
the agent.

This profile uses Fabric 0.5's public runtime command contract. A narrow
executor connects the OpenShell adapter to the pinned OpenShell SDK client.
NemoClaw owns runtime start and stop. VoiceClaw invokes only the supplied
runtime ID and verifies the returned agent, runtime, adapter, harness, and
request identities before accepting a result.

The profile permits ordered, non-durable delegated turns from one connected
VoiceClaw conversation to one pre-provisioned Fabric runtime. A different
VoiceClaw conversation cannot delegate through an already-bound runtime. The
profile also admits only one concurrent browser session. Fabric 0.5 does not
supply evidence that qualifies context continuity or isolation between
invocations, so each delegated goal remains self-contained. A disconnect,
process restart, ambiguous invocation, or runtime replacement does not provide
replay or recovery.

The profile sets `interaction.turn_routing_mode: model`. Delegation requires
the currently advertised `work.delegate` operation; an unavailable or consumed
backend rejects the request. The runtime sends the validated standalone goal
selected by the frontend. Argument, source-text, readiness, and availability
checks still apply; readiness here is local application readiness, not Fabric
agent attestation.

Copy the gateway URL, workspace, and sandbox ID from the operator's deployment.
This fragment uses trusted local plaintext mode:

```yaml
environment:
  VOICECLAW_PORT: "18790"
  VOICECLAW_GATEWAY_URL: "http://172.29.230.2:17681"
  VOICECLAW_WORKSPACE: "ACTUAL_WORKSPACE"
  VOICECLAW_SANDBOX_NAME: assistant
  VOICECLAW_SANDBOX_ID: "ACTUAL_SANDBOX_ID"
  VOICECLAW_AGENT_NAME: main
  VOICECLAW_FABRIC_RUNTIME_ID: "ACTUAL_RUNTIME_ID"
  VOICECLAW_FABRIC_ADAPTER_ID: "ACTUAL_ADAPTER_ID"
  VOICECLAW_FABRIC_HARNESS: "ACTUAL_HARNESS_ID"
  VOICECLAW_FABRIC_ADAPTER_KIND: "ACTUAL_ADAPTER_KIND"
  VOICECLAW_FABRIC_RESULT_POINTER: "/output/response"
  VOICECLAW_AUTH_MODE: none
  VOICECLAW_SPEECH_KEY: "REPLACE_AT_DEPLOY_TIME"
```

All `VOICECLAW_FABRIC_*` values must describe the same provisioned
`fabric_agent`. The generic container environment is literal and does not
discover or interpolate the runtime ID. Use the runtime ID retained for the
applied agent and the exact RunResult identity expected from its adapter. The
result pointer is adapter-specific. These Fabric bindings are a VoiceClaw
profile extension; the pinned generic container contract does not define or
synthesize them.

For HTTPS, set `VOICECLAW_AUTH_MODE: oidcBearer` and supply the application's
service token in `VOICECLAW_GATEWAY_TOKEN`. Literal environment values are
visible in configuration, NemoClaw plans and state, exports, and Docker
inspection. Replace placeholders only in private local configuration and never
commit or share the result. Refer to the [container profile](NEMOCLAW_CONTAINER_V1.md)
for validation rules, the image build, and qualification requirements.

The profile serves the packaged browser console at `GET /` and
`WS /v1/realtime` on `VOICECLAW_PORT`. Publish it only on host loopback or the
approved private deployment network. After NemoClaw apply returns, check local
application readiness on the Linux host:

For a nondefault port, the container service port, publication port,
`VOICECLAW_PORT`, health endpoint, and client URL must all use the same value.

```bash
curl --fail --max-time 15 http://127.0.0.1:18790/health
```

An empty `200` response confirms local application readiness. NemoClaw
deployment readiness and the managed image `HEALTHCHECK` use this endpoint.
`GET /livez` checks only that the VoiceClaw process is serving. Neither
endpoint invokes the agent. Fabric 0.5 has no read-only runtime status command,
so these responses do not prove that the Fabric runtime can answer, provider
credentials work, or a
voice turn completes. `GET /readyz` remains the optional selected-agent
attestation endpoint and returns `503` for this profile.

Earlier manual results used the retired descriptor-based revision and do not
qualify this revision; refer to the
[validation boundaries](NEMOCLAW_CONTAINER_V1.md#validation).

### Open the Installed Browser Console

Use the installed facade directly. The console, browser assets, and Realtime
WebSocket share port `18790`; no separate browser server is required.

1. For a remote installation, forward the Linux host's loopback listener from
   your local computer. Replace `operator@your-linux-host` with your SSH target:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes \
     -L 127.0.0.1:18790:127.0.0.1:18790 operator@your-linux-host
   ```

   Keep the SSH session running while you use the console. The NemoClaw
   `publication` setting must publish the application port on the remote host's
   loopback address first.

2. Open `http://localhost:18790/` locally and select **Connect**. The console
   derives its WebSocket endpoint from this address. Leave **Client secret**
   empty for the private container profile.
3. Wait for **Session ready**, then select **Start listening** and grant
   microphone access. Follow the negotiated turn behavior shown below the
   composer, or enter a typed request and press **Enter**.
4. Ask for a delegated agent task. Inspect the result in **Backend turns** and
   check audible result playback separately. You can send ordered follow-up
   work in the same connected conversation, but keep every delegated goal
   self-contained because native context continuity is not qualified.

Microphone capture requires a secure browser context, such as localhost.
Opening the remote host's plain HTTP address does not provide that context.
If microphone access is unavailable, typed requests remain available.

The result prompt asks for useful spoken material about the outcome, needed
clarification, or a missing requirement. A valid result can still contain
`speech: null`. VoiceClaw displays that result without queuing or inventing
result speech. Display success alone does not prove hosted speech-provider
access or audible playback. Do not retry an ambiguous invocation automatically.
Provision a replacement runtime before continuing because VoiceClaw cannot
recover or safely adopt the prior runtime after losing the invocation outcome.

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
   VOICECLAW_FABRIC_RUNTIME_ID=ACTUAL_RUNTIME_ID
   VOICECLAW_FABRIC_ADAPTER_ID=ACTUAL_ADAPTER_ID
   VOICECLAW_FABRIC_HARNESS=ACTUAL_HARNESS_ID
   VOICECLAW_FABRIC_ADAPTER_KIND=ACTUAL_ADAPTER_KIND
   VOICECLAW_FABRIC_RESULT_POINTER=/output/response
   VOICECLAW_FABRIC_AGENT=main
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

`interaction.turn_routing_mode` accepts `model` or `agent`. The default `model`
mode, also used by the container profile, lets the frontend select a direct reply or
delegation. Explicit `agent` mode requires `work.delegate` for every finalized
request and is intended for bounded qualification. Its private structured model
call is silent; acknowledgements and result speech follow separately after
admission.

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

`/livez` checks the process. Fabric 0.5 does not provide the read-only runtime
status needed for selected-agent attestation, so `/readyz` returns `503` for
this profile. Use an end-to-end delegated turn for agent qualification.

```bash
curl --fail http://127.0.0.1:7860/livez
```

Run a typed delegation check against the configured runtime:

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
  --context-continuity unqualified \
  --input-audio-wav /absolute/path/agent-question.wav \
  --transcript-contains balanced \
  --display-contains balanced \
  --speech-contains balanced \
  --output-audio-wav /absolute/path/agent-result.wav
```

For the developer stack, use port `7860`. The verifier's `unqualified` context
contract matches this Fabric 0.5 profile. Audio input and `--query` are mutually
exclusive; the recorded-input check sends no typed fallback.
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

Start qualification with a pre-provisioned runtime that has not been attached
to another VoiceClaw conversation. Successful ordered turns can use that same
runtime in the original conversation, but this does not qualify native context
continuity. Do not replay a recording after an uncertain invocation; replace
the runtime. Restarting the container cannot recover prior native history or
pending work.

### Troubleshoot Route Selection

In `agent` mode, `operation_unavailable` means the current session does not
advertise the required delegation operation. This rejects the finalized request
before dispatch. After an ambiguous invocation, replace the runtime rather than
replaying the request.

`routing_unavailable` can indicate a failed, incomplete, or canceled private
model route-selection response. This terminal failure stops before delegated
dispatch. Inspect the application container logs for
`Realtime route selection did not complete`.

The warning reports only allowlisted `status`, `type`, `reason`, and `code`
values. Missing or unknown values appear as `unrecognized`. Private goals,
response IDs, provider error messages, and model reasoning remain out of this
warning.

Frontend provider errors use fixed public messages with `llm_provider_error`
or `tts_provider_error` codes. A language-model error identifies failed frontend
generation. For a speech-provider error, inspect **Backend turns** for the text
result before repeating agent work. Provider error details remain private.

## Current Limitations

- The OpenShell profile supports one concurrent VoiceClaw session.
- The container OpenShell/Fabric profile supports ordered non-durable turns for
  one connected VoiceClaw conversation bound to one pre-provisioned runtime.
- Agent results are terminal responses. Backend event streaming, durable
  acceptance, replay, steering, confirmed remote cancellation, and reconnect
  recovery are not available yet.
- The active response-only Work/result lifecycle is process-local. SQLite does
  not make backend execution durable or recoverable after restart.

Refer to the repository [Troubleshooting guide](../../../docs/06-troubleshooting.md)
for model-service and browser-access problems.
