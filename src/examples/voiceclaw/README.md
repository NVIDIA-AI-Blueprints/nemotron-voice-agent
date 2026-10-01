# VoiceClaw

VoiceClaw is a Python package and containerized realtime voice frontend. A
client uses one OpenAI Realtime-compatible WebSocket for audio, text,
transcripts, and application projections. VoiceClaw owns the live media and
presentation loop. In the target managed NemoClaw profile, NemoClaw owns
authoritative agent sessions and durable Work. The developer adapter is
response-only and does not claim that durability.

The package is backend- and frontend-adapter based. The selected realtime
frontend can be the bundled NVA cascade or an external OpenAI
Realtime-compatible service. Model endpoints are registered in YAML; VoiceClaw
does not deploy LLM, ASR, TTS, or speech-to-speech models.

## Architecture

~~~text
Browser or native client
  audio + text + standard Realtime events
              |
              v
VoiceClaw Realtime facade
  session binding, client auth, optional UI
              |
              v
Interaction Manager
  capability-gated routing, local correlation, speech queue,
  display/speech separation, delivery receipts, SQLite projection
       |                                      |
       v                                      v
Realtime frontend                      Backend adapter
  bundled NVA cascade                    NemoClaw selected agent
  or external Realtime API               or another configured backend
       |
       v
ASR / frontend LLM / TTS endpoints
~~~

Rich `display` content is rendered as Markdown by the UI. Bounded `speech`
material is placed in the frontend model's dynamic context so the model can
deliver a natural conversational update; it is never sent directly to TTS.
For the NemoClaw profile, NemoClaw remains authoritative for Work IDs, state,
events, replay, cancellation, and agent credentials. VoiceClaw stores only its
command outbox, materialized projections, delivery state, and presentation
receipts.

## One VoiceClaw YAML

VoiceClaw reads one strict YAML file. For developer runs, start from
`src/examples/voiceclaw/src/voiceclaw/resources/voiceclaw.example.yaml` and select:

| Area | Configures |
| --- | --- |
| `server` | Listener, port, trust boundary, optional client authentication |
| `frontend_profiles` | Bundled NVA cascade or external Realtime endpoint |
| `frontend_profiles.<name>.services.*` | Provider, URL, model, voice, timeouts, provider options |
| `model_contracts` | Stable frontend role and response contracts |
| `interaction_profiles` | Backend capabilities and tool shapes/copy |
| `backend_profiles` | Adapter, endpoint policy, credential reference, deadlines |
| `interaction` | Queue, projection, retention, and routing bounds |
| `state` | Local SQLite projection path |

Unknown fields, duplicate YAML keys, unsupported combinations, inline secret
options, and unresolved references fail before serving. Credentials are
references, not values. In a managed deployment they must be protected files;
developer-only environment references remain available for local testing.

Validate a configuration without opening a listener:

~~~bash
uv sync --project src/examples/voiceclaw --extra server --frozen
uv run --project src/examples/voiceclaw --extra server \
  voiceclaw \
  --config "$PWD/src/examples/voiceclaw/src/voiceclaw/resources/voiceclaw.example.yaml" \
  --check-config
~~~

The packaged example is developer-only: its loopback listeners and endpoints
must never be copied unchanged into the managed volume. Local model services
can run in separate containers or on other hosts; only their bridge-reachable
URLs belong in managed VoiceClaw YAML. Any GPU belongs to those model services.
The managed VoiceClaw container itself requests no GPU and uses private IPC.

## Managed NemoClaw service

The managed NemoClaw profile selects one VoiceClaw integration, one
sandbox agent, and one preloaded immutable image. Build the dedicated non-root
target from a clean repository root:

~~~bash
test -z "$(git status --porcelain --untracked-files=all)"
voiceclaw_source_revision="$(git rev-parse HEAD)"
voiceclaw_version="0.1.0"
voiceclaw_platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
voiceclaw_artifact_dir="/tmp/voiceclaw-artifacts-${voiceclaw_source_revision}"
voiceclaw_wheel="${voiceclaw_artifact_dir}/nemotron_voiceclaw-${voiceclaw_version}-py3-none-any.whl"
voiceclaw_wheel_evidence="${voiceclaw_artifact_dir}/wheel-evidence.json"
test "${voiceclaw_platform%%/*}" = linux
install -d -m 0700 "${voiceclaw_artifact_dir}"
uv build src/examples/voiceclaw --wheel --clear --out-dir "${voiceclaw_artifact_dir}"
python3 src/examples/voiceclaw/scripts/verify-wheel.py \
  "${voiceclaw_wheel}" \
  --expect-version "${voiceclaw_version}" \
  --expect-revision "${voiceclaw_source_revision}" \
  --repository-root "$PWD" > "${voiceclaw_wheel_evidence}"
docker build \
  --target nemoclaw-managed \
  --platform "${voiceclaw_platform}" \
  --build-arg "VOICECLAW_VERSION=${voiceclaw_version}" \
  --build-arg "VOICECLAW_SOURCE_REVISION=${voiceclaw_source_revision}" \
  -f src/examples/voiceclaw/Dockerfile \
  -t "voiceclaw-nemoclaw:${voiceclaw_source_revision}" \
  .

uv run --project src/examples/voiceclaw --extra server --frozen \
  python src/examples/voiceclaw/scripts/verify-image.py \
  "voiceclaw-nemoclaw:${voiceclaw_source_revision}" \
  --runtime-profile nemoclaw-managed \
  --expect-version "${voiceclaw_version}" \
  --expect-revision "${voiceclaw_source_revision}" \
  --expect-source https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent \
  --expect-architecture "${voiceclaw_platform#*/}" \
  --repository-root "$PWD" \
  --wheel "${voiceclaw_wheel}" \
  --wheel-evidence "${voiceclaw_wheel_evidence}" \
  --smoke

docker image inspect --format '{{.Id}}' \
  "voiceclaw-nemoclaw:${voiceclaw_source_revision}"
~~~

Use the returned `sha256:...` image ID in NemoClaw; do not use the mutable tag.
The deterministic image smoke proves the managed process identity, immutable
mode marker, fresh anonymous-volume layout, package identity, liveness, and the
expected pre-projection `503` readiness state. It does not prove a supplied
NemoClaw volume or provider-backed readiness.

The managed target has this fixed contract:

| Item | Value |
| --- | --- |
| Entrypoint | `/usr/local/bin/voiceclaw-runtime` |
| Command | `serve` |
| Process identity | `65532:65532` |
| Port | `18790/tcp` |
| Volume | `/var/lib/voiceclaw` |
| Installer config | `/var/lib/voiceclaw/runtime/config.json` |
| NVIDIA credential | `/var/lib/voiceclaw/credentials/speech` |
| Selected-agent credential | `/var/lib/voiceclaw/credentials/agent` |
| Local projection | `/var/lib/voiceclaw/runtime/state.db` |
| Public client API | `WS /v1/realtime` |
| UI | Disabled in this target |

This setup requires a NemoClaw build that implements the `voiceclaw` managed
service and integration kinds. A build without those kinds rejects this
configuration during schema validation.

NemoClaw configuration:

~~~yaml
spec:
  gateway:
    management: managed
    engine: unix:///var/run/docker.sock
    networkCIDR: 172.20.0.0/24

  services:
    voice-server:
      kind: voiceclaw
      image: "sha256:<64-lowercase-hex-local-image-id>"
      imagePullPolicy: Never
      speech:
        provider: nvidia
        credential:
          env: NVIDIA_API_KEY
      serving:
        port: 18790
        startupTimeoutSeconds: 180

  integrations:
    voice:
      kind: voiceclaw
      serviceRef: voice-server

  sandboxes:
    - name: assistant
      agent:
        name: main
        integrationRefs: [voice]
~~~

`speech.credential.env` names a protected NemoClaw input; its value is projected
to the credential file and is not placed in the Docker runtime specification.
The managed projection exposes no separate frontend-model credential, so this
profile uses that NVIDIA credential for its pinned NVIDIA
LLM, ASR, and TTS services. This is a compatibility assumption for the fixed
profile, not a general credential-routing contract. The nine-field projection
is:

~~~json
{
  "integration": "voice",
  "sandbox": "assistant",
  "agent": "main",
  "port": 18790,
  "speechProvider": "nvidia",
  "speechCredentialPath": "/var/lib/voiceclaw/credentials/speech",
  "agentCredentialPath": "/var/lib/voiceclaw/credentials/agent",
  "agentEndpoint": "http://<managed-gateway-ip>:<gateway-port>",
  "agentRouteHost": "<owned-route>.localhost:<gateway-port>"
}
~~~

NemoClaw writes `runtime/` and `credentials/` as `0700` and their files as
`0600`, owned by `65532:65532`. VoiceClaw rejects duplicate/unknown fields,
symlinks, ACLs, wrong ownership/modes, unsafe addresses, mismatched route ports,
and changes observed during a stable double-read. The agent credential is
reopened for every readiness check and backend admission, so rotation does not
require an image restart for new admissions. An active response-only session
keeps its already-issued session grant; if rotation revokes that grant, the
session may remain unusable until its validated expiry. Configuration or
speech-credential changes drain the active client and rebuild the private
frontend.

### Health and readiness

- PID 1 binds `0.0.0.0:18790` before projection exists.
- `GET /livez` is always empty `200` while the process is alive.
- `GET /readyz` is empty `503` until projection, a non-generative Realtime
  bootstrap to the bundled frontend, selected-agent access, and one-client
  admission are ready; success is empty `200`. The bootstrap may perform the
  frontend's provider prewarm, but it sends no user prompt and creates no
  backend Work.
- The bundled frontend uses a fixed 180-second startup budget. The nine-field
  projection does not support a different `startupTimeoutSeconds` value.
- Readiness uses a scoped non-generative `GET /healthz` with `Host` and
  `X-NemoClaw-Authorization`; it never creates a backend session or sends a
  probe question.
- Managed mode accepts one public Realtime client. The response-only agent
  ingress supports one live session. VoiceClaw rejects client or `argv`
  overrides for configuration, port, authentication, and UI settings.

The nine-field installer projection contains no public-client credential. The
managed listener must stay on NemoClaw's private service network and must not be
published directly to an untrusted LAN or the Internet. The one-UID
installer contract also places the supervisor and bundled frontend in one
container trust domain; `0600` prevents host peers from reading projected
credentials but is not process isolation between those two processes.

### Capability Boundary

The NemoClaw ingress provides one response-only NDJSON exchange per
temporary backend session. VoiceClaw exposes only `work.delegate`, keeps the
exchange asynchronous from the conversation, streams rich display content to
the UI, and queues only the bounded speech field for model-mediated delivery.
It does not advertise durable Work, events, replay, reconnect, cancellation,
steering, or presentation acknowledgement.

NemoClaw remains authoritative for the selected target and durable Work.
VoiceClaw stores only local correlation, delivery, presentation, and materialized
projection state. Agent-specific planning never enters this adapter.

## Developer stack and optional UI

The developer Compose file is intentionally separate from the managed service.
It enables the UI, host networking, and the non-durable response-only
NemoClaw compatibility adapter. Do not use it as evidence for the managed
contract.

### 1. Start optional local models

Put `HF_TOKEN` in the repository-root `.env`, then run:

~~~bash
bash scripts/download-nemo-speech-models.sh
docker compose --profile generic-assistant/single-gpu up -d \
  nvidia-llm-vllm-lightning nemo-speech
~~~

The example expects the LLM at `http://127.0.0.1:18000/v1` and ASR/TTS at
`127.0.0.1:50051`. Use hosted NVIDIA endpoints instead by changing only the
frontend service entries and credential file references in the YAML.

### 2. Prepare the Developer Compatibility Gateway

~~~bash
src/examples/voiceclaw/integrations/nemoclaw/scripts/prepare-nemoclaw.sh
~~~

Onboard the prepared NemoClaw sandbox, obtain its actual `dashboardPort`, and
create the agent and deployment credential files required by the prepared
gateway. Then start:

~~~bash
src/examples/voiceclaw/integrations/nemoclaw/scripts/run-nemoclaw-voice-gateway.sh
~~~

This gateway is response-only and non-durable. It does not provide durable Work,
events, replay, reconnect, cancellation, steering, or presentation
acknowledgement.

### 3. Start VoiceClaw

~~~bash
if ! test -e src/examples/voiceclaw/.env; then
  install -m 0600 src/examples/voiceclaw/.env.example src/examples/voiceclaw/.env
fi
chmod 0600 src/examples/voiceclaw/.env

docker compose \
  --env-file src/examples/voiceclaw/.env \
  -f src/examples/voiceclaw/docker-compose.yml \
  up --build -d
~~~

The `.env` file supplies non-secret Compose substitutions only. Put model,
speech, Realtime, and backend credentials under the mounted operator root and
reference them with `credential.file` in the runtime YAML.

Open `http://127.0.0.1:7860/`. The Realtime endpoint is
`ws://127.0.0.1:7860/v1/realtime?model=nvidia/voiceclaw`.

For a LAN browser demo, copy the YAML, set `server.host: 0.0.0.0`, set
`server.listener_security: tls`, configure ephemeral authentication, and mount
the TLS certificate and key. Browsers require HTTPS or localhost for microphone
access.

The UI is optional. Run the image headlessly with `command: ["serve"]` or by
removing the Compose command override. `command: []` is invalid because the
runtime entrypoint requires a subcommand.

### Browser client authentication

Local loopback development defaults to `auth_mode: none`. For a shared
listener, configure `server.auth_mode: ephemeral` and an owner-controlled
`server.api_key_file`, then issue a short-lived `ek_` client secret:

~~~bash
uv run --project src/examples/voiceclaw voiceclaw-client-secret \
  --master-key-file /absolute/path/to/voiceclaw-public-master \
  --output /absolute/path/to/voiceclaw-client-secret
~~~

Only the short-lived client secret enters the browser. The master key remains
server-side.

## Validate

~~~bash
uv build src/examples/voiceclaw \
  --wheel \
  --out-dir src/examples/voiceclaw/dist \
  --clear
python3 src/examples/voiceclaw/scripts/verify-wheel.py \
  src/examples/voiceclaw/dist/nemotron_voiceclaw-0.1.0-py3-none-any.whl \
  --expect-version 0.1.0 \
  --expect-revision "$(git rev-parse HEAD)" \
  --repository-root "$PWD"
uvx ruff@0.15.6 check src/examples/voiceclaw
uvx ruff@0.15.6 format --check src/examples/voiceclaw
uv run --project src/examples/voiceclaw --group dev pytest -q src/examples/voiceclaw/tests
node --check src/examples/voiceclaw/src/voiceclaw/ui/app.js
~~~

The distribution is `nemotron-voiceclaw`. Server and UI dependencies are in
the optional `server` extra, so the core Python package can be embedded without
publishing the HTTP/WebSocket facade.
