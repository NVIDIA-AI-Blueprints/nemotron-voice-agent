# VoiceClaw

VoiceClaw is a Python package and containerized realtime voice frontend. A
client uses one OpenAI Realtime-compatible WebSocket for audio, text,
transcripts, and application projections. VoiceClaw owns the live media and
presentation loop. In the target managed NemoClaw profile, NemoClaw owns
authoritative agent sessions and durable Work. The current developer adapter is
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

The accepted NemoClaw contract is one referenced VoiceClaw integration, one
preloaded immutable image, one foreground container, one Docker bridge, one
disposable volume, and one TCP port. It does not use Compose, host networking,
sidecars, systemd, automatic restart, or a NemoClaw background monitor.

Build the exact local artifact from a clean repository root and record the
source, input-file digests, native platform, and immutable image ID:

~~~bash
test -z "$(git status --porcelain --untracked-files=all)"
voiceclaw_source_revision="$(git rev-parse HEAD)"
voiceclaw_version="0.1.0"
voiceclaw_platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
test "${voiceclaw_platform%%/*}" = linux
mkdir -p src/examples/voiceclaw/dist
uv build src/examples/voiceclaw \
  --wheel \
  --out-dir src/examples/voiceclaw/dist \
  --clear
python3 src/examples/voiceclaw/scripts/verify-wheel.py \
  src/examples/voiceclaw/dist/nemotron_voiceclaw-0.1.0-py3-none-any.whl \
  --expect-version "${voiceclaw_version}" \
  --expect-revision "${voiceclaw_source_revision}" \
  --repository-root "$PWD" \
  > src/examples/voiceclaw/dist/wheel-evidence.json
sha256sum \
  src/examples/voiceclaw/Dockerfile \
  uv.lock \
  src/examples/voiceclaw/uv.lock
docker build \
  --platform "${voiceclaw_platform}" \
  --build-arg "VOICECLAW_VERSION=${voiceclaw_version}" \
  --build-arg "VOICECLAW_SOURCE_REVISION=${voiceclaw_source_revision}" \
  -f src/examples/voiceclaw/Dockerfile \
  -t "voiceclaw-runtime:poc-${voiceclaw_source_revision}" \
  .
docker image inspect \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}' \
  "voiceclaw-runtime:poc-${voiceclaw_source_revision}"
uv run --project src/examples/voiceclaw --extra server --frozen \
  python src/examples/voiceclaw/scripts/verify-image.py \
  "voiceclaw-runtime:poc-${voiceclaw_source_revision}" \
  --expect-version "${voiceclaw_version}" \
  --expect-revision "${voiceclaw_source_revision}" \
  --expect-source https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent \
  --expect-architecture "${voiceclaw_platform#*/}" \
  --repository-root "$PWD" \
  --wheel src/examples/voiceclaw/dist/nemotron_voiceclaw-0.1.0-py3-none-any.whl \
  --wheel-evidence src/examples/voiceclaw/dist/wheel-evidence.json \
  --smoke --smoke-ui --smoke-realtime \
  > src/examples/voiceclaw/dist/image-evidence.json
python3 src/examples/voiceclaw/scripts/export-image-archive.py \
  "voiceclaw-runtime:poc-${voiceclaw_source_revision}" \
  --image-evidence src/examples/voiceclaw/dist/image-evidence.json \
  --archive src/examples/voiceclaw/dist/voiceclaw-${voiceclaw_source_revision}.docker.tar \
  > src/examples/voiceclaw/dist/archive-evidence.json
python3 src/examples/voiceclaw/scripts/export-image-archive.py \
  --verify-only \
  --image-evidence src/examples/voiceclaw/dist/image-evidence.json \
  --archive-evidence src/examples/voiceclaw/dist/archive-evidence.json \
  --archive src/examples/voiceclaw/dist/voiceclaw-${voiceclaw_source_revision}.docker.tar
~~~

After the final verification command succeeds, a consumer may load that exact
archive with `docker load --input <archive>`. CI builds and retains the ARM64
artifact used by DGX Spark; the local commands above intentionally verify the
native Linux platform reported by the local Docker engine.

The image contract is:

| Item | Value |
| --- | --- |
| Entrypoint | `/usr/local/bin/voiceclaw-runtime` |
| Command | `serve` |
| Port | `18790/tcp` |
| Volume | `/var/lib/voiceclaw` |
| Runtime config | `/var/lib/voiceclaw/config/voiceclaw.yaml` |
| Credentials | Protected files below `/var/lib/voiceclaw/credentials/` |
| Local projection | `/var/lib/voiceclaw/state/state.db` |
| UI | Off; `serve --ui` is developer opt-in only |

The volume root and `config/` and `credentials/` are not writable by either
child process. `credentials/` is root-owned and traverse-only for the facade
group: a selected-agent credential is root-owned, group-readable by the facade
(`0440`), while every speech-provider credential is root-only (`0400`). The
root supervisor stages a speech credential into an ephemeral file owned only
by the private NVA process. Only `state/` is owned by the facade. The facade
and NVA run as different fixed UIDs.

The intended NemoClaw configuration is:

~~~yaml
spec:
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

Here `credential.env` names a protected input to NemoClaw; the secret value
must not become a Docker environment value. NemoClaw must project it as a
root-owned file and render the non-secret VoiceClaw runtime YAML in the one
managed volume.

### Health and readiness

- `GET /livez` is the content-free process-liveness endpoint used by the image
  healthcheck and service managers.
- `GET /health` is an operator-facing diagnostic summary of loaded local
  runtime components; it is not the managed readiness signal.
- `GET /readyz` is selected-agent readiness. It performs a bounded,
  authenticated, non-generative access check and returns empty `200` only when
  the configured agent can be reached with its scoped credential. Failure is
  empty `503`; logs contain only allowlisted reason codes. NemoClaw polls this
  endpoint independently during `check_running()`.
- Readiness never submits a prompt, creates a user session, starts Work, or
  runs the historical arithmetic probe.

### Current cross-repository gate

The VoiceClaw image-side executable, process isolation, configuration loader,
and health/readiness boundary are implemented. The current NemoClaw draft does
not yet provide the runtime YAML, protected speech/agent credential files, or
selected-agent endpoint to the container, so a complete managed
`nemoclaw apply` must remain a draft integration and `/readyz` correctly fails
closed.

The CI Realtime smoke uses isolated loopback fixtures to qualify the image's
public protocol and adapter boundary. It is not evidence of the still-pending
managed NemoClaw bridge, credential projection, or selected-agent readiness.

Two producer details must be frozen with NemoClaw before composed validation:

1. Materialize the non-secret runtime YAML plus protected credential files in
   `/var/lib/voiceclaw`; do not put secret values in YAML, runtime-spec JSON,
   Docker environment values, arguments, logs, errors, or state.
2. Permit the root supervisor only `CHOWN`, `DAC_OVERRIDE`, `KILL`, `SETGID`,
   and `SETUID` after dropping all other capabilities, or provide an equivalent
   isolation design. `KILL` is required only so PID 1 can stop its different-UID
   children cleanly. A generic `cap-drop ALL` container cannot supervise the two
   fixed child identities without this narrow allowance.

The selected-agent credential also needs its own root-owned, facade-readable
protected file and a new managed NemoClaw adapter. It must not be mapped into
the current response-only gateway's deployment-bearer field.

Once those are supplied, NemoClaw owns `plan/apply/destroy`, immutable image
resolution, referenced-service activation, bridge/port/volume creation,
`restart: no`, readiness polling, scoped grant revocation, and owned cleanup.
VoiceClaw does not parse NemoClaw internals or receive OpenClaw/OpenShell
operator credentials.

## Developer stack and optional UI

The developer Compose file is intentionally separate from the managed service.
It enables the UI, host networking, and the current non-durable response-only
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

### 2. Prepare the current compatibility gateway

~~~bash
src/examples/voiceclaw/integrations/nemoclaw/scripts/prepare-nemoclaw.sh
~~~

Onboard the prepared NemoClaw sandbox, obtain its actual `dashboardPort`, and
create the agent and deployment credential files as described by that pinned
compatibility revision. Then start:

~~~bash
src/examples/voiceclaw/integrations/nemoclaw/scripts/run-nemoclaw-voice-gateway.sh
~~~

This gateway is response-only and non-durable. It is not the accepted scoped
agent-ingress contract.

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
