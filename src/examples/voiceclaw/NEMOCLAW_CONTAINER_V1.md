<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: BSD-2-Clause -->

# NemoClaw Container v1 Installer Contract

## Accepted Scope

San Dang accepted VoiceClaw implementation of `voiceclaw.nemoclaw.container.v1`
on October 5, 2026, in the operator conversation. On October 7, San Dang
approved adapting the installer to the latest `feat/voiceclaw_example`
implementation while retaining upstream behavior. These conversation decisions
establish scope and the accountable maintainer; they are not published upstream
issue decisions.

Placement is `src/examples/voiceclaw` in
`NVIDIA-AI-Blueprints/nemotron-voice-agent`. The adaptation baseline is
`fe2402e4ab6377c6eba4ae00a9d2a72f4422f808`. The reason is to preserve the tested
installer interface without retaining a duplicate agent adapter.

The responsibility split is explicit:

| Owner | Responsibility |
| --- | --- |
| NemoClaw | YAML declaration, protected delivery, resolved bindings, dependency ordering, container create/start/health/remove, and owned data removal |
| VoiceClaw | Application image, foreground bootstrap, bundled frontend, local readiness, installer executor, request/result integration, and local shutdown |
| OpenShell | Gateway transport, authentication, authorization, and sandbox isolation |
| Fabric and its adapter | Agent execution, bridge contracts, result/context evidence, and native health capability |
| Issuer/operator | Credential issuance, audience, authority, expiry, revocation, and renewal |

NemoClaw's separate scope acceptance is recorded in
[issue #12567](https://github.com/NVIDIA/NemoClaw/issues/12567#issuecomment-5996200381).
Scope does not include a token issuer, background installer operator, agent
lifecycle manager, Docker socket, host networking, host IPC, or GPU access for
the VoiceClaw application. The default image supervisor, developer Compose
recipe, Python SDK factory, and cloud/external frontend profiles remain separate.

Validation covers descriptor and protected-file rejection, SDK transport,
upstream protocol integration, observation-only local health, state-directory
initialization, no replay, and local child shutdown. Linux image/bootstrap and
explicit voice-turn evidence remain separate gates.

## Installer Inputs

The five application environment inputs remain unchanged:

```text
VOICECLAW_RUNTIME_PROFILE=nemoclaw-container-v1
VOICECLAW_INSTALL_CONTRACT=voiceclaw.nemoclaw.container.v1
VOICECLAW_AGENT_CONNECTION_FILE=/var/lib/voiceclaw/config/agent-connection.json
VOICECLAW_SPEECH_CREDENTIAL_FILE=/var/lib/voiceclaw/credentials/speech
VOICECLAW_STATE_PATH=/var/lib/voiceclaw/state/state.db
```

NemoClaw resolves one connection before starting VoiceClaw. The connection is a
UTF-8 JSON object of at most 64 KiB. This nonsecret example contains invalid
placeholders; replace them with the actual deployment bindings:

```json
{
  "schemaVersion": "nemoclaw.agent-connection.v1",
  "deploymentUid": "<deployment UUID>",
  "service": "voice",
  "gateway": {
    "endpoint": "https://<container-reachable gateway>:<port>",
    "tls": {"trust": "system", "caFile": null}
  },
  "authentication": {
    "mode": "oidcBearer",
    "credentialFile": "/var/lib/voiceclaw/credentials/openshell",
    "refreshMode": "none"
  },
  "target": {
    "workspace": "<resolved workspace>",
    "sandbox": "assistant",
    "sandboxId": "<resolved sandbox UUID>",
    "agent": "main"
  },
  "bridge": {"interfaceVersion": 1, "command": "fabric-agent"},
  "timeouts": {"healthSeconds": 12, "invokeSeconds": 120}
}
```

The parser rejects unknown versions or fields, duplicate JSON keys, malformed
identities, unsafe URLs and paths, and the obsolete installer projection.
There is no automatic fallback or operator credential discovery.

Installer-owned directories use mode `0700`; descriptor and credential files
use `0600`, all owned by UID:GID `65532:65532`. VoiceClaw validates ownership,
modes, regular files, access control lists, stable bounded reads, and no-follow
traversal. Ancestors outside the protected data root retain identity, ownership,
and mode checks; unrelated sibling activity does not invalidate a safe binding.
VoiceClaw does not rewrite protected inputs. Each credential read is bounded to
16 KiB. Speech and OpenShell credential paths must be distinct.

The application owns only its local state. Startup creates missing state parent
directories beneath the verified data root with mode `0700`, preserving a valid
existing `0600` state file. Unsafe existing paths fail startup. This supports a
`NoCopy` volume populated with configuration and credentials but no state
directory.

## Explicit Connection Modes

The installer profile supports two distinct descriptor pairs:

| Gateway | TLS | Authentication |
| --- | --- | --- |
| HTTPS | `trust: system`, `caFile: null` | `mode: oidcBearer`, protected `credentialFile`, `refreshMode: none` |
| Development-only private HTTP | `trust: none`, `caFile: null` | `mode: none`, `credentialFile: null`, `refreshMode: none` |

HTTPS uses a static bearer with verified system trust. It does not implement
upstream client-credentials exchange or refresh. The issuer must approve the
service identity and its finite lifetime, audience, and authority. Do not copy
administrator credentials, operator login state, or refresh tokens. Gateway
rejection fails closed. Private CA delivery requires separate review.

Development HTTP requires a canonical literal IPv4 address from `10.0.0.0/8`,
`172.16.0.0/12`, or `192.168.0.0/16`, with an explicit port. DNS names, IPv6,
loopback, link-local/public addresses, alternate numeric forms, and non-origin
URLs are rejected. No OpenShell credential is read in this mode. Speech
credentials remain required. Mixed modes and automatic downgrade are rejected.

The address parser cannot prove Docker ownership or isolation. NemoClaw must
bind the endpoint to the declared gateway and deployment network and prevent
LAN/public publication. HTTP provides no encryption or peer authentication;
other reachable processes can use gateway RPCs beyond this executor's commands.
This is not production or authenticated qualification.

These modes are installer-specific. The upstream YAML factory retains its
anonymous loopback and remote OAuth client-credentials rules without relaxation.

## Upstream Adapter Integration

The installer executor implements the upstream `SandboxExecutor` port and uses
the image-owned `voiceclaw-openshell-exec` Rust SDK client. The OpenShell SDK
revision remains `6648bd0c290efbc41ba131ee9831ee45cd431f94`, with a checked-in
lockfile. The executor supplies the explicit workspace, sandbox name, and
physical sandbox ID from the protected descriptor.

VoiceClaw composes the existing `OpenShellFabricAdapter`. It retains upstream
request encoding, bridge decoding, result-channel parsing, correlation checks,
and fresh-target context validation. There is no second legacy backend or
fabricated successful bridge check. The executor accepts only the adapter's
fixed `/usr/local/bin/fabric-agent check --agent ... --live` and
`/usr/local/bin/fabric-agent invoke --agent ... --input -` operations. It runs
no login shell or lifecycle command.

Requests use the `nvidia.fabric.openclaw` codec with `{agent, message}` input.
The image selects this OpenClaw adapter; the descriptor adds no adapter override
or new fields. Both the Fabric agent and native OpenClaw agent are bound to
`descriptor.target.agent`. Different Fabric and native agent names are not
supported by this installer contract.
Successful results must preserve the Fabric runtime, adapter, and agent
identities and satisfy the first-turn context shape. The selected response must
parse as `voiceclaw.result.v1` display/speech channels. Plain text is not a
successful fallback.

The adapter supports one concurrent session and one non-durable delegated turn
for each fresh target. A successful or ambiguous invocation consumes that
target. Provision a fresh target before another delegated turn; restarting the
application does not clear native conversation history. There is no automatic
replay, durable backend work, event streaming, steering, confirmed remote
cancellation, or qualified conversation continuity.

The executor bounds stdin and streamed output and enforces the descriptor's
deadlines. Bearer control data enters the trusted SDK child through a private
stdin pipe, separately from the remote request. Credentials do not enter Docker
environment configuration, command arguments, browser state, correlation state,
or logs. The bundled frontend can receive its speech key in a restricted child
environment when required by its provider library.

A sandbox-ID precheck detects an observed replacement but is not atomic
authorization fencing. It cannot eliminate the name-to-ID race. The gateway
identity's authority can exceed the executor's command allowlist.

## Image and Local Readiness

The dedicated `nemoclaw-container-v1` image target owns the following interface:

| Item | Value |
| --- | --- |
| Entrypoint and command | `/usr/local/bin/voiceclaw-runtime serve` |
| Runtime identity | `65532:65532` |
| Listener and API | `0.0.0.0:18790`; `WS /v1/realtime` |
| Browser UI | Disabled |
| Disposable data | `/var/lib/voiceclaw` |
| Writable scratch | `/run/voiceclaw-managed` |
| Stop allowance | SIGTERM; local children stop within 15 seconds |

Build from the repository root after committing all reviewed inputs:

```bash
voiceclaw_source_revision="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)" && \
docker build \
  --target nemoclaw-container-v1 \
  --platform linux/arm64 \
  --build-arg VOICECLAW_VERSION=0.1.0 \
  --build-arg "VOICECLAW_SOURCE_REVISION=${voiceclaw_source_revision}" \
  --build-arg VOICECLAW_SOURCE_URL=https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent \
  -f src/examples/voiceclaw/Dockerfile \
  -t "voiceclaw-nemoclaw:${voiceclaw_source_revision}-arm64" \
  .
```

The build command does not publish an image. Supply an immutable
`repository@sha256:<64-hex-manifest-digest>` to NemoClaw and prove that it
resolves in the selected Docker engine under `imagePullPolicy: Never`. A local
tag or configuration image ID does not establish that manifest reference.

`GET /livez` observes process/listener liveness. Installer-profile `GET /readyz`
returns empty `200` for valid protected inputs, initialized local components, a
live frontend child, and a successful bounded child `/health` observation.
Otherwise it returns empty `503`. The child probe targets
`http://127.0.0.1:7861/health` and requires `200` with `{"status": "ok"}`.
It has a 2-second budget within the 15-second readiness deadline, with 4,096
header bytes and 1,024 body bytes maximum. Docker retains health command output;
the command keeps that output empty.

Local readiness creates no user session, agent invocation, model response,
credential renewal, repair, or lifecycle change. It does not prove provider
credentials or native agent health. Unsupported native health remains distinct;
the upstream adapter still validates the real bridge binding before delegated
work. A validated running binding can pass the upstream binding check while
native health remains unsupported. No unsupported status is fabricated or
relabeled as native healthy.

Docker health uses a 10-second interval, 20-second timeout, 180-second start
period, and three retries. NemoClaw waits up to 300 seconds for startup.
Descriptor health and invoke budgets remain 1–12 and 1–120 seconds respectively.
The image bundles executable dependencies; startup downloads no executable or
package. The hosted frontend preset does not inherit the sandbox's local vLLM
provider automatically.

Application removal deletes its owned disposable volume and preserves the
agent dependency. Deployment-wide destroy follows NemoClaw ownership and can
delete its sandbox. VoiceClaw shutdown stops only local work and children.
The installer runtime closes its SDK executor before draining the facade,
cancelling local helper tasks. It waits up to 2 seconds for helper cleanup,
then terminates the bundled frontend's local process group and reaps the child.
Cleanup still runs when facade shutdown fails. Cancelling a local helper does
not confirm that the remote invocation stopped.
Credential revocation remains the issuer/operator's responsibility.

## Validation and Evidence

Historical manual evidence applies only to the earlier implementation:

| Item | Observed Evidence |
| --- | --- |
| VoiceClaw source | `0e713f7677428c96ec9c59d08f36ec3110dc8275` |
| Reported application manifest | `nc-voiceclaw-test@sha256:b057f38bf971185d1e0cc12e12b6f6476242e2d518831ccc728e8c4c3af06a09` |
| NemoClaw source | `1285a41115945c1f123d377ff580681b9bdb3976` |
| Operator environment | ARM64 Linux DGX Station, NVIDIA GB300, sudo Docker |
| Installer result | NemoClaw apply completed; voice container readiness completed |
| Local probe | HTTP `200` from `http://127.0.0.1:18790/readyz` on October 7 |
| Native health | `fabric_health_unsupported`, not native healthy |

These are operator-supplied results, not authenticated hardware or image
provenance. They prove the earlier manual installation and local readiness, not
an agent/model response or full voice conversation. The earlier pending
HTTP-descriptor coordination was completed on the NemoClaw side; it is not a
remaining parser-only blocker. The upstream adaptation still needs a rebuilt
image and a new complete-input startup test.

Record focused test and hook results for the actual reviewed commit in the PR.
Do not transfer historical test counts or image results to this revision.
Fixture success proves only its assertions. A missing-input image smoke does
not prove complete-input startup.

Remaining qualification requires these distinct results:

1. Build and verify the adapted image, source/package/license/client provenance,
   architecture, and selected-engine manifest resolution.
2. Apply the existing YAML inputs on Linux and observe unchanged apply,
   interrupted setup/retry, invalid protected input rejection, and local
   readiness without native inference.
3. Submit one explicit committed turn against a fresh target after apply exits.
   Prove intended-agent invocation and display/speech results separately from
   hosted speech-provider credential acceptance and audio behavior.
4. Test failed/revoked identity and TLS rejection, ambiguous invocation with no
   replay, secret-free export/reapply, and owned application removal without
   deleting the agent or retained inference storage.

San Dang remains the accountable integration maintainer. Issuer/operator and
provider owners must supply identity and speech qualification evidence.
Supported native health remains upstream work tracked by
[Fabric #298](https://github.com/NVIDIA/NeMo-Fabric/issues/298) and
[NemoClaw #12443](https://github.com/NVIDIA/NemoClaw/issues/12443).
Joint qualification remains tracked by
[NemoClaw #12570](https://github.com/NVIDIA/NemoClaw/issues/12570).
This document does not claim production or full voice qualification.
