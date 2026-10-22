<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: BSD-2-Clause -->

# NemoClaw Container Profile

## Scope

The `nemoclaw-container-v1` image target implements the VoiceClaw settings in
NemoClaw's [Container Service Contract v1](https://github.com/NVIDIA/NemoClaw/blob/bf93570b92c08230cee3816f382b93f97cd519e5/docs/design/container-contract.md),
pinned to the revision reviewed for this profile.

NemoClaw deploys this image as an ordinary `kind: container` service and passes
literal environment values. It does not interpret VoiceClaw settings. There is
no application-specific installer, `agentConnection`, generated connection
descriptor, or credential file. VoiceClaw does not require
`/run/nemoclaw/agent-connection.json` or any other descriptor to start.

The following table assigns each responsibility to one owner:

| Owner | Responsibility |
| --- | --- |
| NemoClaw | Container create/start/health/remove, publication, and declared environment |
| Operator | Gateway and agent creation and verification before deployment, and private handling of literal values |
| VoiceClaw | Image, settings validation, bundled frontend, local health, executor, request/result integration, and local shutdown |
| OpenShell | Gateway transport, authentication, authorization, and sandbox isolation |
| Fabric and its adapter | Agent execution, bridge contracts, RunResult identities, and adapter-owned output |
| Issuer/operator | Service-token issuance, audience, authority, expiry, revocation, and renewal |

The application has no Docker socket, host networking, host interprocess
communication (IPC), GPU access, or agent lifecycle control. The default image
supervisor, developer Compose recipe, Python SDK factory, and cloud/external
frontend profiles remain separate.

## Environment

VoiceClaw reads the following settings from its process environment:

| Variable | Meaning |
| --- | --- |
| `VOICECLAW_PORT` | Listener port for the console, `WS /v1/realtime`, and `GET /health`; default `18790` |
| `VOICECLAW_GATEWAY_URL` | OpenShell gateway URL reachable from the container |
| `VOICECLAW_WORKSPACE` | Selected workspace name |
| `VOICECLAW_SANDBOX_NAME` | Selected sandbox name |
| `VOICECLAW_SANDBOX_ID` | Expected physical sandbox UUID |
| `VOICECLAW_AGENT_NAME` | Agent to invoke in that sandbox |
| `VOICECLAW_FABRIC_RUNTIME_ID` | Runtime ID that NemoClaw received when it provisioned the agent |
| `VOICECLAW_FABRIC_ADAPTER_ID` | Exact adapter ID expected in the Fabric RunResult |
| `VOICECLAW_FABRIC_HARNESS` | Exact harness ID expected in the Fabric RunResult |
| `VOICECLAW_FABRIC_ADAPTER_KIND` | Expected adapter kind: `http`, `native_plugin`, `process`, or `python` |
| `VOICECLAW_FABRIC_RESULT_POINTER` | RFC 6901 pointer to the result-envelope string, such as `/output/response` |
| `VOICECLAW_AUTH_MODE` | `none` or `oidcBearer` |
| `VOICECLAW_GATEWAY_TOKEN` | Service token; required for `oidcBearer` and rejected for `none` |
| `VOICECLAW_SPEECH_KEY` | Hosted speech and LLM credential for the bundled frontend |

All listed values except `VOICECLAW_PORT` and the conditional
`VOICECLAW_GATEWAY_TOKEN` are required. Startup validates all settings before
activation and exits with the fixed message `container-input-invalid` without
echoing a value. There is no default target, host-variable expansion, operator
credential discovery, or unauthenticated fallback.

The five `VOICECLAW_FABRIC_*` values must describe the same `fabric_agent`
resource. NemoClaw's generic `kind: container` environment accepts literal
strings and does not automatically interpolate that resource's runtime ID.
Supply the runtime ID retained for the applied agent, and use the exact adapter
identity from its Fabric configuration. The result pointer is adapter-specific;
`/output/response` is not a universal Fabric output contract. These five
bindings are a VoiceClaw profile extension; the pinned generic container
contract does not define or synthesize them.

Literal credentials are a temporary plaintext approach, not secret storage.
They are visible in YAML, NemoClaw plans and state, exports, Docker inspection,
and the application process environment. Replace placeholders only in private
local configuration; never commit real values or share resulting plans, state,
exports, or logs. Removing the container does not revoke an upstream credential.

VoiceClaw keeps both credentials out of command arguments, browser responses,
and logs. The gateway token reaches only the trusted SDK client through a
private stdin pipe. The speech key reaches only the bundled frontend child
environment and the local credential binding for that frontend.

## Connection Modes

Select exactly one of the following authentication modes:

| `VOICECLAW_AUTH_MODE` | Gateway | Credential |
| --- | --- | --- |
| `oidcBearer` | HTTPS with system certificate verification | `VOICECLAW_GATEWAY_TOKEN` |
| `none` | Development-only private HTTP | No gateway token |

HTTPS uses a static bearer and does not implement client-credentials exchange
or refresh. Do not supply administrator credentials, operator login state, or
refresh tokens. A token's authority is determined by the gateway. Gateway
rejection and TLS errors fail closed.

Development HTTP requires a canonical literal IPv4 address from `10.0.0.0/8`,
`172.16.0.0/12`, or `192.168.0.0/16`, with an explicit port. DNS names, IPv6,
loopback, link-local/public addresses, alternate numeric forms, and non-origin
URLs are rejected. Mixed modes are rejected. The address filter cannot prove
network isolation; HTTP provides no encryption or peer authentication. Use it
only for an explicitly trusted local deployment.

## Agent Adapter Integration

The executor implements the upstream `SandboxExecutor` port and uses the
image-owned `voiceclaw-openshell-exec` Rust SDK client. The OpenShell SDK
revision is `5a7728c304b5528e954e9ae900cf21f84e1a12b1`, with a checked-in
lockfile. The client receives the explicit workspace, sandbox name, and
expected sandbox ID, and verifies that identity before each command. If the
identity changed or cannot be established, the request stops instead of
invoking another sandbox by name. The pinned OpenShell API has no atomic ID
precondition, so this check cannot eliminate the replacement race.

NemoClaw starts and stops the Fabric runtime. VoiceClaw receives the resulting
runtime ID and only invokes that runtime. The executor accepts the fixed Fabric
0.5 command `/opt/fabric/bin/fabric-agent-ctl invoke --agent ... --runtime-id
... --request -`, with a 330-second deadline. It sends one bounded `RunRequest`
object on stdin. The request contains the delegated goal, VoiceClaw request ID,
and non-authoritative correlation context. It runs no login shell, runtime
start, runtime stop, or alternate agent selection.

Fabric 0.5 does not provide a read-only runtime status command. Local health
therefore does not claim native agent readiness. A complete delegated turn is
the qualification check.

The profile sets `interaction.turn_routing_mode: model`. The frontend selects
direct conversation or the advertised `work.delegate` operation. Unavailable or
consumed backends reject locally admitted requests before backend dispatch.
Delegation sends the validated standalone goal produced by the frontend;
acknowledgements and result speech use separate responses.

The adapter validates the direct Fabric RunResult. Its `agent_name`,
`runtime_id`, `adapter_id`, `harness`, `adapter_kind`, and `request_id` must
match the configured binding and dispatched request. The inner RunResult
`status` must also be `succeeded`; outer command success alone is insufficient.
VoiceClaw resolves `VOICECLAW_FABRIC_RESULT_POINTER` against that RunResult and
requires the selected string to parse as `voiceclaw.result.v1` display and
speech channels. Plain text is not a successful fallback.

One connected VoiceClaw conversation can send ordered non-durable turns to the
pre-provisioned runtime. The runtime remains bound to the first VoiceClaw
conversation, and a delegated turn from another conversation is rejected
rather than attached to it. The profile also admits only one concurrent browser
session. Fabric 0.5 does not supply evidence that qualifies context continuity
or isolation between invocations, so every delegated goal remains
self-contained. VoiceClaw does not claim reconnect recovery, replay, event
streaming, steering, confirmed remote cancellation, or durable Work. An
ambiguous invocation fences the binding; do not replay it automatically.

## Image and Health

The `nemoclaw-container-v1` image target provides the following interface:

| Item | Value |
| --- | --- |
| Entrypoint and command | `/usr/local/bin/voiceclaw-runtime serve` |
| Runtime identity | `65532:65532`; other identities are rejected |
| Listener | `0.0.0.0:${VOICECLAW_PORT}`; console at `GET /`, `WS /v1/realtime`, process liveness at `GET /livez`, local readiness at `GET /health`, optional agent attestation at `GET /readyz` |
| Disposable state | `/var/lib/voiceclaw/state/state.db` |
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

The build does not publish an image. Supply NemoClaw an immutable
`repository@sha256:<64-hex-manifest-digest>` that resolves in the selected
Docker engine. A local tag or configuration image ID is not that reference.

`GET /livez` returns a content-free `200` while the VoiceClaw facade is serving.
`GET /health` returns local application readiness; the managed image
`HEALTHCHECK` probes it, and it can remain the NemoClaw `readiness.path`.
Both are service-level checks and create no user session,
agent invocation, model response, or lifecycle change.

`GET /readyz` is the optional selected-agent attestation endpoint. It returns
`503` for this profile because Fabric 0.5 exposes no safe read-only runtime
status operation. The first end-to-end delegated turn is therefore the agent
qualification check and discovers an unavailable runtime without selecting a
different target. None of the health endpoints proves provider credentials,
model responses, or audible delivery.

Docker health uses a 10-second interval, 20-second timeout, 180-second start
period, and three retries. The image bundles executable dependencies; startup
downloads no executable or package. The hosted frontend preset does not
automatically inherit the sandbox's local vLLM provider.

The hosted frontend large language model (LLM) sets `supports_tokenize: false`
because its endpoint does not provide the `/tokenize` API. This disables native
Realtime context truncation without substituting an approximate token counter.
A working `/v1/chat/completions` route does not establish tokenizer support;
advertising an unavailable tokenizer can produce `llm_provider_error` before
completion. This setting does not change the sandbox's local Qwen provider or
qualify a complete voice reply.

Application removal deletes its owned disposable volume and preserves the
agent dependency. Deployment-wide destroy follows NemoClaw ownership and can
delete its sandbox. VoiceClaw shutdown stops only local work and children.
The runtime closes its SDK executor before draining the facade,
cancelling local helper tasks. It waits up to 2 seconds for helper cleanup,
then terminates the bundled frontend's local process group and reaps the child.
Cleanup still runs when facade shutdown fails. Cancelling a local helper does
not confirm that the remote invocation stopped.
Credential revocation remains the issuer/operator's responsibility.

## Validation

Record the VoiceClaw and NemoClaw source revisions, immutable image digests,
platform, and accepted settings for each qualification run:

1. Build and verify the image, source/package/license/client provenance,
   architecture, and selected-engine manifest resolution.
2. Deploy the YAML service on Linux and observe health without native
   inference, invalid-setting rejection, and owned removal.
3. Submit one explicit committed turn after apply exits. Prove intended-agent
   and runtime invocation and display/speech results separately from hosted
   speech-provider credential acceptance and audio behavior. Use the
   [recorded voice verifier](README.md#verify-a-recorded-voice-turn) with
   `--context-continuity unqualified`, and prove that a greeting stays in the
   frontend. Submit an ordered follow-up in the same conversation and prove that
   another conversation cannot delegate through the bound runtime. Audible
   delivery requires separate observation.
4. Test unavailable or replaced agents, failed authentication and TLS
   rejection, ambiguous invocation with no replay, and interruption/retry.

A healthy container, parser fixture, or successful HTTP probe alone is not
qualification. This document does not claim production or full voice
qualification.
