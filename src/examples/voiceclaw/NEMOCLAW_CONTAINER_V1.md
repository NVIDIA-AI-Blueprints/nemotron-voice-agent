<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: BSD-2-Clause -->

# NemoClaw Container Profile

## Scope

The `nemoclaw-container-v1` image target implements the VoiceClaw settings in
NemoClaw's [Container Service Contract v1](https://github.com/NVIDIA/NemoClaw/blob/bf93570b92c08230cee3816f382b93f97cd519e5/docs/design/container-contract.md).
That decision is accepted in NemoClaw, with San Dang as the accountable
maintainer. [NemoClaw PR #13093](https://github.com/NVIDIA/NemoClaw/pull/13093)
provides the generic container base. [PR #12619](https://github.com/NVIDIA/NemoClaw/pull/12619)
is closed, so protected credential-file delivery is deferred.

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
| Fabric and its adapter | Agent execution, bridge contracts, result/context evidence, and native health capability |
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
| `VOICECLAW_AUTH_MODE` | `none` or `oidcBearer` |
| `VOICECLAW_GATEWAY_TOKEN` | Service token; required for `oidcBearer` and rejected for `none` |
| `VOICECLAW_SPEECH_KEY` | Hosted speech and LLM credential for the bundled frontend |

Every value except the port is required. Startup validates all settings before
activation and exits with the fixed message `container-input-invalid` without
echoing a value. There is no default target, host-variable expansion, operator
credential discovery, or unauthenticated fallback.

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
revision remains `6648bd0c290efbc41ba131ee9831ee45cd431f94`, with a checked-in
lockfile. The client receives the explicit workspace, sandbox name, and
expected sandbox ID, and verifies that identity before each command. If the
identity changed or cannot be established, the request stops instead of
invoking another sandbox by name. The pinned OpenShell API has no atomic ID
precondition, so this check cannot eliminate the replacement race.

VoiceClaw composes the existing `OpenShellFabricAdapter`. It retains upstream
request encoding, bridge decoding, result-channel parsing, correlation checks,
and fresh-target context validation. The executor accepts only the adapter's
fixed `/usr/local/bin/fabric-agent check --agent ... --live` and
`/usr/local/bin/fabric-agent invoke --agent ... --input -` operations, with
check and invoke deadlines of 12 and 120 seconds. It bounds stdin and streamed
output and rejects output that echoes the gateway token. It runs no login shell
or lifecycle command. An unavailable agent returns a bounded failure; no other
agent is selected.

The profile sets `interaction.turn_routing_mode: model`. The frontend selects
direct conversation or the advertised `work.delegate` operation. Unavailable or
consumed backends reject delegated requests before admission. Delegation sends
the validated standalone goal produced by the frontend; acknowledgements and
result speech use separate responses.

Requests use the `nvidia.fabric.openclaw` codec with `{agent, message}` input.
Both the Fabric agent and native OpenClaw agent are bound to
`VOICECLAW_AGENT_NAME`. Successful results must preserve the Fabric runtime,
adapter, and agent identities and qualify each appended turn against verified
native history. The selected response must parse as `voiceclaw.result.v1`
display/speech channels. Plain text is not a successful fallback.

First-turn qualification requires exactly one original user prompt, the matching
session identity for the native agent and runtime, and a final assistant response with empty
`tool_calls`. Between the user prompt and final response, the agent can emit
assistant function-call batches and their matching tool results within that
single invocation. Each call must have a unique ID and exactly one matching
result. Results within a batch can arrive in any order, but every pending call
must resolve before another assistant batch or the final response. Unknown or
duplicate IDs, missing results, extra user or system messages, prior conversation
history, and malformed message or call shapes fail qualification.

Intermediate assistant messages have exactly `role`, string `content`, and a
nonempty `tool_calls` list. Each call has exactly `id`, `type`, and `function`,
with `type: function`. The `function` object has exactly string `name` and
`arguments` fields. Call IDs and function names must be nonempty strings without
NUL. Tool results have exactly `role: tool`, string `content`, string
`tool_call_id`, and boolean `is_error`. A reported tool error still requires its
matching call ID.

The final assistant history content and `output.response` can differ only in
leading or trailing JSON whitespace: ASCII space, tab, carriage return, or
newline. The response body and exact user prompt remain unchanged. The selected
response still passes the strict result-envelope parser; JSON-equivalent
rewrites are not accepted as matching history.

The profile permits one non-durable delegated turn against a fresh
target. A successful or ambiguous invocation, an unusable result, or an
abandoned in-flight invocation consumes the target and prevents another
dispatch. VoiceClaw does not adopt existing native history or claim session
continuity, reconnect recovery, replay, event streaming, steering, or confirmed
remote cancellation. Provision a fresh target before delegating again.

## Image and Health

The `nemoclaw-container-v1` image target provides the following interface:

| Item | Value |
| --- | --- |
| Entrypoint and command | `/usr/local/bin/voiceclaw-runtime serve` |
| Runtime identity | `65532:65532`; other identities are rejected |
| Listener | `0.0.0.0:${VOICECLAW_PORT}`; console at `GET /`, `WS /v1/realtime`, `GET /health` |
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

`GET /health` returns empty `200` when local components are initialized, the
bundled frontend child is running, and its `http://127.0.0.1:7861/health`
returns `200` with `{"status": "ok"}`. It returns empty `503` while loading or
after a failure. The child probe has a 2-second budget within a 15-second
health deadline, with 4,096 header bytes and 1,024 body bytes maximum. Use
`readiness.path: /health` in NemoClaw, or rely on the image `HEALTHCHECK`.

Health creates no user session, agent invocation, model response, or lifecycle
change. It does not prove provider credentials or native agent health.
Unsupported native health remains distinct; the upstream adapter still
validates the bridge binding before delegated work.

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

## Validation and Evidence

Historical manual evidence used the retired descriptor-based revision. It
proves only that earlier installation and local readiness, not this revision
or an agent/model response:

| Item | Observed Evidence |
| --- | --- |
| VoiceClaw source | `0e713f7677428c96ec9c59d08f36ec3110dc8275` |
| Reported application manifest | `nc-voiceclaw-test@sha256:b057f38bf971185d1e0cc12e12b6f6476242e2d518831ccc728e8c4c3af06a09` |
| NemoClaw source | `1285a41115945c1f123d377ff580681b9bdb3976` |
| Operator environment | ARM64 Linux DGX Station, NVIDIA GB300, sudo Docker |
| Native health | `fabric_health_unsupported`, not native healthy |

Record focused test results for the reviewed commit. Do not transfer historical
test counts or image results to this revision. A missing-input image smoke does
not prove complete-input startup.

Report the source revision, image digest, platform, and accepted settings for
qualification under [NemoClaw #12570](https://github.com/NVIDIA/NemoClaw/issues/12570):

1. Build and verify the image, source/package/license/client provenance,
   architecture, and selected-engine manifest resolution.
2. Deploy the YAML service on Linux and observe health without native
   inference, invalid-setting rejection, and owned removal.
3. Submit one explicit committed turn against a fresh target after apply exits.
   Prove intended-agent invocation and display/speech results separately from
   hosted speech-provider credential acceptance and audio behavior. Use the
   [recorded voice verifier](README.md#verify-a-recorded-voice-turn) with
   `--context-continuity unqualified`, and prove that a greeting stays in the
   frontend. Audible delivery requires separate observation.
4. Test unavailable or replaced agents, failed authentication and TLS
   rejection, ambiguous invocation with no replay, and interruption/retry.

A healthy container, parser fixture, or successful HTTP probe alone is not
qualification. Supported native health remains upstream work tracked by
[Fabric #298](https://github.com/NVIDIA/NeMo-Fabric/issues/298) and
[NemoClaw #12443](https://github.com/NVIDIA/NemoClaw/issues/12443).
This document does not claim production or full voice qualification.
