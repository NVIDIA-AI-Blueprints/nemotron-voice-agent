# Connect an Agent Harness

The harness backend adapter of the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) runs the delegated work on an external agent, such as Hermes, over the Agent Client Protocol (ACP). For its design, refer to [The Harness](../design/architecture.md#the-harness) on the architecture page.

The repository ships the harness adapter and its configuration, not the agent: you deploy the agent, and the core connects to it through `backend.harness`. The shipped configuration connects to Hermes: the `backend.harness` block of `core.yaml` names the Hermes agent class (`agent: hermes`) and starts `hermes acp` over stdio, and `core.harness.yaml` switches the core to the harness.

> **Note:** In a Docker Compose deployment, the app container has no Hermes, so serve Hermes over WebSocket, as [Connect over WebSocket](#connect-over-websocket) describes. Add a Compose override for the app service, such as `frontend-backend-live`, that mounts the shared `work_root` at the same path on both sides. For Hermes to reach the tool bridge, bind the bridge on `0.0.0.0:8791` in the container, then expose that port on the Compose network, or publish it on the host's loopback address only (`127.0.0.1:8791:8791`) when Hermes runs on the host. The app container runs the server as root, so run Hermes as root as well. Stdio needs a server that runs on the host, or an image of your own that holds Hermes.

## Deploy Hermes

Hermes runs in its own Python environment, outside the app's image. To deploy it:

1. Install [Hermes](https://github.com/nousresearch/hermes-agent) at the tested commit, `af26acab737db4382815aadcd73c06755471ab52`, with its `acp` and `mcp` extras, in a virtual environment of its own. `hermes acp --check` prints `Hermes ACP check OK`. From commit `eb75627faf` on, `hermes acp` connects to the Hermes gateway daemon instead of running the agent itself, so test a later commit before you use it.
2. Give Hermes a home for the voice agent alone, because every session of the core reads the home's identity and configuration: the folder that `HERMES_HOME` names, or `~/.hermes` by default. The home holds Hermes's own `config.yaml`, its `SOUL.md`, and its model key in the home's `.env` (`NVIDIA_API_KEY` for the `nvidia` provider), which Hermes loads itself. The core never passes it a key.
3. Write the home's `SOUL.md`, the identity that opens the system prompt of every Hermes session. Without one, Hermes writes its own persona there on its first run ("You are Hermes Agent, built by Nous Research ..."), and every conductor and worker session reads that persona beside the core's instructions. Use the following identity:

   ```text
   You work behind a voice assistant, for its callers. Follow the instructions of your project context exactly, use only the tools you are given, and do not narrate your tool calls.
   ```

4. Configure Hermes in its `config.yaml`. The following example runs Nemotron 3.5 Lightning on NVIDIA's cloud endpoint:

   ```yaml
   model:
     provider: nvidia
     default: nvidia/nemotron-3.5-lightning-30b-a3b
     base_url: https://integrate.api.nvidia.com/v1
     context_length: 131072
   platform_toolsets:
     acp: [no_mcp]
   tools:
     tool_search:
       enabled: "off"
   agent:
     coding_context: "off"
     task_completion_guidance: false
     parallel_tool_call_guidance: false
     tool_use_enforcement: false
     execution_guidance: false
   memory:
     memory_enabled: false
     user_profile_enabled: false
   auxiliary:
     title_generation:
       enabled: false
     background_review:
       enabled: false
   auth:
     adopt_external_logins: false
   timeouts:
     mcp:
       tool_call: 3600
   ```

The following table explains why each setting matters to the core:

| Setting | Why |
| --- | --- |
| `model.default` and `model.base_url` | The model that Hermes runs. Run the default model of `backend.slot`, because the core accepts requests for that model only. Keep `base_url`: it pins the route that `context_length` describes. Without it, Hermes drops `context_length` when it runs on a route other than the provider's default, such as a local NIM that `NVIDIA_BASE_URL` names. |
| `model.context_length` | The context window that Hermes compresses its own transcript against. Give it at least 65,536 tokens. A worker's session holds the history that the core sends, within `delegation.context_tokens` (24,000 tokens), and its tasks' own rounds. With `backend.harness.session: call`, a kept session holds them until that budget trims the history. Keep the budget well below this window, or Hermes compresses the transcript and drops detail. |
| `platform_toolsets.acp: [no_mcp]` | Only the session's own MCP server, the core's tool bridge, reaches a session. A list without `no_mcp` merges in every MCP server of the configuration. Toolsets that you add after `no_mcp` reach the conductor too, and the commit gate cannot hold them. Hermes warns that the `acp` platform has no valid toolsets (`unknown name(s): no_mcp`) and that tools will be unavailable. This is expected, because the session's own MCP server still reaches the session. Keep `no_mcp`, and do not save the `acp` platform from `hermes tools`, which removes it. |
| No `mcp_servers` | Hermes starts every MCP server that its home lists when it opens its first session, even with `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, which skips only their start at Hermes's startup. `no_mcp` keeps their tools out of the sessions, but not the servers out of the process, and a server that never connects makes each new session wait up to `mcp_discovery_timeout` (1.5 seconds). |
| `tools.tool_search.enabled: "off"` | The session's tools stay in front of the model instead of behind a search tool. |
| `agent` guidance settings and `coding_context` | No coding-agent brief in the system prompt: the core's instructions are the brief. |
| `memory` and `auxiliary` | No memory between calls, and no hidden model calls to title or review each session. |
| `auth.adopt_external_logins: false` | Hermes never borrows another tool's login that it finds on the host. |
| `timeouts.mcp.tool_call` | Above `delegation.client_tool_timeout_seconds` (60 seconds), so a held app call ends by the core's deadline instead of a Hermes timeout. |

At the tested commit, a Hermes ACP session reads no `agent.max_turns`, so the example sets none, and nothing caps the model calls of one turn. `backend.harness.turn_timeout_seconds` (300 seconds, not counting waits for the app) cancels a turn that runs longer.

Hermes keeps each session's transcript in the `state.db` file of its home, which holds the instructions and the conversation of every call. When a provider error fails a model call, it also writes the whole request, with the instructions and the conversation, to a `request_dump_*.json` file in the `sessions` folder of its home. Clear both as your data policy requires.

## Connect over Stdio

With the `stdio` transport, the core starts `hermes acp` itself: one process for the conductor (none at `max_running: 1`) and one for the workers of each conversation, and one more when every one of a process's four turns is taken. To connect:

1. Put `hermes` on the server's `PATH`, or set `backend.harness.command` to its full path.
2. If the home of Hermes is not `~/.hermes`, export `HERMES_HOME` in the server's environment. A started process gets only the variables that `backend.harness.pass_env` names (`PATH`, `LANG`, `LC_ALL`, `TZ`, and `HERMES_HOME` in `core.yaml`), `HERMES_ACP_SKIP_CONFIGURED_MCP=1` from the Hermes agent class, and the values of `backend.harness.env`.
3. Point `CORE_CONFIG_PATH` at `src/examples/frontend_backend_live/core.harness.yaml`, and start the server.

## Connect over WebSocket

To run Hermes on another host or in a container of its own, serve it over WebSocket. For each connection, your server must do the following:

- Start one `hermes acp` process, with Hermes's own environment and `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, as the Hermes agent class gives a process that the core starts.
- Pass each text frame to the process's stdin as one line, and each line of its stdout back as one text frame.
- When the connection closes, end its process: close the process's stdin, and Hermes ends its sessions and exits. Terminate the process if it still runs a few seconds later. The core ends a runtime process only by closing its connection, so a server that keeps the process leaves the processes of every conversation running.
- When the process exits, close the connection. The core sees that a runtime is lost only when its connection closes. A connection left open keeps a running turn waiting until its timeout, and each new session on the process that is gone waits `start_timeout_seconds` and fails.

Then point the harness at it in a profile of your own that sets `transport: websocket` together with `url`, such as the following one. A `url` with the `stdio` transport fails the core's build.

```yaml
extends: core.harness.yaml

backend:
  harness:
    transport: websocket
    url: ws://hermes.internal:8790/acp
    work_root: /var/lib/delegation-core/harness
    bridge:
      bind: "0.0.0.0:8791"
      advertise: "http://voice-agent.internal:8791"
```

The core waits up to `backend.harness.start_timeout_seconds` (60 seconds) for each connection to open, finish its WebSocket handshake, and answer `initialize`, and as long again for each `session/new`. Over stdio, the same limit covers starting the process. `stop_grace_seconds` (5 seconds) bounds only the steps of a close.

Two paths must also reach across:

- Hermes reads each session's instructions from `.hermes.md` in the session's folder, which the core makes under `work_root`. Mount one folder at the same path on both sides, and set `work_root` to that path, because the default, `~/.cache/delegation-core/harness`, is a folder on the core's side only. Run Hermes as the same user as the voice-agent server, or as root. The core makes each runtime process's work folder with mode 0700 for its own user, so a Hermes of another user cannot enter it. Hermes's own image, for example, runs Hermes as the user `hermes`, with UID 10000 unless its `HERMES_UID` names another.
- Hermes calls the core's tools through the tool bridge. Bind the bridge where Hermes can reach it, and set `advertise` to the URL that Hermes uses. A bridge bound on a loopback address, such as the `127.0.0.1` of `core.yaml`, answers only requests whose `Host` header names a loopback address. The MCP SDK's protection against DNS rebinding answers any other request with 421, and Hermes's MCP connect then fails silently, so every task fails. When Hermes reaches the bridge by a name, or through a proxy that keeps the `Host` header, bind a non-loopback address, as the example's `0.0.0.0` does. The bridge serves POST only and answers any other method with 405 at once. Keep that behavior through any proxy, because Hermes probes the URL with HEAD and GET before it connects.

When Hermes cannot see or enter a session's folder, nothing fails. Hermes logs only the warning `configured working directory does not exist` and runs the session without its instructions: the conductor without `texts.conductor_prompt`, and each worker without its composed instructions and the envelope. It can read the project files of its own launch folder instead, such as an `AGENTS.md` there. After you deploy, run one delegation and check that Hermes's log has no such warning.

The core adds no credential to the connection, so serve the agent only where the voice agent alone can reach it. `GET /v1/live/info` shows `backend.harness.url` without its user, password, query, or fragment, because any of them can carry a credential.

## Connect Another ACP Agent

An agent that keeps to the Agent Client Protocol needs no code when it does the following, which the plain agent class relies on:

- It takes an HTTP MCP server with headers in `session/new`, which is an optional capability in ACP. The core gives each session one such server, its door to the tool bridge, with a bearer token in the `Authorization` header.
- It connects to that server and lists its tools within `ready_timeout_seconds` (15 seconds) of `session/new`, before any prompt.
- It needs no `authenticate` call, because its credentials are its own configuration.

An agent that does not do all three fails every delegation. Connect the agent with its own tools turned off: with `agent: acp`, every tool card counts as a call of the session's own MCP server, so neither the commit gate nor the reports see a tool that the agent runs itself. Name the plain agent class, a steering strategy that it can take, and the way to reach it, as the following profile does:

```yaml
extends: core.harness.yaml

backend:
  harness:
    agent: acp
    steer: next_prompt
    command: [my-agent, --acp]
```

The plain agent class puts the instructions at the top of each session's first prompt, takes no mid-turn steer, and passes on every message that the agent streams. An agent with ways of its own, such as a file that it reads its instructions from, a mid-turn steer, or replies of its own that must never be spoken, needs an agent class beside `HermesAgent`. An agent that does not meet the three conditions needs more than configuration too. Refer to [Add an Agent, a Protocol, a Transport, or a Steering Strategy to the Harness](../design/architecture.md#add-an-agent-a-protocol-a-transport-or-a-steering-strategy-to-the-harness).
