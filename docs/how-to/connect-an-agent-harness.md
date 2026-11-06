# Connect an Agent Harness

The harness backend adapter of the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) runs the delegated work on an external agent, such as Hermes, over the Agent Client Protocol (ACP).

The repository ships the harness adapter and its configuration, not the agent: you deploy the agent, and the core connects to it through `backend.harness`. The example's settings are the `config` block of its `frontend-backend-live` entry in [`examples_registry.yaml`](../../examples_registry.yaml). The shipped `core.backend.harness` block connects to Hermes: it names the Hermes agent class (`agent: hermes`) and starts `hermes acp` over stdio. Setting `core.backend.adapter` to `harness`, together with `core.delegation.result_format: speak_display` for the envelope in every worker's instructions, switches the core to the harness.

The adapter needs the packages of the `harness` dependency group in [`pyproject.toml`](../../pyproject.toml), which the app image installs. For a server that you run from the repository, install them with `uv sync --group harness`, or the core's build fails with an error that names the missing packages.

> **Note:** In a Docker Compose deployment, the app container has no Hermes, so serve Hermes over WebSocket, as [Connect over WebSocket](#connect-over-websocket) describes. Add a Compose override for the app service, such as `frontend-backend-live`, that mounts the shared `work_root` at the same path on both sides. For Hermes to reach the tool bridge, bind the bridge on `0.0.0.0:8791` in the container, then expose that port on the Compose network, or publish it on the host's loopback address only (`127.0.0.1:8791:8791`) when Hermes runs on the host. The app container runs the server as root, so run Hermes as root as well. Stdio needs a server that runs on the host, or an image of your own that holds Hermes.

## Deploy Hermes

Hermes runs in its own Python environment, outside the app's image. To deploy it:

1. Install [Hermes](https://github.com/nousresearch/hermes-agent) at the tested commit, `af26acab737db4382815aadcd73c06755471ab52`, with its `acp` and `mcp` extras, in a virtual environment of its own. `hermes acp --check` prints `Hermes ACP check OK`. From commit `eb75627faf` on, `hermes acp` connects to the Hermes gateway daemon instead of running the agent itself, so test a later commit before you use it.
2. Give Hermes a home for the voice agent alone, because every session of the core reads the home's identity and configuration: the folder that `HERMES_HOME` names, or `~/.hermes` by default. The home holds Hermes's own `config.yaml`, its `SOUL.md`, and its model key in the home's `.env` (`NVIDIA_API_KEY` for the `nvidia` provider), which Hermes loads itself. The core never passes it a key.
3. Write the home's `SOUL.md`, the identity that opens the system prompt of every Hermes session. Without one, Hermes writes its own persona there on its first run ("You are Hermes Agent, built by Nous Research ..."), and every conductor and worker session reads that persona beside the core's instructions. Use the following identity:

   ```text
   You work behind a voice assistant, for its callers. Follow the instructions of your project context exactly, use only the tools you are given, and do not narrate your tool calls.
   ```

4. Configure Hermes in its `config.yaml`. Set `model.default` to the backend's default model, which `GET /v1/live/info` reports as `backend.model`. With the `frontend-backend-live` profile, and with `frontend-backend-live/server` while its Super VL NIM runs, that model is Nemotron 3.5 Super VL, `nvidia/nemotron-3.5-super-vl-120b-a12b`. With `frontend-backend-live/single-gpu`, it is Nemotron 3.5 Lightning, which the following example runs on NVIDIA's cloud endpoint:

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
| `model.default` and `model.base_url` | The model that Hermes runs. Run the default model of `backend.slot`, because the core serves that model name only. The core never reads the model that Hermes runs, so it reports no mismatch with that default. A session that names another model gets a nonfatal `responses_validation_failed`, and its tasks still run on Hermes's model, which the server logs as `harness.other_model`. Keep `base_url`: it pins the route that `context_length` describes. Without it, Hermes drops `context_length` when it runs on a route other than the provider's default, such as a local NIM that `NVIDIA_BASE_URL` names. |
| `model.context_length` | The context window that Hermes compresses its own transcript against. Give it at least 65,536 tokens. A worker's session holds the history that the core sends, within `delegation.context_tokens` (24,000 tokens), and its tasks' own rounds. A session that serves the next task of the call holds them until that budget trims the history. Keep the budget well below this window, or Hermes compresses the transcript and drops detail. |
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
2. If the home of Hermes is not `~/.hermes`, export `HERMES_HOME` in the server's environment. A started process gets only the variables that `backend.harness.pass_env` names (`PATH`, `LANG`, `LC_ALL`, `TZ`, and `HERMES_HOME` in the shipped block), `HERMES_ACP_SKIP_CONFIGURED_MCP=1` from the Hermes agent class, and the values of `backend.harness.env`.
3. In the example's `config` block, set `core.backend.adapter` to `harness` and `core.delegation.result_format` to `speak_display`, and start the server.

## Connect over WebSocket

To run Hermes on another host or in a container of its own, serve it over WebSocket. For each connection, your server must do the following:

- Start one `hermes acp` process, with Hermes's own environment and `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, as the Hermes agent class gives a process that the core starts.
- Pass each text frame to the process's stdin as one line, and each line of its stdout back as one text frame.
- When the connection closes, end its process: close the process's stdin, and Hermes ends its sessions and exits. Terminate the process if it still runs a few seconds later. The core ends a runtime process only by closing its connection, so a server that keeps the process leaves the processes of every conversation running.
- When the process exits, close the connection. The core sees that a runtime is lost only when its connection closes. A connection left open keeps a running turn waiting until its timeout, and each new session on the process that is gone waits `start_timeout_seconds` and fails.

Then point the harness at it: in the example's `config` block, set `transport: websocket` together with `url` in `core.backend.harness`, as the following excerpt does. A `url` with the `stdio` transport fails the core's build.

```yaml
config:
  core:
    delegation:
      result_format: speak_display
    backend:
      adapter: harness
      harness:
        agent: hermes
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
- Hermes calls the core's tools through the tool bridge. Bind the bridge where Hermes can reach it, and set `advertise` to the URL that Hermes uses. A bridge bound on a loopback address, such as the default `127.0.0.1`, answers only requests whose `Host` header names a loopback address. The MCP SDK's protection against DNS rebinding answers any other request with 421, and Hermes's MCP connect then fails silently, so every task fails. When Hermes reaches the bridge by a name, or through a proxy that keeps the `Host` header, bind a non-loopback address, as the example's `0.0.0.0` does. The bridge serves POST only and answers any other method with 405 at once. Keep that behavior through any proxy, because Hermes probes the URL with HEAD and GET before it connects.

When Hermes cannot see or enter a session's folder, nothing fails. Hermes logs only the warning `configured working directory does not exist` and runs the session without its instructions: the conductor without `texts.conductor_prompt`, and each worker without its composed instructions and the envelope. It can read the project files of its own launch folder instead, such as an `AGENTS.md` there. After you deploy, run one delegation and check that Hermes's log has no such warning.

The core adds no credential to the connection, so serve the agent only where the voice agent alone can reach it. `GET /v1/live/info` shows `backend.harness.url` without its user, password, query, or fragment, because any of them can carry a credential.

## Connect Another ACP Agent

An agent that keeps to the Agent Client Protocol needs no code when it does the following, which the plain agent class relies on:

- It takes an HTTP MCP server with headers in `session/new`, which is an optional capability in ACP. The core gives each session one such server, its door to the tool bridge, with a bearer token in the `Authorization` header.
- It connects to that server and lists its tools within `ready_timeout_seconds` (15 seconds) of `session/new`, before any prompt.
- It needs no `authenticate` call, because its credentials are its own configuration.

An agent that does not do all three fails every delegation. Connect the agent with its own tools turned off: with `agent: acp`, every tool card counts as a call of the session's own MCP server, so neither the commit gate nor the reports see a tool that the agent runs itself. Replace the shipped `core.backend.harness` block with one that names the plain agent class and the way to reach it, as the following excerpt does:

```yaml
config:
  core:
    delegation:
      result_format: speak_display
    backend:
      adapter: harness
      harness:
        agent: acp
        command: [my-agent, --acp]
```

The plain agent class puts the instructions at the top of each session's first prompt and passes on every message that the agent streams. It takes no mid-turn steer, so a newer request reaches a running task by cancel and re-prompt, as [How the Harness Runs the Work](#how-the-harness-runs-the-work) describes. An agent with ways of its own, such as a file that it reads its instructions from, a mid-turn steer, or replies of its own that must never be spoken, needs an agent class of its own: subclass `AcpAgent` ([`acp/agent.py`](../../src/examples/frontend_backend_live/delegation_core/backend/harness/acp/agent.py)) in a package beside `acp/hermes/`, add it to `AGENTS` in [`acp/agents.py`](../../src/examples/frontend_backend_live/delegation_core/backend/harness/acp/agents.py), and name it in `backend.harness.agent`. An agent that does not meet the three conditions needs more than configuration too.

## How the Harness Runs the Work

For each conversation, the core starts or connects one runtime process for the conductor, which places each new request, and one for the workers, which run the tasks. With `max_running: 1`, no conductor runs, and each request joins the open work, or starts it when none is open. The following rules are the harness's own behavior, with no setting:

- **Newer requests.** The agent class decides how a newer request reaches a running task. Hermes takes it with its own `/steer` at the turn's next model boundary. Any other agent's turn is cancelled when no bridged call is running. After the turn ends with the `cancelled` stop reason, the same session gets the newer request as its next prompt, because the session already holds the task's history. A turn that answers first ends its task, and the newer request starts the next one. If the turn ends in another way after the cancel is sent, the cancel can still reach the agent and stop its next prompt. A fresh session then gets the task's history with the newer request.
- **Sessions.** The session of a task that answered serves the next task of the call when that task's history begins with exactly the items that the session saw. The session then reads only the items that follow. Otherwise, the task opens a fresh session. A session whose task was stopped, failed, or timed out is never reused, and neither is a session whose turn answered before a cancel took effect.
- **Permissions.** The core answers every permission request of the agent with `reject` (`backend.harness.permission`), so no approval request reaches the app.

`backend.harness.agents.hermes` sets two of Hermes's values, and the core's build refuses one that Hermes cannot use:

- `instructions_file` (`.hermes.md`) must be a file that Hermes reads in a session's folder: `.hermes.md`, `HERMES.md`, `AGENTS.override.md`, `AGENTS.md`, `agents.md`, `CLAUDE.md`, `claude.md`, or `.cursorrules`.
- `steer_command` (`"/steer "`) must end with a space, because Hermes reads a command's name up to the first space. Quote it in YAML, which drops the trailing space of an unquoted value.

The text that the core puts before a prompt that starts with `/`, so that Hermes never runs it as a command, is the `backend.harness.agents.hermes.command_guard` entry of `agent_prompts.Core` in the example's `prompts.yaml` (`Request: `). It must not start with `/`.

## Limits

The harness has the following limits:

- With Hermes, after a model call of the turn completed, Hermes's own reply is the turn's answer, because ACP does not mark who wrote a message. Such a reply is an `Error: ...` or the model's reasoning when the model wrote no answer.
- The commit gate, which holds back the calls that the agent makes through the tool bridge while a newer request waits to land, cannot hold the agent's own tools, such as Hermes toolsets that its operator turns on. They run inside the agent. With `hermes`, they are reported after the fact, and with `acp`, they are not reported at all. Give the agent no tools of its own where the gate matters.
- The agent runs one model, the one that its operator configures. The core serves only the default model of `backend.slot`, and the background settings check reports any other model, and any setting that the agent sets itself, such as a reasoning effort, as a nonfatal `responses_validation_failed`. A request's instructions and functions are fixed when its worker starts, so a later change reaches only new workers.
- A session serves the next task only when nothing came between. An item that the session never read, such as a typed message that came with the app's outputs, or a history that `delegation.context_tokens` trimmed, gives the next task a fresh session. Of two tasks that run at once, only the first that answers keeps its session.
