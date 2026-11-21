# Connect an Agent Harness

The harness backend adapter of the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) runs delegated work on an external agent. The agent connects through the Agent Client Protocol (ACP).

The repository ships the harness adapter and its configuration, not the agent: you deploy the agent, and the core connects to it through `backend.harness`. The example's settings are the `config` block of its `frontend-backend-live` entry in [`examples_registry.yaml`](../../examples_registry.yaml). The shipped `core.backend.harness` block connects to Hermes: it names the Hermes agent class (`agent: hermes`) and starts `hermes acp` over stdio. Setting `core.backend.adapter` to `harness`, together with `core.delegation.result_format: speak_display` for the envelope in every worker's instructions, switches the core to the harness.

The repository UI bot keeps the `llm` adapter through its own `ui_bot.core` block, so only Live sessions in [Responses mode](use-live-gateway.md#delegation-modes) reach the agent. To run the UI bot on the agent too, set `ui_bot.core.backend.adapter` to `harness`. Also set `bind` and `advertise` in `ui_bot.core.backend.harness.bridge` to a port of its own, because the Live core's tool bridge keeps its port until the server stops. The UI bot builds a core with its own bridge for each bot session, so run one UI bot session at a time.

ACP needs the `harness` dependency group in [`pyproject.toml`](../../pyproject.toml). For a server that you run from the repository, install it with `uv sync --group harness`. The stock app image includes that group. The group does not install Hermes itself.

> **Note:** The stock app image has no Hermes. For ACP in Docker Compose, serve Hermes over WebSocket, as [Connect over WebSocket](#connect-over-websocket) describes. Add a Compose override for the app service, such as `frontend-backend-live`, that mounts the shared `work_root` at the same path on both sides. For Hermes to reach the tool bridge, bind the bridge on `0.0.0.0:8791` in the container, then expose that port on the Compose network, or publish it on the host's loopback address only (`127.0.0.1:8791:8791`) when Hermes runs on the host. The app container runs the server as root, so run Hermes as root as well. Stdio needs a host-native server or a custom image that contains the agent environment.

## Deploy Hermes

Hermes runs in its own Python environment, outside the app's image. To deploy it:

1. Install [Hermes](https://github.com/nousresearch/hermes-agent) with its `acp` and `mcp` extras, in a virtual environment of its own. `hermes acp --check` prints `Hermes ACP check OK`.
2. Give Hermes a home for the voice agent alone, because every session of the core reads the home's identity and configuration: the folder that `HERMES_HOME` names, or `~/.hermes` by default. The home holds Hermes's own `config.yaml`, its `SOUL.md`, and its model key in the home's `.env` (`NVIDIA_API_KEY` for the `nvidia` provider), which Hermes loads itself. The core never passes it a key.
3. Write the home's `SOUL.md`, the identity that opens the system prompt of every Hermes session. Without one, Hermes writes its own persona there on its first run ("You are Hermes Agent, built by Nous Research ..."), and every conductor and worker session reads that persona beside the core's instructions. Use the following identity:

   ```text
   You work behind a voice assistant, for its callers. Follow the instructions of your project context exactly and use only the tools you are given. Give concise factual updates through the provided work tools when useful.
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
| `model.default` and `model.base_url` | The model that Hermes runs. Run the default model of `backend.slot`, because the core serves that model name only and does not check which model Hermes runs. A session that names another model gets a nonfatal `responses_validation_failed`, and its tasks still run on Hermes's model, which the server logs as `harness.other_model`. Keep `base_url`: it pins the route that `context_length` describes. Without it, Hermes drops `context_length` when it runs on a route other than the provider's default, such as a local NIM that `NVIDIA_BASE_URL` names. |
| `model.context_length` | The context window that Hermes compresses its own transcript against. Give it at least 65,536 tokens. A worker's session holds the history that the core sends, within `delegation.context_tokens` (24,000 tokens), and its tasks' own rounds. A worker session that serves later tasks of the call keeps what it read, also after that budget trims the history, so its transcript can grow until Hermes compresses it. Keep the budget well below this window, or Hermes compresses the transcript and drops detail. |
| `platform_toolsets.acp: [no_mcp]` | Excludes MCP servers configured in the Hermes profile. ACP adds the session's own MCP server, and Hermes can still load native or plugin tools. Hermes can warn about `unknown name(s): no_mcp` while the session bridge still connects. Keep `no_mcp` in the configuration file because saving the ACP platform through `hermes tools` removes it. |
| No `mcp_servers` | Hermes starts every MCP server that its home lists when it opens its first session. `no_mcp` keeps their tools out of the sessions, but a server that never connects makes each new session wait up to `mcp_discovery_timeout` (1.5 seconds). |
| `tools.tool_search.enabled: "off"` | The session's tools stay in front of the model instead of behind a search tool. |
| `agent` guidance settings and `coding_context` | No coding-agent brief in the system prompt: the core's instructions are the brief. |
| `memory` and `auxiliary` | No memory between calls, and no hidden model calls to title or review each session. |
| `auth.adopt_external_logins: false` | Hermes never borrows another tool's login that it finds on the host. |
| `timeouts.mcp.tool_call` | Well above `delegation.client_tool_timeout_seconds` (60 seconds), which counts from the end of the response with the call. That way, a held app call ends by the core's deadline instead of a Hermes timeout. |

Use a Hermes gateway that reports prompt failures and cancellation as typed ACP outcomes. Check both outcomes after ACP initialization. Configure the model in Hermes. To keep request parameters fixed across model calls, including continuations, use the optional [Hermes model binding](../../src/examples/frontend_backend_live/delegation_core/backend/harness/hermes_profile.py) in a dedicated Hermes home.

The example sets no `agent.max_turns`. When a turn still runs after `backend.harness.turn_timeout_seconds` (300 seconds, excluding a worker's waits for the app), the core cancels it. A worker's task fails on timeout, and a conductor gets a new session. Set `backend.harness.conductor_timeout_seconds` to give the conductor a separate limit. This setting is unset by default.

Newer input can interrupt a conductor turn before it applies a decision. After cancellation settles, the conductor retries with the pending inputs and current work records.

The core worker timer does not limit a task whose work [outlives the call](use-live-gateway.md#work-that-outlives-the-call). That work still ends at `delegation.after_call_seconds`, on a stop, or at a limit the agent sets itself. Each bridged call still has `backend.harness.tool_timeout_seconds` (60 seconds).

Hermes keeps each session's transcript in the `state.db` file of its home, which holds the instructions and the conversation of every call. When a provider error fails a model call, it also writes the whole request, with the instructions and the conversation, to a `request_dump_*.json` file in the `sessions` folder of its home. Clear both as your data policy requires.

## Completion and Usage Evidence

ACP reports whether a native prompt completed, failed, or was cancelled. A successful turn still needs a usable answer or an accepted routing decision. The server records per-invocation token usage only when the runtime supplies it. Refer to [Usage Observations](../../src/examples/frontend_backend_live/delegation_core/README.md#usage-observations).

## Connect over Stdio

With the `stdio` transport, the core starts `hermes acp` for each conversation. It starts a worker process and, when `max_running` is above 1, a separate conductor process. It starts another process when all four turn slots of a process are in use.

1. Put `hermes` on the server's `PATH`, or set `backend.harness.command` to its full path.
2. If the home of Hermes is not `~/.hermes`, export `HERMES_HOME` in the server's environment. A started process gets only the variables that `backend.harness.pass_env` names (`PATH`, `LANG`, `LC_ALL`, `TZ`, and `HERMES_HOME` in the shipped block), `HERMES_ACP_SKIP_CONFIGURED_MCP=1` from the Hermes agent class, and the values of `backend.harness.env`.
3. In the example's `config` block, set `core.backend.adapter` to `harness` and `core.delegation.result_format` to `speak_display`, and start the server.

### Enable Native Steering for Local Hermes Workers

For mid-turn corrections over a local stdio connection, export an absolute `HERMES_HOME` path in the server environment. Use a dedicated Hermes home owned by the server user, with directory permissions `0700`. Install the observer from the repository root.

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
import os
from pathlib import Path
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.observer import install_observer

install_observer(Path(os.environ["HERMES_HOME"]))
PY
```

In that home's `config.yaml`, add `voice-agent-steering-evidence` to the existing `plugins.enabled` list. After you update the voice agent, rerun the installation command to update the observer. Restart the profile's Hermes gateway after you install or update the observer.

The adapter uses native steering when the installed observer matches the repository version, Hermes supplies the worker's session identity, and the observer verifies its running Chat Completions request. Every offered tool must belong to that worker's assigned Core bridge.

## Connect over WebSocket

To run Hermes on another host or in a container of its own, serve it over WebSocket. For each connection, your server must do the following:

- Start one `hermes acp` process, with Hermes's own environment and `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, as the Hermes agent class gives a process that the core starts.
- Pass each text frame to the process's stdin as one line, and each line of its stdout back as one text frame.
- When the connection closes, close the process's stdin. Terminate the process if it still runs a few seconds later. If Hermes uses a separate gateway daemon, manage that daemon separately. Closing the viewer confirms only that the ACP connection closed.
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
- Hermes calls the core's tools through the tool bridge. Bind the bridge where Hermes can reach it, and set `advertise` to the URL that Hermes uses. A bridge bound on a loopback address, such as the default `127.0.0.1`, answers any request whose `Host` header does not name a loopback address with 421. Hermes's MCP connect then fails silently, so every task fails. When Hermes reaches the bridge by a name, or through a proxy that keeps the `Host` header, bind a non-loopback address, as the example's `0.0.0.0` does. The bridge serves POST only and answers any other method with 405 at once. Keep that behavior through any proxy, because Hermes probes the URL with HEAD and GET before it connects.

When Hermes cannot see or enter a session's folder, nothing fails. Hermes logs only the warning `configured working directory does not exist` and runs the session without its instructions: the conductor without `texts.conductor_prompt`, and each worker without its composed instructions and the envelope. It can read the project files of its own launch folder instead, such as an `AGENTS.md` there. After you deploy, run one delegation and check that Hermes's log has no such warning.

The core adds no credential to the connection, so serve the agent only where the voice agent alone can reach it. Across hosts, encrypt the connection to the agent and the agent's connection to the tool bridge, because they carry the bridge's bearer token and the conversation. For example, serve the agent at a `wss://` URL and the bridge through a TLS proxy at an `https://` `advertise` URL, or connect the hosts through an encrypted tunnel. `GET /v1/live/info` shows the agent's `url` and the bridge's `advertise` URL without their user, password, query, or fragment, because any of them can carry a credential.

## Connect Another ACP Agent

An agent that keeps to the Agent Client Protocol needs no code when it does the following, which the plain agent class relies on:

- It takes an HTTP MCP server with headers in `session/new`, which is an optional capability in ACP. The core gives each session one such server, its door to the tool bridge, with a bearer token in the `Authorization` header.
- It connects to that server and lists its tools within `ready_timeout_seconds` (15 seconds) of `session/new`, before any prompt.
- It needs no `authenticate` call, because its credentials are its own configuration.
- It calls the tool bridge's tools without asking for permission.

An agent that misses one of the first three fails every delegation. The core declines every permission request, including requests for app functions, caller messages, and conductor controls. An unresolved placement fails rather than guessing a target. Connect the agent with its native tools and native ask-user mechanisms disabled where Core admission must apply. The backend publishes questions through `caller_message` or its answer. A native permission refusal does not mean the caller heard or rejected a question. Replace the shipped `core.backend.harness` block with one that names the plain agent class and its connection, as follows:

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

The core logs each permission request that it answers during a turn as `harness.permission`. After you connect an agent, run one delegation that calls one of the app's functions or a tool source. Check that the server log shows that call and no `harness.permission` line.

The plain agent class puts the instructions at the top of each session's first prompt and passes on every message that the agent streams. When the agent advertises `session/close`, the core closes each session that it stops using. Otherwise, a session lives until its process ends. An agent with its own instructions file or message handling needs an agent class of its own: subclass `AcpAgent` ([`acp/agent.py`](../../src/examples/frontend_backend_live/delegation_core/backend/harness/acp/agent.py)) in a package beside `acp/hermes/`, add it to `AGENTS` in [`acp/agents.py`](../../src/examples/frontend_backend_live/delegation_core/backend/harness/acp/agents.py), and name it in `backend.harness.agent`.

Keep runtime-specific session reply handling, verified steering, and tool-origin checks in your `AcpAgent` subclass. Use the base class's defaults for hooks your agent does not need. The core owns routing and admission of business tool calls. Report support for restricted invocations only when the runtime enforces the allowed tool set.

## How the Harness Runs the Work

The harness runs model placement above `max_running: 1`. At one worker, Core sends new input to the open task or continues the last eligible thread. The [routing decision contract](../../src/examples/frontend_backend_live/delegation_core/README.md#complete-routing-decisions) is shared with the LLM Agent backend.

With [native steering enabled](#enable-native-steering-for-local-hermes-workers), the adapter can send a correction to an active Hermes turn. It records input as applied after it verifies the correction reached the model. When it confirms the active turn ended without receiving the correction, the adapter sends the correction with the task's next prompt. Plain ACP and Hermes without verified steering cancel the obsolete prompt and send the input after that prompt and tracked tool activity settle. The adapter reuses the same native session when safe, or continues from recorded task history in a fresh session.

Stock Hermes ACP cannot enforce invocation-level tool restrictions. Core uses the configured backend LLM for control-only and read-only verification calls when the agent cannot enforce those restrictions. Check `backend.restricted_execution` in `GET /v1/live/info` for the selected execution path.

The harness can reuse an idle native session for its exact task continuation when the instructions and tools match. If the bridge reaches its request retention limit, the next continuation uses a new session with the recorded task history. A task that [outlives the call](use-live-gateway.md#work-that-outlives-the-call) can continue using backend tools. Caller-facing messages and app functions require an attachment.

`backend.harness.agents.hermes.instructions_file` defaults to `.hermes.md`. If you set it to a different file, choose a filename that Hermes reads as session instructions. The `backend.harness.agents.hermes.command_guard` prompt text keeps caller wording that starts with `/` from becoming a Hermes command.

## Limits

- Core authorizes only tools that run through its bridge. Disable native tools and native ask-user mechanisms where Core must control actions. Hermes can load native or plugin tools even with `platform_toolsets.acp: [no_mcp]`.
- The agent runs the model its operator configured. A session that requests settings the agent does not expose can receive a nonfatal `responses_validation_failed` error.
- The worker's initial instructions and offered tools set its permission ceiling. Later requests can restrict those app tools. New tools require a new worker.
