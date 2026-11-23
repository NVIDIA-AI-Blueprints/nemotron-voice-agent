# Connect an Agent Harness

The [Frontend/Backend Live example](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md) can run delegated work on an agent that you operate. The `harness` backend connects to that agent through the Agent Client Protocol (ACP). Use standard input and output (stdio) to start a local agent, or WebSocket to connect to a running deployment.

## Deploy Hermes

1. Install [Hermes](https://github.com/nousresearch/hermes-agent) with the `acp` and `mcp` extras in its own environment. Run `hermes acp --check` and look for `Hermes ACP check OK`.
2. Give the voice agent a dedicated Hermes home through `HERMES_HOME`. Hermes defaults to `~/.hermes`. Hermes reads its model credentials from that home's `.env`. Set the model in `config.yaml` to the `backend.model` value from Live's `GET /v1/live/info`.
3. Set `SOUL.md` in that home to a short identity such as “You work behind a voice assistant. Follow the project instructions and give callers concise, factual updates.” Hermes adds this identity alongside the instructions for each job.

This example uses Nemotron 3.5 Lightning on NVIDIA Cloud with `NVIDIA_API_KEY` in the Hermes home. Use the model, endpoint, and context length of your deployment.

```yaml
model:
  provider: nvidia
  default: nvidia/nemotron-3.5-lightning-30b-a3b
  base_url: https://integrate.api.nvidia.com/v1
  context_length: 131072
platform_toolsets:
  acp: [no_mcp]
tools:
  tool_search: {enabled: "off"}
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
  title_generation: {enabled: false}
  background_review: {enabled: false}
auth:
  adopt_external_logins: false
timeouts:
  mcp:
    tool_call: 3600
```

Keep the home free of shared Model Context Protocol (MCP) servers and disable native tools that callers should not be able to use. Live supplies each agent session with its list of app tools through the MCP bridge. Treat the Hermes home as conversation data, including its `state.db` and request dumps in `sessions/`.

## Connect over Stdio

For a host-native Voice Agent server, install the ACP and MCP dependencies with `uv sync --group harness`. The stock app image includes these dependencies. Use [WebSocket](#connect-over-websocket) with that image, or build an image that includes Hermes for stdio.

In the `pipecat.examples.frontend-backend-live` entry of [`examples_registry.yaml`](../../examples_registry.yaml), merge these settings into the existing `config` block and restart the app.

```yaml
config:
  core:
    delegation:
      result_format: speak_display
    backend:
      adapter: harness
```

The shipped `core.backend.harness` block starts `hermes acp`. Put `hermes` on the server's `PATH` or set `core.backend.harness.command` to the full command. Export `HERMES_HOME` in the server environment if you use a custom location. To pass other environment variables, add their names to `core.backend.harness.pass_env`.

## Connect over WebSocket

Run your own ACP WebSocket endpoint and merge this excerpt into the example's `config` block. The WebSocket endpoint connects the server to Hermes. The MCP bridge lets Hermes call back into the server.

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

- **Share the work directory.** Mount `work_root` at the same absolute path on both sides. Hermes reads each session's instructions from `.hermes.md` in that directory tree. Run Hermes as the same user as the Voice Agent server or as root so it can read the private session folders.
- **Make the bridge reachable.** Set `advertise` to the URL Hermes can reach. If a client sends an HTTP `Origin` header, its scheme, host, and port must match the bridge URL. Use a non-loopback bind address if Hermes reaches it through a host name. A loopback bind can reject the host name with HTTP 421. In Docker Compose, publish the port on host loopback only if Hermes runs on the host, for example `127.0.0.1:8791:8791`. Across hosts, use `wss://` and `https://` or an encrypted private tunnel. Restrict access to the ACP endpoint to the Voice Agent server.
- **Manage the remote process.** If you wrap `hermes acp`, start a process per connection with `HERMES_ACP_SKIP_CONFIGURED_MCP=1`. Reserve stdout for ACP JSON and relay one line per WebSocket text frame. Close the socket when the process exits, and end the process when the socket closes.

## Verify the Connection

Use a Live session in [Responses mode](use-live-gateway.md#delegation-modes) and run a delegation that calls one of your app's function tools. Verify that the tool receives the request and that Live delivers the result. For WebSocket, check the Hermes log for `configured working directory does not exist` if the agent ignores the job instructions. Choose an ACP build that reports failures and cancellations as protocol outcomes so Live can distinguish them from answers. [Troubleshooting](../06-troubleshooting.md#frontendbackend-live)

`core.backend.harness.max_running` defaults to 3 workers per call. The Hermes adapter defaults to 4 concurrent turns per process, including turns waiting for a tool result. Set `max_running: 1` for a single worker. Keep `max_running` within `turns_per_process`. Refer to [harness settings](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/backend/harness/settings.py) for timeouts and other defaults.

The repository UI bot uses its own `ui_bot.core` settings. To use the harness there too, set `ui_bot.core.backend.adapter: harness` and configure `ui_bot.core.backend.harness.bridge` with a port separate from Live's. Run one UI bot session at a time with that bridge.

## Connect Another ACP Agent

Set `core.backend.harness.agent: acp` and supply its `command` for stdio or its `url` for WebSocket. The plain ACP adapter includes instructions in the first prompt. Your agent needs to accept an HTTP MCP server and its authorization header in `session/new`, list its tools before the first prompt, and use its own model credentials. Core declines ACP permission requests. Configure the agent to call the bridge without prompting for permission.

An agent with custom instruction files, tool events, or terminal outcomes can subclass [`AcpAgent`](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/backend/harness/acp/agent.py). Register it in [`AGENTS`](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/backend/harness/acp/agents.py) and select it through `core.backend.harness.agent`.

## Limits

- The harness runs the model and request settings configured in the deployed agent. Set output limits and reasoning settings there. Live bridges app function tools.
- Stock Hermes can load its own native tools. Core uses the configured backend large language model (LLM) for control-only and read-only verification turns when the agent cannot enforce the required tool restrictions. Check `backend.restricted_execution` in `GET /v1/live/info`.
- For corrected input, the ACP adapters let tool calls already in flight settle, cancel the old prompt if it is still active, and send the correction. They reuse a session when safe. For routing and result handling, refer to the [Delegation Core](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/README.md).
