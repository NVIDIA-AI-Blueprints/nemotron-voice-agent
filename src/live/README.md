# Live Gateway

The Live gateway serves the live session protocol, which follows GPT-Live, at `/v1/live`. It owns each session: the
order of its commands and of its close, the wire schema, and the relay of each delegation's events. It knows nothing
about Pipecat or the cascade, and it reaches the frontend adapter and the delegation core only through the ports in
[`ports.py`](ports.py). The [Frontend/Backend Live example](../examples/frontend_backend_live/README.md) plugs its own
parts into it, and its README describes what a live session client sends and receives. For the design, refer to the
[Frontend/Backend Live Architecture](../../docs/design/architecture.md) page.

## Routes

The gateway serves the following routes:

| Route | What it serves |
| --- | --- |
| `POST /v1/live/sessions` | A WebRTC session: the client's SDP offer and settings in, the answer out |
| `WS /v1/live/sessions` | A primary WebSocket session: `session.start` first, then commands and base64 audio events |
| `WS /v1/live/sessions/{id}/attach` | A sideband connection that observes a session and sends trusted commands |
| `GET /v1/live/sessions/{id}/history` | What the session's models hold now: the frontend adapter's half and the delegation part's half |
| `GET /v1/live/info` | The server's voices and session limit, plus what the example's `describe` hook adds |
| `POST /v1/live/delegate` | One backend round for a client app that does its own delegated work, such as the Live Console in client mode |

Any other path under `/v1/live` answers 404. Every route requires the bearer token when `LIVE_API_TOKEN` is set.

## Delegation Modes

A session's `delegation.type` selects the part that handles its delegations:

- **Client mode** (`client`, the default when the session sets no `delegation`). The client delegation
  ([`client_delegation.py`](client_delegation.py)) announces each delegation with `session.delegation.created`. The
  client app does the work and answers with appends that name the delegation. The server builds no delegation core for
  the session, which refuses `response.item.create` and `response.create`.
- **Responses mode** (`responses`). The server delegation ([`server_delegation/`](server_delegation/)) passes each
  delegation to the [delegation core](../delegation_core/README.md), as the next section describes.

## Server Delegation

The server delegation holds no delegation logic: the delegation core decides what happens to each request, and the
frontend adapter decides what the caller hears. For each session, the server delegation does the following:

- It opens the session's conversation in the core with `POST /v1/conversations`, and the session fails to start when
  the core cannot create it within 10 seconds. It deletes the conversation when the session closes.
- It sends each request, in order, to `POST /v1/responses` with the session's `delegation.responses` settings, read
  for each request, so a `session.update` applies to the requests after it.
- It relays each event that comes back to the frontend adapter and, in a `response.event` that names the
  `delegation_id`, to the client. When a stream ends early, the client gets a `response.failed` with `server_error`.
- It holds the client's call results and typed messages from `response.item.create`, at most 512 results and 16
  messages, and sends them at `response.create`.
- It checks the `delegation.responses` settings in the background with `POST /v1/responses/input_tokens`, and it
  reports a setting that the backend cannot honor as a nonfatal `responses_validation_failed` error.

## How an Example Mounts It

[`src/server.py`](../server.py) reads the `live:` block of the serving example's entry in
[`examples_registry.yaml`](../../examples_registry.yaml) and passes its hooks to `mount_live` ([`mount.py`](mount.py)).
The block names the following hooks:

| Hook | What it does |
| --- | --- |
| `factory` | Builds each session's frontend adapter. It is an async function of the session's settings (`SessionConfig`), its transport (`webrtc` or `websocket`), and its WebRTC peer (`None` on a WebSocket) that returns `SessionParts`. It raises `ProtocolError` to refuse a session. `LIVE_ADAPTER_FACTORY` names another factory, as `module:function`. |
| `core` | Builds the gateway's one delegation core client, a `ResponsesClient`, on the first Responses-mode session or console round. At shutdown, after every session has closed, the gateway releases it with `aclose()`. |
| `describe` | Describes the example for `GET /v1/live/info`. |

The Frontend/Backend Live example's hooks are in its [`factory.py`](../examples/frontend_backend_live/factory.py). To
write another frontend adapter, refer to [Add a Frontend Adapter](../../docs/design/architecture.md#add-a-frontend-adapter).

When the frontend adapter reports a failure, the client gets an `error` with code `pipeline_error` if a client command
failed or the session cannot go on, and an `info` notice with the same code if the session recovers and no command
caused it. The gateway also logs each one as a warning.

## Settings

The gateway reads the following environment variables when the server starts:

| Variable | Default | What it sets |
| --- | --- | --- |
| `LIVE_ENABLED` | Unset | Whether the routes are served: `1`, `true`, `yes`, or `on` serves them, and any other value turns them off. When it is unset, they are served when `LIVE_API_TOKEN` is set or when the deployment runs the Live example alone. |
| `LIVE_API_TOKEN` | Unset | The bearer token that every route requires. Without one, the server logs a warning, and anyone who can reach it can use the routes. |
| `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS` | `8`, `1800` | The sessions at once and the length of one session, in seconds. |
| `LIVE_MAX_APPENDS_PER_MINUTE` | `64` | The appends that one session accepts in any 60 seconds, counting `session.instructions.append`, `session.thinking.append`, and `session.commentary.append` together. An append over the limit gets a `rate_limit_exceeded` error, and the gateway neither passes it to the frontend adapter nor acknowledges it. |
| `LIVE_ICE_URLS` | Unset | More ICE servers for the WebRTC peer, comma-separated, beside the TURN server. |
| `LIVE_MAX_CONSOLE_ROUNDS`, `LIVE_ALLOWED_HOSTS` | `2`, unset | The `POST /v1/live/delegate` rounds at once, and, with no token, the host names besides IP addresses and `localhost` that the route answers to. |
| `UVICORN_WORKERS` | `1` | The server's workers. With more than one, no Live route is served, because a session lives in one process. |

For the example's own variables, refer to [Environment Variables](../../docs/design/architecture.md#environment-variables).

## Tests

The gateway's tests in `tests/unit/live/` run offline. Run them from the repository root:

```bash
uv run pytest tests/unit/live -v
```
