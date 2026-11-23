# Nemotron Voice Agent Client

The Nemotron Voice Agent Client is the browser front end for the [Nemotron Voice Agent](../../README.md) blueprint. It gives you a real-time, interruptible voice conversation with the agent, along with controls to switch models and prompts, preview TTS voices, watch live latency metrics, and follow the conversation transcript. The Generic Assistant on LiveKit uses its own browser files in `clients/livekit/`.

It is a React and TypeScript single-page app built with [Vite](https://vite.dev/) and the [Pipecat Client SDK](https://docs.pipecat.ai/client/introduction). The client connects to the Python backend (`src/nemotron_voice_agent/pipecat/runtime/server.py`) over WebRTC or WebSocket and reads its `/api/*` endpoints for session, service, and voice configuration. In a deployed stack the backend serves this client's production build from `clients/pipecat/dist/`, so you normally reach the UI at `https://localhost:7860` rather than running it on its own.

## Features

- **Cascaded pipeline view**: follow the ASR → LLM → TTS flow live.
- **Dual transport**: WebRTC (recommended) or WebSocket.
- **Runtime service switching**: add or remove LLM, ASR, and TTS services without redeploying.
- **Prompt management**: pick a built-in persona or write a custom system prompt.
- **Voice selection**: browse and preview TTS voices with language filtering.
- **Audio visualizers**: real-time input and output waveform display.
- **Metrics dashboard**: latency charts for time to first byte (TTFB) and time to first answer token (TTFAT), token usage, and connection status.
- **Conversation transcript**: live ASR and bot-response display.
- **Webcam vision panel**: live webcam input for the multimodal Omni Subagents example.
- **Event log**: an **EVENT LOG** tab in the developer view that prints the requests and events of the Frontend/Backend Live backend, for an example that lists the `event_log` capability.

The **TTFAT Metrics** section of the **METRICS** tab shows the latest TTFAT value and a chart of the last 10 values for each large language model (LLM) processor. TTFAT runs from the LLM request to the first answer token, or to the start of a tool call. It equals the LLM TTFB plus the time the model thinks before it answers. With reasoning off, TTFAT is almost the same as TTFB. A turn that calls a tool makes two LLM requests, so it reports two TTFAT values. TTFAT does not include speech synthesis time.

## Getting started

### Prerequisites

- Node.js 20 or newer and npm.
- A running Nemotron Voice Agent backend. See the repo [Getting Started](../../docs/01-getting-started.md) guide.

### Run in development

```bash
npm install
npm run dev
```

The Vite dev server starts at `http://localhost:5173` with hot-module reload, which is convenient for fast UI iteration. The full experience also needs the backend running for the `/api/*` endpoints and the WebRTC/WebSocket session, so the simplest way to exercise the complete UI is the backend-served build below.

### Build for production

```bash
npm run build
```

The build is type-checked with `tsc` and emitted to `dist/`. The Python server serves it automatically from `clients/pipecat/dist/`, so after building you reach the UI at `https://localhost:7860`. Rebuild whenever you change the client and redeploy.

### Lint

```bash
npm run lint
```

ESLint runs the TypeScript, React Hooks, and React Refresh rules defined in `eslint.config.js`.

## Event log

An example whose entry in [`examples_registry.yaml`](../../examples_registry.yaml) lists the `event_log` capability gets an **EVENT LOG** tab in the developer view. With that capability, the Frontend/Backend Live UI bot relays the OpenAI-format events of its backend to the browser as RTVI server messages of type `live-event`. Refer to [Event Relay](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#event-relay) for the message and its limits.

The tab is a console of requests and events:

- Filter by type: **Requests**, **Lifecycle**, **Tools**, **Text**, and **Errors**.
- Follow the newest event, and select **Jump to latest** after you scroll away.
- Select **Copy JSON** to copy the events, or **Clear** to empty the list.
- Select a row to see its whole JSON.

The tab keeps the newest 2,000 events. Its code and styles load only for an example that lists the capability, so other examples show nothing of them and do not download their code. The center panel (`src/components/content/CenterPanel.tsx`) loads `src/components/events/EventLogScreen.tsx` lazily after `src/liveEvents/gate.ts` allows it. A small store outside React (`src/liveEvents/store.ts`) holds the current call's events, so the log of a call that just ended stays on screen until the next call starts.

The tests feed the log a call recorded from the real airline tools (`tests/airline-recorded-call.ts`). The file is generated, and a unit test fails when it is out of date. Run this command from the repository root after you change an airline tool, and add `--check` to verify without writing:

```bash
uv run python scripts/export_airline_demo.py      # writes clients/pipecat/tests/airline-recorded-call.ts
```

### Test

The client has no `npm test` script. With Node.js 22.18 or newer, run the tests with `node --test` from `clients/pipecat`:

```bash
node --test tests/*.test.ts
```

The event log tests are in `live-events`. For the Python tests and the scripted caller check, refer to [Verify the Desk](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#verify-the-desk).

## Backend endpoints

The client reads its configuration from the backend (`src/nemotron_voice_agent/pipecat/runtime/server.py`) and starts sessions through these endpoints:

| Endpoint | Purpose |
| --- | --- |
| `/api/deployment` | Active example, available services, and UI capabilities |
| `/api/session-config` | Prompts and default session settings |
| `/api/tts-config` | Available TTS voices and languages |
| `/api/ice-servers` | STUN/TURN configuration for WebRTC |
| `/api/webcam-config` | Webcam capture defaults for multimodal examples |
| `/api/start` | Start a pipeline session |

## Learn more

- [Nemotron Voice Agent](../../README.md): the full blueprint, examples, and deployment guides.
- [Getting Started](../../docs/01-getting-started.md): prerequisites and how to run the backend.
- [Pipecat Client SDK](https://docs.pipecat.ai/client/introduction): the client framework this app is built on.
