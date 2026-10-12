# Nemotron Voice Agent Client

The Nemotron Voice Agent Client is the browser front end for the [Nemotron Voice Agent](../README.md) blueprint. It gives you a real-time, interruptible voice conversation with the agent, along with controls to switch models and prompts, preview TTS voices, watch live latency metrics, and follow the conversation transcript.

It is a React and TypeScript single-page app built with [Vite](https://vite.dev/) and the [Pipecat Client SDK](https://docs.pipecat.ai/client/introduction). The client connects to the Python backend (`src/server.py`) over WebRTC or WebSocket and reads its `/api/*` endpoints for session, service, and voice configuration. In a deployed stack the backend serves this client's production build from `client/dist/`, so you normally reach the UI at `https://localhost:7860` rather than running it on its own.

## Features

- **Cascaded pipeline view**: follow the ASR → LLM → TTS flow live.
- **Dual transport**: WebRTC (recommended) or WebSocket.
- **Runtime service switching**: add or remove LLM, ASR, and TTS services without redeploying.
- **Prompt management**: pick a built-in persona or write a custom system prompt.
- **Voice selection**: browse and preview TTS voices with language filtering.
- **Audio visualizers**: real-time input and output waveform display.
- **Metrics dashboard**: time-to-first-byte latency charts, token usage, and connection status.
- **Conversation transcript**: live ASR and bot-response display.
- **Webcam vision panel**: live webcam input for the multimodal Omni Subagents example.
- **Live Console** (`/live.html`): a developer console for the [Frontend/Backend Live](../src/examples/frontend_backend_live/README.md) example. It is a static page in `public/live/` (no build step of its own) that starts live sessions over WebRTC, edits the session config, shows every protocol event and delegation, shows the talker's and the backend's live conversation history and system prompts (each part marked as coming from the client, the config, or the server), runs client-side delegation, and sends steering commands.

## Getting started

### Prerequisites

- Node.js 20 or newer and npm.
- A running Nemotron Voice Agent backend. See the repo [Getting Started](../docs/01-getting-started.md) guide.

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

The build is type-checked with `tsc` and emitted to `dist/`. The Python server serves it automatically from `client/dist/`, so after building you reach the UI at `https://localhost:7860`. Rebuild whenever you change the client and redeploy.

### Lint

```bash
npm run lint
```

ESLint runs the TypeScript, React Hooks, and React Refresh rules defined in `eslint.config.js`.

## Backend endpoints

The client reads its configuration from the backend (`src/server.py`) and starts sessions through these endpoints:

| Endpoint | Purpose |
| --- | --- |
| `/api/deployment` | Active example, available services, and UI capabilities |
| `/api/session-config` | Prompts and default session settings |
| `/api/tts-config` | Available TTS voices and languages |
| `/api/ice-servers` | STUN/TURN configuration for WebRTC |
| `/api/webcam-config` | Webcam capture defaults for multimodal examples |
| `/api/start` | Start a pipeline session |

## Learn more

- [Nemotron Voice Agent](../README.md): the full blueprint, examples, and deployment guides.
- [Getting Started](../docs/01-getting-started.md): prerequisites and how to run the backend.
- [Pipecat Client SDK](https://docs.pipecat.ai/client/introduction): the client framework this app is built on.
