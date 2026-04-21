# Frontend Directory

This directory contains both WebRTC and WebSocket UI implementations for the Nemotron Voice Agent.

## Structure

```
frontend/
├── Dockerfile           # Unified Dockerfile for both UIs
├── webrtc_ui/          # WebRTC UI (React/Vite application)
└── websocket_ui/       # WebSocket UI (Static HTML)
```

## Usage

The UI type is automatically selected based on the `TRANSPORT` environment variable in your `.env` file:

- `TRANSPORT=WEBRTC` (default) - Uses the WebRTC UI
- `TRANSPORT=WEBSOCKET` - Uses the WebSocket UI

## Running UI Service

The unified Dockerfile builds both UIs and serves the appropriate one based on the `TRANSPORT` variable:

```bash
# Run with WebRTC UI (default)
docker-compose up ui-app

# Run with WebSocket UI
TRANSPORT=WEBSOCKET docker-compose up ui-app -d
```

The UI will be available at `http://localhost:9000`

## Development

### WebRTC UI
The WebRTC UI is a React application built with Vite.

**Key Features:**
- **Real-time Audio Streaming**: Playback of audio from the WebRTC stream
- **Voice Selection**:
  - Default voice selection from available TTS voices
  - Custom voice prompt upload (for zero-shot TTS models)
- **Transcripts Display**: Real-time conversation transcripts showing user and bot messages
- **Prompt Editing**: Edit system prompt for custom usecases
- **State Persistence**: Maintains voice and prompt state across reconnections

For local development:
```bash
cd webrtc_ui
npm install
npm run dev
```

> **Note:** For remote accessibility, run `npm run dev -- --host` instead of `npm run dev` to make the dev server accessible from remote IPs. You may also need to deploy a TURN server. Refer to the [Deploy TURN Server for Remote Access](../docs/01-getting-started.md#optional-deploy-turn-server-for-remote-access) section in the getting started documentation.

### WebSocket UI
The WebSocket UI consists of static HTML files in the `websocket_ui/static/` directory that use WebSocket connections for real-time audio streaming and bidirectional communication with the voice agent. You can serve these files directly with any HTTP server.
