# Choose a Transport Method

By default, the Nemotron Voice Agent blueprint uses web real-time communication (WebRTC). You can switch to WebSocket transport for different deployment scenarios or client requirements.

The following table compares the available transport options:

| Transport | Best For | Latency | Network Requirements |
|-----------|----------|---------|----------------------|
| **WebRTC** (default) | Production voice interactions, lowest latency | ~50-150ms | Requires TURN server for remote access |
| **WebSocket** | Testing, firewall-restricted environments, simpler deployments | ~100-300ms | Works through standard HTTP ports |

## Switch to WebSocket Transport

1. Update `.env` to enable WebSocket transport. If you have not created `.env` yet, copy [config/env.example](../../config/env.example) to `.env`.

    ```bash
    # In .env file
    TRANSPORT=WEBSOCKET
    ```

2. Restart the services to apply the transport change:

    ```bash
    docker compose stop python-app ui-app
    docker compose up -d
    ```

The system automatically loads the appropriate pipeline and UI based on the `TRANSPORT` setting.

## Switch Back to WebRTC Transport

1. Update `.env` to restore the default WebRTC transport:

    ```bash
    # In .env file
    TRANSPORT=WEBRTC
    ```

2. Restart the services:

    ```bash
    docker compose stop python-app ui-app
    docker compose up -d
    ```

## Access URLs

Use the URL for your deployment target after the services restart:

| Deployment | URL |
|------------|-----|
| Workstation or server compose stack | `http://<host-ip>:9000` |
| Jetson compose stack | `http://<jetson-ip>:8081` |

If you access WebRTC remotely, configure TURN as described in [Getting Started](../01-getting-started.md#optional-deploy-turn-server-for-remote-access). WebSocket transport usually avoids TURN requirements because it uses standard HTTP connectivity.
