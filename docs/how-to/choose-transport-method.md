# Choose a Transport Method

By default, the Nemotron Voice Agent blueprint uses web real-time communication (WebRTC). You can switch to WebSocket transport for different deployment scenarios or client requirements.

The following table compares the available transport options:

| Transport | Best For | Latency | Network Requirements |
|-----------|----------|---------|----------------------|
| **WebRTC** (default) | Production voice interactions, lowest latency | ~50-150ms | Requires TURN server for remote access |
| **WebSocket** | Testing, firewall-restricted environments, simpler deployments | ~100-300ms | Works through standard HTTP ports |

## Switch to WebSocket Transport

1. Update your [.env](../../config/env.example) file to enable WebSocket transport:

    ```bash
    # In .env file
    TRANSPORT=WEBSOCKET
    ```

2. Restart the services to apply the transport change:

    ```bash
    docker compose stop python-app ui-app
    docker compose up -d
    ```

The system automatically loads the appropriate pipeline and UI based on the `TRANSPORT` setting. After starting the services, access the web interface through your browser at `http://your-server-ip:9000`.
