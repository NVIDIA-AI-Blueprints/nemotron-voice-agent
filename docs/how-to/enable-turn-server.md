# Enable a TURN Server for Remote Access

Use a TURN server when a firewall or NAT prevents a direct WebRTC connection. Localhost typically does not need it. This guide applies to the Pipecat examples (`nva-pipecat`).

> **Architecture note:** The bundled `turn` profile uses the `instrumentisto/coturn` image, which is supported on **x86_64 (linux/amd64) only**. It is **not** supported on arm64 / aarch64 platforms (for example, NVIDIA Jetson Thor). On arm64 hosts, do not enable `--profile turn`. Instead, point the client at an externally hosted TURN server by setting `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` in `.env` (see the snippet below).

## Deploy the bundled coturn service (x86_64)

The opt-in `turn` profile runs [Coturn](../../docker/sidecars/docker-compose.coturn.yaml) alongside one Pipecat recipe.

1. Set the client-reachable TURN address and credentials in `.env`. Do not rely on the Compose fallback credentials.

   ```env
   TURN_URL=turn:<turn-host-or-ip>:3478
   TURN_USERNAME=<user>
   TURN_PASSWORD=<pass>
   ```

2. Allow clients to reach the TURN listener on port `3478` and the UDP relay range `49160–49200`. The app advertises both UDP and TCP transports on the listener port.
3. Start one recipe with the TURN overlay. This example uses Cloud Generic Assistant:

   ```bash
   docker compose --profile generic-assistant --profile turn up -d
   ```

4. Connect from the remote browser. The UI fetches ICE configuration from `GET /api/ice-servers`.
5. Stop the recipe and overlay when finished:

   ```bash
   docker compose --profile generic-assistant --profile turn down
   ```

If `TURN_URL` is unset, the app derives the host from the request. Behind a reverse proxy, forward the client-reachable host in `X-Forwarded-Host`.

## Use an external TURN server (arm64 / Jetson Thor, or shared infra)

On arm64 hosts the bundled coturn image will not run. Point the client at an externally hosted TURN server by setting the same three variables in `.env`, and **omit** `--profile turn`:

```env
TURN_URL=turn:<turn-host-or-ip>:3478
TURN_USERNAME=<user>
TURN_PASSWORD=<pass>
```

## Live Sessions

The Frontend/Backend Live example uses the same TURN settings for the server's WebRTC peer in each Live session (`POST /v1/live/sessions`). When `TURN_USERNAME` and `TURN_PASSWORD` are set, the server adds the TURN server to that peer's ICE servers. To add more ICE servers, list them in `ice_urls` of the `live.settings` block in `examples_registry.yaml`. Configure your Live client's peer connection with the same TURN server.

See the [Getting Started Guide](../01-getting-started.md) for the full deployment flow.
