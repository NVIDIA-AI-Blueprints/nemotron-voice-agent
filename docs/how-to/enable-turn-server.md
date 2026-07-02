# Enable a TURN Server for Remote Access

A TURN server is only needed when the browser connects from a different network than the host (NAT, restrictive firewall, cloud deployment). Localhost and same-subnet clients work without it.

> **Architecture note:** The bundled `turn` profile uses the `instrumentisto/coturn` image, which is supported on **x86_64 (linux/amd64) only**. It is **not** supported on arm64 / aarch64 platforms (for example, NVIDIA Jetson Thor). On arm64 hosts, do not enable `--profile turn`. Instead, point the client at an externally hosted TURN server by setting `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` in `.env` (see the snippet below).

## Deploy the bundled coturn service (x86_64)

A Coturn service ships in `docker-compose.yml` behind an opt-in `turn` profile. Add `--profile turn` to any deploy command:

```bash
docker compose --profile generic-assistant --profile turn up -d              # cloud-only + TURN
docker compose --profile generic-assistant/workstation --profile turn up -d  # local NIM + TURN
```

- Coturn binds host ports UDP `3478` and UDP `49160-49200`. These must be reachable from clients (open them on your cloud firewall / security group).
- The client auto-fetches ICE config from `GET /api/ice-servers`, so no client-side setup is needed. The endpoint returns TURN credentials to the browser. Do not expose the app or this endpoint on the internet without application authentication.
- Set strong TURN credentials explicitly in `.env` (required whenever TURN is enabled). Do not rely on the compose fallback credentials.

    ```env
    # Required when TURN is enabled. Also set when TURN is deployed on a different host.
    TURN_URL=turn:<turn-host-or-ip>:3478
    TURN_USERNAME=<user>
    TURN_PASSWORD=<pass>
    ```

- Generate a long random password, for example:

    ```bash
    openssl rand -hex 24
    ```

- If `TURN_URL` is unset, the app derives the TURN host from the request. When using a reverse proxy in that mode, ensure it forwards the `X-Forwarded-Host` header so the derived TURN URL resolves to the client-reachable hostname.
  For production, prefer short-lived TURN credentials and gate `/api/ice-servers` behind the same authentication as the application.

## Use an external TURN server (arm64 / Jetson Thor, or shared infra)

On arm64 hosts the bundled coturn image will not run. Point the client at an externally hosted TURN server by setting the same three variables in `.env`, and **omit** `--profile turn`:

```env
TURN_URL=turn:<turn-host-or-ip>:3478
TURN_USERNAME=<user>
TURN_PASSWORD=<pass>
```

See the [Getting Started Guide](../01-getting-started.md) for the full deployment flow.
