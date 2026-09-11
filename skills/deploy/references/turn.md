# TURN

Add `--profile turn` when clients connect from outside the host network. Skip this file unless the user asked for TURN or remote clients.

The bundled coturn profile uses the `instrumentisto/coturn` image, which is **x86_64 (`linux/amd64`) only**. On arm64 (Jetson Thor), do not enable the bundled profile. Set `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` for an external TURN server.

Coturn has compose defaults, but the app publishes ICE servers to clients only when `TURN_USERNAME` and `TURN_PASSWORD` are in `.env`:

```bash
test -f .env || cp .env.example .env
grep -Eq '^TURN_USERNAME=.+$' .env || printf '\nTURN_USERNAME=turn-%s\n' "$(openssl rand -hex 4)" >> .env
grep -Eq '^TURN_PASSWORD=.+$' .env || printf 'TURN_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
```

Set `TURN_URL=turn:<host-or-ip>:3478` when TURN runs on a different host, or when the host derived from the incoming request is not reachable by clients. Open UDP `3478` and UDP `49160-49200` from client networks.

```bash
docker compose --profile <recipe> --profile turn up -d
docker compose ps coturn
# HTTPS by default. If PIPELINE_TLS=false the HTTPS call fails and the HTTP one returns the config.
curl -k https://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers \
  || curl http://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers
```
