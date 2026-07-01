# Troubleshooting

Use this guide when deployment starts but the application is not reachable, services remain unhealthy, or browser audio does not work.

## Check Service Status

Run the following command from the repository root:

```bash
docker compose ps
```

On first startup, model services can remain in `starting` while images download and optimized engines initialize. First startup can take 30-60 minutes depending on network speed, GPU, and model profile.

For workstation deployments, inspect the local ASR, TTS, LLM, application, and UI services:

```bash
docker compose logs -f asr-service
docker compose logs -f tts-service
docker compose logs -f nvidia-llm
docker compose logs -f python-app
docker compose logs -f ui-app
```

For Jetson deployments, include the Jetson compose file. The Jetson stack includes `llm-nvidia-jetson`, `python-app`, and `ui-app`; ASR and TTS endpoints are configured separately in `.env`.

```bash
docker compose -f docker-compose.jetson.yml ps
docker compose -f docker-compose.jetson.yml logs -f python-app llm-nvidia-jetson
```

## NGC Login or Image Pull Fails

If Docker cannot pull images from `nvcr.io`, confirm that your API key is exported and that Docker is logged in to NGC:

```bash
export NGC_API_KEY=<your-nvidia-api-key>
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
```

If `docker compose up` still fails, rerun it after login:

```bash
docker compose up -d
```

## Model Services Stay in `starting`

The first run can take a long time while NIM services build or load optimized engines. This is expected during initial deployment.

For workstation deployments, check progress with:

```bash
docker compose logs -f asr-service tts-service nvidia-llm
```

For Jetson deployments, check the local LLM service and confirm that the ASR and TTS endpoints in `.env` are reachable:

```bash
docker compose -f docker-compose.jetson.yml logs -f llm-nvidia-jetson
```

If startup fails with GPU or memory errors, verify that your system meets the [GPU requirements](./01-getting-started.md#gpu-requirements). The default workstation deployment requires two GPUs, with ASR/TTS on GPU 0 and LLM on GPU 1.

## Python App Is Unhealthy

In the workstation compose stack, `python-app` depends on healthy ASR, TTS, and LLM services. If `python-app` does not start, inspect upstream services first:

```bash
docker compose ps asr-service tts-service nvidia-llm
```

Then inspect the Python app logs:

```bash
docker compose logs -f python-app
```

Common causes include missing `.env` values, an invalid `SYSTEM_PROMPT_SELECTOR`, unavailable model endpoints, or service names removed from `docker-compose.yml` without updating `depends_on`.

For Jetson deployments, inspect `llm-nvidia-jetson` and confirm that `ASR_SERVER_URL`, `TTS_SERVER_URL`, and `NVIDIA_LLM_URL` in `.env` point to reachable endpoints.

## Browser Cannot Access the Microphone

For local HTTP testing in Chrome, add the UI origin to the insecure-origin allowlist:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Use the origin for your deployment:

- Workstation or server: `http://<host-ip>:9000`
- Jetson: `http://<jetson-ip>:8081`

Restart Chrome after changing the flag. For production or shared deployments, use HTTPS instead of browser insecure-origin flags.

## Remote WebRTC Does Not Connect

Remote WebRTC often requires a TURN server. Confirm that both TURN configuration surfaces are set:

- `.env` contains `TURN_SERVER_URL`, `TURN_USERNAME`, and `TURN_PASSWORD`.
- `frontend/webrtc_ui/src/config.ts` contains the matching `RTC_CONFIG` for browser-side ICE servers.

After editing `config.ts`, rebuild the UI image:

```bash
docker compose up --build -d ui-app
```

For Jetson, use the `http://<jetson-ip>:8081` origin in Chrome and TURN-related examples instead of the workstation `http://<host-ip>:9000` origin.

## Configuration Changes Do Not Apply

Most environment variables and prompt files are loaded when `python-app` starts. Restart the service after changes to `.env`, `config/prompt.yaml`, or `config/ipa.json`:

```bash
docker compose restart python-app
```

If you changed service images, compose services, or UI configuration, rebuild or restart the affected services:

```bash
docker compose up --build -d
```

## Evaluation Scripts Cannot Connect

The BigBench and performance scripts use WebSocket transport. If your deployment uses the default WebRTC transport, switch to WebSocket before running evaluation. See [Choose a Transport Method](./how-to/choose-transport-method.md).

Confirm the `nvidia-pipecat` submodule is initialized:

```bash
git submodule update --init
ls nvidia-pipecat/tests/perf
```
