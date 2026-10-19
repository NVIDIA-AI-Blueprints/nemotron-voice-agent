---
name: nemotron-voice-agent-fdh-deploy
description: Build, deploy, and evaluate the Frontend Delegation Hermes (FDH) example on the frontend-delegation-hermes-agent/single-gpu recipe, with the browser UI (default profile) or OpenAI Realtime evaluation runs such as τ² (eval overlay). Use when starting, smoke testing, or troubleshooting the FDH voice service or its Hermes backend (fdh-hermes, by default in an NVIDIA OpenShell sandbox, or the plain baseline overlay).
version: "2.3.0"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [deployment, docker-compose, voice-agent, nemotron, hermes, evaluation]
---

# Frontend Delegation Hermes Deployment

`SKILL.md` is the decision tree. Open a reference only for the profile you chose. Do not open the others.

## When to Use This Skill

Use this skill to build the Frontend Delegation Hermes (FDH) images, start or stop the `frontend-delegation-hermes-agent/single-gpu` recipe, run the smoke tests, point an evaluation client at it, and troubleshoot the containers. The example lives in `src/examples/frontend_delegation_hermes/`. It is registered in `examples_registry.yaml` as `frontend-delegation-hermes-agent`, with the Realtime model `nvidia/nemotron-realtime-frontend-delegation-hermes`.

Do not use this skill for the other registered examples (use `nemotron-voice-agent-deploy`), for `.env`, prompt, or catalog edits of the other examples (use `nemotron-voice-agent-configure-pipeline`), or to change the Pipecat version (use `nemotron-voice-agent-upgrade-pipecat`).

## Rules

- Run every command from the repository root. Set `P=frontend-delegation-hermes-agent/single-gpu`, and `E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml"` for eval mode.
- For eval mode, pass `$E` in **every** Compose command (`up`, `ps`, `logs`, `exec`, `down`). A command without it recreates the containers with the `default` profile and `gateway.yaml`.
- Every benchmark run uses the eval overlay. Only the overlay switches the Hermes gateway to `gateway.eval.yaml`. Realtime sessions on a `default` deployment pair the eval frontend with the wrong gateway.
- The Hermes backend runs in an NVIDIA OpenShell sandbox by default. Use the plain runtime overlay, `docker/docker-compose.fdh-hermes-plain.yaml`, only when the user asks for it (A/B checks, debugging) or the host cannot run OpenShell. Always pass files in this order: `docker-compose.yml`, then the plain overlay, then the eval overlay. For the plain runtime, set `E="-f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml"` (plain `default`) or `E="-f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml -f docker/docker-compose.fdh-eval.yaml"` (plain `eval`), and pass `$E` in every command.
- If, and only if, the user says the run is a VoiceBench evaluation, ask: "Keep the frontend silent for VoiceBench? The filler is then logged only, and the backend's answer is the first response. Default: no." Wait for the answer. If yes, append `-f docker/docker-compose.fdh-voicebench.yaml` to `$E` after the eval overlay. Do not ask for Tau, the browser UI, or any other run, and never add the overlay for them. VoiceBench runs use `eval`. Refer to `references/tau.md`.
- OpenShell has not yet been validated on a live host (the go/no-go spike and the eval gates are pending). If it fails on a host, report the failure and offer the plain runtime. Do not edit `docker/fdh/openshell/` files to work around it without the user's approval.
- `docker-compose.yml` sets every FDH variable of the voice service as a literal. `FDH_*` keys in `.env` have no effect. Do not write them to `.env`.
- The Hermes checkout in `HERMES_AGENT_DIR` must be clean and at the commit in the header of `docker/fdh/requirements.txt`.
- The voice service mounts `src/` read-only, but `fdh-hermes` runs the source built into its image. Keep `--build` on every `up` after a source change.
- Preserve the existing `.env`. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present. Never print their values. Never commit `.env`, `logs/`, traces, or Hermes homes.

## Containers

The recipe runs the services in the following table:

| Service | Image | Role |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `nemotron-voice-agent:latest` (shared app, Python 3.12) | The voice side: browser UI (RTVI) and `/v1/realtime`, on host port `${PIPELINE_APP_PORT:-7860}` |
| `fdh-hermes` | `nemotron-voice-agent-fdh-openshell-cli:local` (built from `docker/fdh/openshell/Dockerfile.cli`) | The controller (`docker/fdh/openshell/controller.sh`). It creates the OpenShell sandbox `fdh-hermes`, which runs the FDH gateway (port 8790 inside) and one Hermes worker per session from `nemotron-voice-agent-fdh:local` (Python 3.14), as UID 1500. On stop it saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and deletes the sandbox |
| `hermes-openshell-gateway` | `ghcr.io/nvidia/openshell/gateway` (pinned in `docker/fdh/openshell/VERSION`) | The OpenShell gateway on `127.0.0.1:8080`. It holds the Docker socket and routes `default--fdh-hermes.openshell.localhost` to the sandbox |
| `hermes-openshell-init` | same as the gateway | One-time: creates the gateway's signing keys in `/var/lib/openshell/tls`, then exits |
| `fdh-hermes-image` | `nemotron-voice-agent-fdh:local` | Build only: builds the Hermes image from `docker/Dockerfile.fdh`, then exits |
| `nemo-speech` | NeMo-Speech.cpp | ASR and TTS on GPU 0, port 50051 |

The voice service reaches Hermes at `FDH_BACKEND_URL`: `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` by default. With the plain overlay, `fdh-hermes` is the FDH gateway container itself (`nemotron-voice-agent-fdh:local`, port 8790 on the Compose network, not published), the URL is `ws://fdh-hermes:8790/v1/backend`, and the two OpenShell services do not start.

## Decision Tree

Follow this tree. Ask only when the request leaves the choice open.

```text
Request names a profile?
├─ no      → ASK: "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?"
├─ default → docker compose --profile $P up -d --build
│            browser UI at https://<host>:7860 (pipecat_demo.yaml + adapters/pipecat.yaml + gateway.yaml)
│            → references/pipecat.md
└─ eval    → docker compose $E --profile $P up -d --build
             ws://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation-hermes
             (pipecat_realtime_eval.yaml + adapters/pipecat.eval.yaml + gateway.eval.yaml), browser sessions refused
             → references/tau.md
```

"Start the FDH demo" or "talk to FDH in the browser" means `default`. "Benchmark", "τ²", "Tau", or "VoiceBench" means `eval`.

## Deploy

### 1. Run the Preflight

Complete `references/prerequisites.md`: the OpenShell host checks, `.env` with `NVIDIA_API_KEY` and `HF_TOKEN` (plus `REALTIME_API_KEY` for Realtime), the Hermes checkout and commit check, and the speech model weights.

### 2. Start the Recipe

```bash
P=frontend-delegation-hermes-agent/single-gpu
mkdir -p logs
docker compose --profile $P config -q
# default (browser UI)
docker compose --profile $P up -d --build
# eval (OpenAI Realtime only)
E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml"
docker compose $E --profile $P up -d --build
# plain runtime (baseline): add the plain overlay before the eval overlay
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml [-f docker/docker-compose.fdh-eval.yaml] --profile $P up -d --build
```

Compose builds `fdh-hermes-image` and the controller image, starts the OpenShell gateway, waits until `fdh-hermes` is healthy (the FDH gateway in the sandbox answers `GET /health` through the OpenShell gateway, 180 s start period), and then starts the voice service. The first controller build compiles the OpenShell CLI from Rust source at the pinned commit and takes several minutes.

### 3. Verify the Deployment

```bash
docker compose [$E] --profile $P ps -a                                    # fdh-hermes-image, hermes-openshell-init: exited (0)
curl -fk https://localhost:7860/health                                    # eval: curl -f http://localhost:7860/health
docker compose [$E] --profile $P exec fdh-hermes \
  curl -s -H 'Host: default--fdh-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
docker compose [$E] --profile $P logs fdh-hermes | grep 'gateway config'  # controller: image, gateway config gateway[.eval].yaml
W=$(docker ps --filter ancestor=nemotron-voice-agent-fdh:local --format '{{.Names}}' | head -1)   # sandbox workload
docker logs "$W" 2>&1 | grep 'FDH gateway'                                # config gateway[.eval].yaml, source <hash>
docker compose [$E] --profile $P exec frontend-delegation-hermes-agent-single-gpu printenv FDH_PROFILE PIPELINE_TLS
```

Plain runtime: the health check is `exec fdh-hermes curl -s http://localhost:8790/health`, and `logs fdh-hermes | grep 'FDH gateway'` shows the gateway line.

Compare the source hash with the checkout: `find src -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1`. A different hash means that the image is stale: rerun `up` with `--build`.

After the first session, read `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=<profile>)` and the `config hash` from `docker compose [$E] --profile $P logs frontend-delegation-hermes-agent-single-gpu`. Report the profile, the gateway configuration, the source hash, and the config hash. Then run the smoke test in the profile's reference.

### 4. Stop

The following command stops the recipe. In the OpenShell runtime, `fdh-hermes` saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and deletes the sandbox (up to 60 s). In the plain runtime, the gateway closes open sessions one after another (up to 75 s). A sandbox left behind by a crashed controller is saved and deleted on the next start:

```bash
docker compose [$E] --profile $P down
```

`down` keeps `/var/lib/openshell` (the OpenShell gateway's keys and database).

## References

Open only what the chosen profile needs:

- Preflight, image rebuild triggers, and speech models: `references/prerequisites.md`
- Browser UI (`default`): `references/pipecat.md`
- Eval mode, Realtime smoke test, benchmark runs, and logs (including OpenShell logs and traces): `references/tau.md`
- Configuration knobs and variables: `references/config-knobs.md`
- Symptoms and fixes: `references/troubleshooting.md`

The user guide is `docs/how-to/deploy-frontend-delegation-hermes-agent.md`.
