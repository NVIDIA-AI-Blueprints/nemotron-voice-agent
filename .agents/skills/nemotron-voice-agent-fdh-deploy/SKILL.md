---
name: nemotron-voice-agent-fdh-deploy
description: Build, deploy, and evaluate the Frontend Delegation Hermes (FDH) example on the frontend-delegation-hermes-agent/single-gpu recipe, with the browser UI (default profile) or OpenAI Realtime evaluation runs such as τ² (eval overlay). Use when starting, smoke testing, or troubleshooting the FDH voice service or its fdh-hermes gateway container.
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
- Every benchmark run uses the eval overlay. Only the overlay switches the `fdh-hermes` gateway to `gateway.eval.yaml`. Realtime sessions on a `default` deployment pair the eval frontend with the wrong gateway.
- `docker-compose.yml` sets every FDH variable of the voice service as a literal. `FDH_*` keys in `.env` have no effect. Do not write them to `.env`.
- The Hermes checkout in `HERMES_AGENT_DIR` must be clean and at the commit in the header of `docker/fdh/requirements.txt`.
- The voice service mounts `src/` read-only, but `fdh-hermes` runs the source built into its image. Keep `--build` on every `up` after a source change.
- Preserve the existing `.env`. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present. Never print their values. Never commit `.env`, `logs/`, traces, or Hermes homes.

## Containers

The recipe runs the services in the following table:

| Service | Image | Role |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `nemotron-voice-agent:latest` (shared app, Python 3.12) | The voice side: browser UI (RTVI) and `/v1/realtime`, on host port `${PIPELINE_APP_PORT:-7860}` |
| `fdh-hermes` | `nemotron-voice-agent-fdh:local` (Python 3.14) | The FDH gateway on port 8790 of the Compose network (not published) and one Hermes worker per session |
| `fdh-hermes-image` | `nemotron-voice-agent-fdh:local` | Build only: builds the Hermes image from `docker/Dockerfile.fdh`, then exits |
| `nemo-speech` | NeMo-Speech.cpp | ASR and TTS on GPU 0, port 50051 |

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

"Start the FDH demo" or "talk to FDH in the browser" means `default`. "Benchmark", "τ²", or "Tau" means `eval`.

## Deploy

### 1. Run the Preflight

Complete `references/prerequisites.md`: `.env` with `NVIDIA_API_KEY` and `HF_TOKEN` (plus `REALTIME_API_KEY` for Realtime), the Hermes checkout and commit check, and the speech model weights.

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
```

Compose builds `fdh-hermes-image`, waits until `fdh-hermes` is healthy (its gateway answers `GET /health`), and then starts the voice service.

### 3. Verify the Deployment

```bash
docker compose [$E] --profile $P ps -a                                    # fdh-hermes-image: exited (0)
curl -fk https://localhost:7860/health                                    # eval: curl -f http://localhost:7860/health
docker compose [$E] --profile $P exec fdh-hermes curl -s http://localhost:8790/health; echo
docker compose [$E] --profile $P logs fdh-hermes | grep 'FDH gateway'     # config gateway[.eval].yaml, source <hash>
docker compose [$E] --profile $P exec frontend-delegation-hermes-agent-single-gpu printenv FDH_PROFILE PIPELINE_TLS
```

Compare the source hash with the checkout: `find src -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1`. A different hash means that the image is stale: rerun `up` with `--build`.

After the first session, read `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=<profile>)` and the `config hash` from `docker compose [$E] --profile $P logs frontend-delegation-hermes-agent-single-gpu`. Report the profile, the gateway configuration, the source hash, and the config hash. Then run the smoke test in the profile's reference.

### 4. Stop

The following command stops the recipe. The gateway closes open sessions one after another, so it can take up to 75 s:

```bash
docker compose [$E] --profile $P down
```

## References

Open only what the chosen profile needs:

- Preflight, image rebuild triggers, and speech models: `references/prerequisites.md`
- Browser UI (`default`): `references/pipecat.md`
- Eval mode, Realtime smoke test, benchmark runs, and logs: `references/tau.md`
- Configuration knobs and variables: `references/config-knobs.md`
- Symptoms and fixes: `references/troubleshooting.md`

The user guide is `docs/how-to/deploy-frontend-delegation-hermes-agent.md`.
