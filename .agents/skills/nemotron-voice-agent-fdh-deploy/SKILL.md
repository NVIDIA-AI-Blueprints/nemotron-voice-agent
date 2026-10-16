---
name: nemotron-voice-agent-fdh-deploy
description: Build, deploy, and evaluate the Frontend Delegation Hermes (FDH) example with its Pipecat adapter (demo or τ² eval) or its τ³ benchmark adapter. Use when starting, smoke testing, or troubleshooting the FDH containers.
version: "2.3.0"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [deployment, docker-compose, voice-agent, nemotron, hermes, evaluation]
---

# Frontend Delegation Hermes Deployment

`SKILL.md` is the decision tree. Open a reference only for the adapter you chose. Do not open the others.

## When to Use This Skill

Use this skill to build the Frontend Delegation Hermes (FDH) image, start or stop one of its two adapters, run the smoke tests, point a τ² or τ³ client at it, and troubleshoot the containers. The example lives in `src/examples/frontend_delegation_hermes/` and has its own Compose file, `docker/docker-compose.frontend-delegation-hermes.yaml`. It is not in `examples_registry.yaml` and has no root Compose recipe.

Do not use this skill for the registered examples (use `nemotron-voice-agent-deploy`), for `.env`, prompt, or catalog edits of the registered examples (use `nemotron-voice-agent-configure-pipeline`), or to change the Pipecat version (use `nemotron-voice-agent-upgrade-pipecat`).

## Rules

- Run every command from the repository root. Set `F=docker/docker-compose.frontend-delegation-hermes.yaml`.
- Start `nemo-speech` from the root project first. The FDH containers join its external network, `nemotron-voice-agent_default`.
- Always name the service. `--profile tau up` without a service name starts both `fdh-voice` and `fdh-tau`.
- Pass `--profile tau` to every `down`. A plain `down` leaves `fdh-tau` running.
- Run one adapter per reportable run. Before a reportable τ³ run, check `docker ps` and stop `fdh-voice` if it runs. Both adapters share `nemo-speech` and the Inference Hub.
- The Hermes checkout in `HERMES_AGENT_DIR` must be clean and at the commit in the header of `docker/fdh/requirements.txt`.
- Preserve the existing `.env`. Never commit `.env`, `logs/`, traces, or Hermes homes.
- `src/` is mounted read-only. Code and prompt changes need a container restart, not a rebuild. `up -d` does not recreate a container when only a mounted file changed, so run `docker compose -f $F restart <service>`, or `up -d --force-recreate <service>`. Rebuild only for the triggers in `references/prerequisites.md`.

## Adapters

The following table compares the two adapters:

| Adapter | Service | Host port | Use |
| --- | --- | --- | --- |
| Pipecat (default) | `fdh-voice` | 8776 | Browser demo, or τ² evaluation over OpenAI Realtime with Pipecat-native VAD and turns |
| τ³ | `fdh-tau` (Compose profile `tau`) | 8775 | τ³ benchmarking only, with the client-driven audio clock and 800 ms VAD silence |

## Decision Tree

Follow this tree. Ask only when the request leaves the choice open.

```text
Request names an adapter?
├─ no      → ASK: "Which adapter: Pipecat (the default) or τ³?"
├─ Pipecat → ASK: "Demo or eval?"  (skip when the request already says)
│    ├─ demo → fdh-voice: pipecat_demo.yaml + gateway.yaml + adapters/pipecat.yaml (Pipecat defaults), https://<host>:8776/
│    │         → references/pipecat.md (Demo)
│    └─ eval → fdh-voice: FDH_PIPECAT_PROFILE=pipecat_realtime_eval.yaml FDH_PIPECAT_GATEWAY_CONFIG=gateway.eval.yaml
│              FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml (0.8 s audio-time turn end) FDH_PIPECAT_TLS=false,
│              ws://<host>:8776/v1/realtime → references/pipecat.md (Eval)
└─ τ³      → ask nothing more: fdh-tau with --profile tau, realtime_eval.yaml + gateway.eval.yaml,
             ws://<host>:8775/v1/realtime → references/tau.md
```

"Deploy the FDH demo" goes straight to the Pipecat demo.

## Deploy

### 1. Run the Preflight

Complete `references/prerequisites.md`: `.env` with `NVIDIA_API_KEY` and `HERMES_AGENT_DIR`, the Hermes commit check, the speech models, and `nemo-speech`.

### 2. Start the Chosen Adapter

The following commands start each adapter. `--build` is needed on a fresh host and after a rebuild trigger.

```bash
F=docker/docker-compose.frontend-delegation-hermes.yaml
# Pipecat demo
docker compose --env-file .env -f $F up -d --build fdh-voice
# Pipecat eval
FDH_PIPECAT_PROFILE=pipecat_realtime_eval.yaml FDH_PIPECAT_GATEWAY_CONFIG=gateway.eval.yaml \
FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml FDH_PIPECAT_TLS=false \
  docker compose --env-file .env -f $F up -d --build fdh-voice
# τ³
docker compose --env-file .env -f $F --profile tau up -d --build fdh-tau
```

### 3. Verify the Deployment

Each container starts its own backend gateway on `127.0.0.1:8790` inside the container, then the voice server. The following commands check both:

```bash
docker compose -f $F --profile tau ps
docker exec fdh-voice curl -s 127.0.0.1:8790/health; echo      # fdh-tau for τ³
curl -sk https://localhost:8776/health; echo                    # Pipecat demo
curl -s http://localhost:8776/health; echo                      # Pipecat eval (TLS off)
curl -s http://localhost:8775/health; echo                      # τ³
```

Then run the smoke tests in the adapter's reference.

### 4. Stop

The following command stops both adapters and their gateways and workers (up to 75 s):

```bash
docker compose -f $F --profile tau down
```

## References

Open only what the chosen adapter needs:

- Preflight, image rebuild triggers, and `nemo-speech`: `references/prerequisites.md`
- Pipecat adapter, demo and eval: `references/pipecat.md`
- τ³ adapter, gates, and logs: `references/tau.md`
- Configuration knobs, variables, and ports: `references/config-knobs.md`
- Symptoms and fixes: `references/troubleshooting.md`
