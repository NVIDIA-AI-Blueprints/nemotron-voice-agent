---
name: nemotron-voice-agent-fda-deploy
description: Build, deploy, and evaluate the Frontend Delegation Agent (FDA) example on the frontend-delegation-agent/single-gpu recipe, with the browser UI (default profile) or OpenAI Realtime benchmark runs (eval overlay). Use when starting, smoke testing, or troubleshooting the FDA voice service or its backend: Hermes (fda-hermes, by default in an NVIDIA OpenShell sandbox, or the plain baseline overlay) or the lightweight LLM backend inside the voice container (docker/docker-compose.fda-llm.yaml).
version: "2.3.0"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [deployment, docker-compose, voice-agent, nemotron, hermes, evaluation]
---

# Frontend Delegation Agent Deployment

`SKILL.md` is the decision tree. Open a reference only for the profile and backend you chose. Do not open the others.

## When to Use This Skill

Use this skill to build the Frontend Delegation Agent (FDA) images, start or stop the `frontend-delegation-agent/single-gpu` recipe, run the smoke tests, point an evaluation client at it, and troubleshoot the containers. The example lives in `src/examples/frontend_delegation_agent/`. It is registered in `examples_registry.yaml` as `frontend-delegation-agent`, with the Realtime model `nvidia/nemotron-realtime-frontend-delegation`.

Do not use this skill for the other registered examples (use `nemotron-voice-agent-deploy`), for `.env`, prompt, or catalog edits of the other examples (use `nemotron-voice-agent-configure-pipeline`), or to change the Pipecat version (use `nemotron-voice-agent-upgrade-pipecat`).

## Rules

- Run every command from the repository root. Set `P=frontend-delegation-agent/single-gpu`, and set `E` to the Compose files of the chosen backend and profile (table below). Pass `$E` in **every** Compose command (`up`, `ps`, `logs`, `exec`, `down`). An `up` without the eval overlay recreates the containers with the `default` profile and `gateway.yaml`; an `up` without the LLM overlay recreates them with Hermes.
- **Always ask which backend to start**, unless the request already names it: "Which backend: `hermes` (the default: Hermes in its own `fda-hermes` container, in an OpenShell sandbox) or `llm` (the lightweight backend inside the voice container: no Hermes checkout, no Hermes image, no OpenShell)?" Suggest `hermes`, wait for the answer, and never pick one silently. "Hermes", "OpenShell", "sandbox", or "plain runtime" names `hermes`; "LLM backend", "lightweight backend", "single container", "without Hermes", or `FDA_BACKEND=llm` names `llm`. Ask it together with the profile question when both are open.
- Every benchmark run uses the eval overlay, with either backend. Only the overlay switches the gateway to `gateway.eval.yaml` (on `fda-hermes` for Hermes, on the voice service for `llm`). Realtime sessions on a `default` deployment pair the eval frontend with the wrong gateway.
- The LLM backend uses `docker/docker-compose.fda-llm.yaml`, always after the eval overlay and before the silent-frontend overlay. Never combine it with the plain runtime overlay. It sets `FDA_BACKEND=llm` and `FDA_BACKEND_LINK=in_process` on the voice service and keeps the four Hermes and OpenShell services from starting. Do not write `FDA_BACKEND` or `FDA_BACKEND_LINK` to `.env`.
- The Hermes backend runs in an NVIDIA OpenShell sandbox by default. Use the plain runtime overlay, `docker/docker-compose.fda-hermes-plain.yaml`, only when the user asks for it (A/B checks, debugging) or the host cannot run OpenShell. Always pass files in this order: `docker-compose.yml`, then the plain overlay, then the eval overlay. For the plain runtime, set `E="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml"` (plain `default`) or `E="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml -f docker/docker-compose.fda-eval.yaml"` (plain `eval`), and pass `$E` in every command.
- If, and only if, the user says the run is a VoiceBench evaluation, ask two questions and wait for both answers: "Keep the frontend silent? The filler is then logged only, and the backend's answer is the first response. Default: no." and "Which prompt profile? Suggested: `general_qa`." If the first answer is yes, append `-f docker/docker-compose.fda-silent-frontend.yaml` to `$E` last (after the eval overlay, and after the LLM overlay for `llm`). VoiceBench still needs the backend question. Prefix every Compose command that recreates containers with the chosen profile, for example `FDA_PROMPT_PROFILE=general_qa docker compose $E --profile $P up -d --build`. VoiceBench runs use `eval`. Refer to `references/benchmarking.md`.
- Every other run (other benchmarks, the browser UI, and so on) uses the mode's default prompt profile without asking: `generic` for `default`, `customer_service` for `eval`. Do not ask the two questions, never add the silent-frontend overlay, and do not set `FDA_PROMPT_PROFILE`. If `.env` sets `FDA_PROMPT_PROFILE`, show the value and ask the user to confirm or remove it.
- OpenShell has not yet been validated on a live host (the go/no-go spike and the eval gates are pending). If it fails on a host, report the failure and offer the plain runtime. Do not edit `docker/fda/openshell/` files to work around it without the user's approval.
- `docker-compose.yml` sets every FDA variable of the voice service as a literal. `FDA_*` keys in `.env` have no effect, with one exception: `FDA_PROMPT_PROFILE` (the prompt profile, `config/prompts/<profile>/`), which Compose passes to both the voice service and `fda-hermes`. Do not write any other `FDA_*` key to `.env`. Never set `FDA_PROMPT_PROFILE` to an empty value. An unknown name fails with an error that names the missing prompt file.
- Hermes only: the Hermes checkout in `HERMES_AGENT_DIR` must be clean and at the commit in the header of `docker/fda/requirements.txt`. The LLM backend needs no checkout and no OpenShell host checks.
- The voice service mounts `src/` read-only, but `fda-hermes` runs the source built into its image. With Hermes, keep `--build` on every `up` after a source or prompt change (the worker protocol is version 2; an older image fails with `protocol version 2 is not supported (expected 1)`). With `llm`, a restart of the voice service applies a source or prompt change.
- Each backend reads its own backend prompt file, `config/prompts/<profile>/backend/hermes.yaml` or `backend/llm.yaml`. They are independent copies: a prompt change meant for both backends edits both.
- Preserve the existing `.env`. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present. Never print their values. Never commit `.env`, `logs/`, traces, or Hermes homes.

## Containers

With the Hermes backend, the recipe runs the services in the following table:

| Service | Image | Role |
| --- | --- | --- |
| `frontend-delegation-agent-single-gpu` | `nemotron-voice-agent:latest` (shared app, Python 3.12) | The voice side: browser UI (RTVI) and `/v1/realtime`, on host port `${PIPELINE_APP_PORT:-7860}` |
| `fda-hermes` | `nemotron-voice-agent-fda-openshell-cli:local` (built from `docker/fda/openshell/Dockerfile.cli`) | The controller (`docker/fda/openshell/controller.sh`). It creates the OpenShell sandbox `fda-hermes`, which runs the FDA gateway (port 8790 inside) and one Hermes worker per session from `nemotron-voice-agent-fda:local` (Python 3.14), as UID 1500. On stop it saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and deletes the sandbox |
| `hermes-openshell-gateway` | `ghcr.io/nvidia/openshell/gateway` (pinned in `docker/fda/openshell/VERSION`) | The OpenShell gateway on `127.0.0.1:8080`. It holds the Docker socket and routes `default--fda-hermes.openshell.localhost` to the sandbox |
| `hermes-openshell-init` | same as the gateway | One-time: creates the gateway's signing keys in `/var/lib/openshell/tls`, then exits |
| `fda-hermes-image` | `nemotron-voice-agent-fda:local` | Build only: builds the Hermes image from `docker/Dockerfile.fda`, then exits |
| `nemo-speech` | NeMo-Speech.cpp | ASR and TTS on GPU 0, port 50051 |

The voice service reaches Hermes at `FDA_BACKEND_URL`: `ws://default--fda-hermes.openshell.localhost:8080/v1/backend` by default. With the plain overlay, `fda-hermes` is the FDA gateway container itself (`nemotron-voice-agent-fda:local`, port 8790 on the Compose network, not published), the URL is `ws://fda-hermes:8790/v1/backend`, and the two OpenShell services do not start.

With the LLM backend overlay, only the voice service and `nemo-speech` run. The voice service runs the gateway and one OpenAI SDK tool loop per session in its own process (`FDA_BACKEND_LINK=in_process`), loads the gateway file named by `FDA_GATEWAY_CONFIG` (`gateway.yaml`; `gateway.eval.yaml` with the eval overlay), and calls `BACKEND_LLM_MODEL` (`nvidia/nvidia/nemotron-3-ultra`) on `BACKEND_LLM_BASE_URL` with `NVIDIA_API_KEY`. There is no sandbox: the backend runs with the voice container's real key and network.

The following table lists `E` for each backend and profile. Append `-f docker/docker-compose.fda-silent-frontend.yaml` last for a silent VoiceBench run:

| Backend | `default` | `eval` |
| --- | --- | --- |
| `hermes`, OpenShell (default) | `E="-f docker-compose.yml"` | `E="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml"` |
| `hermes`, plain runtime | `E="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml"` | `E="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml -f docker/docker-compose.fda-eval.yaml"` |
| `llm` | `E="-f docker-compose.yml -f docker/docker-compose.fda-llm.yaml"` | `E="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml -f docker/docker-compose.fda-llm.yaml"` |

## Decision Tree

Follow this tree. Ask only when the request leaves a choice open. Ask both open questions in one message.

```text
1. Request names a backend?
   ├─ no     → ASK: "Which backend: `hermes` (default; Hermes in fda-hermes, OpenShell sandbox)
   │                 or `llm` (lightweight backend inside the voice container)?"  Suggest hermes; wait.
   ├─ hermes → OpenShell runtime (plain overlay only on request or when OpenShell cannot run)
   │           preflight: all of references/prerequisites.md
   └─ llm    → docker/docker-compose.fda-llm.yaml after the eval overlay; never the plain overlay
               preflight: references/prerequisites.md sections 1 (shared host needs), 2, and 6 only
2. Request names a profile?
   ├─ no      → ASK: "Which profile: `default` (for the browser UI) or `eval` (for benchmark runs)?"
   ├─ default → docker compose $E --profile $P up -d --build      (E from the table above)
   │            browser UI at https://<host>:7860 (pipecat_demo.yaml + adapters/pipecat.yaml + gateway.yaml)
   │            → references/pipecat.md
   └─ eval    → docker compose $E --profile $P up -d --build      (E includes the eval overlay)
                ws://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation
                (pipecat_realtime_eval.yaml + adapters/pipecat.eval.yaml + gateway.eval.yaml), browser sessions refused
                → references/benchmarking.md
```

"Start the FDA demo" or "talk to FDA in the browser" means `default`, but names no backend: ask. "Benchmark", "evaluation", or "VoiceBench" means `eval`, but names no backend: ask.

## Deploy

### 1. Run the Preflight

Complete `references/prerequisites.md`. Hermes: the OpenShell host checks, `.env` with `NVIDIA_API_KEY` and `HF_TOKEN` (plus `REALTIME_API_KEY` for Realtime), the Hermes checkout and commit check, and the speech model weights. LLM backend: `.env` with the same keys and the speech model weights only; skip the OpenShell checks and the Hermes checkout.

### 2. Start the Recipe

```bash
P=frontend-delegation-agent/single-gpu
mkdir -p logs
# Hermes (default), OpenShell runtime
E="-f docker-compose.yml"                                                          # default (browser UI)
E="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml"                    # eval (OpenAI Realtime only)
# Hermes, plain runtime (baseline): the plain overlay before the eval overlay
E="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml [-f docker/docker-compose.fda-eval.yaml]"
# LLM backend: the LLM overlay after the eval overlay
E="-f docker-compose.yml -f docker/docker-compose.fda-llm.yaml"                     # default
E="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml -f docker/docker-compose.fda-llm.yaml"   # eval
# then, with the one E you chose
docker compose $E --profile $P config -q
docker compose $E --profile $P up -d --build
```

Hermes: Compose builds `fda-hermes-image` and the controller image, starts the OpenShell gateway, waits until `fda-hermes` is healthy (the FDA gateway in the sandbox answers `GET /health` through the OpenShell gateway, 180 s start period), and then starts the voice service. The first controller build compiles the OpenShell CLI from Rust source at the pinned commit and takes several minutes.

LLM backend: Compose builds only the shared app image and starts the voice service and `nemo-speech`. The voice service does not wait for a backend service and does not check a gateway at startup.

### 3. Verify the Deployment

Run the block of the chosen backend. Hermes:

```bash
docker compose $E --profile $P ps -a                                    # fda-hermes-image, hermes-openshell-init: exited (0)
curl -fk https://localhost:7860/health                                    # eval: curl -f http://localhost:7860/health
docker compose $E --profile $P exec fda-hermes \
  curl -s -H 'Host: default--fda-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
docker compose $E --profile $P logs fda-hermes | grep 'gateway config'    # controller: image, gateway config gateway[.eval].yaml
W=$(docker ps --filter ancestor=nemotron-voice-agent-fda:local --format '{{.Names}}' | head -1)   # sandbox workload
docker logs "$W" 2>&1 | grep 'FDA gateway'                                # config gateway[.eval].yaml, source <hash>
docker compose $E --profile $P exec frontend-delegation-agent-single-gpu printenv FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE
docker compose $E --profile $P exec fda-hermes printenv FDA_PROMPT_PROFILE   # the same prompt profile on both sides
```

Plain runtime: the health check is `exec fda-hermes curl -s http://localhost:8790/health`, and `logs fda-hermes | grep 'FDA gateway'` shows the gateway line.

LLM backend (health on the voice service only; there is no `fda-hermes`, no gateway health route, and no image source hash):

```bash
docker compose $E --profile $P ps -a                                      # only the voice service and nemo-speech; no fda-hermes
curl -fk https://localhost:7860/health                                    # eval: curl -f http://localhost:7860/health
docker compose $E --profile $P exec frontend-delegation-agent-single-gpu \
  printenv FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG                 # llm, in_process, gateway.yaml (eval: gateway.eval.yaml)
docker compose $E --profile $P exec frontend-delegation-agent-single-gpu printenv FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE
# after the first session: the session_open record names the backend
grep -o '"agent_kind": *"[a-z]*"' logs/fda_gateway_events.pipecat.jsonl | tail -n 1   # "agent_kind": "llm"
```

If `FDA_BACKEND` prints nothing or `ps` lists `fda-hermes`, the `up` ran without the LLM overlay: rerun it with `$E`.

Hermes only: compare the source hash with the checkout: `find src -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1`. A different hash means that the image is stale: rerun `up` with `--build`.

After the first session, read `frontend-delegation-agent ready (adapter=pipecat, profile=<profile>)` and the `config hash` from `docker compose $E --profile $P logs frontend-delegation-agent-single-gpu`. Report the backend, the profile, the prompt profile, the gateway configuration, and the config hash, plus the source hash for Hermes or `FDA_BACKEND`/`FDA_BACKEND_LINK` for `llm`. The `config hash` differs between the backends. Then run the smoke test in the profile's reference.

### 4. Stop

The following command stops the recipe, with the same `$E` that started it. In the OpenShell runtime, `fda-hermes` saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and deletes the sandbox (up to 60 s). In the plain runtime, the gateway closes open sessions one after another (up to 75 s). A sandbox left behind by a crashed controller is saved and deleted on the next start. With the LLM backend, `down` stops the voice service and `nemo-speech`, and the backend stops with the voice service:

```bash
docker compose $E --profile $P down
```

`down` keeps `/var/lib/openshell` (the OpenShell gateway's keys and database).

## References

Open only what the chosen profile needs:

- Preflight, image rebuild triggers, and speech models: `references/prerequisites.md`
- Browser UI (`default`): `references/pipecat.md`
- Eval mode, Realtime smoke test, benchmark runs, and logs (including OpenShell logs and traces): `references/benchmarking.md`
- Configuration knobs, variables, backends, and prompt profiles: `references/config-knobs.md`
- Symptoms and fixes: `references/troubleshooting.md`

The user guide is `docs/how-to/deploy-frontend-delegation-agent.md`.
