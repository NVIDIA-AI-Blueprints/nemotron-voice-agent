# Prerequisites

Complete these steps once per host, from the repository root. Both adapters need all of them.

## 1. Host Setup

### 1.1 Environment File

`.env` must define `NVIDIA_API_KEY`, an Inference Hub key. The frontend model (`nvidia/nvidia/nemotron-3.5-lightning`) and the backend model (`nvidia/nvidia/nemotron-3-ultra`) both run on the Inference Hub. Keep any existing `.env`:

```bash
test -f .env || cp .env.example .env      # then set NVIDIA_API_KEY
grep -c '^NVIDIA_API_KEY=.\+' .env        # expect 1
```

### 1.2 Hermes Checkout

The image installs Hermes from a **clean** checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at the commit recorded in the header of `docker/fdh/requirements.txt`. Compose passes the checkout as the named build context `hermes`. Set its absolute path in `.env` as `HERMES_AGENT_DIR`. A relative path resolves against `docker/`, and the default is `../../hermes-agent`, a sibling of this repository.

The following commands clone Hermes at the pinned commit, record the path, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fdh/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
set -a; . ./.env; set +a
git -C "$HERMES_AGENT_DIR" status --porcelain                  # expect no output
git -C "$HERMES_AGENT_DIR" rev-parse HEAD; grep -m1 'Hermes commit' docker/fdh/requirements.txt
```

The two commit hashes must match. Hermes always runs inside the container, so a deployment needs no host Hermes environment.

### 1.3 Optional Host Hermes Environment

You need a host Hermes environment only for the host gateway (refer to the host gateway appendix in `tau.md`) and for the Hermes tests on the host. It is a separate virtual environment under `~/.cache/fdh`, so the checkout's own `.venv` stays untouched. The Python version depends on the checkout's `requires-python`: use `3.14` when it allows 3.14 and its lockfile supports only 3.14, and `3.13` when it caps Python below 3.14.

```bash
export HERMES_REPO=$HERMES_AGENT_DIR
grep -m1 requires-python "$HERMES_REPO/pyproject.toml"; grep -A1 -m1 resolution-markers "$HERMES_REPO/uv.lock"
export FDH_HERMES_PY=3.14                                                  # or 3.13
export FDH_HERMES_VENV=$HOME/.cache/fdh/hermes-venv-${FDH_HERMES_PY/./}  # hermes-venv-314 or hermes-venv-313
(cd "$HERMES_REPO" && UV_PROJECT_ENVIRONMENT=$FDH_HERMES_VENV uv sync --python $FDH_HERMES_PY)
(cd "$HERMES_REPO" && $FDH_HERMES_VENV/bin/python -c "import run_agent; print('hermes ok')")
```

`uv sync` with the wrong version fails with `The current Python platform is not compatible with the lockfile's supported environments`. A failed `uv sync` still creates an empty environment. Remove it with `rm -rf $FDH_HERMES_VENV` before you switch versions. If `import run_agent` fails from another directory, set `FDH_HERMES_REPO=$HERMES_REPO` when you start the host gateway.

### 1.4 Image Rebuild Triggers

Both services use one image, `nemotron-voice-agent-fdh:latest`, built from `docker/Dockerfile.fdh` with Python 3.14.7 and Pipecat 1.11.0. `src/` is mounted read-only, so code and prompt changes need only a restart. Rebuild the image after a change to any of the following:

- `docker/fdh/requirements.txt` or `docker/fdh/overrides.txt`
- `docker/Dockerfile.fdh`
- `docker/fdh/entrypoint.sh`
- the Hermes checkout or `HERMES_AGENT_DIR`

The following command rebuilds the image once for both services:

```bash
docker compose --env-file .env -f docker/docker-compose.frontend-delegation-hermes.yaml --profile tau build fdh-tau
```

The build runs import smoke tests for both adapters and the gateway, imports Hermes, and checks the dependency set with `uv pip check`.

### 1.5 Shell Variables

The following variables are used by the host gateway and the browser URL:

```bash
export DOCKER_HOST_IP=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)   # for example 172.17.0.1
export HOST_IP=$(hostname -I | awk '{print $1}')                                     # for the browser URL
mkdir -p logs
```

## 2. Speech Models and NeMo-Speech

Both adapters use the `nemo-speech` container (ASR and TTS on port 50051), through the single-GPU section of `src/examples/frontend_backend_verdict/services.local.yaml`. The following commands download the models and start the container. The download is idempotent and fetches only what is missing or fails its checksum:

```bash
bash scripts/download-nemo-speech-models.sh
ls models/nemo-speech
docker compose --profile frontend-backend-verdict-agent/single-gpu up -d nemo-speech
docker logs -f nemotron-voice-agent-nemo-speech-1 2>&1 | grep -m1 'listening on 0.0.0.0:50051'
```

Starting `nemo-speech` also creates the network `nemotron-voice-agent_default`, which the FDH Compose file declares as external.
