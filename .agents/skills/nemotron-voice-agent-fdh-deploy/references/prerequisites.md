# Prerequisites

Complete these steps once per host, from the repository root. Both profiles need all of them.

## 1. Host

- An x86_64 Linux host with one NVIDIA GPU for `nemo-speech` (GPU 0), NVIDIA driver 580.x or newer, and the NVIDIA Container Toolkit. The Hermes image lock (`docker/fdh/requirements.txt`) is resolved for x86_64 Linux, so DGX Spark and Jetson Thor are not supported. The LLMs run on the NVIDIA Inference Hub, not on the local GPU.
- Docker Compose v2.20 or newer, Git, Python 3, and either the Hugging Face `hf` CLI or `uvx`.
- Free host ports `7860` (or `PIPELINE_APP_PORT`) and `50051`. The gateway port 8790 is not published.

## 2. Environment File

`.env` must define `NVIDIA_API_KEY`, an Inference Hub key, and `HF_TOKEN` for the speech model download. Realtime clients need `REALTIME_API_KEY`. The frontend model (`nvidia/nvidia/nemotron-3.5-lightning`) and the backend model (`nvidia/nvidia/nemotron-3-ultra`) both run on the Inference Hub. `NVIDIA_API_KEY` is the only `.env` key that reaches `fdh-hermes`, and `docker/fdh/gateway.sh` refuses to start without it.

Keep any existing `.env`. Check presence only, never values:

```bash
test -f .env || cp .env.example .env      # then set NVIDIA_API_KEY, HF_TOKEN, REALTIME_API_KEY
for k in NVIDIA_API_KEY HF_TOKEN REALTIME_API_KEY; do printf '%s ' "$k"; grep -c "^$k=.\+" .env; done   # expect 1 each
```

Do not write `FDH_*`, `UVICORN_WS_PING_INTERVAL`, or `FDH_PROFILE` to `.env`. Compose sets them as literals on the voice service, and the eval overlay changes them.

## 3. Hermes Checkout

The Hermes image installs Hermes from a **clean** checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at the commit recorded in the header of `docker/fdh/requirements.txt`. The `fdh-hermes-image` service passes the checkout as the named build context `hermes`. Set its absolute path in `.env` as `HERMES_AGENT_DIR`. A relative path resolves against the repository root, and the default is `../hermes-agent`, a sibling of this repository.

The following commands clone Hermes at the pinned commit, record the path, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fdh/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
HERMES_AGENT_DIR=$(grep -m1 '^HERMES_AGENT_DIR=' .env | cut -d= -f2-)
git -C "$HERMES_AGENT_DIR" status --porcelain                  # expect no output
git -C "$HERMES_AGENT_DIR" rev-parse HEAD; grep -m1 'Hermes commit' docker/fdh/requirements.txt
```

The two commit hashes must match. Hermes runs only in the `fdh-hermes` container, so a deployment needs no host Hermes environment.

## 4. Image Rebuild Triggers

The recipe uses two images:

- `nemotron-voice-agent:latest`, the shared app image (`docker/Dockerfile`, `uv sync --frozen --extra fdh`). The voice service mounts `src/` read-only, so voice-side code and prompt changes need only a restart. Rebuild it after a dependency change.
- `nemotron-voice-agent-fdh:local`, the Hermes image (`docker/Dockerfile.fdh`, Python 3.14.7). It copies `src/` at build time, writes the hash of `src/` to `/app/SOURCE_SHA256`, and does not mount `src/`.

Rebuild with `up -d --build` after a change to any of the following:

- anything under `src/` that the gateway or the Hermes workers run (`core/sidecar`, `core/worker`, `config/gateway*.yaml`, `config/prompts.backend.yaml`, and so on);
- `docker/fdh/requirements.txt`, `docker/fdh/overrides.txt`, `docker/fdh/gateway.sh`, or `docker/Dockerfile.fdh`;
- the Hermes checkout or `HERMES_AGENT_DIR`.

The build runs import smoke tests for the gateway and the worker, imports Hermes, and checks the dependency set with `uv pip check`. `docker/fdh/gateway.env` is read at container start, so a change to it needs only `up -d`.

## 5. Optional Host Hermes Environment

You need a host Hermes environment only for the Hermes tests on the host (`test_fdh_hermes_multisession.py`, `test_fdh_hermes_adapter.py`). It is a separate virtual environment under `~/.cache/fdh`, so the checkout's own `.venv` stays untouched. The Python version depends on the checkout's `requires-python`: use `3.14` when it allows 3.14 and its lockfile supports only 3.14, and `3.13` when it caps Python below 3.14.

```bash
export HERMES_REPO=$HERMES_AGENT_DIR
grep -m1 requires-python "$HERMES_REPO/pyproject.toml"; grep -A1 -m1 resolution-markers "$HERMES_REPO/uv.lock"
export FDH_HERMES_PY=3.14                                                  # or 3.13
export FDH_HERMES_VENV=$HOME/.cache/fdh/hermes-venv-${FDH_HERMES_PY/./}  # hermes-venv-314 or hermes-venv-313
(cd "$HERMES_REPO" && UV_PROJECT_ENVIRONMENT=$FDH_HERMES_VENV uv sync --python $FDH_HERMES_PY)
(cd "$HERMES_REPO" && $FDH_HERMES_VENV/bin/python -c "import run_agent; print('hermes ok')")
```

`uv sync` with the wrong version fails with `The current Python platform is not compatible with the lockfile's supported environments`. A failed `uv sync` still creates an empty environment. Remove it with `rm -rf $FDH_HERMES_VENV` before you switch versions.

## 6. Speech Models

The recipe starts `nemo-speech` itself (ASR and TTS on port 50051). The voice side reads the endpoints from the `singlegpu` section of `src/examples/frontend_delegation_hermes/services.local.yaml`. Download the weights once, as your user, without `sudo`. The download is idempotent and fetches only what is missing or fails its checksum:

```bash
bash scripts/download-nemo-speech-models.sh
ls models/nemo-speech
mkdir -p logs
```
