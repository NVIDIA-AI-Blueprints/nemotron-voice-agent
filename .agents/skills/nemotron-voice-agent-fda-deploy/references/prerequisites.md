# Prerequisites

Complete these steps once per host, from the repository root. Both profiles need them. The backend decides which ones apply:

- Hermes backend (default): every section.
- LLM backend (`docker/docker-compose.fda-llm.yaml`): the shared host needs in section 1 (not the OpenShell items), section 2, the shared app image in section 4, and section 6. Skip the OpenShell checks, section 3 (Hermes checkout), and section 5. `HERMES_AGENT_DIR` is not needed.

## 1. Host

- An x86_64 Linux host with one NVIDIA GPU for `nemo-speech` (GPU 0), NVIDIA driver 580.x or newer, and the NVIDIA Container Toolkit. The Hermes image lock (`docker/fda/requirements.txt`) is resolved for x86_64 Linux, so DGX Spark and Jetson Thor are not supported. The LLMs run on the NVIDIA Inference Hub, not on the local GPU.
- Docker Engine 28 or newer, Docker Compose 2.24.4 or newer (the plain and LLM overlays use `!override` and `!reset`), Git (Hermes only), Python 3, and either the Hugging Face `hf` CLI or `uvx`.
- Free host ports `7860` (or `PIPELINE_APP_PORT`) and `50051`. The FDA gateway port 8790 is not published.
- For the Hermes backend in the default OpenShell runtime: Linux kernel 6.2 or newer with Landlock turned on, a free host port `127.0.0.1:8080`, and no other OpenShell gateway on the host (port 8080 and `/var/lib/openshell` are host-wide; Docker creates `/var/lib/openshell` as root). Hosts that cannot meet these use the plain overlay, `docker/docker-compose.fda-hermes-plain.yaml`.

Check the host needs. The `uname`, `lsm`, and port 8080 lines are for the Hermes OpenShell runtime only:

```bash
uname -r                                                  # 6.2 or newer
cat /sys/kernel/security/lsm                              # must list landlock
docker version --format '{{.Server.Version}}'             # 28 or newer
docker compose version                                    # 2.24.4 or newer
ss -ltn 'sport = :8080'                                   # expect no listener
```

## 2. Environment File

`.env` must define `NVIDIA_API_KEY`, an Inference Hub key, and `HF_TOKEN` for the speech model download. Realtime clients need `REALTIME_API_KEY`. The frontend model (`nvidia/nvidia/nemotron-3.5-lightning`) and the backend model (`nvidia/nvidia/nemotron-3-ultra`) both run on the Inference Hub. Hermes: apart from the optional `FDA_PROMPT_PROFILE`, `NVIDIA_API_KEY` is the only `.env` key that reaches `fda-hermes`, and `fda-hermes` refuses to start without it. In the OpenShell runtime the key goes to the OpenShell provider, never into the sandbox. LLM backend: the voice container reads the key from `.env` and sends it for both LLMs.

Keep any existing `.env`. Check presence only, never values:

```bash
test -f .env || cp .env.example .env      # then set NVIDIA_API_KEY, HF_TOKEN, REALTIME_API_KEY
for k in NVIDIA_API_KEY HF_TOKEN REALTIME_API_KEY; do printf '%s ' "$k"; grep -c "^$k=.\+" .env; done   # expect 1 each
```

Do not write `FDA_*` (including `FDA_BACKEND` and `FDA_BACKEND_LINK`; choose the backend with the overlay), `UVICORN_WS_PING_INTERVAL`, or `FDA_PROFILE` to `.env`. Compose sets them as literals on the voice service, and the eval overlay changes them. The one exception is `FDA_PROMPT_PROFILE`, the prompt profile, which `.env` or the shell can set; prefer the shell for a single run, and never set it empty (refer to `config-knobs.md`, section Prompt Profiles).

## 3. Hermes Checkout

Hermes backend only. Skip this section for the LLM backend.

The Hermes image installs Hermes from a **clean** checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at the commit recorded in the header of `docker/fda/requirements.txt`. The `fda-hermes-image` service passes the checkout as the named build context `hermes`. Set its absolute path in `.env` as `HERMES_AGENT_DIR`. A relative path resolves against the repository root, and the default is `../hermes-agent`, a sibling of this repository.

The following commands clone Hermes at the pinned commit, record the path, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fda/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
HERMES_AGENT_DIR=$(grep -m1 '^HERMES_AGENT_DIR=' .env | cut -d= -f2-)
git -C "$HERMES_AGENT_DIR" status --porcelain                  # expect no output
git -C "$HERMES_AGENT_DIR" rev-parse HEAD; grep -m1 'Hermes commit' docker/fda/requirements.txt
```

The two commit hashes must match. Hermes runs only in the Hermes image (the OpenShell sandbox, or the plain `fda-hermes` container), so a deployment needs no host Hermes environment.

## 4. Image Rebuild Triggers

With Hermes, the recipe builds three images. With the LLM backend, it builds only the first:

- `nemotron-voice-agent:latest`, the shared app image (`docker/Dockerfile`, `uv sync --frozen --extra fda`, which includes the OpenAI SDK that the LLM backend uses). The voice service mounts `src/` read-only, so voice-side code changes need only a restart. With the LLM backend, that includes the backend code, the gateway configuration, and `config/prompts/<profile>/backend/llm.yaml`. With Hermes, a backend prompt change (`backend/hermes.yaml`) reaches the Hermes image only through a rebuild, so keep both sides in step with `up -d --build`. Rebuild it after a dependency change.
- `nemotron-voice-agent-fda:local`, the Hermes image (`docker/Dockerfile.fda`, Python 3.14.7). It copies `src/` at build time, writes the hash of `src/` to `/app/SOURCE_SHA256`, and does not mount `src/`. It has a `sandbox` user (UID/GID 1500) and `WORKDIR /sandbox` for OpenShell, but no `USER` line, so the plain runtime still runs as root.
- `nemotron-voice-agent-fda-openshell-cli:local`, the OpenShell controller image (`docker/fda/openshell/Dockerfile.cli`). It builds the OpenShell CLI from source at the commit in `docker/fda/openshell/VERSION`, because prerelease tags publish no CLI download. The first build compiles Rust and takes several minutes.

Rebuild with `up -d --build` after a change to any of the following:

- anything under `src/` that the gateway or the Hermes workers run (`core/sidecar`, `core/worker`, `core/backends`, `core/delegation/protocol.py`, `config/gateway*.yaml`, `config/prompts/<profile>/`, and so on). The worker protocol is version 2: an image built from older source fails with `protocol version 2 is not supported (expected 1)`;
- `docker/fda/requirements.txt`, `docker/fda/overrides.txt`, `docker/fda/gateway.sh`, or `docker/Dockerfile.fda`;
- the Hermes checkout or `HERMES_AGENT_DIR`.

The build runs import smoke tests for the gateway and the worker, imports Hermes, and checks the dependency set with `uv pip check`. `docker/fda/gateway.env` is read at container start, so a change to it needs only `up -d`. For changes to the files in `docker/fda/openshell/`, refer to `config-knobs.md`, section OpenShell Runtime. A change to `docker/fda/openshell/VERSION` or `Dockerfile.cli` needs `up -d --build`.

## 5. Optional Host Hermes Environment

Hermes backend only. You need a host Hermes environment only for the Hermes tests on the host (`test_fda_hermes_multisession.py`, `test_fda_hermes_adapter.py`). It is a separate virtual environment under `~/.cache/fda`, so the checkout's own `.venv` stays untouched. The Python version depends on the checkout's `requires-python`: use `3.14` when it allows 3.14 and its lockfile supports only 3.14, and `3.13` when it caps Python below 3.14.

```bash
export HERMES_REPO=$HERMES_AGENT_DIR
grep -m1 requires-python "$HERMES_REPO/pyproject.toml"; grep -A1 -m1 resolution-markers "$HERMES_REPO/uv.lock"
export FDA_HERMES_PY=3.14                                                  # or 3.13
export FDA_HERMES_VENV=$HOME/.cache/fda/hermes-venv-${FDA_HERMES_PY/./}  # hermes-venv-314 or hermes-venv-313
(cd "$HERMES_REPO" && UV_PROJECT_ENVIRONMENT=$FDA_HERMES_VENV uv sync --python $FDA_HERMES_PY)
(cd "$HERMES_REPO" && $FDA_HERMES_VENV/bin/python -c "import run_agent; print('hermes ok')")
```

`uv sync` with the wrong version fails with `The current Python platform is not compatible with the lockfile's supported environments`. A failed `uv sync` still creates an empty environment. Remove it with `rm -rf $FDA_HERMES_VENV` before you switch versions.

## 6. Speech Models

The recipe starts `nemo-speech` itself (ASR and TTS on port 50051). The voice side reads the endpoints from the `singlegpu` section of `src/examples/frontend_delegation_agent/services.local.yaml`. Download the weights once, as your user, without `sudo`. The download is idempotent and fetches only what is missing or fails its checksum:

```bash
bash scripts/download-nemo-speech-models.sh
ls models/nemo-speech
mkdir -p logs
```
