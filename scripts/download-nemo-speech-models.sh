#!/usr/bin/env bash
# Download NeMo-Speech.cpp GGUF weights for */single-gpu recipes.
#
# Run as your user, never sudo. The script:
#   - reads HF_TOKEN from the repo .env
#   - creates models/nemo-speech (or $NEMO_SPEECH_MODEL_LOC / $1)
#   - if Docker already created that path or ~/.cache/huggingface as root,
#     reclaims ownership with a one-shot container (no sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this script as root/sudo." >&2
  echo "A missing Docker bind-mount is created as root; sudo then hides hf/uvx on your PATH." >&2
  echo "Re-run as your user. This script reclaims root-owned directories automatically." >&2
  exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

DEST="${1:-${NEMO_SPEECH_MODEL_LOC:-${ROOT}/models/nemo-speech}}"
DEST="$(python3 -c 'import os, sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "${DEST}")"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_CACHE="$(python3 -c 'import os, sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "${HF_CACHE}")"
export PATH="${HOME}/.local/bin:${PATH}"

case "${DEST}" in
  / | "${HOME}" | "${HOME}/.cache" | "${ROOT}" | "${ROOT}/models")
    echo "Refusing unsafe model destination: ${DEST}" >&2
    echo "Choose a dedicated model directory." >&2
    exit 1
    ;;
esac

if [[ "$(dirname "${DEST}")" == "/" ]]; then
  echo "Refusing top-level model destination: ${DEST}" >&2
  echo "Choose a dedicated model directory below a user-managed path." >&2
  exit 1
fi

case "${HF_CACHE}" in
  / | "${HOME}" | "${HOME}/.cache" | "${ROOT}" | "${ROOT}/models")
    echo "Refusing unsafe Hugging Face cache: ${HF_CACHE}" >&2
    echo "Choose a dedicated cache directory." >&2
    exit 1
    ;;
esac

if [[ "$(dirname "${HF_CACHE}")" == "/" ]]; then
  echo "Refusing top-level Hugging Face cache: ${HF_CACHE}" >&2
  echo "Choose a dedicated cache directory below a user-managed path." >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
  HF=(uvx --from huggingface_hub hf)
else
  echo "Neither 'hf' nor 'uvx' is available. Install uv (https://docs.astral.sh/uv/) or the Hugging Face CLI." >&2
  exit 1
fi

# If Docker created a missing bind-mount, the host path is root-owned. Reclaim
# it with the Docker daemon instead of sudo, so hf/uvx stay on the user PATH.
reclaim_if_unwritable() {
  local path="$1"
  if [[ -e "${path}" && ! -w "${path}" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "Cannot write to ${path} (often root-owned after docker compose created a missing bind-mount)." >&2
      echo "Fix: sudo chown -R $(id -u):$(id -g) '${path}'" >&2
      exit 1
    fi
    echo "Reclaiming ownership of ${path} (created as root by Docker)..."
    docker run --rm -v "${path}:/fix" alpine:latest chown -R "$(id -u):$(id -g)" /fix
  fi
}

reclaim_if_unwritable "${DEST}"
reclaim_if_unwritable "${HF_CACHE}"
mkdir -p "${DEST}/magpie-tts/extracted" "${DEST}/nano-codec" "${HF_CACHE}"

if [[ ! -w "${DEST}" || ! -w "${HF_CACHE}" ]]; then
  echo "Still cannot write to ${DEST} or ${HF_CACHE}." >&2
  exit 1
fi

"${HF[@]}" download nvidia/nemotron-speech-streaming-en-0.6b \
  nemotron-speech-streaming-en-0.6b.q8_0.gguf --local-dir "${DEST}"

"${HF[@]}" download nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --local-dir "${DEST}"

"${HF[@]}" download nvidia/magpie_tts_multilingual_357m \
  --include magpie_tts_multilingual_357m.v2602.f16.gguf \
  --include magpie_tts_multilingual_357m.nemo \
  --local-dir "${DEST}/magpie-tts"

tar -xf "${DEST}/magpie-tts/magpie_tts_multilingual_357m.nemo" \
  -C "${DEST}/magpie-tts/extracted"

"${HF[@]}" download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
  nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --local-dir "${DEST}/nano-codec"

chmod -R a+rX "${DEST}"

echo "Models ready at ${DEST}"
