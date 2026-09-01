#!/usr/bin/env bash
# Download NeMo-Speech.cpp GGUF weights and TTS TN grammars for */single-gpu recipes.
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

# Magpie TTS text-normalization grammars live on the NeMo-Speech.cpp GitHub
# release, not Hugging Face. Pin matches nvcr.io/nvidia/nemo-speech.cpp:0.1.0.
NEMO_SPEECH_CPP_RELEASE="${NEMO_SPEECH_CPP_RELEASE:-v0.1.0}"
TN_ARCHIVE="tn_configs.tar.bz2"
TN_SHA256="${NEMO_SPEECH_TN_SHA256:-2ca242c6d29f551eba3663d7e508c0d9dad10440e287628234154c6d1a72c7bc}"
TN_URL="https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/${NEMO_SPEECH_CPP_RELEASE}/${TN_ARCHIVE}"

download_url() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    if curl -fL --retry 3 --connect-timeout 30 --max-time 180 -o "${dest}" "${url}"; then
      return 0
    fi
    echo "curl failed for ${url}; trying another downloader..." >&2
    rm -f "${dest}"
  fi
  if command -v wget >/dev/null 2>&1; then
    if wget -O "${dest}" "${url}"; then
      return 0
    fi
    echo "wget failed for ${url}; trying another downloader..." >&2
    rm -f "${dest}"
  fi
  if command -v gh >/dev/null 2>&1; then
    if gh release download "${NEMO_SPEECH_CPP_RELEASE}" \
      --repo NVIDIA/NeMo-Speech.cpp \
      --pattern "${TN_ARCHIVE}" \
      --dir "$(dirname "${dest}")"; then
      return 0
    fi
    echo "gh release download failed for ${TN_ARCHIVE}." >&2
  fi
  echo "Need a working curl, wget, or gh to download ${url}" >&2
  return 1
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Checksum mismatch for ${file}" >&2
    echo "  expected ${expected}" >&2
    echo "  got      ${actual}" >&2
    return 1
  fi
}

# Best-effort install of the Magpie TTS text-normalization grammars. Returns
# non-zero (without exiting) if the release asset is unavailable, corrupt, or
# incomplete, so the GGUF setup still counts as a success.
install_tn_grammars() {
  local tar="${DEST}/${TN_ARCHIVE}"

  if [[ -f "${tar}" ]]; then
    local have
    have="$(sha256sum "${tar}" | awk '{print $1}')"
    if [[ "${have}" != "${TN_SHA256}" ]]; then
      echo "Existing ${tar} has an unexpected checksum; re-downloading..."
      rm -f "${tar}"
    fi
  fi

  if [[ ! -f "${tar}" ]]; then
    echo "Downloading TTS text-normalization grammars (${TN_ARCHIVE})..."
    download_url "${TN_URL}" "${tar}" || return 1
  fi

  verify_sha256 "${tar}" "${TN_SHA256}" || return 1

  rm -rf "${DEST}/tn_configs"
  tar -xjf "${tar}" -C "${DEST}" || return 1

  [[ -f "${DEST}/tn_configs/en/tokenize_and_classify.far" && \
     -f "${DEST}/tn_configs/en/verbalize.far" ]] || return 1
}

warn_tn_unavailable() {
  rm -rf "${DEST}/tn_configs"
  cat >&2 <<EOF

============================================================================
WARNING: TTS text-normalization (TN) grammars could not be installed.

  Source: ${TN_URL}

  Without them, single-GPU Magpie TTS reads digits, dates, and currency
  literally (for example "2" instead of "two").

  The rest of the speech models downloaded successfully. Re-run this script
  to retry the TN download. If the GitHub release asset moved, pin the new
  checksum with NEMO_SPEECH_TN_SHA256.
============================================================================

EOF
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

tn_ok=true
if ! install_tn_grammars; then
  warn_tn_unavailable
  tn_ok=false
fi

chmod -R a+rX "${DEST}"

if [[ "${tn_ok}" == true ]]; then
  echo "Models ready at ${DEST} (TTS text normalization enabled)."
else
  echo "Models ready at ${DEST} (TTS text normalization NOT installed; see warning above)."
fi
