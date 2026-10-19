#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Download NeMo-Speech.cpp GGUF weights and TTS TN grammars for */single-gpu recipes.
#
# Run as your user, never sudo. The script:
#   - reads HF_TOKEN from the repo .env
#   - creates models/nemo-speech (or $NEMO_SPEECH_MODEL_LOC / $1)
#   - if Docker already created that path or ~/.cache/huggingface as root,
#     reclaims ownership with a one-shot container (no sudo)
#   - installs the asr/, diarization/, and vad/ models that the NeMo-Speech.cpp
#     diarization setup in generic-assistant/single-gpu loads
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

MAGPIE_REPO="nvidia/magpie_tts_multilingual_357m"
MAGPIE_NEMO_REVISION="9e73758a7db5ff6895f66cfc7a02f4a2f8c2acef"
MAGPIE_NEMO="magpie_tts_multilingual_357m.nemo"
MAGPIE_NEMO_SHA256="ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953"
MAGPIE_GGUF_REVISION="446bff55e1fe3d3aa29c771574e7d2b0fa322990"
MAGPIE_GGUF="magpie_tts_multilingual_357m.v2607.f16.gguf"
MAGPIE_GGUF_SHA256="fb8e9297138dcc7b5fc51332f361e3cddfd433d4e0bc1d87e16e6aed21f618fc"

# Magpie TTS text-normalization grammars live on the NeMo-Speech.cpp GitHub
# release, not Hugging Face. Pin matches nvcr.io/nvidia/nemo-speech.cpp:0.2.0.
TN_PINNED_RELEASE="v0.2.0"
TN_PINNED_SHA256="2ca242c6d29f551eba3663d7e508c0d9dad10440e287628234154c6d1a72c7bc"
NEMO_SPEECH_CPP_RELEASE="${NEMO_SPEECH_CPP_RELEASE:-${TN_PINNED_RELEASE}}"
TN_ARCHIVE="tn_configs.tar.bz2"
TN_SHA256="${NEMO_SPEECH_TN_SHA256:-${TN_PINNED_SHA256}}"
TN_URL="https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/${NEMO_SPEECH_CPP_RELEASE}/${TN_ARCHIVE}"

# generic-assistant/single-gpu also runs diarization. It loads its ASR,
# diarizer, and VAD models from asr/, diarization/, and vad/.
ASR_EN_GGUF="nemotron-speech-streaming-en-0.6b.q8_0.gguf"
DIAR_REPO="nvidia/Nemotron-3-Diarization"
DIAR_REVISION="f667ed73aee57d40cc39428eb768b4fd87a0a29e"
DIAR_GGUF="Nemotron-3-Diarization.q8_0.gguf"
DIAR_GGUF_SHA256="08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"

# NeMo-Speech.cpp publishes no Silero VAD GGUF. Its converter builds one from
# the whisper.cpp Silero checkpoint; the output checksum pins the result.
VAD_REPO="ggml-org/whisper-vad"
VAD_REVISION="9ffd54a1e1ee413ddf265af9913beaf518d1639b"
VAD_SOURCE="ggml-silero-v6.2.0.bin"
VAD_SOURCE_SHA256="2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
VAD_GGUF="silero-v6.2.0.gguf"
VAD_GGUF_SHA256="7adbffd75f1de85b20a03c1c8960f2c43d8129995f0581b1c1804d50e39cb661"
VAD_CONVERTER_REVISION="6a3ca369370782acee0dd155e34394ba60b920c4"
VAD_CONVERTER_URL="https://github.com/NVIDIA/NeMo-Speech.cpp/archive/${VAD_CONVERTER_REVISION}.tar.gz"
VAD_CONVERTER_SHA256="2db94a47ad61a16e6ff3da1786fc4097747233dedf2f12671201782af640165c"
VAD_CONVERTER_PYTHON="3.12"
VAD_CONVERTER_GGUF_VERSION="0.19.0"
VAD_CONVERTER_NUMPY_VERSION="2.2.6"

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
  if [[ "${url}" == "${TN_URL}" ]] && command -v gh >/dev/null 2>&1; then
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

require_nonempty() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "Required speech model missing or empty: ${path}" >&2
    echo "Do not start */single-gpu until this file is present." >&2
    exit 1
  fi
}

sha256_matches() {
  local file="$1"
  local expected="$2"
  [[ -s "${file}" ]] || return 1
  local actual
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]]
}

download_magpie_tts() {
  local dir="${DEST}/magpie-tts"
  local gguf="${dir}/${MAGPIE_GGUF}"
  local nemo="${dir}/${MAGPIE_NEMO}"

  if sha256_matches "${nemo}" "${MAGPIE_NEMO_SHA256}"; then
    echo "Magpie TTS ${MAGPIE_NEMO} already present and checksum matches."
  else
    echo "Downloading ${MAGPIE_NEMO} from ${MAGPIE_REPO}@${MAGPIE_NEMO_REVISION}..."
    "${HF[@]}" download "${MAGPIE_REPO}" "${MAGPIE_NEMO}" \
      --revision "${MAGPIE_NEMO_REVISION}" \
      --local-dir "${dir}"
    require_nonempty "${nemo}"
    verify_sha256 "${nemo}" "${MAGPIE_NEMO_SHA256}" || exit 1
  fi

  if sha256_matches "${gguf}" "${MAGPIE_GGUF_SHA256}"; then
    echo "Magpie TTS ${MAGPIE_GGUF} already present and checksum matches."
  else
    echo "Downloading ${MAGPIE_GGUF} from ${MAGPIE_REPO}@${MAGPIE_GGUF_REVISION}..."
    "${HF[@]}" download "${MAGPIE_REPO}" "${MAGPIE_GGUF}" \
      --revision "${MAGPIE_GGUF_REVISION}" \
      --local-dir "${dir}"
    require_nonempty "${gguf}"
    verify_sha256 "${gguf}" "${MAGPIE_GGUF_SHA256}" || exit 1
  fi

  tar -xf "${nemo}" -C "${dir}/extracted"
  if [[ -z "$(find "${dir}/extracted" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Magpie tokenizer extract is empty: ${dir}/extracted" >&2
    echo "The tokenizer lives in the .nemo archive, not the GGUF." >&2
    exit 1
  fi
}

download_diarizer() {
  local gguf="${DEST}/diarization/${DIAR_GGUF}"

  if sha256_matches "${gguf}" "${DIAR_GGUF_SHA256}"; then
    echo "Diarizer ${DIAR_GGUF} already present and checksum matches."
    return
  fi
  echo "Downloading ${DIAR_GGUF} from ${DIAR_REPO}@${DIAR_REVISION}..."
  "${HF[@]}" download "${DIAR_REPO}" "${DIAR_GGUF}" \
    --revision "${DIAR_REVISION}" \
    --local-dir "${DEST}/diarization"
  require_nonempty "${gguf}"
  verify_sha256 "${gguf}" "${DIAR_GGUF_SHA256}" || exit 1
}

# The diarization setup reads the English ASR model from asr/, so link the
# existing GGUF instead of storing a second copy.
link_diarization_asr() {
  mkdir -p "${DEST}/asr"
  ln -sfn "../${ASR_EN_GGUF}" "${DEST}/asr/${ASR_EN_GGUF}"
}

run_vad_converter() {
  if command -v uv >/dev/null 2>&1; then
    uv run --no-project --quiet --python "${VAD_CONVERTER_PYTHON}" \
      --with "gguf==${VAD_CONVERTER_GGUF_VERSION}" \
      --with "numpy==${VAD_CONVERTER_NUMPY_VERSION}" \
      python3 "$@"
  elif python3 - "${VAD_CONVERTER_GGUF_VERSION}" "${VAD_CONVERTER_NUMPY_VERSION}" 2>/dev/null <<'PY'
import sys
from importlib.metadata import version
sys.exit(0 if (version("gguf"), version("numpy")) == tuple(sys.argv[1:3]) else 1)
PY
  then
    python3 "$@"
  else
    echo "Converting the VAD model needs uv, or python3 with gguf==${VAD_CONVERTER_GGUF_VERSION} and numpy==${VAD_CONVERTER_NUMPY_VERSION}." >&2
    return 1
  fi
}

# Best-effort build of the Silero VAD GGUF. Returns non-zero (without exiting)
# on failure, because only generic-assistant/single-gpu needs it.
install_vad_model() {
  local gguf="${DEST}/vad/${VAD_GGUF}"
  local stage="${DEST}/.vad.staging"

  if sha256_matches "${gguf}" "${VAD_GGUF_SHA256}"; then
    echo "VAD ${VAD_GGUF} already present and checksum matches."
    return 0
  fi
  echo "Building ${VAD_GGUF} from ${VAD_REPO}@${VAD_REVISION}..."
  rm -rf "${stage}"
  mkdir -p "${stage}" "${DEST}/vad"
  if "${HF[@]}" download "${VAD_REPO}" "${VAD_SOURCE}" \
       --revision "${VAD_REVISION}" --local-dir "${stage}" && \
     verify_sha256 "${stage}/${VAD_SOURCE}" "${VAD_SOURCE_SHA256}" && \
     download_url "${VAD_CONVERTER_URL}" "${stage}/converter.tar.gz" && \
     verify_sha256 "${stage}/converter.tar.gz" "${VAD_CONVERTER_SHA256}" && \
     tar -xzf "${stage}/converter.tar.gz" -C "${stage}" --strip-components=1 \
       --wildcards "*/convert_model.py" "*/conversion/*" && \
     run_vad_converter "${stage}/convert_model.py" silero \
       --outfile "${stage}/${VAD_GGUF}" \
       --from-whisper-ggml "${stage}/${VAD_SOURCE}" && \
     verify_sha256 "${stage}/${VAD_GGUF}" "${VAD_GGUF_SHA256}"; then
    mv "${stage}/${VAD_GGUF}" "${gguf}"
    rm -rf "${stage}"
    return 0
  fi
  rm -rf "${stage}"
  return 1
}

warn_vad_unavailable() {
  cat >&2 <<EOF

============================================================================
WARNING: The Silero VAD model could not be built.

  generic-assistant/single-gpu runs NeMo-Speech.cpp with diarization and
  cannot start without:
    ${DEST}/vad/${VAD_GGUF}

  The build needs uv (or python3 with gguf==${VAD_CONVERTER_GGUF_VERSION} and
  numpy==${VAD_CONVERTER_NUMPY_VERSION}), huggingface.co, and
  github.com. Re-run this script to retry.

  Other */single-gpu recipes do not need this model.
============================================================================

EOF
}

# Best-effort install of the Magpie TTS text-normalization grammars. Returns
# non-zero (without exiting) if the release asset is unavailable, corrupt, or
# incomplete, so the GGUF setup still counts as a success.
install_tn_grammars() {
  local tar="${DEST}/${TN_ARCHIVE}"

  if [[ "${NEMO_SPEECH_CPP_RELEASE}" != "${TN_PINNED_RELEASE}" && "${TN_SHA256}" == "${TN_PINNED_SHA256}" ]]; then
    echo "NEMO_SPEECH_CPP_RELEASE=${NEMO_SPEECH_CPP_RELEASE} but the checksum is still the ${TN_PINNED_RELEASE} pin;" >&2
    echo "set NEMO_SPEECH_TN_SHA256 for ${NEMO_SPEECH_CPP_RELEASE} or the download will fail verification." >&2
  fi

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

  verify_sha256 "${tar}" "${TN_SHA256}" || { rm -f "${tar}"; return 1; }

  local stage="${DEST}/.tn_configs.staging"
  rm -rf "${stage}"
  mkdir -p "${stage}"
  if ! tar -xjf "${tar}" -C "${stage}"; then
    rm -rf "${stage}"
    return 1
  fi
  if [[ ! -f "${stage}/tn_configs/en/tokenize_and_classify.far" || \
        ! -f "${stage}/tn_configs/en/verbalize.far" ]]; then
    rm -rf "${stage}"
    return 1
  fi
  rm -rf "${DEST}/tn_configs"
  mv "${stage}/tn_configs" "${DEST}/tn_configs"
  rm -rf "${stage}"
}

warn_tn_unavailable() {
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

download_magpie_tts
download_diarizer

"${HF[@]}" download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
  nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --local-dir "${DEST}/nano-codec"

require_nonempty "${DEST}/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
require_nonempty "${DEST}/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
require_nonempty "${DEST}/magpie-tts/${MAGPIE_GGUF}"
require_nonempty "${DEST}/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"
link_diarization_asr

tn_ok=true
if ! install_tn_grammars; then
  warn_tn_unavailable
  tn_ok=false
fi

vad_ok=true
if ! install_vad_model; then
  warn_vad_unavailable
  vad_ok=false
fi

chmod -R a+rX "${DEST}"

if [[ "${tn_ok}" == true ]]; then
  echo "Models ready at ${DEST} (TTS text normalization enabled)."
else
  echo "Models ready at ${DEST} (TTS text normalization NOT installed; see warning above)."
fi
if [[ "${vad_ok}" != true ]]; then
  echo "generic-assistant/single-gpu cannot start until the VAD model is installed; see warning above."
fi
