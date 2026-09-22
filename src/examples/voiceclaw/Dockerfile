# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Build from the repository root:
# docker build -f src/examples/voiceclaw/Dockerfile -t voiceclaw .
FROM nvcr.io/nvidia/base/ubuntu:noble-20260610@sha256:a6015cf73597a174b8624ef0a9f78937f79a0862c078df4922dc35f26cec8fad

ARG VOICECLAW_VERSION=0.1.0
ARG VOICECLAW_SOURCE_REVISION=development
ARG VOICECLAW_SOURCE_URL=https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent

RUN case "${VOICECLAW_VERSION}" in \
      ""|*[!0-9A-Za-z.+-]*) echo "VOICECLAW_VERSION is invalid" >&2; exit 1 ;; \
    esac \
    && if [ "${VOICECLAW_SOURCE_REVISION}" != development ]; then \
      [ "${#VOICECLAW_SOURCE_REVISION}" -eq 40 ] \
      && case "${VOICECLAW_SOURCE_REVISION}" in *[!0-9a-f]*) false ;; *) true ;; esac \
      || { echo "VOICECLAW_SOURCE_REVISION must be development or a full lowercase Git revision" >&2; exit 1; }; \
    fi \
    && case "${VOICECLAW_SOURCE_URL}" in \
      https://*) \
        case "${VOICECLAW_SOURCE_URL}" in \
          *"@"*|*"?"*|*"#"*|*" "*) echo "VOICECLAW_SOURCE_URL must not contain authority credentials, a query, fragment, or spaces" >&2; exit 1 ;; \
        esac ;; \
      *) echo "VOICECLAW_SOURCE_URL must be an https URL" >&2; exit 1 ;; \
    esac

LABEL org.opencontainers.image.title="VoiceClaw"
LABEL org.opencontainers.image.description="VoiceClaw with selectable bundled or external Realtime frontends"
LABEL org.opencontainers.image.vendor="NVIDIA"
LABEL org.opencontainers.image.licenses="BSD-2-Clause"
LABEL org.opencontainers.image.version="${VOICECLAW_VERSION}"
LABEL org.opencontainers.image.revision="${VOICECLAW_SOURCE_REVISION}"
LABEL org.opencontainers.image.source="${VOICECLAW_SOURCE_URL}"

ENV PYTHONUNBUFFERED=1 \
    VOICECLAW_CONFIG=/var/lib/voiceclaw/config/voiceclaw.yaml \
    VOICECLAW_FRONTEND_RUNTIME_DIR=/run/voiceclaw/frontend \
    NLTK_DATA=/usr/local/share/nltk_data \
    XDG_CACHE_HOME=/var/cache/voiceclaw

ARG UV_VERSION=0.11.16
ARG UV_INSTALLER_SHA256=b9f925505899533f36a3acfdf8684c661ff2d5c8735f759fca768367b5996123
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    python3.12 \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 3 --connect-timeout 10 --max-time 120 \
      --output /tmp/uv-installer.sh \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" \
    && echo "${UV_INSTALLER_SHA256}  /tmp/uv-installer.sh" | sha256sum --check --strict \
    && sh /tmp/uv-installer.sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -f /tmp/uv-installer.sh

WORKDIR /app

RUN mkdir -p /app/legal
COPY LICENSE /app/legal/LICENSE
COPY third_party_oss_license.txt /app/legal/third_party_oss_license.txt

# Preserve the repository image's Ubuntu source-package compliance artifact.
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils dpkg-dev \
    && mkdir -p /app/legal/source \
    && if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
         sed -i -E 's/^(Types:.*deb)(\s*)$/\1 deb-src\2/; s/^Types: deb$/Types: deb deb-src/' \
           /etc/apt/sources.list.d/ubuntu.sources; \
       fi \
    && apt-get update \
    && cd /app/legal/source \
    && dpkg -l | awk '/^ii/{print $2}' | cut -d: -f1 | xargs -r apt-get source --download-only 2>/dev/null || true
RUN apt-get remove -y xz-utils dpkg-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv venv --python python3.12 && uv sync --frozen --no-dev

# Pipecat imports its Punkt sentence tokenizer during server startup. Fetch
# immutable, checksum-pinned data and its license during the image build.
RUN mkdir -p /usr/local/share/nltk_data/tokenizers /usr/share/licenses/nltk_data \
    && curl --fail --location --retry 3 --connect-timeout 10 --max-time 120 \
      --output /tmp/punkt_tab.zip \
      https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip \
    && curl --fail --location --retry 3 --connect-timeout 10 --max-time 120 \
      --output /usr/share/licenses/nltk_data/LICENSE \
      https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/LICENSE \
    && echo "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106  /tmp/punkt_tab.zip" \
      | sha256sum --check --strict \
    && echo "8d030ab5afc58f0b6a1f4207c12fd9553de6da2294efede65a0c58f9a6495fcc  /usr/share/licenses/nltk_data/LICENSE" \
      | sha256sum --check --strict \
    && python3.12 -c 'import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); root = "punkt_tab/english/"; files = ("abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt"); [archive.extract(root + name, sys.argv[2]) for name in files]' \
      /tmp/punkt_tab.zip /usr/local/share/nltk_data/tokenizers \
    && rm /tmp/punkt_tab.zip \
    && chmod -R a=rX /usr/local/share/nltk_data /usr/share/licenses/nltk_data

RUN groupadd --gid 10001 voiceclaw \
    && useradd --create-home --uid 10001 --gid voiceclaw voiceclaw \
    && groupadd --gid 10002 voiceclaw-nva \
    && useradd --create-home --uid 10002 --gid voiceclaw-nva voiceclaw-nva \
    && install -d -o root -g root -m 0755 /var/lib/voiceclaw \
    && install -d -o root -g root -m 0700 /var/lib/voiceclaw/config \
    && install -d -o root -g voiceclaw -m 0710 /var/lib/voiceclaw/credentials \
    && install -d -o voiceclaw -g voiceclaw -m 0700 /var/lib/voiceclaw/state \
    && install -d -m 0700 /var/cache/voiceclaw /var/cache/voiceclaw-nva \
    && touch /etc/voiceclaw-container-runtime \
    && chmod 0444 /etc/voiceclaw-container-runtime \
    && chown -R voiceclaw:voiceclaw /var/cache/voiceclaw \
    && chown -R voiceclaw-nva:voiceclaw-nva /var/cache/voiceclaw-nva \
    && chmod 0755 /var/lib/voiceclaw \
    && chmod 0700 /var/lib/voiceclaw/config /var/lib/voiceclaw/state \
        /var/cache/voiceclaw /var/cache/voiceclaw-nva

COPY src/ src/
COPY examples_registry.yaml examples_registry.yaml
RUN uv sync \
      --project /app/src/examples/voiceclaw \
      --extra server \
      --no-dev \
      --frozen \
    && ln -s /app/src/examples/voiceclaw/.venv/bin/voiceclaw-runtime /usr/local/bin/voiceclaw-runtime

VOLUME ["/var/lib/voiceclaw"]
EXPOSE 18790/tcp
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=20s --timeout=3s --start-period=90s --retries=3 \
    CMD ["/usr/local/bin/voiceclaw-runtime", "healthcheck"]

ENTRYPOINT ["/usr/local/bin/voiceclaw-runtime"]
CMD ["serve"]
