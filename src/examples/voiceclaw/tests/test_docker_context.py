# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from pathlib import Path


def test_example_docker_context_excludes_local_secrets_and_worktrees() -> None:
    example = Path(__file__).resolve().parents[1]
    patterns = set((example / "Dockerfile.dockerignore").read_text(encoding="utf-8").splitlines())

    assert {
        ".git",
        ".env",
        "**/.env",
        "**/.runtime",
        "**/operator/*",
        "**/.venv",
        "**/*.docx",
        "models",
        "src/examples/voiceclaw/scripts",
        "src/examples/voiceclaw/tests",
    } <= patterns


def test_container_exposes_the_exact_managed_runtime_contract() -> None:
    example = Path(__file__).resolve().parents[1]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["/usr/local/bin/voiceclaw-runtime"]' in dockerfile
    assert 'CMD ["serve"]' in dockerfile
    assert 'CMD ["/usr/local/bin/voiceclaw-runtime", "healthcheck"]' in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile
    runtime_link = "ln -s /app/src/examples/voiceclaw/.venv/bin/voiceclaw-runtime /usr/local/bin/voiceclaw-runtime"
    assert runtime_link in dockerfile
    assert "EXPOSE 18790/tcp" in dockerfile
    assert "VOICECLAW_CONFIG=/var/lib/voiceclaw/config/voiceclaw.yaml" in dockerfile
    assert "https://127.0.0.1:7860/health" not in dockerfile
    assert "VOICECLAW_FRONTEND_RUNTIME_DIR=/run/voiceclaw/frontend" in dockerfile
    assert "VOICECLAW_INTERNAL_REALTIME_PORT" not in dockerfile
    assert "REALTIME_UPSTREAM_ENDPOINT=" not in dockerfile
    assert "install -d -o root -g root -m 0755 /var/lib/voiceclaw" in dockerfile
    assert "install -d -o root -g root -m 0700 /var/lib/voiceclaw/config" in dockerfile
    assert "install -d -o root -g voiceclaw -m 0710 /var/lib/voiceclaw/credentials" in dockerfile
    assert "install -d -o voiceclaw -g voiceclaw -m 0700 /var/lib/voiceclaw/state" in dockerfile


def test_container_pins_runtime_tokenizer_data_and_license() -> None:
    example = Path(__file__).resolve().parents[1]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")

    revision = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
    assert f"nltk/nltk_data/{revision}/packages/tokenizers/punkt_tab.zip" in dockerfile
    assert f"nltk/nltk_data/{revision}/LICENSE" in dockerfile
    assert "nltk/nltk_data/gh-pages/" not in dockerfile
    assert "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106" in dockerfile
    assert "8d030ab5afc58f0b6a1f4207c12fd9553de6da2294efede65a0c58f9a6495fcc" in dockerfile
    assert 'root = "punkt_tab/english/"' in dockerfile
    assert all(
        name in dockerfile
        for name in ("abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt")
    )
    assert "/usr/share/licenses/nltk_data/LICENSE" in dockerfile


def test_container_records_validated_oci_build_provenance_without_secret_arguments() -> None:
    example = Path(__file__).resolve().parents[1]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")

    for argument in ("VOICECLAW_VERSION", "VOICECLAW_SOURCE_REVISION", "VOICECLAW_SOURCE_URL"):
        assert f"ARG {argument}" in dockerfile
    for label in (
        "org.opencontainers.image.version",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.source",
        "org.opencontainers.image.licenses",
    ):
        assert f"LABEL {label}" in dockerfile
    argument_lines = [line.upper() for line in dockerfile.splitlines() if line.startswith("ARG ")]
    assert not any(
        secret_name in line
        for line in argument_lines
        for secret_name in ("API_KEY", "BEARER", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")
    )


def test_compose_mounts_one_operator_owned_configuration_file() -> None:
    example = Path(__file__).resolve().parents[1]
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")

    assert "VOICECLAW_CONFIG_FILE_HOST" in compose


def test_compose_does_not_bulk_inject_the_substitution_environment() -> None:
    example = Path(__file__).resolve().parents[1]
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")

    assert "env_file:" not in compose
    assert "NVIDIA_API_KEY" not in compose
    assert "HOSTED_REALTIME_API_KEY" not in compose
    assert "target: /run/voiceclaw/config/voiceclaw.yaml" in compose
    assert "VOICECLAW_CONFIG: /run/voiceclaw/config/voiceclaw.yaml" in compose


def test_compose_mounts_one_profile_neutral_operator_root_read_only() -> None:
    example = Path(__file__).resolve().parents[1]
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")

    assert "VOICECLAW_OPERATOR_FILES_HOST" in compose
    assert "target: /run/voiceclaw/operator" in compose
    assert "/run/secrets/nemoclaw_voice_gateway_bearer" not in compose
    assert "VOICECLAW_TLS_CERTFILE:" not in compose
    assert "VOICECLAW_TLS_KEYFILE:" not in compose
    assert (example / "operator" / ".gitignore").is_file()


def test_compose_prevents_children_from_regaining_privilege_on_exec() -> None:
    example = Path(__file__).resolve().parents[1]
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")

    assert "security_opt:" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:\n      - ALL" in compose
    for capability in ("CHOWN", "DAC_OVERRIDE", "KILL", "SETGID", "SETUID"):
        assert f"      - {capability}" in compose


def test_onboarding_keeps_dotenv_and_public_master_out_of_shell_evaluation_and_argv() -> None:
    example = Path(__file__).resolve().parents[1]
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert "chmod 0600 src/examples/voiceclaw/.env" in readme
    assert ". src/examples/voiceclaw/.env" not in readme
    assert "Authorization: Bearer ${VOICECLAW_REALTIME_API_KEY}" not in readme
    assert "voiceclaw-client-secret" in readme
    assert "--master-key-file" in readme
