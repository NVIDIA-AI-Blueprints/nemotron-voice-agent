# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import re
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


def test_container_exposes_the_generic_runtime_contract() -> None:
    example = Path(__file__).resolve().parents[1]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["/usr/local/bin/voiceclaw-runtime"]' in dockerfile
    assert 'CMD ["serve"]' in dockerfile
    assert 'CMD ["/usr/local/bin/voiceclaw-runtime", "healthcheck"]' in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile
    runtime_link = "ln -s /app/src/examples/voiceclaw/.venv/bin/voiceclaw-runtime /usr/local/bin/voiceclaw-runtime"
    assert runtime_link in dockerfile
    assert "EXPOSE 7860/tcp" in dockerfile
    assert "VOICECLAW_CONFIG=/etc/voiceclaw/voiceclaw.yaml" in dockerfile
    assert "https://127.0.0.1:7860/health" not in dockerfile
    assert "VOICECLAW_FRONTEND_RUNTIME_DIR=/run/voiceclaw/frontend" in dockerfile
    assert "VOICECLAW_INTERNAL_REALTIME_PORT" not in dockerfile
    assert "REALTIME_UPSTREAM_ENDPOINT=" not in dockerfile
    assert "install -d -o root -g root -m 0755 /var/lib/voiceclaw" in dockerfile
    assert "install -d -o root -g root -m 0700 /var/lib/voiceclaw/config" in dockerfile
    assert "install -d -o root -g voiceclaw -m 0710 /var/lib/voiceclaw/credentials" in dockerfile
    assert "install -d -o voiceclaw -g voiceclaw -m 0700 /var/lib/voiceclaw/state" in dockerfile
    assert "--extra openshell" in dockerfile
    assert "voiceclaw.container.yaml" in dockerfile


def test_container_has_one_generic_runtime_target() -> None:
    example = Path(__file__).resolve().parents[1]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM voiceclaw-base AS voiceclaw-runtime" in dockerfile
    assert dockerfile.rstrip().endswith('CMD ["serve"]')
    stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert stages[-1] == "FROM voiceclaw-base AS voiceclaw-runtime"


def test_ci_uses_only_current_image_and_verifier_contracts() -> None:
    example = Path(__file__).resolve().parents[1]
    repository = example.parents[2]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")
    workflow = (repository / ".github" / "workflows" / "voiceclaw.yml").read_text(encoding="utf-8")

    stages = {
        match.group(1)
        for match in re.finditer(r"^FROM\s+\S+(?:\s+AS\s+([a-z0-9_-]+))?\s*$", dockerfile, re.MULTILINE)
        if match.group(1) is not None
    }
    workflow_targets = set(re.findall(r"--target\s+([a-z0-9_-]+)", workflow))
    assert workflow_targets <= stages
    assert "--runtime-profile" not in workflow
    assert "--smoke-realtime" not in workflow
    assert "--extra server --extra openshell --group dev --frozen" in workflow
    assert '"${wheel}[server,openshell]"' in workflow
    assert "from voiceclaw.adapters.openshell_fabric import build_openshell_fabric_backend" in workflow


def test_container_installs_the_hash_locked_openshell_release_without_git() -> None:
    example = Path(__file__).resolve().parents[1]
    repository = example.parents[2]
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (example / "pyproject.toml").read_text(encoding="utf-8")
    lock = (example / "uv.lock").read_text(encoding="utf-8")
    third_party_licenses = (repository / "third_party_oss_license.txt").read_text(encoding="utf-8")

    assert '"openshell==0.1.2"' in pyproject
    assert 'name = "openshell"\nversion = "0.1.2"\nsource = { registry = "https://pypi.org/simple" }' in lock
    assert 'name = "openshell", marker = "extra == \'openshell\'", specifier = "==0.1.2"' in lock
    assert "sha256:8c409da4f176d42418d92366fe201f47cceef2c0fa432bfbce2bf938649d59cf" in lock
    assert "--extra openshell" in dockerfile
    assert "git+https://" not in pyproject
    assert "git \\" not in dockerfile
    assert "License for openshell" in third_party_licenses
    assert "License for cloudpickle" in third_party_licenses


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
    assert "VOICECLAW_OPENSHELL_CLIENT_SECRET:" not in compose
    assert "target: /run/voiceclaw/config/voiceclaw.yaml" in compose
    assert "VOICECLAW_CONFIG: /run/voiceclaw/config/voiceclaw.yaml" in compose
    assert "VOICECLAW_OPENSHELL_ENDPOINT:?Set connection.gatewayEndpoint URL" in compose
    assert "VOICECLAW_OPENSHELL_WORKSPACE:?Set the applied workspace" in compose
    assert "VOICECLAW_OPENSHELL_SANDBOX:?Set the authored sandbox name" in compose
    assert "127.0.0.1:8080" not in compose
    assert "VOICECLAW_OPENSHELL_WORKSPACE:-default" not in compose


def test_compose_mounts_one_profile_neutral_operator_root_read_only() -> None:
    example = Path(__file__).resolve().parents[1]
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")

    assert "VOICECLAW_OPERATOR_FILES_HOST" in compose
    assert "target: /run/voiceclaw/operator" in compose
    assert "VOICECLAW_OPENSHELL_CLIENT_SECRET:" not in compose
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


def test_onboarding_keeps_dotenv_and_secrets_out_of_shell_evaluation_and_argv() -> None:
    example = Path(__file__).resolve().parents[1]
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert "chmod 0600 src/examples/voiceclaw/.env" in readme
    assert ". src/examples/voiceclaw/.env" not in readme
    assert "VOICECLAW_OPENSHELL_CLIENT_SECRET=" not in readme
    assert "voiceclaw-client-secret" in readme
    assert "--master-key-file" in readme


def test_loopback_onboarding_uses_no_backend_secret_and_preserves_operator_group() -> None:
    example = Path(__file__).resolve().parents[1]
    readme = (example / "README.md").read_text(encoding="utf-8")
    compose = (example / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (example / "Dockerfile").read_text(encoding="utf-8")

    assert "useradd --create-home --uid 10001 --gid voiceclaw voiceclaw" in dockerfile
    assert 'VOICECLAW_FILE_GID="$(id -g)" docker compose' in readme
    assert "install -d -m 0710 src/examples/voiceclaw/operator" in readme
    assert "requires no OpenShell client secret" in readme
    assert '      - "${VOICECLAW_FILE_GID:-1000}"' in compose
    assert "VOICECLAW_OPENSHELL_CLIENT_SECRET_FILE" not in compose
    assert "VOICECLAW_OPENSHELL_CLIENT_ID" not in compose
    assert "VOICECLAW_OPENSHELL_ISSUER" not in compose
