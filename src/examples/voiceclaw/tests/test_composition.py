# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from voiceclaw.adapters.nemoclaw.committed_turn import DEFAULT_RESULT_DISPLAY_BUDGET_BYTES, EndpointPolicy
from voiceclaw.adapters.nemoclaw.factory import build_committed_turn_backend
from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.interaction import InteractionCoordinator
from voiceclaw.backends import BackendPluginError, DurableRuntimeUnavailableError
from voiceclaw.composition import BackendComposition, BackendFactory, compose_backends
from voiceclaw.config import BackendProfile, ConfigurationError, load_config
from voiceclaw.domain.models import (
    BackendCapabilities,
    BackendEvent,
    BackendOperation,
    CapabilitySource,
    Durability,
    EventDelivery,
)
from voiceclaw.interaction_profiles import load_interaction_profile_catalog
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.interaction import (
    AttachRequest,
    BackendAdmission,
    BackendAttachment,
    BackendCommandReceipt,
    DetachRequest,
    ReconcileRequest,
    WorkCommand,
)
from voiceclaw.ports.turns import CommittedTurnBackend, CommittedTurnRequest, CommittedTurnResult

EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"
BASE_ENV = {
    "REALTIME_UPSTREAM_ENDPOINT": "ws://127.0.0.1:7861/v1/realtime",
}
TEST_BEARER = "voiceclaw-test-deployment-bearer-0001"


def _example_environment(tmp_path: Path) -> dict[str, str]:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    return {
        **BASE_ENV,
        "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": str(credential_file),
    }


def _file_profile(path: Path) -> BackendProfile:
    return BackendProfile(
        kind="nemoclaw",
        credential_file=str(path),
        settings={"mode": "response_only", "endpoint": "http://127.0.0.1:18800", "endpoint_policy": "loopback_only"},
    )


def test_example_configuration_enables_only_the_response_only_backend(tmp_path: Path) -> None:
    environment = _example_environment(tmp_path)
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    composition = compose_backends(config, environ=environment)

    assert composition.turn_backend is not None
    assert composition.turn_status == "response_only"


def test_selected_agent_readiness_requires_an_active_interaction_port() -> None:
    class Readiness:
        async def check_selected_agent(self) -> None:
            return None

    with pytest.raises(BackendPluginError, match="requires an active interaction port"):
        BackendComposition(
            turn_backend=None,
            turn_status="disabled",
            selected_agent_readiness=Readiness(),
        )


def test_stable_nemoclaw_kind_uses_the_local_endpoint_default(tmp_path: Path) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    profile = BackendProfile(
        kind="nemoclaw",
        credential_file=str(credential_file),
        settings={"mode": "response_only"},
    )

    with patch("voiceclaw.adapters.nemoclaw.factory.NemoClawCommittedTurnAdapter") as adapter_type:
        build_committed_turn_backend(profile, {})

    assert adapter_type.call_args.kwargs["origin"] == "http://127.0.0.1:18800"


def test_ambient_nemoclaw_endpoint_cannot_override_yaml_or_default(tmp_path: Path) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    profile = BackendProfile(
        kind="nemoclaw",
        credential_file=str(credential_file),
        settings={"mode": "response_only"},
    )

    with patch("voiceclaw.adapters.nemoclaw.factory.NemoClawCommittedTurnAdapter") as adapter_type:
        build_committed_turn_backend(profile, {"NEMOCLAW_VOICE_GATEWAY_ORIGIN": "http://127.0.0.1:19000"})
        build_committed_turn_backend(
            replace(profile, settings={**profile.settings, "endpoint": "http://127.0.0.1:19001"}),
            {"NEMOCLAW_VOICE_GATEWAY_ORIGIN": "http://127.0.0.1:19000"},
        )

    assert adapter_type.call_args_list[0].kwargs["origin"] == "http://127.0.0.1:18800"
    assert adapter_type.call_args_list[1].kwargs["origin"] == "http://127.0.0.1:19001"


@pytest.mark.parametrize("settings", [{"mode": "agent_session_v1alpha1"}, {"typo": True}])
def test_nemoclaw_rejects_unsupported_modes_and_unknown_settings(tmp_path: Path, settings: dict[str, object]) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    profile = BackendProfile(kind="nemoclaw", credential_file=str(credential_file), settings=settings)

    with pytest.raises(ConfigurationError):
        build_committed_turn_backend(profile, {})


def test_nemoclaw_rejects_environment_backed_credentials(tmp_path: Path) -> None:
    environment = {**_example_environment(tmp_path), "NEMOCLAW_API_KEY": TEST_BEARER}
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    config = replace(
        config,
        default_backend="default",
        backend_profiles={
            **config.backend_profiles,
            "default": BackendProfile(
                kind="nemoclaw_committed_turn",
                credential_env="NEMOCLAW_API_KEY",
                settings={"endpoint": "http://127.0.0.1:8080", "endpoint_policy": "loopback_only"},
            ),
        },
    )
    with pytest.raises(ConfigurationError, match="must use credential.file"):
        compose_backends(config, environ=environment)


class _CustomTurnBackend:
    async def inspect(self) -> CommittedTurnBackend:
        return CommittedTurnBackend(
            label="Custom",
            target_ref="custom-target",
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label="Custom",
                revision="custom-test-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
            ),
            capability_source=CapabilitySource.OPERATOR_CONFIGURED,
            capability_source_id="custom_response_only",
        )

    async def commit_turn(self, _request: CommittedTurnRequest) -> CommittedTurnResult:
        raise AssertionError("not exercised by composition")

    async def stream_turn(self, _request: CommittedTurnRequest):
        if False:
            yield  # pragma: no cover


def test_caller_supplied_factory_extends_registry_without_core_branch(tmp_path: Path) -> None:
    environment = _example_environment(tmp_path)
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    profile = BackendProfile(kind="custom_response_only")
    config = replace(
        config,
        default_backend="custom",
        backend_profiles={**config.backend_profiles, "custom": profile},
    )
    backend = _CustomTurnBackend()
    calls: list[tuple[BackendProfile, dict[str, str], ModelContractCatalog]] = []

    def build(selected: BackendProfile, environ: dict[str, str], contracts: ModelContractCatalog):
        calls.append((selected, environ, contracts))
        return backend, "custom_response_only"

    factories: dict[str, BackendFactory] = {"custom_response_only": build}
    composition = compose_backends(config, environ=environment, factories=factories)

    assert composition.turn_backend is backend
    assert composition.turn_status == "custom_response_only"
    assert calls[0][:2] == (profile, environment)
    assert calls[0][2].profile == "default"


class _DurableBackend:
    def __init__(self) -> None:
        self.attach_requests: list[AttachRequest] = []

    async def attach(self, request: AttachRequest) -> BackendAttachment:
        self.attach_requests.append(request)
        return BackendAttachment(
            attachment_id="attachment-custom",
            backend_session_id="backend-session-custom",
            capabilities=BackendCapabilities(
                backend_kind="custom_durable",
                target_label="Custom durable backend",
                revision="custom-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
                durability=Durability.BACKEND,
                event_delivery=EventDelivery.ORDERED_REPLAY,
                sessionful=True,
            ),
        )

    async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
        return BackendCommandReceipt(
            command_id=command.command_id,
            admission=BackendAdmission.ACCEPTED,
            work_id="work-custom",
        )

    async def reconcile(self, request: ReconcileRequest) -> BackendCommandReceipt:
        return BackendCommandReceipt(
            command_id=request.command.command_id,
            admission=BackendAdmission.ACCEPTED,
            work_id="work-custom",
        )

    async def events(self, attachment_id: str, *, after_sequence: int | None) -> AsyncIterator[BackendEvent]:
        del attachment_id, after_sequence
        if False:
            yield  # pragma: no cover

    async def detach(self, request: DetachRequest) -> None:
        del request


def test_caller_factory_supplies_a_durable_port_while_core_owns_coordination(tmp_path: Path) -> None:
    environment = _example_environment(tmp_path)
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    profile = BackendProfile(kind="custom_durable")
    config = replace(
        config,
        default_backend="custom",
        backend_profiles={**config.backend_profiles, "custom": profile},
    )
    backend = _DurableBackend()

    def build(
        selected: BackendProfile,
        environ: dict[str, str],
        contracts: ModelContractCatalog,
    ) -> BackendComposition:
        assert selected is profile
        assert environ == environment
        assert contracts.profile == "default"
        return BackendComposition(
            turn_backend=None,
            turn_status="durable_agent_session",
            agent_backend=backend,
        )

    composition = compose_backends(
        config,
        environ=environment,
        factories={"custom_durable": build},
    )

    assert composition.turn_backend is None
    assert composition.agent_backend is backend
    assert composition.turn_status == "durable_agent_session"
    with SqliteStateStore(":memory:") as store:
        coordinator = composition.create_interaction_coordinator(
            state_store=store,
            model_contracts=load_model_contract_catalog(),
            interaction_profile=load_interaction_profile_catalog().resolve("stateless"),
        )
        assert isinstance(coordinator, InteractionCoordinator)
        asyncio.run(
            coordinator.attach(
                AttachRequest(
                    session_id="session-custom",
                    conversation_id="conversation-custom",
                    backend_profile="custom",
                )
            )
        )
        assert backend.attach_requests == [
            AttachRequest(
                session_id="session-custom",
                conversation_id="conversation-custom",
                backend_profile="custom",
            )
        ]
        with pytest.raises(DurableRuntimeUnavailableError, match="core Realtime session runtime"):
            composition.create_runtime(backend_profile="custom", state_store=store)


def test_backend_composition_does_not_accept_a_plugin_owned_runtime_factory() -> None:
    with pytest.raises(TypeError, match="runtime_factory"):
        BackendComposition(  # type: ignore[call-arg]
            turn_backend=None,
            turn_status="custom_durable",
            runtime_factory=lambda *_args: object(),
        )


def test_backend_composition_rejects_ambiguous_or_missing_active_ports() -> None:
    with pytest.raises(BackendPluginError, match="cannot mix ephemeral and durable"):
        BackendComposition(
            turn_backend=_CustomTurnBackend(),
            turn_status="ambiguous",
            agent_backend=_DurableBackend(),
        )
    with pytest.raises(BackendPluginError, match="active backend composition must expose"):
        BackendComposition(turn_backend=None, turn_status="missing")
    with pytest.raises(BackendPluginError, match="disabled backend composition cannot expose"):
        BackendComposition(
            turn_backend=None,
            turn_status="disabled",
            agent_backend=_DurableBackend(),
        )
    with pytest.raises(TypeError, match="must implement AgentInteractionPort"):
        BackendComposition(
            turn_backend=None,
            turn_status="invalid_agent",
            agent_backend=object(),  # type: ignore[arg-type]
        )


def test_unknown_backend_kind_is_rejected_by_registry_lookup(tmp_path: Path) -> None:
    environment = _example_environment(tmp_path)
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    config = replace(
        config,
        default_backend="unknown",
        backend_profiles={**config.backend_profiles, "unknown": BackendProfile(kind="not_registered")},
    )

    with pytest.raises(ConfigurationError, match="unsupported default backend kind"):
        compose_backends(config, environ=environment)


def test_credential_file_is_read_without_leaking_it_into_environment(tmp_path: Path) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)

    with patch("voiceclaw.adapters.nemoclaw.factory.NemoClawCommittedTurnAdapter") as adapter_type:
        contracts = load_model_contract_catalog()
        backend, status = build_committed_turn_backend(_file_profile(credential_file), {}, contracts)

    assert backend is adapter_type.return_value
    assert status == "response_only"
    adapter_type.assert_called_once_with(
        origin="http://127.0.0.1:18800",
        deployment_bearer=TEST_BEARER,
        endpoint_policy=EndpointPolicy.LOOPBACK_ONLY,
        exchange_deadline_seconds=125.0,
        result_display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
        model_contracts=contracts,
    )


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), "125"])
def test_nemoclaw_exchange_deadline_must_be_a_finite_positive_number(tmp_path: Path, value: object) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    profile = BackendProfile(
        kind="nemoclaw_committed_turn",
        credential_file=str(credential_file),
        settings={
            "endpoint": "http://127.0.0.1:18800",
            "endpoint_policy": "loopback_only",
            "exchange_deadline_seconds": value,
        },
    )

    with pytest.raises(ConfigurationError, match="exchange_deadline_seconds"):
        build_committed_turn_backend(profile, {})


@pytest.mark.parametrize("value", [True, 0, -1, 128_001, 8192.0, "8192"])
def test_nemoclaw_result_display_budget_must_be_a_bounded_integer(tmp_path: Path, value: object) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    profile = BackendProfile(
        kind="nemoclaw_committed_turn",
        credential_file=str(credential_file),
        settings={
            "endpoint": "http://127.0.0.1:18800",
            "endpoint_policy": "loopback_only",
            "result_display_budget_bytes": value,
        },
    )

    with pytest.raises(ConfigurationError, match="result_display_budget_bytes"):
        build_committed_turn_backend(profile, {})


def test_credential_file_symlink_is_rejected(tmp_path: Path) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    symlink = tmp_path / "credential-link"
    symlink.symlink_to(credential_file)

    with pytest.raises(ConfigurationError, match="could not be read securely"):
        build_committed_turn_backend(_file_profile(symlink), {})


def test_credential_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    credential_fifo = tmp_path / "nemoclaw-deployment-bearer"
    os.mkfifo(credential_fifo, mode=0o600)

    with pytest.raises(ConfigurationError, match="bounded regular file"):
        build_committed_turn_backend(_file_profile(credential_fifo), {})


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b" ",
        b"deployment secret that is deliberately long",
        b"voiceclaw-test-deployment-bearer-0001\nsecond-value",
        b"voiceclaw-test-deployment-bearer-0001\r\n",
        b"\xff",
        b"x" * 4098,
    ],
)
def test_malformed_credential_file_is_rejected(tmp_path: Path, content: bytes) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_bytes(content)
    credential_file.chmod(0o600)

    with pytest.raises(ConfigurationError, match="credential file"):
        build_committed_turn_backend(_file_profile(credential_file), {})


@pytest.mark.parametrize("mode", [0o620, 0o604, 0o701])
def test_credential_file_rejects_unsafe_permissions(tmp_path: Path, mode: int) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(mode)

    with pytest.raises(ConfigurationError, match="inaccessible to others"):
        build_committed_turn_backend(_file_profile(credential_file), {})


def test_credential_file_rejects_posix_access_acl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    monkeypatch.setattr(
        "voiceclaw.adapters.nemoclaw.factory.os.getxattr",
        lambda *_args, **_kwargs: b"extended-acl",
        raising=False,
    )

    with pytest.raises(ConfigurationError, match="must not have a POSIX access ACL"):
        build_committed_turn_backend(_file_profile(credential_file), {})
