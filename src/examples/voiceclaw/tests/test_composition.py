# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.interaction import InteractionCoordinator
from voiceclaw.backends import BackendPluginError, DurableRuntimeUnavailableError
from voiceclaw.composition import BackendComposition, BackendFactory, compose_backends
from voiceclaw.config import BackendProfile, ConfigurationError, parse_config
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


def _base_config():
    return parse_config(
        {
            "schema_version": "voiceclaw.config.v3",
            "frontend_profiles": {
                "default": {
                    "kind": "openai_realtime",
                    "endpoint": "ws://127.0.0.1:7861/v1/realtime",
                    "model": "test/realtime",
                }
            },
            "default_frontend": "default",
            "backend_profiles": {"default": {"kind": "none", "settings": {}}},
            "default_backend": "default",
        }
    )


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


class _LifecycleTurnBackend(_CustomTurnBackend):
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.close_calls = 0

    async def check_selected_agent(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _LifecycleReadiness:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.close_calls = 0

    async def check_selected_agent(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_backend_composition_shutdown_releases_distinct_lifecycle_resources() -> None:
    backend = _LifecycleTurnBackend()
    readiness = _LifecycleReadiness()
    composition = BackendComposition(
        turn_backend=backend,
        turn_status="response_only",
        selected_agent_readiness=readiness,
    )

    asyncio.run(composition.shutdown())

    assert (backend.shutdown_calls, readiness.shutdown_calls) == (1, 1)
    assert (backend.close_calls, readiness.close_calls) == (0, 0)


def test_backend_composition_deduplicates_shared_lifecycle_resource_by_identity() -> None:
    backend = _LifecycleTurnBackend()
    composition = BackendComposition(
        turn_backend=backend,
        turn_status="response_only",
        selected_agent_readiness=backend,
    )

    composition.close()

    assert backend.close_calls == 1
    assert backend.shutdown_calls == 0


def test_backend_composition_attempts_every_cleanup_before_propagating_failure() -> None:
    class FailingBackend(_LifecycleTurnBackend):
        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            raise RuntimeError("shutdown failed")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close failed")

    backend = FailingBackend()
    readiness = _LifecycleReadiness()
    composition = BackendComposition(
        turn_backend=backend,
        turn_status="response_only",
        selected_agent_readiness=readiness,
    )

    with pytest.raises(RuntimeError, match="shutdown failed"):
        asyncio.run(composition.shutdown())
    assert (backend.shutdown_calls, readiness.shutdown_calls) == (1, 1)

    with pytest.raises(RuntimeError, match="close failed"):
        composition.close()
    assert (backend.close_calls, readiness.close_calls) == (1, 1)


def test_caller_supplied_factory_extends_registry_without_core_branch() -> None:
    environment = {"UNRELATED": "value"}
    profile = BackendProfile(kind="custom_response_only")
    config = replace(
        _base_config(),
        default_backend="custom",
        backend_profiles={"custom": profile},
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


def test_caller_factory_supplies_a_durable_port_while_core_owns_coordination() -> None:
    environment = {"UNRELATED": "value"}
    profile = BackendProfile(kind="custom_durable")
    config = replace(
        _base_config(),
        default_backend="custom",
        backend_profiles={"custom": profile},
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


def test_unknown_backend_kind_is_rejected_by_registry_lookup() -> None:
    config = replace(
        _base_config(),
        default_backend="unknown",
        backend_profiles={"unknown": BackendProfile(kind="not_registered")},
    )

    with pytest.raises(ConfigurationError, match="unsupported default backend kind"):
        compose_backends(config, environ={})
