# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.interaction import InteractionCoordinator
from voiceclaw.backends import (
    BACKEND_ENTRY_POINT_GROUP,
    BACKEND_PLUGIN_API_VERSION,
    BackendAdapterRegistration,
    BackendComposition,
    DurableRuntimeUnavailableError,
)
from voiceclaw.composition import compose_backends
from voiceclaw.config import BackendProfile, ConfigurationError, VoiceClawConfig, load_config
from voiceclaw.interaction_profiles import load_interaction_profile_catalog
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog

EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"
TEST_BEARER = "voiceclaw-test-deployment-bearer-0001"


class _InstalledAgentBackend:
    async def attach(self, _request: object) -> object:
        raise AssertionError("not exercised by plugin composition")

    async def execute(self, _command: object) -> object:
        raise AssertionError("not exercised by plugin composition")

    async def reconcile(self, _request: object) -> object:
        raise AssertionError("not exercised by plugin composition")

    async def events(self, _attachment_id: str, *, after_sequence: int | None):
        del after_sequence
        if False:
            yield  # pragma: no cover

    async def detach(self, _request: object) -> None:
        raise AssertionError("not exercised by plugin composition")


@dataclass
class _FakeEntryPoint:
    name: str
    exported: object
    group: str = BACKEND_ENTRY_POINT_GROUP
    load_count: int = 0

    def load(self) -> object:
        self.load_count += 1
        return self.exported


def _environment(tmp_path: Path) -> dict[str, str]:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    return {
        "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": str(credential_file),
        "REALTIME_UPSTREAM_ENDPOINT": "ws://127.0.0.1:7861/v1/realtime",
    }


def _select(config: VoiceClawConfig, kind: str) -> VoiceClawConfig:
    return replace(
        config,
        default_backend="installed",
        backend_profiles={**config.backend_profiles, "installed": BackendProfile(kind=kind)},
    )


def test_only_selected_installed_adapter_is_loaded_and_receives_trusted_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "custom_installed")
    calls: list[tuple[BackendProfile, Any, ModelContractCatalog]] = []
    backend = _InstalledAgentBackend()

    def factory(profile: BackendProfile, environ: Any, contracts: ModelContractCatalog) -> BackendComposition:
        calls.append((profile, environ, contracts))
        return BackendComposition(
            turn_backend=None,
            turn_status="custom_installed",
            agent_backend=backend,
        )

    selected = _FakeEntryPoint(
        name="custom_installed",
        exported=BackendAdapterRegistration(
            name="custom_installed",
            api_version=BACKEND_PLUGIN_API_VERSION,
            factory=factory,
        ),
    )
    unselected = _FakeEntryPoint(name="unselected_adapter", exported=object())
    discovery_calls: list[dict[str, object]] = []

    def entry_points(**kwargs: object) -> list[_FakeEntryPoint]:
        discovery_calls.append(dict(kwargs))
        return [unselected, selected]

    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", entry_points)

    composition = compose_backends(config, environ=environment)

    assert composition.turn_status == "custom_installed"
    assert composition.agent_backend is backend
    assert selected.load_count == 1
    assert unselected.load_count == 0
    assert discovery_calls == [{"group": BACKEND_ENTRY_POINT_GROUP}]
    assert calls[0][0] is config.backend_profiles["installed"]
    assert calls[0][1] == environment
    assert calls[0][2].profile == config.model_contracts.profile


def test_selected_durable_plugin_supplies_only_the_agent_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "durable_installed")
    backend = _InstalledAgentBackend()
    entry = _FakeEntryPoint(
        name="durable_installed",
        exported=BackendAdapterRegistration(
            name="durable_installed",
            api_version=BACKEND_PLUGIN_API_VERSION,
            factory=lambda *_args: BackendComposition(
                turn_backend=None,
                turn_status="durable_installed",
                agent_backend=backend,
            ),
        ),
    )
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: [entry])

    composition = compose_backends(config, environ=environment)

    assert entry.load_count == 1
    assert composition.agent_backend is backend
    assert composition.turn_backend is None
    assert not hasattr(composition, "runtime_factory")
    with SqliteStateStore(":memory:") as store:
        coordinator = composition.create_interaction_coordinator(
            state_store=store,
            model_contracts=load_model_contract_catalog(),
            interaction_profile=load_interaction_profile_catalog().resolve("stateless"),
        )
        assert isinstance(coordinator, InteractionCoordinator)
        with pytest.raises(DurableRuntimeUnavailableError, match="core Realtime session runtime"):
            composition.create_runtime(backend_profile="installed", state_store=store)


def test_duplicate_selected_entry_points_are_rejected_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "duplicate_adapter")
    entries = [
        _FakeEntryPoint(name="duplicate_adapter", exported=object()),
        _FakeEntryPoint(name="duplicate_adapter", exported=object()),
    ]
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: entries)

    with pytest.raises(ConfigurationError, match="multiple installed backend adapters"):
        compose_backends(config, environ=environment)

    assert [entry.load_count for entry in entries] == [0, 0]


def test_installed_adapter_cannot_shadow_a_builtin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "disabled")
    entry = _FakeEntryPoint(name="disabled", exported=object())
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: [entry])

    with pytest.raises(ConfigurationError, match="shadows a trusted registry entry"):
        compose_backends(config, environ=environment)

    assert entry.load_count == 0


def test_caller_supplied_factory_cannot_shadow_a_builtin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = load_config(EXAMPLE_CONFIG, environ=environment)
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: [])

    with pytest.raises(ConfigurationError, match="shadows a built-in"):
        compose_backends(config, environ=environment, factories={"disabled": lambda *_args: None})


def test_installed_adapter_api_version_must_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "future_adapter")
    entry = _FakeEntryPoint(
        name="future_adapter",
        exported=BackendAdapterRegistration(
            name="future_adapter",
            api_version="2",
            factory=lambda *_args: BackendComposition(turn_backend=None, turn_status="future_adapter"),
        ),
    )
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: [entry])

    with pytest.raises(ConfigurationError, match="unsupported API version '2'"):
        compose_backends(config, environ=environment)

    assert entry.load_count == 1


@pytest.mark.parametrize(
    ("exported", "message"),
    [
        (object(), "must export BackendAdapterRegistration"),
        (
            BackendAdapterRegistration(
                name="different_adapter",
                api_version=BACKEND_PLUGIN_API_VERSION,
                factory=lambda *_args: BackendComposition(turn_backend=None, turn_status="different_adapter"),
            ),
            "exported registration 'different_adapter'",
        ),
    ],
)
def test_installed_registration_shape_and_identity_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exported: object,
    message: str,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "selected_adapter")
    entry = _FakeEntryPoint(name="selected_adapter", exported=exported)
    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", lambda **_kwargs: [entry])

    with pytest.raises(ConfigurationError, match=message):
        compose_backends(config, environ=environment)


def test_installed_factory_must_return_normalized_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), "legacy_adapter")
    registration = BackendAdapterRegistration(
        name="legacy_adapter",
        api_version=BACKEND_PLUGIN_API_VERSION,
        factory=lambda *_args: (None, "legacy_adapter"),  # type: ignore[arg-type,return-value]
    )
    monkeypatch.setattr(
        "voiceclaw.backends.importlib_metadata.entry_points",
        lambda **_kwargs: [_FakeEntryPoint(name="legacy_adapter", exported=registration)],
    )

    with pytest.raises(ConfigurationError, match="factory must return BackendComposition"):
        compose_backends(config, environ=environment)


@pytest.mark.parametrize("kind", ["Uppercase", "has-hyphen", "double__underscore", "a" * 65])
def test_configured_adapter_name_is_validated_before_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    environment = _environment(tmp_path)
    config = _select(load_config(EXAMPLE_CONFIG, environ=environment), kind)
    discovery_calls = 0

    def entry_points(**_kwargs: object) -> list[object]:
        nonlocal discovery_calls
        discovery_calls += 1
        return []

    monkeypatch.setattr("voiceclaw.backends.importlib_metadata.entry_points", entry_points)

    with pytest.raises(ConfigurationError, match="configured backend kind"):
        compose_backends(config, environ=environment)

    assert discovery_calls == 0
