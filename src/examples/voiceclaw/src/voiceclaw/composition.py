# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Application composition without importing a media framework."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from voiceclaw.adapters.nemoclaw.factory import build_nemoclaw_backend
from voiceclaw.backends import (
    BackendComposition,
    BackendPluginError,
    discover_backend_registration,
    validate_backend_name,
)
from voiceclaw.config import BackendProfile, ConfigurationError, VoiceClawConfig
from voiceclaw.model_contracts import (
    ModelContractCatalog,
    ModelContractError,
    load_model_contract_catalog,
)
from voiceclaw.ports.turns import EphemeralCommittedTurnPort

BackendFactory = Callable[
    [BackendProfile, Mapping[str, str], ModelContractCatalog],
    BackendComposition | tuple[EphemeralCommittedTurnPort | None, str],
]


def _disabled_backend(
    _profile: BackendProfile,
    _environ: Mapping[str, str],
    _model_contracts: ModelContractCatalog,
) -> BackendComposition:
    return BackendComposition(turn_backend=None, turn_status="disabled")


_BUILTIN_BACKEND_FACTORIES: Mapping[str, BackendFactory] = MappingProxyType(
    {
        "none": _disabled_backend,
        "disabled": _disabled_backend,
        "nemoclaw": build_nemoclaw_backend,
        "nemoclaw_committed_turn": build_nemoclaw_backend,
    }
)


def compose_backends(
    config: VoiceClawConfig,
    *,
    environ: Mapping[str, str],
    factories: Mapping[str, BackendFactory] | None = None,
    model_contracts: ModelContractCatalog | None = None,
) -> BackendComposition:
    """Build the selected adapter from trusted or selected installed factories."""
    profile = config.backend_profiles[config.default_backend]
    if model_contracts is None:
        try:
            model_contracts = load_model_contract_catalog(
                config.model_contracts.path,
                profile=config.model_contracts.profile,
            )
        except ModelContractError as error:
            raise ConfigurationError(str(error)) from error
    try:
        selected_kind = validate_backend_name(profile.kind, field="configured backend kind")
        registry = dict(_BUILTIN_BACKEND_FACTORIES)
        if factories is not None:
            for name, supplied_factory in factories.items():
                name = validate_backend_name(name, field="caller-supplied backend adapter name")
                if name in registry:
                    raise BackendPluginError(f"caller-supplied backend adapter {name!r} shadows a built-in")
                if not callable(supplied_factory):
                    raise BackendPluginError(f"caller-supplied backend adapter {name!r} is not callable")
                registry[name] = supplied_factory
        installed = discover_backend_registration(selected_kind, reserved_names=frozenset(registry))
    except BackendPluginError as error:
        raise ConfigurationError(str(error)) from error

    factory = registry.get(selected_kind)
    installed_factory = False
    if installed is not None:
        factory = installed.factory
        installed_factory = True
    if factory is None:
        raise ConfigurationError(f"unsupported default backend kind: {selected_kind}")
    environment_view = MappingProxyType(dict(environ))
    try:
        built = factory(profile, environment_view, model_contracts)
    except ConfigurationError:
        raise
    except Exception as error:
        raise ConfigurationError(f"backend adapter {selected_kind!r} could not be composed") from error
    if type(built) is BackendComposition:
        return built
    if installed_factory:
        raise ConfigurationError(f"installed backend adapter {selected_kind!r} factory must return BackendComposition")
    if not isinstance(built, tuple) or len(built) != 2:
        raise ConfigurationError(f"backend adapter {selected_kind!r} returned an invalid composition")
    turn_backend, turn_status = built
    try:
        return BackendComposition(turn_backend=turn_backend, turn_status=turn_status)
    except (TypeError, ValueError, BackendPluginError) as error:
        raise ConfigurationError(f"backend adapter {selected_kind!r} returned an invalid composition") from error


__all__ = ["BackendComposition", "BackendFactory", "compose_backends"]
