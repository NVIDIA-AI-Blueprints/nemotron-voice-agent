# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Capability- and state-gated semantic tools for frontend models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from voiceclaw.domain.models import (
    AgentQueryKind,
    BackendCapabilities,
    BackendOperation,
    Durability,
    WorkState,
)
from voiceclaw.interaction_profiles import (
    ArgumentBinding,
    InteractionProfile,
    SessionScope,
    WorkCardinality,
    load_interaction_profile_catalog,
)

_MAX_PROJECTED_IDENTIFIERS = 32
_STEERABLE_STATES = frozenset(
    {
        WorkState.SUBMITTED,
        WorkState.ACCEPTED,
        WorkState.QUEUED,
        WorkState.RUNNING,
        WorkState.WAITING_INPUT,
        WorkState.WAITING_PERMISSION,
    }
)
_CANCELLABLE_STATES = _STEERABLE_STATES


def _identifier(value: str, name: str) -> str:
    """Normalize one server-owned identifier used for tool projection."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized or "\x00" in normalized or len(normalized.encode("utf-8")) > 512:
        raise ValueError(f"{name} is invalid")
    return normalized


@dataclass(frozen=True, slots=True)
class ToolWork:
    """Minimum backend-authored Work fact required for tool admission."""

    work_id: str
    state: WorkState
    agent_target: str | None = None

    def __post_init__(self) -> None:
        """Validate the normalized Work identity and optional target."""
        object.__setattr__(self, "work_id", _identifier(self.work_id, "work_id"))
        object.__setattr__(self, "state", WorkState(self.state))
        if self.agent_target is not None:
            object.__setattr__(self, "agent_target", _identifier(self.agent_target, "agent_target"))

    @property
    def consumes_capacity(self) -> bool:
        """Return whether this Work still occupies a backend execution slot."""
        return not self.state.terminal

    @property
    def steerable(self) -> bool:
        """Return whether steering is valid for this normalized state."""
        return self.state in _STEERABLE_STATES

    @property
    def cancellable(self) -> bool:
        """Return whether cancellation is valid for this normalized state."""
        return self.state in _CANCELLABLE_STATES


@dataclass(frozen=True, slots=True)
class PendingToolQuery:
    """Backend-authored pending query plus local duplicate-suppression evidence."""

    query_id: str
    work_id: str
    kind: AgentQueryKind
    blocking: bool
    claimed: bool = False

    def __post_init__(self) -> None:
        """Validate immutable query-routing facts."""
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        object.__setattr__(self, "work_id", _identifier(self.work_id, "work_id"))
        object.__setattr__(self, "kind", AgentQueryKind(self.kind))
        if not isinstance(self.blocking, bool) or not isinstance(self.claimed, bool):
            raise TypeError("query blocking and claimed flags must be booleans")

    @property
    def operation(self) -> BackendOperation:
        """Return the canonical response operation for this query kind."""
        return {
            AgentQueryKind.INFORMATION: BackendOperation.ANSWER_QUERY,
            AgentQueryKind.PERMISSION: BackendOperation.RESPOND_PERMISSION,
        }[self.kind]


@dataclass(frozen=True, slots=True)
class ToolProjectionState:
    """Immutable facts used to decide which capability is valid right now."""

    control_revision: int = 0
    applied_sequence: int = 0
    works: tuple[ToolWork, ...] = ()
    pending_queries: tuple[PendingToolQuery, ...] = ()
    reserved_work_ids: tuple[str, ...] = ()
    anonymous_capacity_reservations: int = 0
    cancel_claimed_work_ids: frozenset[str] = frozenset()
    command_recovery_pending: bool = False

    def __post_init__(self) -> None:
        """Validate snapshot identity, cardinality, and query correlation."""
        if isinstance(self.control_revision, bool) or self.control_revision < 0:
            raise ValueError("control_revision must be a non-negative integer")
        if isinstance(self.applied_sequence, bool) or self.applied_sequence < 0:
            raise ValueError("applied_sequence must be a non-negative integer")
        works = tuple(self.works)
        queries = tuple(self.pending_queries)
        reserved = tuple(_identifier(work_id, "reserved_work_id") for work_id in self.reserved_work_ids)
        cancel_claims = frozenset(
            _identifier(work_id, "cancel_claimed_work_id") for work_id in self.cancel_claimed_work_ids
        )
        if not all(isinstance(work, ToolWork) for work in works):
            raise TypeError("works must contain ToolWork values")
        if not all(isinstance(query, PendingToolQuery) for query in queries):
            raise TypeError("pending_queries must contain PendingToolQuery values")
        work_ids = tuple(work.work_id for work in works)
        query_ids = tuple(query.query_id for query in queries)
        if len(work_ids) != len(set(work_ids)):
            raise ValueError("Work IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("pending query IDs must be unique")
        if len(reserved) != len(set(reserved)):
            raise ValueError("reserved Work IDs must be unique")
        if set(reserved) & set(work_ids):
            raise ValueError("reserved Work IDs must not duplicate projected Work")
        if isinstance(self.anonymous_capacity_reservations, bool) or self.anonymous_capacity_reservations < 0:
            raise ValueError("anonymous_capacity_reservations must be a non-negative integer")
        if not isinstance(self.command_recovery_pending, bool):
            raise TypeError("command_recovery_pending must be a boolean")
        by_id = {work.work_id: work for work in works}
        for query in queries:
            work = by_id.get(query.work_id)
            if work is None or not work.consumes_capacity:
                raise ValueError("pending AgentQueries must reference non-terminal Work")
        object.__setattr__(self, "works", works)
        object.__setattr__(self, "pending_queries", queries)
        object.__setattr__(self, "reserved_work_ids", reserved)
        object.__setattr__(self, "cancel_claimed_work_ids", cancel_claims)

    @property
    def known_work_ids(self) -> tuple[str, ...]:
        """Return a bounded deterministic set of backend Work IDs."""
        identifiers = tuple(work.work_id for work in self.works) + self.reserved_work_ids
        return identifiers[-_MAX_PROJECTED_IDENTIFIERS:]

    @property
    def capacity_work(self) -> tuple[ToolWork, ...]:
        """Return non-terminal Work, including unknown and cancelling states."""
        return tuple(work for work in self.works if work.consumes_capacity)

    @property
    def steerable_work(self) -> tuple[ToolWork, ...]:
        """Return Work that may accept steering now."""
        return tuple(work for work in self.works if work.steerable)

    @property
    def cancellable_work(self) -> tuple[ToolWork, ...]:
        """Return Work that may accept cancellation now."""
        return tuple(
            work for work in self.works if work.cancellable and work.work_id not in self.cancel_claimed_work_ids
        )

    @property
    def capacity_in_use(self) -> int:
        """Return projected Work plus receipt-backed admission reservations."""
        return len(self.capacity_work) + len(self.reserved_work_ids) + self.anonymous_capacity_reservations

    @property
    def unclaimed_queries(self) -> tuple[PendingToolQuery, ...]:
        """Return backend queries without a persisted local response claim."""
        return tuple(query for query in self.pending_queries if not query.claimed)


@dataclass(frozen=True, slots=True)
class SemanticTool:
    """Provider-neutral function tool definition."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach the schema from the caller's mutable mapping."""
        object.__setattr__(self, "input_schema", MappingProxyType(dict(self.input_schema)))


class CapabilityToolRegistry:
    """Project normalized Work tools from trusted capability and state facts."""

    def __init__(self, *, profile: InteractionProfile | None = None) -> None:
        """Bind one validated interaction profile to this capability view."""
        self._profile = profile or load_interaction_profile_catalog().resolve("stateless")

    @property
    def profile(self) -> InteractionProfile:
        """Return the immutable profile used by this registry."""
        return self._profile

    def project(
        self,
        capabilities: BackendCapabilities,
        *,
        state: ToolProjectionState | None = None,
    ) -> tuple[SemanticTool, ...]:
        """Return tools both authorized and applicable to the captured state."""
        current = self._projection_state(capabilities, state)
        return tuple(
            self._definition(tool_name, capabilities, current)
            for tool_name in sorted(name for name in self._profile.tools if name != "conversation.respond")
            if self._is_projectable(
                tool_name,
                capabilities,
                self._applicable_operations(tool_name, capabilities, current),
            )
        )

    def operation_for_tool(
        self,
        tool_name: str,
        capabilities: BackendCapabilities,
        *,
        state: ToolProjectionState | None = None,
        arguments: Mapping[str, Any] | None = None,
        resolved_operation: BackendOperation | None = None,
    ) -> BackendOperation:
        """Resolve one operation from the same authoritative state snapshot."""
        current = self._projection_state(capabilities, state)
        if tool_name not in self._profile.tools or tool_name == "conversation.respond":
            raise KeyError(tool_name)
        operations = self._applicable_operations(
            tool_name,
            capabilities,
            current,
            arguments=arguments or {},
        )
        if resolved_operation is not None:
            if resolved_operation not in operations:
                raise KeyError(tool_name)
            return resolved_operation
        if len(operations) == 1:
            return operations[0]
        if not operations:
            raise KeyError(tool_name)
        raise ValueError(f"{tool_name} requires authoritative server-side route context")

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        capabilities: BackendCapabilities | None = None,
        state: ToolProjectionState | None = None,
        resolved_operation: BackendOperation | None = None,
    ) -> Mapping[str, Any]:
        """Validate one model call against its topology and current state.

        A missing state performs shape-only validation for immutable command
        retries. New command admission always supplies a captured state.
        """
        if tool_name not in self._profile.tools or tool_name == "conversation.respond":
            raise KeyError(tool_name)
        if capabilities is None:
            capabilities = BackendCapabilities(
                backend_kind="validation",
                target_label="configured",
                sessionful=self._profile.session_scope is not SessionScope.NONE,
                agent_targets=("configured",) if self._profile.session_scope is SessionScope.TARGET else (),
            )
        if state is None:
            schema = self._base_definition(tool_name).input_schema
        else:
            self._validate_state(capabilities, state)
            operations = self._applicable_operations(tool_name, capabilities, state, arguments=arguments)
            if not self._is_projectable(tool_name, capabilities, operations):
                raise KeyError(tool_name)
            schema = self._definition(tool_name, capabilities, state).input_schema
        properties = schema["properties"]
        required = set(schema["required"])
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments for {tool_name}: {', '.join(unknown)}")
        missing = sorted(required - set(arguments))
        if missing:
            raise ValueError(f"missing arguments for {tool_name}: {', '.join(missing)}")

        validated = {name: self._validate_value(name, value, properties[name]) for name, value in arguments.items()}
        try:
            json.dumps(validated, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"arguments for {tool_name} must be finite JSON values") from error
        return MappingProxyType(validated)

    @staticmethod
    def _validate_value(name: str, value: Any, schema: Mapping[str, Any]) -> Any:
        expected = schema.get("type")
        if expected == "string":
            if not isinstance(value, str):
                raise ValueError(f"argument {name} must be a string")
            normalized = value.strip()
            if "\x00" in normalized or len(normalized) < int(schema.get("minLength", 0)):
                raise ValueError(f"argument {name} is too short")
            try:
                normalized.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"argument {name} is not valid Unicode text") from error
            maximum = int(schema.get("maxLength", 64 * 1024))
            if len(normalized) > maximum:
                raise ValueError(f"argument {name} is too long")
            value = normalized
        elif expected == "object":
            if not isinstance(value, Mapping):
                raise ValueError(f"argument {name} must be an object")
            value = dict(value)
        allowed = schema.get("enum")
        if allowed is not None and value not in allowed:
            raise ValueError(f"argument {name} must be one of: {', '.join(allowed)}")
        return value

    def _projection_state(
        self,
        capabilities: BackendCapabilities,
        state: ToolProjectionState | None,
    ) -> ToolProjectionState:
        if state is None:
            if self._profile.session_scope is not SessionScope.NONE or capabilities.durability is not Durability.NONE:
                raise ValueError("durable and stateful tool projection requires an authoritative state snapshot")
            state = ToolProjectionState()
        self._validate_state(capabilities, state)
        return state

    def _is_projectable(
        self,
        tool_name: str,
        capabilities: BackendCapabilities,
        operations: tuple[BackendOperation, ...],
    ) -> bool:
        if not operations:
            return False
        binding = self._profile.tool(tool_name).argument_binding
        if binding is ArgumentBinding.QUERY_ID_REQUIRED:
            return True
        if binding is ArgumentBinding.TARGET_GOAL_REQUIRED:
            return True
        return len(operations) == 1

    def _applicable_operations(
        self,
        tool_name: str,
        capabilities: BackendCapabilities,
        state: ToolProjectionState,
        *,
        arguments: Mapping[str, Any] | None = None,
    ) -> tuple[BackendOperation, ...]:
        if tool_name not in self._profile.tools or tool_name == "conversation.respond":
            raise KeyError(tool_name)
        arguments = arguments or {}
        if state.command_recovery_pending and tool_name != "work.status":
            return ()
        if tool_name == "work.delegate":
            operations = self._delegate_operations(capabilities, state, arguments)
        elif tool_name == "work.answer_agent":
            operations = self._answer_operations(capabilities, state, arguments)
        elif tool_name == "work.cancel":
            if not capabilities.supports(BackendOperation.CANCEL):
                return ()
            binding = self._profile.tool(tool_name).argument_binding
            if binding is ArgumentBinding.TARGET_REQUIRED:
                targets = self._cancellable_targets(state)
                target = arguments.get("agent")
                if target is not None and target not in targets:
                    return ()
                operations = (BackendOperation.CANCEL,) if targets else ()
            else:
                cancellable = tuple(work.work_id for work in state.cancellable_work)
                work_id = arguments.get("work_id")
                if work_id is not None and work_id not in cancellable:
                    return ()
                if self._profile.work_cardinality is WorkCardinality.SINGLE and len(cancellable) != 1:
                    return ()
                operations = (BackendOperation.CANCEL,) if cancellable else ()
        elif tool_name == "work.status":
            if not capabilities.supports(BackendOperation.STATUS):
                return ()
            if self._profile.work_cardinality is WorkCardinality.SINGLE:
                operations = (BackendOperation.STATUS,) if state.known_work_ids else ()
            elif self._profile.tool(tool_name).argument_binding in {
                ArgumentBinding.WORK_ID_OPTIONAL,
                ArgumentBinding.TARGET_OPTIONAL,
            }:
                operations = (BackendOperation.STATUS,)
            else:
                operations = ()
        else:
            raise KeyError(tool_name)
        return tuple(
            operation for operation in operations if self._operation_authorized(tool_name, capabilities, operation)
        )

    def _operation_authorized(
        self,
        tool_name: str,
        capabilities: BackendCapabilities,
        operation: BackendOperation,
    ) -> bool:
        spec = self._profile.tool(tool_name)
        return capabilities.supports(operation) and any(
            candidate.value == operation.value for candidate in spec.operations
        )

    def _delegate_operations(
        self,
        capabilities: BackendCapabilities,
        state: ToolProjectionState,
        arguments: Mapping[str, Any],
    ) -> tuple[BackendOperation, ...]:
        if self._profile.session_scope is SessionScope.NONE:
            if capabilities.supports(BackendOperation.SUBMIT) and self._has_submit_capacity(capabilities, state):
                return (BackendOperation.SUBMIT,)
            return ()
        if self._profile.work_cardinality is WorkCardinality.SINGLE:
            blocking = tuple(query for query in state.pending_queries if query.blocking)
            if blocking:
                return ()
            if state.capacity_work:
                if len(state.capacity_work) != 1 or len(state.steerable_work) != 1:
                    return ()
                return (BackendOperation.STEER,) if capabilities.supports(BackendOperation.STEER) else ()
            if state.capacity_in_use:
                return ()
            return (BackendOperation.SUBMIT,) if capabilities.supports(BackendOperation.SUBMIT) else ()
        if self._profile.work_cardinality is WorkCardinality.PER_TARGET:
            eligible = self._eligible_specialized_targets(capabilities, state)
            target = arguments.get("agent")
            if target is not None:
                operation = eligible.get(target)
                return (operation,) if operation is not None else ()
            return tuple(dict.fromkeys(eligible.values()))

        operations: list[BackendOperation] = []
        if capabilities.supports(BackendOperation.SUBMIT) and self._has_submit_capacity(capabilities, state):
            operations.append(BackendOperation.SUBMIT)
        if capabilities.supports(BackendOperation.STEER) and len(state.steerable_work) == 1:
            operations.append(BackendOperation.STEER)
        return tuple(operations)

    def _answer_operations(
        self,
        capabilities: BackendCapabilities,
        state: ToolProjectionState,
        arguments: Mapping[str, Any],
    ) -> tuple[BackendOperation, ...]:
        if "work.answer_agent" not in self._profile.tools:
            return ()
        query_id = arguments.get("query_id")
        queries = state.unclaimed_queries
        if query_id is not None:
            queries = tuple(query for query in queries if query.query_id == query_id)
        operations: list[BackendOperation] = []
        for query in queries:
            if capabilities.supports(query.operation) and query.operation not in operations:
                operations.append(query.operation)
        return tuple(operations)

    @staticmethod
    def _has_submit_capacity(capabilities: BackendCapabilities, state: ToolProjectionState) -> bool:
        active_count = state.capacity_in_use
        if active_count == 0:
            return True
        return capabilities.supports_parallel_work and active_count < capabilities.max_parallel_work

    @staticmethod
    def _work_by_target(state: ToolProjectionState) -> dict[str, ToolWork]:
        return {work.agent_target: work for work in state.capacity_work if work.agent_target is not None}

    def _eligible_specialized_targets(
        self,
        capabilities: BackendCapabilities,
        state: ToolProjectionState,
    ) -> dict[str, BackendOperation]:
        if state.reserved_work_ids or state.anonymous_capacity_reservations:
            return {}
        active = self._work_by_target(state)
        can_submit = self._has_submit_capacity(capabilities, state)
        eligible: dict[str, BackendOperation] = {}
        for target in capabilities.agent_targets:
            work = active.get(target)
            if work is not None and work.steerable and capabilities.supports(BackendOperation.STEER):
                eligible[target] = BackendOperation.STEER
            elif work is None and can_submit and capabilities.supports(BackendOperation.SUBMIT):
                eligible[target] = BackendOperation.SUBMIT
        return eligible

    @staticmethod
    def _cancellable_targets(state: ToolProjectionState) -> tuple[str, ...]:
        return tuple(work.agent_target for work in state.cancellable_work if work.agent_target is not None)[
            -_MAX_PROJECTED_IDENTIFIERS:
        ]

    def _base_definition(
        self,
        logical_name: str,
    ) -> SemanticTool:
        if logical_name not in self._profile.tools or logical_name == "conversation.respond":
            raise KeyError(logical_name)
        contract = self._profile.tool(logical_name)
        return SemanticTool(
            name=logical_name,
            description=contract.description,
            input_schema=contract.render_input_schema(),
        )

    def _definition(
        self,
        logical_name: str,
        capabilities: BackendCapabilities,
        state: ToolProjectionState,
    ) -> SemanticTool:
        contract = self._profile.tool(logical_name)
        enum_values: dict[str, tuple[str, ...]] = {}
        binding = contract.argument_binding
        if binding is ArgumentBinding.TARGET_GOAL_REQUIRED:
            enum_values["agent"] = tuple(self._eligible_specialized_targets(capabilities, state))
        elif binding is ArgumentBinding.TARGET_REQUIRED:
            enum_values["agent"] = self._cancellable_targets(state)
        elif logical_name == "work.answer_agent":
            query_ids = tuple(
                query.query_id for query in state.unclaimed_queries if capabilities.supports(query.operation)
            )[-_MAX_PROJECTED_IDENTIFIERS:]
            enum_values["query_id"] = query_ids
        elif binding is ArgumentBinding.WORK_ID_REQUIRED:
            work_ids = tuple(work.work_id for work in state.cancellable_work)[-_MAX_PROJECTED_IDENTIFIERS:]
            enum_values["work_id"] = work_ids
        elif binding is ArgumentBinding.WORK_ID_OPTIONAL and state.known_work_ids:
            enum_values["work_id"] = state.known_work_ids
        elif binding is ArgumentBinding.TARGET_OPTIONAL and capabilities.agent_targets:
            enum_values["agent"] = capabilities.agent_targets
        return SemanticTool(
            name=logical_name,
            description=contract.description,
            input_schema=contract.render_input_schema(enum_values=enum_values),
        )

    def _validate_state(self, capabilities: BackendCapabilities, state: ToolProjectionState) -> None:
        if self._profile.session_scope is SessionScope.NONE and capabilities.sessionful:
            raise ValueError("a sessionless interaction profile cannot use a sessionful backend")
        if self._profile.session_scope is not SessionScope.NONE and not capabilities.sessionful:
            raise ValueError("this interaction profile requires a sessionful backend")
        if self._profile.work_cardinality is WorkCardinality.SINGLE and len(state.capacity_work) > 1:
            raise ValueError("single_stateful topology cannot project more than one active Work")
        targets = tuple(work.agent_target for work in state.works if work.agent_target is not None)
        if self._profile.session_scope is SessionScope.TARGET:
            if not capabilities.agent_targets:
                raise ValueError("a target-scoped interaction profile requires negotiated targets")
            configured = set(capabilities.agent_targets)
            if any(work.agent_target is None for work in state.works):
                raise ValueError("specialized Work requires backend-authored agent_target evidence")
            if not set(targets) <= configured:
                raise ValueError("specialized Work target was not negotiated by the attachment")
            active_targets = tuple(work.agent_target for work in state.capacity_work if work.agent_target is not None)
            if len(active_targets) != len(set(active_targets)):
                raise ValueError("specialized topology allows at most one active Work per agent target")
        elif targets:
            raise ValueError("agent_target evidence is valid only for a target-scoped profile")
