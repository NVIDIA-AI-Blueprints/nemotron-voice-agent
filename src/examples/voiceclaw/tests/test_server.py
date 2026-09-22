# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from voiceclaw.adapters.state.sqlite import SqliteStateStore
from voiceclaw.backends import BackendComposition, DurableRuntimeUnavailableError
from voiceclaw.config import (
    ConfigurationError,
    ListenerSecurity,
    ServerConfig,
    StateConfig,
    load_config,
)
from voiceclaw.ports.readiness import SelectedAgentReadinessError
from voiceclaw.server import (
    _listener_port,
    _public_key,
    _speech_delivery_capabilities,
    _tls_listener_files,
    _upstream_bearer,
    _with_listener_host,
    create_app,
    validate_configuration,
)
from voiceclaw.server import _parser as _server_parser

EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"
TEST_BEARER = "voiceclaw-test-deployment-bearer-0001"


@pytest.fixture
def base_env(tmp_path: Path) -> dict[str, str]:
    credential_file = tmp_path / "nemoclaw-deployment-bearer"
    credential_file.write_text(f"{TEST_BEARER}\n", encoding="ascii")
    credential_file.chmod(0o600)
    return {
        "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": str(credential_file),
        "REALTIME_UPSTREAM_ENDPOINT": "ws://127.0.0.1:7861/v1/realtime",
        "REALTIME_UPSTREAM_API_KEY": "internal-upstream-key",
    }


def _config(*, server: ServerConfig, environ: dict[str, str]) -> object:
    return replace(
        load_config(EXAMPLE_CONFIG, environ=environ),
        server=server,
        state=StateConfig(path=":memory:"),
    )


class _DurableBackend:
    async def attach(self, _request: object) -> object:
        raise AssertionError("durable facade startup must fail before attachment")

    async def execute(self, _command: object) -> object:
        raise AssertionError("durable facade startup must fail before execution")

    async def reconcile(self, _request: object) -> object:
        raise AssertionError("durable facade startup must fail before reconciliation")

    async def events(self, _attachment_id: str, *, after_sequence: int | None):
        del after_sequence
        if False:
            yield  # pragma: no cover

    async def detach(self, _request: object) -> None:
        raise AssertionError("durable facade startup must fail before detachment")


class _SelectedAgentReadiness:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    async def check_selected_agent(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure


class _TurnBackend:
    async def inspect(self):  # pragma: no cover - readiness does not inspect or execute work
        raise AssertionError("readiness must not create a backend turn")

    async def commit_turn(self, _request):  # pragma: no cover - readiness does not inspect or execute work
        raise AssertionError("readiness must not create a backend turn")

    async def stream_turn(self, _request):  # pragma: no cover - readiness does not inspect or execute work
        raise AssertionError("readiness must not create a backend turn")


def test_durable_plugin_fails_before_the_facade_opens_state(
    base_env: dict[str, str],
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    composition = BackendComposition(
        turn_backend=None,
        turn_status="durable_test",
        agent_backend=_DurableBackend(),
    )

    with (
        patch("voiceclaw.server.SqliteStateStore") as store_type,
        pytest.raises(DurableRuntimeUnavailableError, match="core Realtime session runtime"),
    ):
        create_app(config, environ=base_env, composition=composition)

    store_type.assert_not_called()


@pytest.mark.parametrize("profile_name", ["single_stateful", "conductor", "specialized"])
def test_response_only_backend_rejects_stateful_interaction_profiles_before_opening_state(
    base_env: dict[str, str],
    profile_name: str,
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    backend = replace(config.backend_profiles[config.default_backend], interaction_profile=profile_name)
    config = replace(config, backend_profiles={config.default_backend: backend})

    with (
        patch("voiceclaw.server.SqliteStateStore") as store_type,
        pytest.raises(ConfigurationError, match="response-only backends require a sessionless interaction profile"),
    ):
        create_app(config, environ=base_env)

    store_type.assert_not_called()


def test_headless_server_does_not_publish_ui_routes(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    app = create_app(config, environ=base_env)

    async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health"), await client.get("/"), await client.get("/app.js")

    try:
        health, page, script = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert health.status_code == 200
    assert page.status_code == 404
    assert script.status_code == 404


def test_liveness_is_content_free(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    app = create_app(config, environ=base_env)

    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/livez")

    try:
        response = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["cache-control"] == "no-store"


def test_readiness_is_content_free_and_requires_selected_agent_attestation(
    base_env: dict[str, str],
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    readiness = _SelectedAgentReadiness()
    app = create_app(
        config,
        environ=base_env,
        composition=BackendComposition(
            turn_backend=_TurnBackend(),
            turn_status="response_only",
            selected_agent_readiness=readiness,
        ),
    )

    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/readyz")

    try:
        response = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["cache-control"] == "no-store"
    assert readiness.calls == 1


def test_readiness_fails_closed_without_disclosing_backend_failures(
    base_env: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)

    for readiness in (
        None,
        _SelectedAgentReadiness(failure=SelectedAgentReadinessError("selected_agent_access_denied")),
        _SelectedAgentReadiness(failure=SelectedAgentReadinessError("selected_agent_private_customer_123")),
        _SelectedAgentReadiness(failure=TimeoutError("private timeout detail")),
        _SelectedAgentReadiness(failure=ValueError("untrusted adapter detail")),
    ):
        app = create_app(
            config,
            environ=base_env,
            composition=BackendComposition(
                turn_backend=_TurnBackend(),
                turn_status="response_only",
                selected_agent_readiness=readiness,
            ),
        )

        async def exercise(target_app=app) -> httpx.Response:
            transport = httpx.ASGITransport(app=target_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/readyz")

        try:
            response = asyncio.run(exercise())
        finally:
            app.state.voiceclaw_state_store.close()

        assert response.status_code == 503
        assert response.content == b""
        assert response.headers["cache-control"] == "no-store"
        assert b"selected_agent" not in response.content

    assert "selected_agent_readiness_unsupported" in caplog.text
    assert "selected_agent_access_denied" in caplog.text
    assert "selected_agent_unavailable" in caplog.text
    assert "selected_agent_private_customer_123" not in caplog.text
    assert "selected_agent_readiness_timeout" in caplog.text
    assert "readiness_internal_error" in caplog.text
    assert "private timeout detail" not in caplog.text
    assert "untrusted adapter detail" not in caplog.text


def test_server_ui_flag_is_opt_in() -> None:
    assert _server_parser().parse_args([]).ui is False
    assert _server_parser().parse_args(["--ui"]).ui is True


def test_application_speech_and_playback_capabilities_are_explicit(
    base_env: dict[str, str],
) -> None:
    bundled = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    assert _speech_delivery_capabilities(bundled) == {
        "acknowledgement": "model_mediated",
        "result": "model_mediated",
        "failure": "model_mediated",
        "playback_receipt": "conversation.item.truncate.v1",
        "playback_receipt_authority": "client_reported_validated",
        "response_done": "generation_only",
        "speech_floor": "waits_for_playback_receipt",
    }


def test_listener_port_rejects_invalid_cli_override(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)

    assert _listener_port(config, None) == 7860
    assert _listener_port(config, 443) == 443
    for invalid in (0, 65536):
        with pytest.raises(ConfigurationError, match="between 1 and 65535"):
            _listener_port(config, invalid)


def test_package_configuration_check_requires_bundled_loopback_key(
    base_env: dict[str, str],
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    environment = {key: value for key, value in base_env.items() if key != "REALTIME_UPSTREAM_API_KEY"}

    with (
        patch("voiceclaw.server.SqliteStateStore") as store_type,
        pytest.raises(ConfigurationError, match="realtime upstream credential environment is not set"),
    ):
        validate_configuration(config, environ=environment)

    store_type.assert_not_called()


def test_supervised_bundled_check_uses_the_provisioned_loopback_key(
    base_env: dict[str, str],
) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)

    with patch("voiceclaw.server.bind_nva_credential") as bind_nva_credential:
        validate_configuration(config, environ=base_env, bundled_nva_supervised=True)

    bind_nva_credential.assert_not_called()


def test_configuration_check_validates_public_issuer_constraints(base_env: dict[str, str]) -> None:
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=7860,
            auth_mode="ephemeral",
            api_key_env="VOICECLAW_REALTIME_API_KEY",
        ),
        environ=base_env,
    )

    with pytest.raises(ConfigurationError, match="24 to 4096 characters"):
        validate_configuration(
            config,
            environ={**base_env, "VOICECLAW_REALTIME_API_KEY": "too-short"},
        )


def test_configuration_check_does_not_create_new_state_database(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "new" / "voiceclaw.db"
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    validate_configuration(config, environ=base_env)

    assert not state_path.exists()
    assert not state_path.parent.exists()


def test_configuration_check_rejects_impossible_new_state_path(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(blocking_file / "voiceclaw.db")),
    )

    with pytest.raises(ConfigurationError, match="parent is not a directory"):
        validate_configuration(config, environ=base_env)


def test_configuration_check_rejects_existing_state_symlink(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.db"
    store = SqliteStateStore(str(target))
    store.close()
    state_path = tmp_path / "state.db"
    state_path.symlink_to(target)
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    with pytest.raises(ConfigurationError, match="must not contain symbolic links"):
        validate_configuration(config, environ=base_env)


def test_configuration_check_rejects_symlinked_missing_state_parent(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target, target_is_directory=True)
    state_path = linked_parent / "new" / "state.db"
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    with pytest.raises(ConfigurationError, match="must not contain symbolic links"):
        validate_configuration(config, environ=base_env)


def test_configuration_check_rejects_invalid_sqlite_without_mutating_it(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "invalid.db"
    original = b"not a sqlite database"
    state_path.write_bytes(original)
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    with pytest.raises(ConfigurationError, match="usable VoiceClaw SQLite database"):
        validate_configuration(config, environ=base_env)

    assert state_path.read_bytes() == original


def test_configuration_check_rejects_unsupported_state_schema_without_migrating_it(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "future.db"
    connection = sqlite3.connect(state_path)
    connection.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO schema_version(version) VALUES (999)")
    connection.commit()
    connection.close()
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    with pytest.raises(ConfigurationError, match="unsupported VoiceClaw state schema version: 999"):
        validate_configuration(config, environ=base_env)

    connection = sqlite3.connect(f"{state_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("SELECT version FROM schema_version").fetchone() == (999,)
    finally:
        connection.close()


def test_configuration_check_accepts_existing_state_without_changing_schema(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    store = SqliteStateStore(str(state_path))
    store.close()
    before = state_path.stat().st_mtime_ns
    config = replace(
        _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env),
        state=StateConfig(path=str(state_path)),
    )

    validate_configuration(config, environ=base_env)

    assert state_path.stat().st_mtime_ns == before
    connection = sqlite3.connect(f"{state_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("SELECT version FROM schema_version").fetchone() == (6,)
    finally:
        connection.close()


def test_external_realtime_credential_file_is_resolved_server_side(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    secret = tmp_path / "realtime-provider-key"
    secret.write_text("external-provider-secret\n", encoding="utf-8")
    secret.chmod(0o600)
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    assert config.realtime is not None
    config = replace(
        config,
        realtime=replace(config.realtime, credential_env=None, credential_file=str(secret)),
    )

    assert _upstream_bearer(config, {}) == "external-provider-secret"


def test_external_realtime_credential_file_rejects_unsafe_permissions(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    secret = tmp_path / "realtime-provider-key"
    secret.write_text("external-provider-secret\n", encoding="utf-8")
    secret.chmod(0o604)
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    assert config.realtime is not None
    config = replace(
        config,
        realtime=replace(config.realtime, credential_env=None, credential_file=str(secret)),
    )

    with pytest.raises(ConfigurationError, match="accessible by others"):
        _upstream_bearer(config, {})


def test_loopback_dev_serves_health_and_packaged_ui_without_auth(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    app = create_app(config, environ=base_env, ui=True)

    async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return (
                await client.get("/health"),
                await client.get("/"),
                await client.get("/app.js"),
                await client.get("/marked.min.js"),
            )

    try:
        health, page, script, marked = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert health.json()["status"] == "ok"
    assert health.json()["realtime_upstream"]["readiness"] == "checked_per_session"
    assert health.json()["realtime_upstream"]["speech_delivery"] == {
        "acknowledgement": "model_mediated",
        "result": "model_mediated",
        "failure": "model_mediated",
        "playback_receipt": "conversation.item.truncate.v1",
        "playback_receipt_authority": "client_reported_validated",
        "response_done": "generation_only",
        "speech_floor": "waits_for_playback_receipt",
    }
    assert health.json()["turn_backend"] == "response_only"
    assert health.json()["model_contracts"]["schema"] == "voiceclaw.model-contracts.v1"
    assert health.json()["model_contracts"]["profile"] == "default"
    assert health.json()["model_contracts"]["digest"].startswith("sha256:")
    assert health.json()["interaction_profiles"]["schema"] == "voiceclaw.interaction-profiles.v2"
    assert health.json()["interaction_profiles"]["profile"] == "stateless"
    assert health.json()["interaction_profiles"]["digest"].startswith("sha256:")
    assert health.json()["interaction_profiles"]["resolved_digest"].startswith("sha256:")
    assert health.json()["state"] == "running"
    assert page.status_code == 200
    assert "VoiceClaw" in page.text
    assert 'id="endpoint"' in page.text and "readonly" in page.text
    assert 'id="route-auto"' not in page.text
    assert 'id="route-agent-work"' not in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert "connect-src 'self' ws:" not in page.headers["content-security-policy"]
    assert page.text.index('src="./marked.min.js"') < page.text.index('src="./app.js"')
    assert marked.status_code == 200
    assert "marked v15.0.12" in marked.text
    assert "window.marked?.parse?.(value, MARKDOWN_OPTIONS)" in script.text
    assert "function sanitizeMarkedHtml" in script.text
    assert "function renderMinimalMarkdown" not in script.text
    assert "function markdownTableDefinition" not in script.text
    assert "function scheduleProjectionRender" in script.text
    assert "projectionRenderFrame" in script.text

    assert "new WebSocket(endpoint, protocols)" in script.text
    assert "voiceclaw_work_delegate" not in script.text
    assert "tool_choice" not in script.text
    assert 'type: "conversation.item.truncate"' in script.text
    assert "audio_end_ms: truncation.audioEndMs" in script.text
    interruption = script.text[
        script.text.index("function interruptActiveResponse") : script.text.index("function secureMicrophoneContext")
    ]
    cancel_position = interruption.index('type: "response.cancel"')
    truncate_position = interruption.index("dispatchPlaybackTruncations(truncations)")
    assert cancel_position < truncate_position
    assert "pendingPlaybackTruncations" not in script.text
    assert "state.pendingClientEvents.get(error.event_id)" in script.text
    assert "const benignTerminalCancel" in script.text
    assert "response id is not owned by this session" in script.text
    assert "state.speechDeliveryQueueDepth - (state.outputActive ? 1 : 0)" not in script.text
    assert "const waiting = Math.max(0, state.speechDeliveryQueueDepth);" in script.text
    assert "const INITIAL_PLAYOUT_LEAD_SECONDS = 0.45" in script.text
    assert "const MIN_REBUFFER_LEAD_SECONDS = 0.12" in script.text
    assert "playbackScheduleTail: Promise.resolve()" in script.text
    assert "state.playbackScheduleTail.then" in script.text
    assert 'case "response.output_audio.done"' in script.text
    assert "markPlaybackStreamDone(event)" in script.text
    assert "voice.output_state && !hasLocalPlaybackActivity()" in script.text
    assert "projection.waiting_depth" in script.text
    assert "Voice replies queued" in page.text
    assert 'id="speech-queue" class="queue-indicator empty"' in page.text
    assert (
        'title="VoiceClaw voice responses waiting to start; excludes the response currently being delivered"'
        in page.text
    )
    assert 'format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE }' in script.text
    assert 'output: { format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE } }' in script.text
    assert "usesPcm24" in script.text
    session_patch = script.text[
        script.text.index("const patch = {") : script.text.index(
            "const instructions = elements.instructions.value.trim()"
        )
    ]
    assert "turn_detection:" not in session_patch
    assert "state.automaticTurnDetection" in script.text
    assert 'elements.talkButton.addEventListener("click", () => void toggleListening())' in script.text
    assert 'elements.muteButton.addEventListener("click", () => setMuted(!state.muted))' in script.text
    assert 'elements.talkButton.addEventListener("pointerdown"' not in script.text
    assert 'elements.talkButton.addEventListener("pointerup"' not in script.text
    assert 'elements.talkButton.addEventListener("pointercancel"' not in script.text
    assert 'state.automaticTurnDetection ? "Stop listening" : "Stop & send"' in script.text
    start_listening = script.text[
        script.text.index("async function startListening()") : script.text.index("function stopListening()")
    ]
    clear_position = start_listening.index('type: "input_audio_buffer.clear"')
    capture_position = start_listening.index("state.capturing = true")
    assert clear_position < capture_position
    manual_start = start_listening[
        start_listening.index("if (!state.automaticTurnDetection)") : start_listening.index("state.captureBytes = 0")
    ]
    assert 'type: "input_audio_buffer.clear"' in manual_start
    assert "const MIN_LISTENING_SECONDS = 0.2" in script.text
    assert "const MIN_CAPTURE_BYTES = INPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * MIN_LISTENING_SECONDS" in (
        script.text
    )
    stop_listening = script.text[
        script.text.index("function stopListening()") : script.text.index("async function toggleListening()")
    ]
    automatic_stop = stop_listening[
        stop_listening.index("if (state.automaticTurnDetection) {") : stop_listening.index(
            "if (!hadCapture || capturedBytes < MIN_CAPTURE_BYTES)"
        )
    ]
    assert 'type: "input_audio_buffer.clear"' not in automatic_stop
    assert 'type: "input_audio_buffer.commit"' not in automatic_stop
    assert 'type: "response.create"' not in automatic_stop
    assert "AUTOMATIC_STOP_TIMEOUT_MS" in stop_listening
    short_capture_position = stop_listening.index("if (!hadCapture || capturedBytes < MIN_CAPTURE_BYTES)")
    commit_position = stop_listening.index('type: "input_audio_buffer.commit"', short_capture_position)
    response_position = stop_listening.index('type: "response.create"', commit_position)
    short_turn = stop_listening[short_capture_position:commit_position]
    assert 'type: "input_audio_buffer.clear"' in short_turn
    assert 'type: "input_audio_buffer.commit"' not in short_turn
    assert "return;" in short_turn
    assert commit_position < response_position
    assert 'errorCode === "input_audio_transcription_empty"' in script.text
    assert "VAD_SILENCE_BYTES" not in script.text
    assert 'id="delegation-list"' in page.text
    assert 'id="delegation-announcement"' in page.text
    assert 'id="mute-button"' in page.text
    assert 'state.backendCapabilities.has("live_delegated_context")' in script.text
    assert 'record.status = "outcome_unknown"' in script.text
    assert "stopPlayback(false, true)" in script.text
    assert "state.muteTruncationReported" in script.text
    assert '["Backend response", record.correlation.response_id]' in script.text
    assert 'developerSummary.textContent = "Developer details"' in script.text
    assert 'succeeded: "Response received"' in script.text
    assert 'resultLabel.textContent = failed ? "Failure" : "Response"' in script.text
    assert "record.result = resultPresentation" in script.text
    assert "projectionCorrelation(root)" in script.text
    assert "const previous = state.targetProjection || {}" in script.text
    assert "previous.status" in script.text
    assert 'state.gatewayReachable ? "gateway_reachable"' in script.text
    assert '<p class="eyebrow">Backend turns</p>' in page.text
    speech_started = script.text[
        script.text.index('case "input_audio_buffer.speech_started"') : script.text.index(
            'case "input_audio_buffer.speech_stopped"'
        )
    ]
    assert "dispatchPlaybackTruncations(stopPlayback(true, true))" in speech_started
    assert 'type: "response.cancel"' not in speech_started
    speech_stopped = script.text[
        script.text.index('case "input_audio_buffer.speech_stopped"') : script.text.index(
            'case "input_audio_buffer.committed"'
        )
    ]
    assert "finishAutomaticStop()" in speech_stopped
    assert "function finishAutomaticStop()" in script.text
    assert "if (!state.stopAfterSpeech) return" in script.text
    assert "Hold to talk" not in page.text


def test_non_loopback_none_auth_does_not_require_public_key(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="0.0.0.0", port=7860, auth_mode="none"), environ=base_env)
    app = create_app(
        config,
        environ={**base_env, "VOICECLAW_REALTIME_API_KEY": "stale-public-secret"},
    )

    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://test") as client:
            return await client.post("/v1/realtime/client_secrets")

    try:
        response = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert response.status_code == 404
    assert response.json() == {"error": "authentication_disabled"}


def test_ephemeral_auth_requires_configured_public_key(base_env: dict[str, str]) -> None:
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=7860,
            auth_mode="ephemeral",
            api_key_env="VOICECLAW_REALTIME_API_KEY",
        ),
        environ=base_env,
    )

    with pytest.raises(ConfigurationError, match="VOICECLAW_REALTIME_API_KEY"):
        create_app(config, environ=base_env)


def test_ephemeral_auth_resolves_an_owner_controlled_public_key_file(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "voiceclaw-public-master"
    key_file.write_text("voiceclaw-test-master-key-that-is-long-enough\n", encoding="ascii")
    key_file.chmod(0o600)
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=7860,
            auth_mode="ephemeral",
            api_key_file=str(key_file),
        ),
        environ=base_env,
    )

    assert _public_key(config, base_env) == "voiceclaw-test-master-key-that-is-long-enough"


def test_ephemeral_auth_rejects_a_public_key_file_accessible_by_other_users(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "voiceclaw-public-master"
    key_file.write_text("voiceclaw-test-master-key-that-is-long-enough\n", encoding="ascii")
    key_file.chmod(0o604)
    config = _config(
        server=ServerConfig(auth_mode="ephemeral", api_key_file=str(key_file)),
        environ=base_env,
    )

    with pytest.raises(ConfigurationError, match="accessible by others"):
        _public_key(config, base_env)


def test_ephemeral_auth_rejects_ambiguous_direct_server_configuration(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    config = _config(
        server=ServerConfig(
            auth_mode="ephemeral",
            api_key_env="VOICECLAW_REALTIME_API_KEY",
            api_key_file=str(tmp_path / "public-master"),
        ),
        environ=base_env,
    )

    with pytest.raises(ConfigurationError, match="exactly one"):
        _public_key(config, {**base_env, "VOICECLAW_REALTIME_API_KEY": "a" * 32})


def test_cli_host_override_cannot_silently_cross_the_loopback_boundary(base_env: dict[str, str]) -> None:
    loopback = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    exposed = _with_listener_host(loopback, "0.0.0.0")

    app = create_app(exposed, environ=base_env)
    app.state.voiceclaw_state_store.close()

    with pytest.raises(ConfigurationError, match="loopback requires"):
        _tls_listener_files(exposed, "", "")


def test_loopback_listener_can_use_plain_http(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)

    assert _tls_listener_files(config, "", "") == (None, None)


def test_private_network_listener_explicitly_allows_plain_bridge_http(base_env: dict[str, str]) -> None:
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=18790,
            listener_security=ListenerSecurity.PRIVATE_NETWORK,
        ),
        environ=base_env,
    )

    assert _tls_listener_files(config, "", "") == (None, None)


def test_tls_listener_requires_a_complete_existing_identity(base_env: dict[str, str], tmp_path: Path) -> None:
    config = _config(
        server=ServerConfig(host="0.0.0.0", port=18790, listener_security=ListenerSecurity.TLS),
        environ=base_env,
    )
    with pytest.raises(ConfigurationError, match="required when server.listener_security is tls"):
        _tls_listener_files(config, "", "")

    certificate = tmp_path / "cert.pem"
    private_key = tmp_path / "key.pem"
    certificate.touch()
    private_key.touch()
    assert _tls_listener_files(config, str(certificate), str(private_key)) == (
        str(certificate),
        str(private_key),
    )


def test_configured_upstream_credential_is_required(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    environment = {key: value for key, value in base_env.items() if key != "REALTIME_UPSTREAM_API_KEY"}

    with pytest.raises(ConfigurationError, match="REALTIME_UPSTREAM_API_KEY"):
        create_app(config, environ=environment)


def test_state_store_and_runtime_are_created_once_per_application(base_env: dict[str, str]) -> None:
    config = _config(server=ServerConfig(host="127.0.0.1", port=7860), environ=base_env)
    stores: list[SqliteStateStore] = []

    def open_store(path: str) -> SqliteStateStore:
        store = SqliteStateStore(path)
        stores.append(store)
        return store

    with patch("voiceclaw.server.SqliteStateStore", side_effect=open_store):
        app = create_app(config, environ=base_env)

    assert len(stores) == 1
    assert app.state.voiceclaw_state_store is stores[0]
    assert app.state.voiceclaw_runtime is not None
    stores[0].close()


def test_master_authorized_client_secret_endpoint_returns_ephemeral_key(base_env: dict[str, str]) -> None:
    key = "voiceclaw-test-master-key-that-is-long-enough"
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=7860,
            auth_mode="ephemeral",
            api_key_env="VOICECLAW_REALTIME_API_KEY",
            client_secret_lifetime_seconds=60,
        ),
        environ=base_env,
    )
    app = create_app(config, environ={**base_env, "VOICECLAW_REALTIME_API_KEY": key})

    async def exercise() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.post("/v1/realtime/client_secrets")
            authorized = await client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": f"Bearer {key}"},
            )
            return unauthorized, authorized

    try:
        unauthorized, authorized = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["value"].startswith("ek_")


def test_file_backed_master_authorizes_client_secret_endpoint(
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    key = "voiceclaw-test-file-master-key-that-is-long-enough"
    key_file = tmp_path / "voiceclaw-public-master"
    key_file.write_text(f"{key}\n", encoding="ascii")
    key_file.chmod(0o600)
    config = _config(
        server=ServerConfig(
            host="0.0.0.0",
            port=7860,
            auth_mode="ephemeral",
            api_key_file=str(key_file),
            client_secret_lifetime_seconds=60,
        ),
        environ=base_env,
    )
    app = create_app(config, environ=base_env)

    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": f"Bearer {key}"},
            )

    try:
        response = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()

    assert response.status_code == 200
    assert response.json()["value"].startswith("ek_")
