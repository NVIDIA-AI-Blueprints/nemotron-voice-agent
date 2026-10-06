# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Source-level guards for the dependency-free packaged browser client.

The UI deliberately ships as plain JavaScript without a Node dependency tree.
These checks protect the event-contract decisions that can be verified in the
Python package suite; real microphone, WebSocket, DOM, and Web Audio behavior
still belongs in a browser E2E suite.
"""

from pathlib import Path

_UI = Path(__file__).parents[1] / "src" / "voiceclaw" / "ui"
_APP = _UI / "app.js"
_INDEX = _UI / "index.html"


def _source() -> str:
    return _APP.read_text(encoding="utf-8")


def _html() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _assert_contract(source: str, *, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> None:
    for value in required:
        assert value in source
    for value in forbidden:
        assert value not in source


def _between(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


def test_result_updates_do_not_mutate_request_lifecycle() -> None:
    """Keep result availability separate from request admission state."""
    source = _source()
    update_request = _between(source, "function updateRequest(", "function updateResult(")
    update_result = _between(source, "function updateResult(", "function submitText(")

    assert "record.status = status" in update_request
    assert "record.resultStatus = status" in update_result
    assert "record.status = status" not in update_result
    assert "recordDelegationPhase(" not in update_result


def test_negotiated_vad_tracks_commit_response_and_interruption_independently() -> None:
    """Honor both negotiated VAD response controls independently."""
    source = _source()
    negotiation = _between(source, "function applyNegotiatedTurnDetection(", "function updateQueue(")
    server_events = _between(source, "function handleServerEvent(", "function onSessionCreated(")

    assert "turnDetection.create_response !== false" in negotiation
    assert "turnDetection.interrupt_response !== false" in negotiation
    committed = _between(
        server_events,
        'case "input_audio_buffer.committed":',
        'case "conversation.item.input_audio_transcription.delta":',
    )
    assert "state.automaticTurnDetection && !state.automaticResponseCreation" in committed
    assert '{ type: "response.create", response: {} }' in committed
    speech_started = _between(
        server_events,
        'case "input_audio_buffer.speech_started":',
        'case "input_audio_buffer.speech_stopped":',
    )
    assert "if (state.automaticResponseInterruption)" in speech_started
    assert "stopPlayback(true, true)" in speech_started
    assert "dispatchPlaybackTruncations(stopPlayback(true, true))" in speech_started
    assert 'type: "response.cancel"' not in speech_started

    speech_stopped = _between(
        server_events,
        'case "input_audio_buffer.speech_stopped":',
        'case "input_audio_buffer.committed":',
    )
    assert "finishAutomaticStop()" in speech_stopped

    start_listening = _between(source, "async function startListening()", "function stopListening()")
    assert start_listening.index('type: "input_audio_buffer.clear"') < start_listening.index("state.capturing = true")
    manual_start = _between(start_listening, "if (!state.automaticTurnDetection)", "state.captureBytes = 0")
    assert 'type: "input_audio_buffer.clear"' in manual_start

    stop_listening = _between(source, "function stopListening()", "async function toggleListening()")
    automatic_stop = _between(
        stop_listening,
        "if (state.automaticTurnDetection) {",
        "if (!hadCapture || capturedBytes < MIN_CAPTURE_BYTES)",
    )
    _assert_contract(
        automatic_stop,
        required=(),
        forbidden=(
            'type: "input_audio_buffer.clear"',
            'type: "input_audio_buffer.commit"',
            'type: "response.create"',
        ),
    )
    assert "AUTOMATIC_STOP_TIMEOUT_MS" in stop_listening
    short_capture = stop_listening.index("if (!hadCapture || capturedBytes < MIN_CAPTURE_BYTES)")
    commit = stop_listening.index('type: "input_audio_buffer.commit"', short_capture)
    response = stop_listening.index('type: "response.create"', commit)
    short_turn = stop_listening[short_capture:commit]
    _assert_contract(
        short_turn,
        required=('type: "input_audio_buffer.clear"', "return;"),
        forbidden=('type: "input_audio_buffer.commit"',),
    )
    assert commit < response

    session_patch = _between(source, "const patch = {", "const instructions = elements.instructions.value.trim()")
    _assert_contract(
        source,
        required=(
            'format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE }',
            'output: { format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE } }',
            "usesPcm24",
            "state.automaticTurnDetection",
            "const MIN_LISTENING_SECONDS = 0.2",
            "const MIN_CAPTURE_BYTES = INPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * MIN_LISTENING_SECONDS",
            'errorCode === "input_audio_transcription_empty"',
            "function finishAutomaticStop()",
            "if (!state.stopAfterSpeech) return",
            'elements.talkButton.addEventListener("click", () => void toggleListening())',
            'elements.muteButton.addEventListener("click", () => setMuted(!state.muted))',
            'state.automaticTurnDetection ? "Stop listening" : "Stop & send"',
        ),
        forbidden=(
            "VAD_SILENCE_BYTES",
            'elements.talkButton.addEventListener("pointerdown"',
            'elements.talkButton.addEventListener("pointerup"',
            'elements.talkButton.addEventListener("pointercancel"',
        ),
    )
    assert "turn_detection:" not in session_patch


def test_streaming_markdown_is_throttled_and_latest_result_stays_open() -> None:
    """Bound streamed Markdown work without hiding the completed result."""
    source = _source()
    scheduler = _between(source, "function scheduleProjectionRender(", "function appendResponseText(")
    update_request = _between(source, "function updateRequest(", "function updateResult(")
    update_result = _between(source, "function updateResult(", "function submitText(")

    assert "STREAMING_MARKDOWN_RENDER_INTERVAL_MS" in scheduler
    assert "projectionRenderTimer" in scheduler
    assert "window.setTimeout(requestRenderFrame, delay)" in scheduler
    assert "record.expanded = record.id === state.latestDelegationId" in update_request
    assert "record.id === state.latestDelegationId" in update_result
    _assert_contract(
        source,
        required=(
            "window.marked?.parse?.(value, MARKDOWN_OPTIONS)",
            "function sanitizeMarkedHtml",
        ),
        forbidden=("function renderMinimalMarkdown", "function markdownTableDefinition"),
    )


def test_delegation_card_uses_only_the_server_projected_goal_summary() -> None:
    """Never relabel the latest raw user transcript as the delegated goal."""
    source = _source()
    metadata = _between(source, "function projectionFromMetadata(", "function projectionBelongsToActiveSession(")
    summary = _between(source, "function stableRequestSummary(", "function projectionCorrelation(")
    update_request = _between(source, "function updateRequest(", "function updateResult(")

    assert 'key === "request_summary" ? rawValue : parseMetadataValue(rawValue)' in metadata
    assert "state.recentUserTurns" not in summary
    assert '"Delegated goal summary unavailable"' in summary
    assert "function fullRequestText(" not in source
    assert 'queryLabel.textContent = "Delegated goal summary"' in source
    assert 'record.query === "Delegated goal summary unavailable" ? summary : record.query' in update_request


def test_target_contract_uses_projection_body_instead_of_metadata_capacity() -> None:
    """Keep capabilities in standard text while metadata remains a small routing header."""
    source = _source()
    parser = _between(source, "function backendTargetBody(", "function applyProjection(")
    application = _between(source, "function applyProjection(", "function updateBackend(")

    assert "JSON.parse(value)" in parser
    assert "projection.capabilities = body.capabilities" in parser
    assert "projection.frontend_tools = body.frontend_tools" in parser
    assert 'target_state: "target_state"' in parser
    assert 'projection.kind === "backend_target"' in application
    assert "backendTargetBody(responseText)" in application


def test_assistant_message_keeps_response_start_time_after_streaming_completes() -> None:
    """Render one immutable response start time instead of completion time."""
    source = _source()
    messages = _between(source, "function createMessage(", "function removeMessage(")
    begin_response = _between(source, "function beginResponse(", "function bindOutputItem(")
    finish_response = _between(source, "function finishResponse(", "function parseMetadataValue(")

    assert "wrapper.dataset.startedAt" in messages
    assert "timeLabel(wrapper.dataset.startedAt)" in messages
    assert "startedAt: new Date()" in begin_response
    assert "track.startedAt" in finish_response


def test_conversation_and_client_identity_memory_are_bounded() -> None:
    """Prevent an indefinitely open browser session from retaining every turn."""
    source = _source()
    pruning = _between(source, "function pruneConversationMessages(", "function scrollConversation(")

    assert "MAX_CONVERSATION_MESSAGES = 200" in source
    assert "MAX_USER_ITEM_IDS = MAX_CONVERSATION_MESSAGES" in source
    assert "while (state.messageElements.size > MAX_CONVERSATION_MESSAGES)" in pruning
    assert "while (state.userItemIds.size > MAX_USER_ITEM_IDS)" in pruning
    assert "id === state.activeResponseId || state.userItemIds.has(id)" in pruning
    assert "state.userItemIds.delete(item.id)" in source


def test_standard_truncate_reports_partial_and_fully_played_audio() -> None:
    """Use one standard event for interrupted and fully drained playback receipts."""
    source = _source()
    partial = _between(source, "function playbackTruncationsAt(", "function dispatchPlaybackTruncations(")
    dispatch = _between(source, "function dispatchPlaybackTruncations(", "function playbackIdentity(")
    identity = _between(source, "function playbackIdentity(", "function streamHasScheduledSource(")
    completed = _between(source, "function reportCompletedPlayback(", "function hasLocalPlaybackActivity(")
    failed = _between(source, "function reportFailedPlayback(", "function hasLocalPlaybackActivity(")
    response_done = _between(source, "function finishResponse(", "function parseMetadataValue(")

    assert "audioEndMs: heardThroughMs" in partial
    assert "event_id: truncation.receiptId" in dispatch
    assert 'type: "conversation.item.truncate"' in dispatch
    assert 'voiceclaw_playback_receipt_required === "true"' in identity
    assert "track.metadata.voiceclaw_playback_receipt_id" in identity
    assert 'type: "conversation.item.truncate"' in completed
    assert "event_id: progress.receiptId" in completed
    assert "audio_end_ms: progress.audioEndMs" in completed
    assert "progress.playedThroughMs < progress.audioEndMs" in completed
    assert "streamHasScheduledSource(key)" in completed
    assert "completePlaybackResponse(track.id)" in response_done
    assert 'type: "conversation.item.truncate"' in failed
    assert "audio_end_ms: heardThroughMs" in failed
    assert "reportFailedPlayback(stream)" in source

    interruption = _between(source, "function interruptActiveResponse", "function secureMicrophoneContext")
    assert interruption.index('type: "response.cancel"') < interruption.index(
        "dispatchPlaybackTruncations(truncations)"
    )
    _assert_contract(
        source,
        required=(
            "state.pendingClientEvents.get(error.event_id)",
            "const benignTerminalCancel",
            "response id is not owned by this session",
            "const waiting = Math.max(0, state.speechDeliveryQueueDepth);",
            "const INITIAL_PLAYOUT_LEAD_SECONDS = 0.45",
            "const MIN_REBUFFER_LEAD_SECONDS = 0.12",
            "playbackScheduleTail: Promise.resolve()",
            "state.playbackScheduleTail.then",
            'case "response.output_audio.done"',
            "markPlaybackStreamDone(event)",
            "voice.output_state && !hasLocalPlaybackActivity()",
            "projection.waiting_depth",
            "stopPlayback(false, true)",
            "state.muteTruncationReported",
        ),
        forbidden=(
            "pendingPlaybackTruncations",
            "state.speechDeliveryQueueDepth - (state.outputActive ? 1 : 0)",
        ),
    )


def test_transport_and_delegation_ui_keep_public_and_backend_contracts_separate() -> None:
    """Keep backend projections on the standard public Realtime surface."""
    source = _source()
    html = _html()

    _assert_contract(
        source,
        required=(
            "new WebSocket(endpoint, protocols)",
            'continuity === "shared_target"',
            "Context continuity and isolation are not qualified",
            'record.status = "outcome_unknown"',
            '["Backend invocation reference", record.correlation.response_id]',
            'developerSummary.textContent = "Developer details"',
            'succeeded: "Response received"',
            'resultLabel.textContent = failed ? "Failure" : "Response"',
            "record.result = resultPresentation",
            "projectionCorrelation(root)",
            "const previous = state.targetProjection || {}",
            "previous.status",
            'state.targetBindingVerified ? "binding_verified"',
        ),
        forbidden=("voiceclaw_work_delegate", "tool_choice"),
    )
    _assert_contract(
        html,
        required=(
            "Voice replies queued",
            'id="speech-queue" class="queue-indicator empty"',
            'title="VoiceClaw voice responses waiting to start; excludes the response currently being delivered"',
            'id="delegation-list"',
            'id="delegation-announcement"',
            'id="mute-button"',
            '<p class="eyebrow">Backend turns</p>',
        ),
        forbidden=("Hold to talk",),
    )
