# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Source-level guards for the dependency-free packaged browser client.

The UI deliberately ships as plain JavaScript without a Node dependency tree.
These checks protect the event-contract decisions that can be verified in the
Python package suite; real microphone, WebSocket, DOM, and Web Audio behavior
still belongs in a browser E2E suite.
"""

from pathlib import Path

_APP = Path(__file__).parents[1] / "src" / "voiceclaw" / "ui" / "app.js"


def _source() -> str:
    return _APP.read_text(encoding="utf-8")


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
