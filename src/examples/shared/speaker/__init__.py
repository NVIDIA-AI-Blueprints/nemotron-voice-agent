# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker attribution from ASR word metadata."""

from examples.shared.speaker.transcript import (
    SpeakerDiarizationProcessor,
    SpeakerRun,
    TranscriptFragment,
    continues_previous_word,
    extract_speaker_id,
    extract_word_span,
    finalize_speaker_runs,
    finalized_turn_payloads,
    forward_speaker_events,
    group_fragments_by_speaker,
    interim_payload,
    join_cut_word,
    progress_payloads,
    reassign_first_word,
    resolve_asr_speaker_support,
    send_rtvi_payloads,
    speaker_diarization_enabled,
    speaker_diarization_requested,
    split_transcript_by_speaker,
)

__all__ = [
    "SpeakerDiarizationProcessor",
    "SpeakerRun",
    "TranscriptFragment",
    "continues_previous_word",
    "extract_speaker_id",
    "extract_word_span",
    "finalize_speaker_runs",
    "finalized_turn_payloads",
    "forward_speaker_events",
    "group_fragments_by_speaker",
    "interim_payload",
    "join_cut_word",
    "progress_payloads",
    "reassign_first_word",
    "resolve_asr_speaker_support",
    "send_rtvi_payloads",
    "speaker_diarization_enabled",
    "speaker_diarization_requested",
    "split_transcript_by_speaker",
]
