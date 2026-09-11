# VoiceBench Evaluation

This directory contains the Blueprint-side contract for VoiceBench-compatible
evaluation. The benchmark must measure the deployed ASR and LLM, but not TTS:
the normal WebSocket transport is audio-in/audio-out and re-transcribing its
audio response would incorrectly include TTS quality in VoiceBench scores.

## Start the Generic Assistant

Configure and start the server normally. The VoiceBench client creates one
session per sample with this body:

```json
{
  "pipeline_mode": "generic-assistant",
  "benchmark_text_only": true
}
```

`benchmark_text_only` is intentionally session-scoped. It keeps the deployed
ASR and LLM configuration, skips TTS readiness and construction, disables
audio output, and emits the final assistant response as an RTVI server message:

```json
{"type": "benchmark-assistant-response", "transcript": "..."}
```

The client should record the target repository SHA, `GET /api/services`, and
the session body together with every result. Those fields identify the actual
ASR and LLM deployment used for the score.

## Protocol

1. `POST /api/session-config` using the body above.
2. Connect to `/api/ws?session_id=<id>`.
3. Stream the VoiceBench input audio as 16 kHz mono PCM protobuf audio frames,
   followed by silence to finalize the ASR turn.
4. Read protobuf `message` frames until receiving the
   `benchmark-assistant-response` RTVI message.

Use the official [VoiceBench](https://github.com/HITsz-TMG/VoiceBench)
dataset and scoring code for the dataset-specific judges and metrics.
