# Pipecat 1.7 to 1.11 Upgrade Changelog

This document records the adapter-specific changes this repository needed to
move from `pipecat-ai==1.7.0` to `pipecat-ai==1.11.0`. It complements
`CHANGELOG.md`, which records product-level, user-facing changes. This page
covers the internal Pipecat call-site migrations that a future upgrade or a
troubleshooting session needs, and that a product changelog entry would not
carry.

Use the
[`nemotron-voice-agent-upgrade-pipecat`](../.agents/skills/nemotron-voice-agent-upgrade-pipecat/SKILL.md)
skill for a future Pipecat version bump. This page documents what that
migration actually touched in this codebase, not a general upgrade
procedure.

## Dependency Change

```text
pipecat-ai[nvidia,silero,runner,webrtc,websocket,openai,mcp]==1.7.0
  -> pipecat-ai[nvidia,silero,runner,webrtc,websocket,openai,mcp]==1.11.0
```

Upstream `pipecat-ai` dropped the `mcp` extra between 1.7.0 and 1.11.0. This
repository's Realtime MCP runtime (`src/realtime/mcp.py`) still needs it, so
the extra stays explicit in `pyproject.toml` rather than following upstream's
removal. The `mcp` package itself moved to 2.2.0, which changed several of the
call sites documented below.

## Call-Site Changes

### `FrameProcessorSetup` replaces task-manager-based setup

Pipecat 1.11 initializes processors through a `setup(FrameProcessorSetup)`
lifecycle method instead of passing a task manager into `process_frame` on
first use. Three services needed their initialization moved:

- `RealtimeNvidiaSTTService` (`src/realtime/asr.py`) now creates its NVIDIA
  client and recognition config in `setup()`, not in `start()`. `start()`
  keeps only per-turn state resets.
- `RealtimeVADInputProcessor` (`src/realtime/vad.py`) now initializes its VAD
  controller in `setup()`. The previous code lazily called
  `self._vad_controller.setup(self.task_manager)` on the first processed
  frame; `task_manager` no longer exists as a processor-visible attribute in
  1.11.
- `src/realtime/serializer.py` already used `FrameProcessorSetup`; this
  migration removed an unused `TTSUpdateSettingsFrame` import left over from
  the 1.7 code path.

### Settings and sentinel imports moved out of `pipecat.services.settings`

`NOT_GIVEN`, `assert_given`, and `is_given` moved from
`pipecat.services.settings` to `pipecat.utils.types` in 1.11.
`src/examples/shared/nvidia_llm.py` and `src/realtime/asr.py` updated their
imports accordingly. Two more call sites previously imported `NOT_GIVEN` from
the `openai` package directly (`src/realtime/frames.py`,
`src/realtime/tool_projection.py`, `src/realtime/transport.py`); those now
import Pipecat's own `NOT_GIVEN` from `pipecat.utils.types` so a sentinel
`is`-comparison used across a context boundary cannot compare Pipecat's value
against OpenAI's.

### `RealtimeResponseContextFrame` is keyword-only

`LLMContextFrame`, which `RealtimeResponseContextFrame` extends, gained a
defaulted field in 1.11. A dataclass with defaulted parent fields and
non-defaulted child fields cannot use positional construction, so
`src/realtime/frames.py` marks `RealtimeResponseContextFrame`
`@dataclass(kw_only=True)`. Every construction site already used keyword
arguments; this only makes the ordering requirement explicit and enforced.

### Removed `CANCEL_ASYNC_TOOL_NAME`

Pipecat 1.11 removed `pipecat.utils.async_tool_cancellation.CANCEL_ASYNC_TOOL_NAME`.
`RealtimeClientToolProjection._provider_reserved_names()`
(`src/realtime/client_tools.py`) no longer reserves it; the provider-reserved
name set is empty unless the active LLM service itself reserves names.

### `httpx.TimeoutException` narrowed to `TIMEOUT_EXCEPTIONS`

`NvidiaLLMService._process_context` (`src/examples/shared/nvidia_llm.py`)
caught `httpx.TimeoutException` directly. Pipecat 1.11 introduces
`pipecat.utils.http.TIMEOUT_EXCEPTIONS`, a tuple Pipecat itself uses to
classify timeouts across its supported HTTP clients. Catching that tuple
instead of a single `httpx` exception type keeps the deferred-inference retry
path aligned with how Pipecat's own services classify a timeout.

### Omni turn-state attribute rename

`NvidiaOmniLLMService._update_settings`
(`src/examples/omni_assistant/nvidia_omni_multimodal_service.py`) checked
`self._user_speaking`. The parent class's Pipecat-1.10-era turn-state
attribute is `self._audio_user_speaking`; the 1.7 code path against an older
Pipecat happened not to need this attribute. Reading the stale name always
evaluated as unset, which silently skipped the audio-buffer overflow guard
during simultaneous user speech.

### `mcp` 2.2.0 API rename

`src/realtime/mcp.py` follows the `mcp` package's 2.2.0 renames:

- `McpError` is now `MCPError`.
- `Tool.inputSchema` is now `Tool.input_schema`.
- `CallToolResult.isError` is now `CallToolResult.is_error`.

### Frontend/Backend latency reporting

`turn_contribution_lines()` replaces `chronological_events()` on the stage
latency breakdown used by `src/examples/frontend_backend_agent/pipeline.py`.

### Realtime Talker composition moved out of the RTVI import path

`ReliableRealtimeNvidiaLLMService` (the Realtime-protocol-aware Talker
service) moved from `reliable_talker.py` into a new module,
`src/examples/frontend_backend_agent/src/reliable_realtime_talker.py`, and
`pipeline.py` imports it lazily, only inside the `is_realtime` branch. The
class composes `examples.shared.nvidia_llm.NvidiaLLMService`, which pulls in
the Realtime protocol stack. Importing it unconditionally meant an ordinary
RTVI (non-Realtime) session initialized Realtime machinery it never used.

### Ordinary Frontend/Backend service lookup

The non-Realtime path in `pipeline.py` selected default LLM, TTS, ASR, and
Thinker-LLM catalog entries through a `pipeline_mode`-scoped registry key
(`_registry_default_service_key(pipeline_mode, ...)`). That indirection is
gone; the non-Realtime path now calls `load_service_entry(category, "")`
directly, matching the Realtime path's plain default lookup, and adds a
`domain_service_loader` closure so an empty booking-server key falls back to
the catalog's registered default while the Realtime path is unaffected.

### Text-encoded tool calls: multiple calls in one stream

`harvest_text_tool_calls` (`src/examples/shared/text_tool_calls.py`) handles
a Nemotron completion that encodes more than one `<tool_call>` block in a
single text stream. Previously it emitted one synthesized chunk carrying every
harvested `ChoiceDeltaToolCall`, which downstream Pipecat 1.11 aggregation
only consumed the first tool call from. It now emits one chunk per tool call
(`role="assistant"` on the first, `role=None` on the rest, matching how a
native streamed multi-call response is shaped), followed by a single terminal
chunk carrying `finish_reason="tool_calls"` with an empty delta.

Synthesized tool-call chunks also now reach the stage-metrics observer
(`_collect_stream` in `reliable_talker.py`), so a text-encoded call is timed
and correlated the same way a natively streamed one is; previously only
native tool-call chunks were observed.

### `GENERIC_TALKER_STREAM_TIMEOUT_SECONDS` default: 40.0 to 15.0

Not a Pipecat API change, but adjusted during this migration and worth
recording alongside it. The Talker stream bound
(`_TALKER_STREAM_TIMEOUT_SECONDS` in `reliable_talker.py`) retries once, so
two attempts at the old 40 s default could alone consume 80 s, leaving no
headroom against a 90 s caller response deadline once the up-to-40 s backend
deadline (`GENERIC_BACKEND_TIMEOUT_SECONDS`) is added. At 15 s per attempt,
two attempts plus the backend deadline total at most 70 s. See
`src/examples/frontend_backend_agent/README.md#configure-the-shared-pipeline`
for the full budget breakdown.

## Verification

This migration was validated against:

- `uvx ruff@0.15.6 check .` and `uvx ruff@0.15.6 format --check .`
- The full unit suite (`uv run pytest tests/ -v`)
- A container build and health check with
  `EXAMPLE_SELECTION=generic-frontend-backend-agent`
- `scripts/tau_realtime_smoke.py` in the `voice-agent-evaluation` repository,
  which exercises the client-tool round trip, delegate-tool ownership, and
  the timeout budget above against a live Realtime session

Record actual pass/fail evidence for each in the pull request's Documentation
Writer Review receipt and commit history rather than treating this page as
that evidence itself.
