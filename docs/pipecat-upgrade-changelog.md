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

## Verification

This migration was validated against:

- `uvx ruff@0.15.6 check .` and `uvx ruff@0.15.6 format --check .`
- The full unit suite (`uv run pytest tests/ -v`)
- A container build and health check
- A single-task Tau airline run from the `voice-agent-evaluation` repository,
  which exercises the client-tool round trip and delegate-tool ownership
  against a live Realtime session

Record actual pass/fail evidence for each in the pull request's Documentation
Writer Review receipt and commit history rather than treating this page as
that evidence itself.
