# Voice-safe LLM output

Mandatory derive-domain Step4+fake-data-and-tools. User hears LLM→TTS only.

Failures: speaks product_id/session_id UUID | price Q speaks id not price | symptoms filler silence disclaimer | causes prompt lists params dev msg session_id metadata not tools Nemotron thinking silent wrong handler signature sequential tools.

SYSTEM_INSTRUCTION reply_language rules:
1 customer-facing names qty prices local currency words
2 never speak product_id session_id order UUID tool names JSON keys markdown emoji English tech labels paths raw errors
3 after tool 2-4 sentences voice_hint meaning not disclaimer-only
4 purchase search→mutate→confirm name+price speech
5 price Q tool then currency amount not ids
6 order numbers short spoken checkout only never raw UUID

Wire ids internal: resolve `session_id` per request/transport invocation and pass into handlers; FORBID module-global `_active_session_id` or transport-global defaults. Schemas domain fields only product_id description internal never say omit session_id. Handlers return voice_hint paraphrase not field names.

Nemotron thinking→llm-reasoning.md gate wiring default OFF leaks→silence redacted_thinking TTS.

Pipecat turn: silent after STT fix LLM logs | false interrupt VADUserTurnStart+MinWords | WebRTC **Omni** greeting: REQ `MuteUntilFirstBotComplete` on connect (see bot.omni-runner.md); cascaded WebRTC may omit mute when barge-in required | reconnect worker.cancel per-session SileroVAD hard refresh

Checklist: prohibitions in prompt | session_id handlers only | voice_hint | FunctionCallParams only | one combined main tool | no filler every on_function_calls_started | thinking false extra_body | smoke name+price no ids | answer ~5s post-tool
