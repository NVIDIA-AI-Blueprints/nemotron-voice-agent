# Common voice debug | cross-platform

REF first-try-workstation-webrtc.md canonical WebRTC REQ. Platform-specific→dgx-spark,jetson-thor.

| symptom | cause | fix |
| Connect spins,ICE fail | no TURN off-LAN | turn_support+coturn POST /start turn: → first-try |
| POST /start 400 | missing [webrtc] extra | first-try row1 |
| language_code crash Connect | Pipecat1.3 TTS API | language=Language.EN_US first-try row9 |
| developer role 400 | supports_developer_role | WorkstationNvidiaLLMService False first-try row7 |
| text OK no voice | ASR likely OK; TTS/transport/bot output path broken | check TTS health/port, transport output, bot logs — not ASR port |
| VAD speaking no reply (cascaded) | STT silent | verify ASR gRPC before bot |
| VAD speaking no reply (Omni) | turn not finalized / omni request stuck | check turn-stop + Omni request logs — no STT service |
| silence after welcome | bot TypeError/NameError | bot logs |
| LLM 400 unexpected kwarg | chat_template_kwargs top-level | nest extra_body llm-reasoning |
