# Pipecat | scaffold+iterate

FORBID pipeline from memory. Live sources first. Iterate→iterate.md.

## Doc source order (REQ)

1. **MCP pipecat-docs FIRST** when the host agent exposes that MCP server (id `pipecat-docs`; Kapa endpoint daily-docs.mcp.kapa.ai). Use MCP tools if present — do not assume a specific config file path. Query cascaded|omni audio-in,runner,/client,NvidiaSTT/LLM/TTS. Omni also LLMService,speech-input,smart-turn,user-turn-strategies pipeline/omni.md. FORBID pipecat-ai-mcp-server scaffold.
2 **No MCP:** fetch docs.pipecat.ai/llms.txt then pages: pipeline,runner guide,running-bots-locally,nvidia LLM/STT/TTS,quickstart,speech-input,service-settings,metrics,flows intro.
3 CLI optional: `uv tool install pipecat-ai-cli; pc init --list-options`
4 Examples last resort: github pipecat-ai/pipecat-examples fetch one file.

WebSocket UDP blocked user request(not default):
bot.py websocket|WebsocketServerTransport grep -i websocket | optional bot_ssh_server.py :7860 WS:8765 | pyproject websocket extra | handoff note UDP blocked. FORBID WebRTC-only if user asked WS.

Both WebRTC+WebSocket (gate choice **Both**, recommended):
`transport_params` with **webrtc and websocket** keys; pyproject `[webrtc,websocket]`; run `bot.py` **without** `-t` so POST `/start` and Pipecat `/client` can switch `transport`. Same bot.py for either path → networking/transport-selection.md. Ship turn_support if remote WebRTC leg. FORBID `-t webrtc` when user chose Both.

Blocked: report failure STOP no invent imports.

Post-scaffold: change→iterate fetch first | symptom→troubleshoot | runtime→service-settings | omni→pipeline/omni,bot.omni-* | local LLM profile→nim-llm-profiles
FORBID Pipecat API change without docs page or MCP query.
