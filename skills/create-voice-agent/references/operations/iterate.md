# Iterate | post-scaffold changes

Trigger: change|update|modify|recreate|rebuild|iterate|adjust|slow|swap model|tune|quality complaint. Requires bot.py|agent.py. FORBID memory-only edits; fetch docs BEFORE code.

```text
PROC: read bot.py|agent.py,pyproject,.env.example,domain_store,compose → framework index → map category → fetch ALL docs → minimal diff preserve DEPLOYMENT_PLATFORM/constants → Pipecat runtime prefer *UpdateSettingsFrame → run.md verify → symptom troubleshoot
```

Index: Pipecat bot.py→MCP pipecat-docs FIRST when host MCP tools available; else docs.pipecat.ai/llms.txt | LiveKit agent.py→MCP livekit FIRST when host MCP tools available; else docs.livekit.io/llms.txt

## Gate partial

framework|deployment|transport|model slots→WAIT open rows | prompt|VAD|tools|domain|latency|infra→no gate | wipe→full gate

## Category→docs

| cat | triggers | ref |
| --- | --- | --- |
| LLM swap local | different Nemotron3 | model-selection,catalog,nim-llm,hardware-probe |
| LLM config local | precision,OOM,max-model-len | nim-llm-profiles |
| LLM reasoning | thinking,budget,leak | llm-reasoning |
| ASR/TTS | swap | nemotron-speech model-selection,language-routing |
| LLM cloud | swap | catalog,GET /v1/models |
| prompt | tone,persona | voice-and-llm-output+framework prompt |
| domain/tools | vertical,functions | derive-domain,fake-data,bot.domain |
| speech customization | glossary,boost | speech-customization |
| turn/VAD | barge,cut off | troubleshoot turn+speech-input |
| latency | slow | troubleshoot lat,perf |
| transport | WebRTC,WebSocket,Both,TURN,SSH | pipecat+transport-selection+networking OR livekit llms |
| runtime settings | voice,temp | service-settings OR livekit audio customization |
| pipeline structure | processor,flows | pipecat pipeline/flows OR livekit architecture |
| pipeline mode | cascaded↔omni | partial gate+pipeline/omni or cascaded |
| omni | audio-in LLM | pipeline/omni,bot.omni-* |
| platform/NIM | compose,GPU | deployment,platform refs,nemotron-speech |
| observability | logs,metrics | pipecat metrics/debug OR livekit observability |

## Subjective→layer

slow→troubleshoot lat | dumb/wrong→LLM/prompt | robotic→TTS | cuts off→turn | ASR→ASR matrix/customization
"LLM"+different model→model-selection | "LLM"+precision/OOM same model→nim-llm

## Runtime vs rebuild

UpdateSettingsFrame: voice/temp/lang Pipecat mid-session | VAD/turn: aggregator params | LLM different model: model-selection→new image→nim-llm | LLM precision/OOM: nim-llm | reasoning: llm-reasoning constants restart | ASR/TTS local: matrix+compose | LLM cloud: constants | framework change: SKILL checklist

Recreate: regen from assets+locked gate+delta; keep pyproject unless extras; keep .env secrets; compose unless NIM images change.
Verify: run.md min health→agent→connect→one turn.
