# bot.domain | domain wiring

Content derive-domain.md extends bot.workstation-runner.md pipecat-docs tools TTS API.

Constants bot.py:
```python
DOMAIN_ID,PRIMARY_LANGUAGE,LOCALE="<derived>"
DOMAIN_STORE_PATH="domain_store.json"; DEPLOYMENT_PLATFORM="<gate>"
ASR_MODEL,TTS_MODEL,TTS_LANGUAGE_CODE,TTS_VOICE,LLM_MODEL_ID="..."  # discovery
SYSTEM_INSTRUCTION="""<derive Step4>"""
SPEECH_GLOSSARY_PATH="speech_glossary.json"  # if customization
```

Speech glossary only if wizard complete. Wizard Step3 must display this full shape inline in chat and receive explicit user confirmation before any `speech_glossary.json` or bot wiring:
```json
{"boost_words":["term1","term2"],"boost_score":20.0,"pronunciations":{"OBD":"oʊ biː diː"}}
```
```python
import json
from pathlib import Path

def _load_speech_glossary():
    p=Path(SPEECH_GLOSSARY_PATH)
    if not p.exists(): return [],20.0,{}
    d=json.loads(p.read_text())
    return d.get("boost_words")or[],float(d.get("boost_score",20.0)),d.get("pronunciations")or{}
SPEECH_BOOST_WORDS,SPEECH_BOOST_SCORE,SPEECH_PRONUNCIATIONS=_load_speech_glossary()
```
STT boosted_lm_words/score if words. TTS custom_dictionary if pronunciations. Jetson voice= not voice_id=.

Session: resolve per-invocation session_id from runner/transport context on connect; inject into tool dispatcher only — not schemas/dev msgs. FORBID module-global `_active_session_id`.

Nemotron cloud: LLM_EXTRA llm-reasoning.md LLM_REASONING_ENABLED budget Settings extra=LLM_EXTRA; max_completion_tokens=512 **only when** LLM_REASONING_ENABLED is explicitly False; when reasoning ON derive from LLM_REASONING_BUDGET (+ headroom for final answer, prefer ≥1024)

Greeting: TTSSpeakFrame on_client_ready after set_bot_ready preferred.

Tools Pipecat: FunctionCallParams only register_function ToolsSchema one combined tool main intent no filler every on_function_calls_started SYSTEM_INSTRUCTION main tool once 2-4 sentence answer.

Pipeline: transport.input→stt→user_aggregator→llm→tts→transport.output→assistant_aggregator
Files: bot.py domain_store.json optional speech_glossary.json domain_tools.py pyproject.toml .env.example
