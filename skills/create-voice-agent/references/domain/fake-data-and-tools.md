# Fake data + tools impl

Content per derive-domain.md. HOW to implement.

Principles: domain_store.json matches model | pure Python handlers no network JSON returns | schemas match SYSTEM_INSTRUCTION | voice_hint reply_language | session_id code only not schemas/dev msg | one tool round/turn combine sequential | voice_hint not ok:true only. Confirm API pipecat-docs FORBID invent.

Pipecat failures:

| fail | symptom | fix |
| handler(params,field) | missing arg IN_PROGRESS stuck | FunctionCallParams only params.arguments |
| 3+ sequential tools | filler silence | combined tool |
| on_function_calls_started filler each | repeated filler | omit or once |
| multi-step prompt | sequential rounds | combined_tool once then speak |
| substring match | false positives | exact match |
| slow post-tools | pause | smaller Nemotron3 Nano |

Handler:

```python
from pipecat.services.llm_service import FunctionCallParams
async def _handle(params: FunctionCallParams):
    session_id = getattr(params, "session_id", None) or getattr(getattr(params, "context", None), "session_id", None)
    if not session_id:
        await params.result_callback({"error": "missing session context"})
        return
    result=fn(session_id,params.arguments[...])
    await params.result_callback(result)
llm.register_function("name",_handle); tools=ToolsSchema(standard_tools=[...])
```

FORBID async def handler(params, symptoms: str). Combined tool: screen+record+search→one voice_hint dict.

domain_store.json project root utf-8 ensure_ascii=False agent seed. Handlers load/save store keyed by the **per-invocation** session id from the active request/transport context; fail closed when unavailable. FORBID module-global or transport-global session state. Schemas no session_id wire bot.domain.md. SYSTEM_INSTRUCTION derive Step4 only.

Fallback no tools: embed short catalog SYSTEM_INSTRUCTION note limitation. Smoke optional: `uv run python -c "from domain_tools import fn; print(fn(...))"`
