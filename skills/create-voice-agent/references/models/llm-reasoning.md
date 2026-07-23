# LLM reasoning | Nemotron3 thinking on/off+budget

Trigger: Nemotron3 LLM locked(cascaded or omni reasoning id); user asks reasoning/thinking/budget; post-deploy toggle; **or** delegated rec from usecase-before-rec included Reasoning row (then gate pre-answered unless user swaps).
Voice default: OFF enable_thinking:false (lowest latency; thinking not to TTS).
REF: usecase-before-rec(tier defaults), voice-and-llm-output, nim-llm-profiles(local NIM), iterate(post-deploy), pipeline/omni(omni)

## Applies
Nemotron3 Nano/Super/Ultra cascaded|omni reasoning ids→gate YES toggle enable_thinking | no thinking support→skip row

## Pre-scaffold gate (blocks bot.py unless pre-answered: reasoning off|on|on budget N)
FORBID LLM_REASONING_* in bot.py before gate. OFF=recommended option in question not skip ask.

| user | gate |
| --- | --- |
| silent | ask Step1 WAIT |
| reasoning off/no reasoning | OFF |
| reasoning on/thinking on | ON→Step2 budget |
| reasoning on budget 512 | ON+512 |
| you choose(voice) | ask OFF recommended; second you choose→OFF after gate turn |
| rec locked reasoning from usecase tier | use rec values; skip Step1 unless user swaps on confirm |

Tier defaults (when rec set reasoning): general→OFF | balanced→OFF (ON 512–2048 only if user/complexity needs) | specialized→ON budget 8192 (16384 heavy tools). See usecase-before-rec.md.

Use **structured MCQ** for Step1 (OFF|ON|you choose) and Step2 budget (512|2048|8192|16384|32768) when gate runs and host supports MCQ — especially if bundled with transport/model confirm per gate.md.

Step1 TPL (fallback if MCQ UI unavailable): `LLM reasoning: OFF(recommended voice) or ON(hard tasks, latency)? off|on|you choose`
Step2 if ON budgets: 512 light | 2048 moderate | 8192 deep(default voice) | 16384 heavy | 32768 max. Custom OK; warn >16384 voice.

## Wire bot.py
```python
LLM_REASONING_ENABLED = False
LLM_REASONING_BUDGET = 8192
def _llm_extra() -> dict:
    body = {"chat_template_kwargs": {"enable_thinking": LLM_REASONING_ENABLED}}
    if LLM_REASONING_ENABLED: body["reasoning_budget"] = LLM_REASONING_BUDGET
    return {"extra_body": body}
```
Pass extra=_llm_extra() to NvidiaLLMService. FORBID chat_template_kwargs outside extra_body.
Reasoning ON: max_completion_tokens≥512 prefer 1024+.

Local NIM compose REQ when ON:
```yaml
NIM_PASSTHROUGH_ARGS: --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --max-num-seqs 64
```
Without --reasoning-parser nemotron_v3→TTS speaks redacted_thinking or silence. Cloud NVCF: still set enable_thinking extra_body.

## Post-deploy iterate
Edit constants+restart (not mid-session UpdateSettingsFrame). Toggle ON/OFF|change budget|recommend OFF/512-2048 for latency.

## Troubleshoot
thinking leak→local add parser or OFF | silence after turn ON→raise max_completion_tokens lower budget or OFF | slow→OFF/lower budget | 400→nest extra_body only

## Anti-patterns
silent OFF at scaffold | reasoning in .env | ON without parser local | model confirm=reasoning confirm
