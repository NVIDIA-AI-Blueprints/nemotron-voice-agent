# Derive domain | infer from intent

FORBID copy skill restaurants/menus/prompts/tools. Infer state in handoff. Read before domain_store,handlers,SYSTEM_INSTRUCTION.
Vertical→speech-customization before scaffold. Generic skips.

Step1 DomainSpec: domain_id snake_case | domain_label | primary_language ISO639 default en | locale BCP47 | reply_language=primary | persona | user_goal | capabilities 3-6 verbs | constraints safety brevity voice fake data only. Missing→default ask only if ambiguous.

Step2 Fake data: patterns catalog+transaction|scheduling|Q&A|search+commit. Rules domain keys 8-15 seed reply_language locale currency no PII unique domain_store.json.

Step3 Tools: capability→list_/search_/add_/confirm_ minimal schema no session_id inject code handler domain_store.json only. Voice latency fixed flow→one combined tool domain_tools.py fake-data-and-tools.

Step4 SYSTEM_INSTRUCTION from DomainSpec: role reply_language goal tool names+when voice rules voice-and-llm-output no spoken ids/UUIDs/tool names tool workflow main tool once speak names/prices ≥2 domain rules failure handling. Write reply_language if ≠en FORBID skill examples.

Step5 Greeting 1-2 sentences reply_language connect context.add_message or TTSSpeakFrame unique domain.

Step6 Constants bot.py: DOMAIN_ID PRIMARY_LANGUAGE LOCALE SYSTEM_INSTRUCTION. Models model-selection language-routing if non-EN catalog if cloud.

Handoff: DomainSpec data+tool rationale prompt+greeting language models example utterance.
