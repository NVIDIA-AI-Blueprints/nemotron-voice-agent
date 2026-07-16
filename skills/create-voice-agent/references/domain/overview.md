# Domain overview | vertical agents

Trigger: vertical|language|both. Agent derives data/tools/prompts FORBID copy skill. Gate still applies framework/deployment/transport.

Rules: infer DomainSpec→data→tools→SYSTEM_INSTRUCTION→greeting | FORBID copy menus/tools/prompts | author domain_store.json | state derivations handoff | mandatory derive-domain.md first

Triggers: vertical→derive→fake-data→speech-customization | language→language-routing | you choose non-EN→language-routing+model-selection | generic EN→catalog | boost/pronunciation→speech-customization+live NVIDIA pages

Workflow: derive-domain → gate → vertical speech Step0 WAIT wizard if yes → model-selection+language-routing → domain_store → handlers SYSTEM_INSTRUCTION → voice-and-llm-output → glossary if customization (Step3 show full boost_words+pronunciations inline before WAIT) → bot.domain.md → handoff DomainSpec rationale models speech example utterance

DomainSpec fields: domain_id,domain_label,primary_language,locale,reply_language,persona,user_goal,capabilities,constraints,deployment
Layout: bot.py/agent.py domain_store.json optional speech_glossary.json domain_tools.py. LiveKit same derivation livekit.md
