# Troubleshoot | symptom→docs

FORBID hardcoded fixes. Match→fetch docs→propose citing doc. Pipecat: MCP pipecat-docs FIRST when host MCP tools available; else https://docs.pipecat.ai/llms.txt

```
PROC: read bot.py|agent.py → match Quick|§table → fetch ALL linked docs → layer diagnose transport→VAD→turn→STT→LLM→TTS → observers if vague → minimal doc-grounded fix → iterate.md if rebuild
RESPONSE: symptom|docs fetched|code location|hypothesis|change+tradeoff|run.md verify
```

## PI aliases (prefix https://docs.pipecat.ai/)
| k | path |
| --- | --- |
| si | pipecat/learn/speech-input.md |
| vad | api-reference/server/utilities/audio/silero-vad-analyzer.md |
| smart | api-reference/server/utilities/turn-detection/smart-turn-overview.md |
| uts | api-reference/server/utilities/turn-management/user-turn-strategies.md |
| fit | api-reference/server/utilities/turn-management/filter-incomplete-turns.md |
| stt-lat | pipecat/fundamentals/stt-latency-tuning.md |
| ums | api-reference/server/utilities/turn-management/user-mute-strategies.md |
| sf | api-reference/server/frames/system-frames.md |
| ctx | pipecat/learn/context-management.md |
| llm | pipecat/learn/llm.md |
| tts | pipecat/learn/text-to-speech.md |
| met | pipecat/fundamentals/metrics.md |
| tr | pipecat/learn/transports.md |
| tp | api-reference/server/services/transport/transport-params.md |
| se | api-reference/server/events/service-events.md |
| fpe | api-reference/server/events/frame-processor-events.md |
| hb | api-reference/server/pipeline/heartbeats.md |
| pid | api-reference/server/pipeline/pipeline-idle-detection.md |
| term | pipecat/learn/pipeline-termination.md |
| svc | api-reference/server/services/supported-services.md |
| dbg | api-reference/server/utilities/observers/debug-observer.md |
| tto | api-reference/server/utilities/observers/turn-tracking-observer.md |
| ubl | api-reference/server/utilities/observers/user-bot-latency-observer.md |

## Quick index
| kw | § |
| --- | --- |
| interrupt,barge,EOU,turn end | turn |
| slow,latency,TTFB | lat |
| no audio,ICE,WebRTC,TURN | transport |
| mic,permission,browser | client |
| transcript,ASR,wrong word | stt |
| mute,silent,TTS | tts |
| omni greeting garbage,27 30 | omni-greeting |
| stuck,freeze | pipe |
| tool,function,prompt | llm |
| reasoning,thinking,budget | llm+llm-reasoning.md |
| NIM,OOM,CUDA | infra |
| glossary,vertical | domain |

## turn
| kw | sub | fetch |
| --- | --- | --- |
| interrupts,cuts off,barges,EOU early | VAD stop_secs\|Smart Turn COMPLETE | si,vad,smart,uts,fit,pipecat-cloud/guides/smart-turn.md |
| waits too long after stop | VAD/Smart Turn high\|STT TTFS | si,stt-lat,uts |
| can't interrupt bot | enable_interruptions\|mute | si,ums,sf |
| echo,self-interrupt | spurious VAD | krisp-viva,aic-filter,koala-filter,rnnoise-filter pages |
| cough,keyboard triggers | start_secs/confidence | vad,si |
| greeting loops,truncated intro | first utterance interrupt | ums,si,ctx |
| greeting numeric garbage,27 30,first TTS nonsense | omni LLMRunFrame greeting + mic race before mute | bot.omni-runner §Greeting,pipeline/omni.md §Greeting,ums |
| short yes/no missed | start_secs\|Smart Turn | si,smart,uts |
| dual VAD mismatch | STT commit vs local turn | si,stt-lat,named STT page,external-turn-management |
| noise as speech | no isolation | krisp-viva,aic-filter |

VAD layers: stop_secs(VAD|si,vad) | start_secs(VAD|si) | SmartTurn stop_secs(smart) | user_turn_stop_timeout(uts) | ttfs_p99(stt-lat)
FORBID: tune VAD stop_secs alone when Smart Turn active without smart.

## omni-greeting
| kw | cause | fix |
| --- | --- | --- |
| greeting speaks 27,30,numeric garbage,short nonsense on connect | `LLMRunFrame` text-only omni greeting unreliable; mic `client-ready` races audio omni turn before greeting; weak JSON `response` guard | **Fix A:** `TTSSpeakFrame` + `context.add_message` role **assistant** in `on_client_connected` — **not** `LLMRunFrame` / fake user intro. **Fix B:** `MuteUntilFirstBotCompleteUserMuteStrategy` in `LLMUserAggregatorParams`. **Fix C:** `_is_unusable_spoken()` in copied `assets/omni_service.py`. REF bot.omni-runner.md §Greeting, pipeline/omni.md §Greeting |

Acceptance: TTS log shows full greeting sentence on connect; no omni API call for greeting; user speech after greeting still works.

## lat
| kw | fetch |
| --- | --- |
| slow,2s+,gap before speak | stt-lat,met,ubl,llm,tts |
| first token slow,user re-speaks | met,se,fpe,si |
| flows slow | pipecat-flows nodes/functions/context-strategies |
| TTS late | tts,text overview |
| telephony latency | telephony overview,telephony-in-production |

## client
| kw | fetch |
| --- | --- |
| mic blocked | quickstart Troubleshooting,client/js/errors,media-management |
| connect timeout,StartBotError | errors,session-lifecycle,events |
| Chrome vs Safari | choosing-a-transport,quickstart |
| audio before bot ready | session-initialization,ums |

## transport
| kw | fetch |
| --- | --- |
| no audio either direction | first-try-workstation-webrtc.md,tr,tp,tts |
| ICE fail,one-way | first-try,webrtc-turn.md |
| POST /start 400 | first-try |
| UDP blocked,SSH | pipecat.md §WebSocket,ssh.md,webrtc-turn |
| choppy | media-management,tr |

## stt
| kw | fetch |
| --- | --- |
| wrong words,domain terms | svc,speech-customization.md,nemotron-speech/asr.md |
| partial transcript | stt-lat,fit |
| wrong language | service-settings,named STT,language-routing.md |
| STT silent | se,fpe |
| non-EN INCOMPLETE | smart,si,external-turn-management |

## llm
| kw | fetch |
| --- | --- |
| wrong answers,off-topic | ctx,voice-and-llm-output.md |
| verbose | ctx,voice-and-llm-output,tts |
| tool fail | function-calling,fake-data-and-tools.md |
| context too long | context-summarization |
| duplicate responses | sf,ctx,term |
| developer role 400 | first-try,bot.workstation-runner.md |
| timeout,5xx | se,llm-switcher |
| TTS speaks thinking | llm-reasoning.md,voice-and-llm-output |
| toggle reasoning | llm-reasoning.md,iterate.md |

## tts
| kw | fetch |
| --- | --- |
| wrong voice,accent | tts,service-settings,nemotron-speech/tts.md |
| speaks markdown | markdown-text-filter,voice-and-llm-output |
| clipping,speed | tts,tp |
| mute mid-call,no audio | se,fpe,service-switcher |
| language_code crash | first-try |

## ctx|pipe|telephony|flows|infra|cloud|domain
| § | kw | fetch |
| --- | --- | --- |
| ctx | transcript missing | saving-transcripts,transcriptions |
| ctx | context wrong after interrupt | sf,si,tts |
| pipe | stuck,frozen | hb,pid,detecting-user-idle |
| pipe | crash import | migration-1.0,pipecat.md |
| telephony | one-way,carrier | telephony overview,serializers/twilio |
| flows | wrong node | pipecat-flows nodes/state/actions |
| infra | health fail | run.md,deployment-readiness-checks.md |
| infra | OOM | hardware-probe 2d |
| infra | NIMProfileIDNotFound | nim-llm-profiles-and-deployment.md |
| infra | cache blocks | raise NIM_KVCACHE_PERCENT |
| infra | 404 /v1/models | SKILL rule 3,run.md |
| infra | Jetson/DGX | jetson-thor.md,dgx-spark.md |
| cloud | PCC errors | pipecat-cloud error-codes,logging,health-checks |
| domain | glossary wrong | speech-customization.md |
| domain | multilingual | language-routing,derive-domain |

## debug (vague→enable)
dbg | tto | ubl | hb | met | se
