# Eval prompts | create-voice-agent test cases

Invoke create-voice-agent skill or load skills/create-voice-agent. Format: prompt | EXPECT | WRONG

## Step 0 probe (all create paths)

Any create/scaffold request → EXPECT **run hardware probe commands first** (nvidia-smi, DMI, device-tree) before gate AskQuestion | WRONG ask deployment before probe | no GPU → EXPECT auto cloud + inform user, no deployment MCQ | GPU → EXPECT Local|Cloud MCQ with GPU/platform label

## Env secrets

Cloud/LiveKit create with empty .env → EXPECT NVIDIA_API_KEY missing message + link build.nvidia.com/settings/api-keys + STOP WAIT | WRONG scaffold without key | Jetson local → HF_TOKEN WAIT | LiveKit → LIVEKIT_* WAIT + cloud.livekit.io / lk cloud auth guide

## Transport MCQ

Open transport → EXPECT AskQuestion WebRTC | WebSocket | Both (recommended) | WRONG infer WebRTC default at scaffold | Both → transport_params webrtc+websocket, pyproject both extras, run without `-t`

## Ambiguous Pipecat

`Build voice assistant in pipecat_test` → EXPECT probe then STOP bundled Q framework transport models (deployment auto if no GPU) no files | WRONG scaffold | WRONG deployment ask on no-GPU host

## Model slot MCQ (rows 4–6)

Open LLM/ASR/TTS → EXPECT **three separate AskQuestions** with skill options + `(recommended)` + Other | WRONG `name models or I'll choose` | WRONG single bundled models MCQ

Follow-up `Pipecat WebRTC you choose all`→usecase WAIT→LLM MCQ→ASR MCQ→TTS MCQ→disclosure table with exact ids | WRONG delegate-only rec without per-slot MCQ

Partial `Pipecat cloud NVCF pipecat_test`→transport/models only

## Cloud you choose

`Create voice agent cloud endpoint you pick models Pipecat` → probe→transport→usecase→LLM→ASR→TTS MCQs→disclosure no bot.py turn1 | WRONG `name models or you choose` single ask

## Delegated use case tiers

`Pipecat cloud WebRTC you choose` → usecase Q WAIT | then LLM MCQ (Nano/Super/Ultra + Other) → ASR MCQ (Nemotron Streaming / Parakeet RNNT variants + Other) → TTS MCQ (Magpie + Other) → disclosure | `specialized`+Ultra pick → reasoning ON in table

Follow-up `WebRTC reasoning on budget 512 looks good`→scaffold wired constants

## LiveKit explicit

`LiveKit livekit_test NVIDIA cloud cascaded agent.py models in file console` → probe+env-secrets then implement rec unless named; default LiveKit Cloud LIVEKIT_* in .env | WRONG self-hosted localhost default without user ask

## Workstation H100

`Pipecat pipecat_local_h100 local GPU bot.py docker-compose.nim.workstation.yml WebRTC start NIMs+bot` → workstation compose NIM bring-up

## SSH one-shot

`Pipecat cascaded workstation-test local GPU running NIMs SSH Mac` → skip gate docker ps bot.workstation-runner --host 0.0.0.0 -t webrtc handoff VM:7860/client

## Jetson

Explicit jetson template HF_TOKEN you choose | one-shot SSH rec WAIT jetson-thor Riva 50052 vLLM 18000 TURN

## Hindi food domain

`Pipecat cloud WebRTC Hindi food you choose start bot` → skip main gate usecase WAIT rec multilingual WAIT (auto-detect→Parakeet RNNT Multilingual; locked Hindi→Nemotron ASR Streaming Multilingual) speech Step0 derive-domain no canned menu | WRONG EN-only Nemotron Streaming skip usecase when delegated

## Medical customization

`Pipecat cloud WebRTC medical clinic clinic_agent you choose` → rec WAIT speech Step0 wizard if yes fetch NVIDIA docs | WRONG hardcode matrix Whisper boost

## Partial domain / one-shot delegated

`Hindi food ordering you choose everything else` → if `one_shot_delegated` + `I confirm your top picks`: framework/deployment/transport only, then disclosure with pre-locked models (skip MCQ) | else: framework/deployment/transport + per-slot MCQs when models open
