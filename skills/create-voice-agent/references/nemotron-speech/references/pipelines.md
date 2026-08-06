# ASR pipeline riva-build | deploy-time

All riva-build inside NIM container --entrypoint /bin/bash. Fetch pipeline-configuration.html params not this file.

Docs: asr/customization/pipeline-configuration, customization runtime custom_configuration, protos, catalog.ngc.nvidia.com nim models, in-container riva-build -h

Prereq asr-custom container NGC_API_KEY deployable .riva from NGC _finetune

Config: --config-name=streaming|offline then components
Decoders: CTC/Conformer greedy|flashlight LM | RNNT/TDT/Nemotron nemo + use_stateful_decoding streaming
Chunk size server pipeline redeploy required client chunk_duration_ms ≠ server chunk_size
Components fetch page: LM ARPA/KenLM/NeMo VAD Silero endpointing diarization Sortformer
Runtime tune without rebuild→customization.html custom_configuration keys
Next asr-custom custom .nemo deploy
