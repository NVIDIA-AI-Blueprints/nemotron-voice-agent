# ASR custom model deploy | 4 phases

When prebuilt NIM insufficient. Phase N/4. Fetch riva-build syntax pipeline-configuration Notes per model.

Docs: pipeline-configuration nemo2riva inline, catalog.ngc _finetune, support-matrix/asr, prerequisites, riva-build -h in container
Prereq setup.md. Default deployable_vX.Y .riva NGC _finetune. trainable .nemo only if user fine-tuned.

Phase1 .riva|.nemo: A ngc download _finetune deployable .riva | B user .nemo riva-build inline nemo2riva copy Notes architecture
Phase2 riva-build RMIR inside container streaming|offline decoder/VAD/LM per pipelines.md output /riva_build_deploy/
Phase3 riva-deploy model repo
Phase4 docker run NIM_MODEL_REPO custom repo or compose override

Runtime without rebuild→customization.html. Build-time→pipelines.md. Cloud custom acoustic not supported local only.
