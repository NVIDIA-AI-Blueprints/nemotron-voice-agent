# Record Conversations and Post-Process Them Offline

Session recording stores every conversation of the multilingual assistant: its
timeline, per-service metrics, transcripts, the exact LLM inputs (including
images), and audio. You can then compare pipeline variants, and a background
runner (the *dreamer*) post-processes past sessions while nobody is talking to
the agent, for example to extract memories about the people it talks to.

Recording has two parts, a hot path and a cold path:

- **Hot path (inside the voice pipeline).** An observer and a few event hooks
  only put small records on a queue. A single writer task batches them to
  SQLite and the filesystem in a worker thread, so the pipeline never waits on
  I/O. If storage can't keep up, records are dropped and counted rather than
  adding latency.
- **Cold path (the `dreamer` container).** Jobs run only when no session is
  live. A job checks between steps whether a conversation has started. If one
  has, it stops at that point and resumes from its checkpoint later.

## Enable recording

Set these in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITORING_ENABLED` | `false` | Record sessions |
| `MONITORING_DATA_DIR` | `data` | Root for the database and artifacts (relative to project root, or absolute) |
| `MONITORING_DB_URL` | `sqlite:///<data dir>/voice_agent.db` | SQLAlchemy URL; `postgresql+psycopg://…` works once the driver is installed |
| `RECORD_AUDIO_TURNS` | `true` | Per-turn user WAVs (16 kHz, the raw ASR input) and bot WAVs |
| `RECORD_AUDIO_STEREO` | `true` | Whole-call `conversation.wav`, user on the left channel and bot on the right, written in 10 s chunks |
| `RECORD_VIDEO` | `off` | `off`, `keyframes` (JPEG snapshots of the input video) or `full` (chunked MP4, for development only). Only takes effect once the transport enables video input (`video_in_enabled`); the multilingual transport doesn't enable it yet. Images in the LLM context and images returned by tools are recorded regardless. |
| `RECORD_VIDEO_FPS` | `1.0` | Keyframe rate for `RECORD_VIDEO=keyframes` |
| `RECORD_SYSTEM_METRICS` | `true` | Sample host GPU load, GPU temperature, power, RAM, and CPU load once per second while a session is live. Refer to [System Metrics](#system-metrics). |
| `GIT_SHA` | unset | Code revision stored in each session snapshot. Set it for the app container, which does not contain `.git`. On a host checkout, the recorder reads the revision from `.git`. Refer to [Record the Code Revision](#record-the-code-revision). |

When recording is enabled, it replaces the legacy `ENABLE_*_AUDIO_DUMP` recorder
for the multilingual assistant.

Compose mounts `./data` into the app container. The container runs as root, so
files under `data/` are root-owned on the host. Run the tools below inside a
container (`docker compose exec dreamer …`), or `chown` the directory.

## What is stored

Everything is keyed by the server `session_id`. The same id is passed to
pipecat as the OpenTelemetry `conversation_id`, so Phoenix traces can be joined
to these rows.

| Table | Content |
|-------|---------|
| `sessions` | Start and end, `end_reason`, heartbeat, and a **config snapshot** (models, servers, language, voice, turn detection, client tools, git sha). The snapshot is the A/B key. |
| `turns` | One row per exchange: user and bot text, timestamps, `interrupted`, and `barge_in`. Turn 0 is the greeting. Refer to [Turn Timestamps and Flags](#turn-timestamps-and-flags). |
| `turn_metrics` | One row per user turn, written when the turn ends: turn kind, response and voice latency, the stage breakdown, token and call counts, and GPU load. Refer to [Per-Turn Metrics](#per-turn-metrics). |
| `tool_calls` | One row per tool or intent call: name, trigger (`llm` or `intent`), target (`client` or `home_assistant`), send time, duration, outcome (`ok`, `error`, `timeout`, or `cancelled`), and the `perceivable` flag |
| `system_samples` | Host-wide samples, one per second while a session is live, with the number of live sessions |
| `events` | Timeline: VAD and bot speaking, interruptions, final ASR, function calls (requested, sent, result, cancelled), latency breakdowns, errors |
| `metrics` | Flat numeric samples: `ttfb`, `processing`, `llm_usage.*`, `tts_usage`, `user_bot_latency`, `first_bot_speech_latency`, `turn_duration`, … Zero-valued `ttfb` and `processing` samples, which Pipecat emits at startup, are not stored. |
| `llm_calls` | The full LLM input, tools, output, function calls, TTFT, tokens, and `n_images` / `image_pixels` for each call of the conversation LLM |
| `media` | Audio clips, images and video, deduplicated by SHA-256. `source` is `context`, `tool:<name>`, `user_image` or `camera:<track>`. |
| `annotations` | Job outputs and human labels: `(target, source, kind, value)` |
| `jobs` | Post-processing queue with checkpointed `progress` |
| `people` | People the agent talks to: name and an `archived` flag |
| `speaker_assignments` | Who spoke: one row for the whole session (`turn_idx=-1`) or per-turn overrides. `source` is `live:picker`, `live:voice-id:<model key>` (suffixed with `:verified` when the speaker's face confirmed the voice, or with `:low` or `:unknown` for an uncertain attribution), or `human:<name>`. |
| `voice_embeddings` | Voice and face ID enrollment data: speaker and face embeddings per person and model key, written by the `enroll_speaker` tool. `source` is `live:enroll:voice` or `live:enroll:face`. |
| `memories` | Facts about a person: text, category, confidence, `status` (`proposed`, `active`, `forgotten` or `superseded`), source, and review fields. Rows are never deleted. |
| `memory_evidence` | The session, turn and quote that support each memory |
| `memory_uses` | Which memories were injected into which live session |

Images are never stored inline. Any `data:` image found in the LLM context or in a
tool result is replaced by `{"type": "media_ref", "sha256": …}`, and the file is
written once to
`data/artifacts/sessions/<date>/<session_id>/media/<sha256>.<ext>`. The
`llm_calls` rows can therefore be replayed against another LLM or VLM.

```text
data/
├── voice_agent.db
└── artifacts/sessions/2026-09-26/<session_id>/
    ├── conversation.wav          # stereo, user L / bot R
    ├── turns/001_user_00.wav     # VAD segment 0 of user turn 1 (16 kHz)
    ├── turns/001_bot_00.wav
    ├── media/<sha256>.jpg        # images the LLM saw / tools returned
    └── video/0000.mp4            # RECORD_VIDEO=full only
```

### Turn Timestamps and Flags

The `turns` table keeps two pairs of timestamps. Use the pair that matches your
question.

| Column | Meaning |
|--------|---------|
| `user_speech_stopped_at` | The user stopped speaking: the voice activity detection (VAD) decision minus its silence window |
| `user_stopped_at` | The turn was released to the LLM, after turn detection and ASR finalization |
| `bot_started_at` | The LLM response started |
| `bot_speech_started_at` | The first bot audio was sent |
| `interrupted` | The assistant response of this turn was cut short: an interruption arrived while the response was open, or the session ended during it |
| `barge_in` | The user started this turn while the bot was still speaking |

Every user turn sends an interruption through the pipeline, so an `interruption`
event is not a barge-in.

### Per-Turn Metrics

Each `turn_metrics` row carries two latencies:

- **Voice latency** runs from the end of user speech to the first bot audio.
- **Response latency** runs from the end of user speech to the first perceivable
  response. That is the first bot audio, or the moment a perceivable tool call is
  sent, whichever comes first. Both latencies are equal when the turn has no
  perceivable action. Refer to
  [Client-Executed Tools](../../src/examples/multilingual/README.md#client-executed-tools)
  for the `perceivable` flag.

The stages split the time up to the first response along the critical path.
Stages never overlap, and `unexplained_secs` holds the time that no stage owns,
so the stages and the remainder add up to `total_secs`.

| Column | Stage |
|--------|-------|
| `asr_secs` | End of speech to the final transcript |
| `turn_detection_secs` | The rest of the wait until the turn is released. ASR finalization and turn detection run concurrently, and the time goes to whichever finishes last. |
| `intent_match_secs` | Intent engine matching, when the engine is enabled |
| `llm_first_secs` | First LLM call: start to first token, or start to the tool call when the call ends in one |
| `tool_secs` | Tool round trips and intent target calls |
| `llm_later_secs` | Later LLM calls, up to the first token of the call that speaks |
| `text_aggregation_secs` | First token to the first sentence handed to the TTS |
| `tts_secs` | TTS time to first byte |
| `unexplained_secs` | Remainder |

The turn `kind` lets you compare like with like. The first matching rule wins:
`intent` (the intent engine answered), `vision` (an LLM call of the turn had
image input), `tool` (at least one tool or intent call), and `plain`.

The recorder writes these rows when a turn ends and again when the session
closes. To fill them for sessions recorded before this table existed, run the
backfill. It is idempotent and uses the same code as the live recorder.

```bash
PYTHONPATH=src uv run python -m monitoring.turn_metrics                 # every session
PYTHONPATH=src uv run python -m monitoring.turn_metrics --session <id>  # one session
```

Backfilled sessions have no `perceivable` flags, so their response latency
equals their voice latency.

### Repair Sessions From Earlier Recorders

Earlier recorders stored the text of an interrupted assistant response on the
following turn, together with `interrupted` and `bot_stopped_at`. The recorder
now keeps them on the turn that opened the response, and marks new sessions with
`recorder_version: 2` in the config snapshot. To repair older sessions, preview
the changes and then apply them:

```bash
PYTHONPATH=src uv run python -m monitoring.turn_metrics --repair-attribution --dry-run
PYTHONPATH=src uv run python -m monitoring.turn_metrics --repair-attribution
```

The command prints each changed turn before and after, moves the response back
to its own turn, and then recomputes the per-turn metrics. For a response that
was cut, `bot_stopped_at` becomes the start of the next user turn. You can run
the command again safely: each repaired session gets a
`turn_attribution_repair` annotation that lists what moved, and the command
skips sessions that have one. Memories extracted earlier by the `dream` job are
not updated. Re-run `dream` from the session page if you want it to read the
corrected turns.

When the database lives in a root-owned `data/` directory, run the commands in
a container, for example `docker compose exec dreamer uv run python -m
monitoring.turn_metrics --repair-attribution`.

### System Metrics

The sampler runs in its own thread, only while at least one session is live.
All values are host-wide. The sampler skips a source that is not available and
logs one line that lists the active and missing sources.

| Metric | Source | On the host | In the app container |
|--------|--------|-------------|----------------------|
| CPU load | `/proc/stat` | Yes | Yes |
| RAM used | `/proc/meminfo` (`MemTotal - MemAvailable`; includes GPU memory on Jetson) | Yes | Yes |
| GPU temperature | `/sys/class/thermal` zone `gpu-thermal`, else `nvidia-smi` | Jetson, or any host with `nvidia-smi` | Jetson, or a local recipe |
| Board input power | `/sys/class/hwmon` INA238 monitor (`VIN`): the power drawn by the whole module | Jetson Thor | Jetson Thor |
| GPU power | `nvidia-smi` power draw of GPU 0, used only when no board monitor exists | Hosts with `nvidia-smi` | Local recipes |
| GPU load | `nvidia-smi` utilization of GPU 0 | Hosts with `nvidia-smi` | Local recipes |

Board input power and GPU power are different quantities. Each sample stores
which one it holds in `power_source` (`board` or `gpu`), and the review UI labels
the value **Board input power** or **GPU power** accordingly. Do not compare the
two across hosts.

The app services of the local recipes (`<example>/server` and
`<example>/single-gpu`) reserve the NVIDIA devices with
`NVIDIA_DRIVER_CAPABILITIES=utility`, which makes `nvidia-smi` available in the
container without CUDA libraries. The app allocates no GPU memory. The cloud
recipes do not request the NVIDIA runtime, so they start on hosts without a GPU
and record CPU load and RAM only. When `nvidia-smi` is missing or fails, the
sampler drops that source and keeps the others.

### Record the Code Revision

Each session snapshot stores the code revision as `git_sha`, which the
**Metrics** page uses for the per-revision trend. A server started on the host
reads it from `.git`. The app container has no `.git`, so pass the revision
through `GIT_SHA`:

```bash
GIT_SHA=$(git rev-parse HEAD) docker compose --profile multilingual-assistant/single-gpu up -d
```

The local recipes forward `GIT_SHA` from your shell or from `.env`. The cloud
recipes read it from `.env` only. When `GIT_SHA` is unset, the deployment starts
normally and sessions are recorded without a revision. Because Compose mounts
`./src` into the container, set the value again after you update the checkout
and restart the service.

## Compare pipeline variants

```bash
PYTHONPATH=src uv run python -m monitoring.report                      # grouped by llm/asr/tts model + language
PYTHONPATH=src uv run python -m monitoring.report --since-days 7 \
    --group-by llm.model,turn_detection.silero_vad_only --csv report.csv
```

Each group shows:

- user→bot latency (p50/p90), also split by turn kind (`turns[tool]`, `user_bot_p50[tool]`, …), and first-speech latency;
- per-service TTFB, with LLM TTFT reported separately for text-only and image-bearing calls;
- prompt tokens and interruption rate;
- WER per ASR source, once `reasr` has run.

The turn-kind split reads the `turn_metrics` table. Run the backfill in
[Per-Turn Metrics](#per-turn-metrics) for older sessions.

## Review in the web UI

Click **Review** in the header of the web client, or open `#/review` directly.
It opens a separate tab, so a call in progress isn't interrupted. The review
API (`/api/review/*`) is only served when `MONITORING_ENABLED=true`.

- **Sessions.** Recorded conversations with their models, review progress and
  live WER. Opening a session shows each turn with:
  - a summary of the host load during the session: GPU and CPU load, GPU
    temperature, RAM, and board input power or GPU power, depending on the source;
  - the response and voice latency, the turn kind, a barge-in badge, and a
    timeline of the latency stages next to the GPU and CPU load during the turn;
  - the tool and intent calls with their duration and outcome;
  - the user audio, every transcript diffed against the reference, and the human reference;
  - images, and the LLM calls (the full stored input on click);
  - the assistant reply, plus the whole-call stereo recording;
  - a speaker selector for the whole session and for each turn. Changing who
    spoke re-queues the `dream` job for that session.
- **People.** Create, rename, and archive the people the agent talks to. Each
  person shows their session and memory counts, their enrolled voice and face
  samples, and how many turns voice ID attributed to them (and how many of
  those a face verified). Reconcile people who turn out to be the same person:
  - **Possible duplicates** lists pairs whose enrolled voices or faces are
    similar (cosine similarity of their centroids, per model) or who share a
    name. The scores help you decide; nothing is merged automatically.
  - **Keep** one person of a pair, or on a person's page pick **Same person
    as**, to merge the other into them. The turns, voice and face samples, and
    memories of the merged person move to the one you keep, who keeps their
    name; the merged person is deleted. A merge cannot be undone.
  - On a person's page, **Forget** deletes their voice or face samples for one
    model, for example when a sample belongs to someone else. They must enroll
    again to be recognized.
- **Memories.** A queue of memories that nobody has reviewed yet. Each memory
  shows its evidence turns (audio and quote), its confidence, and how often it
  was used ("used in N sessions / N replies"). You can:
  - **approve** it, so that it becomes `active`;
  - **correct** its text or reassign it to another person. This creates a
    `human:<name>` memory that supersedes the original;
  - **forget** it, so that it is never used again.
- **ASR reference.** Every recorded user turn, with the ones where the live
  ASR and the reference model (Voxtral) disagree first.
  - Listen, then take the live (`1`) or reference (`2`) text, or edit it (`e`),
    and save (`Enter`). Saving rescores WER immediately, on the CPU only.
  - Turns where both agree can be accepted in bulk.
  - Your annotator name is kept in the browser and stored as
    `source="human:<name>"`.
- **Metrics.** Compares pipeline variants over a time range (24 h, 7 days,
  30 days, all). A variant is the combination of the config fields you
  compare by: LLM, ASR, TTS, voice, language, turn detection, transport, git
  revision, intent engine, prompt. Each variant keeps its color across filter
  changes. A turn-kind filter (all, plain, tool, intent-handled, vision) applies
  to the latency views, and every view shows its sample count. The page shows:
  - response latency as the headline number, with voice latency next to it;
  - response latency per variant: median with a p10–p90 whisker, plus a
    per-session trend (click a point to open the session);
  - the stage breakdown per variant, and a table per variant and turn kind;
  - a trend per day and per git revision with latency and stage percentiles. A
    day or revision is marked "slower" when its median response latency is more
    than 20% and 0.1 s above the previous one, with at least 5 turns in both;
  - tool and intent calls: count, duration, and failure, timeout, and
    cancellation rates;
  - time to first byte per service;
  - LLM time to first token, text-only vs with images;
  - ASR WER per transcript source;
  - a table with all the numbers.
- **Dreamer.**
  - **Status:** what the worker is doing and why (idle, waiting for a live
    session to end, running a job, paused), its heartbeat, and job counts.
  - **Controls:** Pause/Resume (a running job stops at its next checkpoint and
    resumes later), the state of the on-demand model containers, and the job
    queue filtered by status, with Cancel for pending jobs and Retry/Re-run for
    the others.
  - **Queue for all sessions:** runs a job kind on every ended session that has
    never had it.
  - **How it works:** the dreamer publishes its state and reads the pause flag
    through the database (`kv` table). The voice server therefore needs no
    Docker access.

## Offline post-processing (dreamer)

```bash
docker compose --profile multilingual-assistant/single-gpu --profile dreamer up -d
docker compose --profile dreamer-models build    # once: Voxtral image
docker compose --profile dreamer-models create   # once: on-demand model containers
docker compose exec dreamer uv run python -m monitoring.jobs status
```

How the runner behaves:

- **Scheduling.** Every ended session gets the jobs listed in
  `auto_enqueue` (default `[reasr, dream]`). A job runs only when no session has a recent heartbeat and
  the last activity is older than `idle_grace_secs`.
- **On-demand models.** A job can require a compose service from the
  `dreamer-models` profile, for example `dreamer-asr-en`. The runner starts it
  through the Docker socket only while idle, and stops it when the queue is
  empty or a conversation starts. After `docker compose down`, run
  `docker compose --profile dreamer-models create --force-recreate` again:
  containers created before keep a reference to the removed network and fail to
  start with `network ... not found`.
- **Crash recovery.** Sessions orphaned by a crash are closed after
  `orphan_after_secs`.

Configure the runner in
[`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml)
(or point `DREAMER_CONFIG` at your own copy).

### `reasr`: evaluate another ASR on real conversations

For each recorded user turn, `reasr` transcribes the audio with the configured
`reference` and `candidates` endpoints. It then scores the live transcript and
each candidate by word error rate (WER):

- **Reference.** The latest **human correction** is used when there is one.
  Otherwise the `reference` model serves as a pseudo ground truth.
- **Output.** Per-turn `kind="wer"` annotations, plus a session-level
  `kind="wer_summary"`: total errors divided by total reference words.

Endpoints can use either protocol:

- `protocol: riva` (default): a Riva-compatible gRPC server, such as Riva, a NIM
  or NeMo-Speech.cpp. Set `server: host:port`.
- `protocol: openai`: an OpenAI-compatible `POST /v1/audio/transcriptions`
  endpoint. Set `base_url` and `model`.

Choose which models run with `enabled` on each endpoint (default `true`). A
disabled endpoint stays configured, but `reasr` skips it and its container is
never started. By default only the Voxtral reference runs: the shipped
`nemotron-speech-streaming-en-0.6b` candidate has `enabled: false`. Set it to
`true` to compare that model on English sessions; it starts `dreamer-asr-en`.

The default reference is **Voxtral Mini 4B**
(`mistralai/Voxtral-Mini-4B-Realtime-2602`). It is served by vLLM in the
on-demand `dreamer-asr-voxtral` container and covers French, English, German,
Spanish and more. Its image (`docker/voxtral.Dockerfile`) adds the audio
decoders missing from the stock vLLM image. Build it once with
`docker compose --profile dreamer-models build dreamer-asr-voxtral`. You can
tune it with `VOXTRAL_MODEL`, `VOXTRAL_GPU_MEMORY_UTILIZATION` (default `0.15`,
since it shares unified memory with the idle live stack) and
`VOXTRAL_MAX_MODEL_LEN`.

To add human corrections:

```bash
docker compose exec dreamer uv run python -m monitoring.transcripts_csv export /app/data/turns.csv
# fill the "corrected" column, then:
docker compose exec dreamer uv run python -m monitoring.transcripts_csv import /app/data/turns.csv --annotator alice
docker compose exec dreamer uv run python -m monitoring.jobs enqueue reasr <session_id> ...
```

### `dream`: extract memories about people

For each ended session attributed to a person, `dream` extracts durable facts
about that person, such as their name, preferences, relationships, routines,
events, and health information they shared. The job works as follows:

- **Skipped sessions.** A session without an attributed person is marked done
  with `skipped` progress. Attribute it on the session page to run the job again.
- **Transcript.** For each user turn, the job uses the human reference, then
  the `reasr` reference model transcript, then the live ASR text.
- **LLM.** By default, the job calls the session's own LLM (the `base_url` and
  `model` stored in the session config), for example the always-on vLLM of a
  `*/single-gpu` recipe. Set `dream.llm` to use another OpenAI-compatible
  endpoint, or `dream.service` to start an on-demand `dreamer-models`
  container.
- **Status.** A memory with a confidence at or above `auto_use_threshold`
  (default `0.8`) becomes `active` and is used in live prompts right away. You
  review it afterwards. A memory below the threshold stays `proposed` until you
  approve it. A memory that would replace a human-reviewed memory is always
  `proposed`.
- **Re-runs.** The job never changes a reviewed memory. Before extracting
  again, it supersedes the unreviewed memories whose only evidence is that
  session.

The `dream` section of
[`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml)
also sets `reasoning` and `temperature`.

### Add a job

```python
from monitoring.jobs.base import Job, JobContext, register

@register
class SafetyJob(Job):
    kind = "safety"

    def run(self, ctx: JobContext) -> None:
        for turn in ctx.store.rows("turns", ctx.job["target"]):
            if turn["idx"] < ctx.progress.get("next", 0):
                continue
            ctx.check_preempted()            # yields to live sessions
            verdict = ...                    # e.g. call the (idle) vLLM endpoint
            ctx.store.add_annotations([{ "session_id": ctx.job["target"], "target_type": "turn",
                "target_id": f"{ctx.job['target']}:{turn['idx']}", "source": "guard-model",
                "kind": "safety", "value": verdict }])
            ctx.save_progress(next=turn["idx"] + 1)
```

Import the module in `src/monitoring/jobs/__init__.py`, then add its `kind` to
`auto_enqueue`. Planned jobs follow the same pattern:

- `safety`: kid-safety policy checks on text and images;
- `rl_export`: JSONL with implicit rewards such as interruptions, rephrasing,
  tool success and latency;
- `revlm`: replay of `llm_calls` against another model.

## Remember people between conversations

The multilingual assistant can remember facts about the people who talk to it.
This feature requires `MONITORING_ENABLED=true`. A client that supports voice
identification attributes each turn automatically. Refer to
[Identify Speakers by Voice](#identify-speakers-by-voice). With the browser
client, you attribute sessions manually:

1. Create people on the **People** page of the review UI.
2. Before you connect, pick the person in the **Who's talking** menu in the
   header of the live client. The browser remembers your choice and sends it
   as `person_id` in the session config.
3. At session start, the system prompt names the person and lists up to 30 of
   their `active` memories, most confident first. The prompt block is
   `person_memory_addon` in
   [`src/examples/multilingual/prompts.yaml`](../../src/examples/multilingual/prompts.yaml).
   The session is attributed to the person (source `live:picker`), and the
   memories used are logged in `memory_uses`.
4. After the session ends, the dreamer runs `dream` to extract new memories.
   Review them in the **Memories** activity.

An unknown or archived person, or a database error, never blocks a session.
The session then starts without memories.

Memories can contain personal and health information. They stay in the local
monitoring database. The `dream` job sends the session transcript to the LLM it
calls. With a cloud LLM, either as the session LLM or through `dream.llm`, the
transcript leaves your machine.

## Identify Speakers by Voice

A client that computes speaker embeddings, such as the Reachy Mini conversation
app, can tell the multilingual assistant who speaks in each turn. The server
stores the embedding gallery and applies the result. The browser client does
not send speaker updates, so its sessions behave as before. The wire format is
specified in [Voice ID protocol](../voice-id-protocol.md).

Voice identification requires `MONITORING_ENABLED=true`, because people and
their embeddings live in the monitoring database. It works as follows:

1. The client fetches `GET /api/voice-id/gallery?model=<model key>`. The
   response lists one centroid per non-archived person who has embeddings for
   that model key. The endpoint returns `503` when monitoring is disabled.
2. During a session, the client sends `speaker-update` messages with its
   running estimate. The first valid update activates voice identification for
   the session: the `voice_id_addon` prompt block is added to the system
   prompt and the `enroll_speaker` tool is offered to the LLM.
3. When a user turn is committed, the pipeline attributes it by using the
   latest update received for that turn. Without one, it reuses the previous
   speaker at tier `low`.
4. The user message that the LLM receives starts with a tag such as
   `[speaker: Alice]`, `[speaker: Alice (uncertain)]`, or
   `[speaker: unknown guest]`. The recorded transcript is not tagged.
5. The turn is attributed in `speaker_assignments` with the source
   `live:voice-id:<model key>`, so the `dream` job extracts memories for the
   right person.
6. The `person_memory_addon` block follows the speaker. A confident match
   (tier `high`) loads that person's memories. A different or unknown speaker
   removes them. An uncertain turn of the same person keeps them.

To enroll a new voice, the person tells the assistant their name and agrees
that it remembers their voice. The assistant then calls `enroll_speaker`,
which runs on the server. The tool binds the embeddings of the current speaker
to the person with that name, or creates the person, and sends
`speaker-enrolled` to the client. To enroll a person you created in the review
UI, have them give the same name. The tool refuses to rename a voice that is
confidently recognized as another person.

Per-person policies must use `VoiceIdSession.current_speaker()` in
[`src/examples/shared/voice_id.py`](../../src/examples/shared/voice_id.py).
Its `trusted_person_id` is set only for a confident match. For every other
speaker it is `None`, and guest policies apply. A voice match is not
authentication: a recording of a voice can pass it.

Speaker embeddings are biometric data. They stay in the monitoring database
and are never sent to the LLM. To delete the embeddings of a person, use
**Forget** on their page in the review UI, or call
`DELETE /api/voice-id/people/<person id>/embeddings`. Archiving a person
removes them from the gallery but keeps the embeddings.

A client loads the gallery when it connects, so after a merge it keeps
reporting the merged person's old id until it reconnects. The merge is
recorded as an alias (`person_alias:<old id>` in the `kv` table), and the
server resolves it wherever a live session turns an id into a person: the
speaker is tagged, trusted and attributed as the person you kept.

Voice identification has the following limits:

- A turn of an unknown speaker has no row in `speaker_assignments`. If you
  also picked a person in **Who's talking**, the review UI and the `dream` job
  attribute that turn to the picked person.
- The assistant does not enroll voices on its own. Embeddings are stored only
  through `enroll_speaker`.
- Only the multilingual assistant handles `speaker-update` messages.

### Confirm Speakers by Face

A client with a camera can also recognize faces. Face identification never
replaces the voice: it confirms it and reports who is in view. No image leaves
the client. The client sends only face embeddings, which are stored next to the
voice embeddings under their own model key and served by the same gallery
endpoint.

- **Tier `verified`.** A `speaker-update` can carry the face that the client
  linked to the speaker. When the voice and that face name the same person, the
  tier is `verified`. The server treats `verified` like `high` for memories and
  trust, and records the turn with the source
  `live:voice-id:<model key>:verified`. A policy that needs both factors must
  check `current_speaker().verified`. The server downgrades a `verified` update
  whose face does not name the same person to `high`.
- **Who is in view.** The client sends `presence-update` messages with the
  people it sees. The first face message adds the `face_id_addon` prompt block.
  The speaker tag then lists the other people in view, for example
  `[speaker: Alice; also in view: Bob, unknown guest]`.
- **Greeting.** When a person recognized at face tier `high` comes into view
  for the first time in a session, and nobody speaks and no reply is pending,
  the server adds `[presence: Alice came into view]` to the context and runs
  the LLM once so that the assistant can greet them by name. A person is
  greeted at most once per session. A person who already spoke, or who appeared
  while the conversation stayed busy for 10 seconds, is not greeted.
- **Enrollment.** `enroll_speaker` also binds the face embeddings that the
  client linked to the current speaker. The assistant asks for consent to
  remember both voice and face. Enrollment still works when no face was seen.

Presence alone never loads the memories of a person, never changes
`current_speaker()`, and never grants a policy. There is no liveness check: a
photo of an enrolled person produces a face match. For this reason a face
without a matching voice is never trusted.

## Remote storage later

- `SessionStore` uses SQLAlchemy Core. Upserts are dialect-aware for SQLite and
  PostgreSQL, so moving the database only means changing `MONITORING_DB_URL`.
- `ArtifactStore` is a five-method protocol (`put`, `get`, `exists`,
  `local_path`, `commit`). An S3 or MinIO implementation can stage streamed
  files locally and upload them on `commit`.
- The dreamer decides idleness from the database alone, so it can run on
  another machine as soon as both stores are remote.
