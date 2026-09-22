// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

(() => {
  "use strict";

  const MAX_EVENTS = 240;
  const MAX_ACTIVITIES = 36;
  const MAX_REQUEST_SUMMARIES = 128;
  const MAX_DELEGATIONS = 128;
  const MAX_CONVERSATION_MESSAGES = 200;
  const MAX_USER_ITEM_IDS = MAX_CONVERSATION_MESSAGES;
  const INPUT_SAMPLE_RATE = 24000;
  const PCM16_BYTES_PER_SAMPLE = 2;
  const MIN_LISTENING_SECONDS = 0.2;
  const MIN_CAPTURE_BYTES = INPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * MIN_LISTENING_SECONDS;
  const AUTOMATIC_STOP_TIMEOUT_MS = 8000;
  const INITIAL_PLAYOUT_LEAD_SECONDS = 0.45;
  const MIN_REBUFFER_LEAD_SECONDS = 0.12;
  const PLAYBACK_CURSOR_EPSILON_SECONDS = 0.005;
  const STREAMING_MARKDOWN_RENDER_INTERVAL_MS = 100;
  const MARKDOWN_OPTIONS = Object.freeze({ async: false, breaks: false, gfm: true });
  const MARKDOWN_TAGS = new Set([
    "A", "BLOCKQUOTE", "BR", "CODE", "DEL", "EM", "H1", "H2", "H3", "H4", "H5", "H6", "HR", "LI",
    "OL", "P", "PRE", "STRONG", "TABLE", "TBODY", "TD", "TH", "THEAD", "TR", "UL",
  ]);
  const byId = (id) => document.getElementById(id);
  const elements = {
    authRequirement: byId("auth-requirement"),
    backendBadge: byId("backend-badge"),
    capabilityList: byId("capability-list"),
    clearActivity: byId("clear-activity"),
    clearEvents: byId("clear-events"),
    clientSecret: byId("client-secret"),
    composer: byId("composer"),
    composerHelp: byId("composer-help"),
    connectionDot: byId("connection-dot"),
    connectionForm: byId("connection-form"),
    connectionLabel: byId("connection-label"),
    connectionPanel: byId("connection-panel"),
    connectionToggle: byId("connection-toggle"),
    connectButton: byId("connect-button"),
    conversationLog: byId("conversation-log"),
    diagnostics: byId("diagnostics"),
    delegationCount: byId("delegation-count"),
    delegationAnnouncement: byId("delegation-announcement"),
    delegationEmpty: byId("delegation-empty"),
    delegationList: byId("delegation-list"),
    delegationsNote: byId("delegations-note"),
    emptyConversation: byId("empty-conversation"),
    endpoint: byId("endpoint"),
    eventCount: byId("event-count"),
    eventLog: byId("event-log"),
    inputSignal: byId("input-signal"),
    inputState: byId("input-state"),
    instructions: byId("instructions"),
    messageInput: byId("message-input"),
    modelSignal: byId("model-signal"),
    modelState: byId("model-state"),
    muteButton: byId("mute-button"),
    muteLabel: byId("mute-label"),
    notice: byId("notice"),
    outputSignal: byId("output-signal"),
    outputState: byId("output-state"),
    queueIndicator: byId("speech-queue"),
    queueCount: byId("queue-count"),
    secretNote: byId("secret-note"),
    sendButton: byId("send-button"),
    talkButton: byId("talk-button"),
    talkLabel: byId("talk-label"),
    targetAvatar: byId("target-avatar"),
    targetName: byId("target-name"),
    targetReference: byId("target-reference"),
    targetStatus: byId("target-status"),
    activityList: byId("activity-list"),
  };

  const state = {
    socket: null,
    authMode: "unknown",
    connected: false,
    protocolReady: false,
    gatewayReachable: false,
    frontendTools: new Set(),
    backendCapabilities: new Set(),
    targetProjection: null,
    session: null,
    conversationId: null,
    activeResponseId: null,
    responses: new Map(),
    messageElements: new Map(),
    userItemIds: new Set(),
    recentUserTurns: [],
    activities: [],
    wireEvents: [],
    totalEvents: 0,
    eventRenderTimer: null,
    noticeTimer: null,
    audioContext: null,
    outputGain: null,
    playbackCursor: 0,
    playbackSources: new Set(),
    playbackSourceMetadata: new Map(),
    playbackProgress: new Map(),
    playbackStreams: new Map(),
    playbackScheduleTail: Promise.resolve(),
    playbackEpoch: 0,
    completedPlaybackResponses: new Set(),
    suppressedPlaybackResponses: new Set(),
    muteTruncationReported: new Set(),
    reportedPlaybackReceipts: new Set(),
    speechDeliveryQueueDepth: 0,
    pendingClientEvents: new Map(),
    requestSummaries: new Map(),
    delegations: new Map(),
    latestDelegationId: null,
    sessionId: null,
    micStream: null,
    micSource: null,
    micProcessor: null,
    micSink: null,
    captureRequested: false,
    capturing: false,
    captureBytes: 0,
    automaticTurnDetection: false,
    automaticResponseCreation: false,
    automaticResponseInterruption: false,
    vadSpeechActive: false,
    stopAfterSpeech: false,
    stopListeningTimer: null,
    muted: false,
    inputMode: "idle",
  };

  function defaultEndpoint() {
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${window.location.host}/v1/realtime`;
    }
    return "ws://localhost:7860/v1/realtime";
  }

  function makeId(prefix) {
    const random = window.crypto?.randomUUID?.().replaceAll("-", "") || `${Date.now()}${Math.random()}`.replace(".", "");
    return `${prefix}_${random}`;
  }

  function timeLabel(value = new Date()) {
    const date = value instanceof Date ? value : new Date(value);
    return Number.isNaN(date.getTime())
      ? "now"
      : new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
  }

  function compactId(value) {
    if (typeof value !== "string" || !value) return "—";
    return value.length > 22 ? `${value.slice(0, 12)}…${value.slice(-6)}` : value;
  }

  function titleCase(value) {
    if (typeof value !== "string" || !value) return "Unknown";
    return value.replaceAll(/[._-]+/g, " ").replaceAll(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function normalizeStateClass(value) {
    const normalized = String(value || "neutral").toLowerCase().replaceAll(/[^a-z]+/g, "-");
    const allowed = new Set([
      "active", "attached", "cancelled", "completed", "dispatching", "error", "failed", "gateway-reachable", "locally-queued", "outcome-unknown", "queued", "reachable", "ready", "received", "running", "started", "submitting", "succeeded", "unavailable", "waiting", "waiting-for-response",
    ]);
    return allowed.has(normalized) ? normalized : "neutral";
  }

  function setStatusPill(element, label, stateName = label) {
    element.textContent = label || "Unknown";
    element.className = `status-pill ${normalizeStateClass(stateName)}`;
  }

  function setConnection(mode, detail) {
    state.connected = mode === "connected";
    elements.connectionDot.className = `connection-dot ${mode}`;
    elements.connectionLabel.textContent = detail || titleCase(mode);
    elements.connectButton.textContent = state.connected ? "Disconnect" : mode === "connecting" ? "Connecting…" : "Connect";
    elements.connectButton.classList.toggle("disconnect", state.connected);
    elements.connectButton.disabled = mode === "connecting";
    elements.messageInput.disabled = !state.protocolReady;
    elements.sendButton.disabled = !state.protocolReady;
    updateTalkControl();
    elements.muteButton.disabled = !state.protocolReady;
    elements.endpoint.disabled = state.connected || mode === "connecting";
    elements.clientSecret.disabled = state.authMode === "none" || state.connected || mode === "connecting";
    elements.instructions.disabled = state.connected || mode === "connecting";
  }

  function setSignal(kind, label, active = false) {
    const signal = elements[`${kind}Signal`];
    const text = elements[`${kind}State`];
    signal.dataset.state = active ? "active" : "idle";
    text.textContent = label;
  }

  function applyNegotiatedTurnDetection(session) {
    const turnDetection = session?.audio?.input?.turn_detection;
    state.automaticTurnDetection = turnDetection !== null && typeof turnDetection === "object";
    state.automaticResponseCreation = state.automaticTurnDetection && turnDetection.create_response !== false;
    state.automaticResponseInterruption = state.automaticTurnDetection && turnDetection.interrupt_response !== false;
  }

  function updateQueue() {
    const waiting = Math.max(0, state.speechDeliveryQueueDepth);
    elements.queueCount.textContent = String(waiting);
    elements.queueCount.setAttribute("aria-label", `${waiting} voice ${waiting === 1 ? "reply" : "replies"} queued`);
    elements.queueIndicator?.classList.toggle("empty", waiting === 0);
  }

  function updateListeningHelp() {
    elements.composerHelp.textContent = state.automaticTurnDetection
      ? "Click once to keep listening; VoiceClaw detects each turn automatically · Click Stop listening when finished · Mute controls playback only · Enter sends text"
      : "Click once to record a voice turn, then click Stop & send · Mute controls playback only · Enter sends text";
  }

  function updateTalkControl() {
    const finishing = state.inputMode === "finishing";
    const transcribing = state.inputMode === "transcribing";
    elements.talkButton.disabled = !state.protocolReady || finishing || transcribing;
    elements.talkButton.classList.toggle("recording", state.capturing);
    elements.talkButton.setAttribute("aria-pressed", String(state.capturing));
    if (finishing) {
      elements.talkButton.setAttribute("aria-label", "Finishing the current voice turn");
      elements.talkLabel.textContent = "Finish speaking…";
    } else if (transcribing) {
      elements.talkButton.setAttribute("aria-label", "Transcribing the voice turn");
      elements.talkLabel.textContent = "Transcribing…";
    } else if (state.capturing) {
      const label = state.automaticTurnDetection ? "Stop listening" : "Stop listening and send voice turn";
      elements.talkButton.setAttribute("aria-label", label);
      elements.talkLabel.textContent = state.automaticTurnDetection ? "Stop listening" : "Stop & send";
    } else {
      elements.talkButton.setAttribute("aria-label", "Start listening");
      elements.talkLabel.textContent = "Start listening";
    }
  }

  function showNotice(message, kind = "warning", timeout = 7000) {
    window.clearTimeout(state.noticeTimer);
    elements.notice.textContent = message;
    elements.notice.className = `notice${kind === "error" ? " error" : ""}`;
    elements.notice.hidden = false;
    if (timeout > 0) {
      state.noticeTimer = window.setTimeout(() => {
        elements.notice.hidden = true;
      }, timeout);
    }
  }

  function clearNotice() {
    window.clearTimeout(state.noticeTimer);
    elements.notice.hidden = true;
  }

  function addActivity(label, detail = "", result = "neutral", timestamp = new Date()) {
    state.activities.unshift({ label, detail, result, timestamp });
    state.activities.splice(MAX_ACTIVITIES);
    renderActivities();
  }

  function renderActivities() {
    elements.activityList.replaceChildren();
    if (!state.activities.length) {
      const empty = document.createElement("li");
      empty.className = "activity-empty";
      empty.textContent = "Session events will appear here.";
      elements.activityList.append(empty);
      return;
    }
    for (const activity of state.activities) {
      const item = document.createElement("li");
      if (["success", "completed", "succeeded", "reachable", "ready", "attached"].includes(activity.result)) item.classList.add("success");
      if (["failure", "failed", "error", "cancelled"].includes(activity.result)) item.classList.add("failure");
      const label = document.createElement("strong");
      label.textContent = activity.label;
      const detail = document.createElement("span");
      detail.textContent = [activity.detail, timeLabel(activity.timestamp)].filter(Boolean).join(" · ");
      item.append(label, detail);
      elements.activityList.append(item);
    }
  }

  function redact(value, key = "") {
    if (/secret|token|authorization|credential|password|api[_-]?key|grant/i.test(key)) return "[redacted]";
    if (Array.isArray(value)) return value.map((entry) => redact(entry));
    if (value && typeof value === "object") {
      const clean = {};
      for (const [childKey, childValue] of Object.entries(value)) clean[childKey] = redact(childValue, childKey);
      return clean;
    }
    if (typeof value === "string") {
      return value
        .replaceAll(/ek_[A-Za-z0-9._-]+/g, "ek_[redacted]")
        .replaceAll(/Bearer\s+[^\s,}\]]+/gi, "Bearer [redacted]");
    }
    return value;
  }

  function diagnosticsPayload(payload) {
    let bounded = payload;
    if (payload?.type === "input_audio_buffer.append" && typeof payload.audio === "string") {
      bounded = { ...payload, audio: `[PCM16 audio omitted · ${payload.audio.length} base64 chars]` };
    }
    if (payload?.type === "response.output_audio.delta" && typeof payload.delta === "string") {
      bounded = { ...payload, delta: `[PCM16 audio omitted · ${payload.delta.length} base64 chars]` };
    }
    return redact(bounded);
  }

  function logWire(direction, payload) {
    state.totalEvents += 1;
    state.wireEvents.unshift({ direction, payload: diagnosticsPayload(payload), timestamp: new Date() });
    state.wireEvents.splice(MAX_EVENTS);
    elements.eventCount.textContent = String(state.totalEvents);
    if (elements.diagnostics.open && state.eventRenderTimer === null) {
      state.eventRenderTimer = window.setTimeout(() => {
        state.eventRenderTimer = null;
        renderWireEvents();
      }, 120);
    }
  }

  function renderWireEvents() {
    elements.eventCount.textContent = String(state.totalEvents);
    elements.eventLog.replaceChildren();
    for (const entry of state.wireEvents) {
      const item = document.createElement("li");
      const direction = document.createElement("span");
      direction.className = `event-direction ${entry.direction === "outbound" ? "outbound" : ""}`;
      direction.textContent = entry.direction === "outbound" ? "↑" : "↓";
      const main = document.createElement("div");
      main.className = "event-main";
      const type = document.createElement("strong");
      type.textContent = entry.payload?.type || "websocket.event";
      const raw = document.createElement("pre");
      raw.textContent = JSON.stringify(entry.payload, null, 2);
      main.append(type, raw);
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = timeLabel(entry.timestamp);
      item.append(direction, main, time);
      elements.eventLog.append(item);
    }
  }

  function sendEvent(payload, pending = null) {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
      showNotice("The Realtime connection is not open.", "error");
      return null;
    }
    const event = { ...payload };
    if (!event.event_id) event.event_id = makeId("event");
    try {
      state.socket.send(JSON.stringify(event));
    } catch (error) {
      showNotice(`The Realtime event could not be sent: ${error.message}`, "error", 0);
      return null;
    }
    if (pending) {
      while (state.pendingClientEvents.size >= MAX_EVENTS) {
        state.pendingClientEvents.delete(state.pendingClientEvents.keys().next().value);
      }
      state.pendingClientEvents.set(event.event_id, pending);
    }
    logWire("outbound", event);
    return event.event_id;
  }

  function forgetFirstPending(kind, predicate = () => true) {
    for (const [eventId, pending] of state.pendingClientEvents) {
      if (pending.kind === kind && predicate(pending)) {
        state.pendingClientEvents.delete(eventId);
        return pending;
      }
    }
    return null;
  }

  function forgetPending(kind, predicate = () => true) {
    for (const [eventId, pending] of state.pendingClientEvents) {
      if (pending.kind === kind && predicate(pending)) state.pendingClientEvents.delete(eventId);
    }
  }

  async function prepareAudioContext() {
    if (!state.audioContext || state.audioContext.state === "closed") {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error("This browser does not provide the Web Audio API.");
      state.audioContext = new AudioContextClass({ latencyHint: "interactive" });
      state.outputGain = state.audioContext.createGain();
      state.outputGain.gain.value = state.muted ? 0 : 1;
      state.outputGain.connect(state.audioContext.destination);
    }
    if (state.audioContext.state === "suspended") await state.audioContext.resume();
    state.playbackCursor = Math.max(state.playbackCursor, state.audioContext.currentTime);
  }

  async function loadDeploymentPolicy() {
    try {
      const response = await fetch("/health", { cache: "no-store", credentials: "same-origin" });
      if (!response.ok) throw new Error("health request failed");
      const health = await response.json();
      const mode = health?.authentication?.mode;
      if (mode !== "none" && mode !== "ephemeral") throw new Error("unknown authentication mode");
      state.authMode = mode;
    } catch {
      state.authMode = "unknown";
    }

    if (state.authMode === "none") {
      elements.clientSecret.value = "";
      elements.clientSecret.placeholder = "Not required";
      elements.authRequirement.textContent = "disabled for this deployment";
      elements.secretNote.textContent = "This deployment accepts the Realtime connection without a browser key.";
    } else if (state.authMode === "ephemeral") {
      elements.clientSecret.placeholder = "ek_…";
      elements.authRequirement.textContent = "required";
      const secretPrefix = document.createElement("code");
      secretPrefix.textContent = "ek_";
      elements.secretNote.replaceChildren(
        document.createTextNode("Paste a short-lived "),
        secretPrefix,
        document.createTextNode(
          " client secret. It is kept in memory only, removed from this form on connect, and redacted from diagnostics.",
        ),
      );
    } else {
      elements.authRequirement.textContent = "deployment policy unavailable";
      elements.secretNote.textContent = "If this deployment enforces authentication, provide its short-lived ek_ client secret.";
    }
    elements.clientSecret.disabled = state.authMode === "none" || state.connected;
  }

  function connect() {
    if (state.connected) {
      disconnect(1000, "user disconnected");
      return;
    }
    if (state.socket && state.socket.readyState === WebSocket.CONNECTING) return;

    const endpoint = defaultEndpoint();
    elements.endpoint.value = endpoint;

    const secret = elements.clientSecret.value.trim();
    if (state.authMode === "ephemeral" && !secret) {
      showNotice("This deployment requires a short-lived browser client secret.", "error");
      elements.clientSecret.focus();
      return;
    }
    if (state.authMode === "none" && secret) {
      showNotice("Browser authentication is disabled for this deployment; leave the client secret empty.", "error");
      return;
    }
    if (secret && !secret.startsWith("ek_")) {
      showNotice("Browser authentication requires a short-lived client secret beginning with ek_.", "error");
      elements.clientSecret.focus();
      return;
    }

    clearNotice();
    resetSessionView();
    setConnection("connecting", "Connecting");
    void prepareAudioContext().catch(() => undefined);

    try {
      const protocols = secret ? ["realtime", `openai-insecure-api-key.${secret}`] : ["realtime"];
      state.socket = new WebSocket(endpoint, protocols);
      elements.clientSecret.value = "";
    } catch (error) {
      elements.clientSecret.value = "";
      setConnection("disconnected", "Disconnected");
      showNotice(`Could not open the Realtime socket: ${error.message}`, "error", 0);
      return;
    }

    const socket = state.socket;
    socket.addEventListener("open", () => {
      if (state.socket !== socket) return;
      setConnection("connected", "Transport connected");
      addActivity("Realtime transport connected", "Waiting for session", "success");
    });
    socket.addEventListener("message", (message) => {
      if (state.socket !== socket) return;
      handleSocketMessage(message);
    });
    socket.addEventListener("error", () => {
      if (state.socket !== socket) return;
      showNotice("The Realtime WebSocket reported a connection error. Open diagnostics for protocol details.", "error", 0);
    });
    socket.addEventListener("close", (event) => {
      if (state.socket !== socket) return;
      const wasReady = state.protocolReady;
      cleanupConnection();
      const description = event.reason ? `${event.code} · ${event.reason}` : String(event.code);
      setConnection("disconnected", "Disconnected");
      addActivity("Realtime transport closed", description, event.code === 1000 ? "neutral" : "failure");
      if (event.code !== 1000 || !wasReady) showNotice(`Socket closed (${description}).`, "error", 0);
    });
  }

  function disconnect(code = 1000, reason = "normal closure") {
    if (state.socket && state.socket.readyState < WebSocket.CLOSING) state.socket.close(code, reason);
    cleanupConnection();
    setConnection("disconnected", "Disconnected");
  }

  function cleanupConnection() {
    state.connected = false;
    state.protocolReady = false;
    state.gatewayReachable = false;
    state.frontendTools.clear();
    state.backendCapabilities.clear();
    state.targetProjection = null;
    state.session = null;
    state.conversationId = null;
    state.activeResponseId = null;
    cancelProjectionRenders();
    state.responses.clear();
    state.speechDeliveryQueueDepth = 0;
    state.pendingClientEvents.clear();
    state.requestSummaries.clear();
    state.completedPlaybackResponses.clear();
    state.suppressedPlaybackResponses.clear();
    state.muteTruncationReported.clear();
    state.reportedPlaybackReceipts.clear();
    state.automaticTurnDetection = false;
    state.automaticResponseCreation = false;
    state.automaticResponseInterruption = false;
    state.vadSpeechActive = false;
    state.stopAfterSpeech = false;
    state.inputMode = "idle";
    state.socket = null;
    state.sessionId = null;
    stopPlayback();
    releaseMicrophone();
    setSignal("input", "Ready");
    setSignal("model", "Idle");
    setSignal("output", state.muted ? "Muted" : "Idle");
    for (const record of state.delegations.values()) {
      if (!["locally_queued", "dispatching", "waiting_for_response", "running", "queued"].includes(record.status)) {
        continue;
      }
      record.status = "outcome_unknown";
      record.updatedAt = new Date();
      recordDelegationPhase(record, "outcome_unknown", "Connection closed · outcome unavailable");
    }
    renderDelegations();
    resetDisconnectedView();
  }

  function resetDisconnectedView() {
    elements.backendBadge.className = "backend-badge";
    elements.backendBadge.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    elements.backendBadge.append(dot, document.createTextNode("Backend not attached"));
    elements.targetAvatar.textContent = "A";
    elements.targetName.textContent = "No active runtime attachment";
    elements.targetReference.textContent = "Reconnect to load backend capabilities";
    setStatusPill(elements.targetStatus, "Disconnected", "neutral");
    elements.capabilityList.replaceChildren();
    const placeholder = document.createElement("span");
    placeholder.className = "capability muted";
    placeholder.textContent = "Backend capabilities unavailable while disconnected";
    elements.capabilityList.append(placeholder);
    updateListeningHelp();
    updateTalkControl();
    updateQueue();
  }

  function resetSessionView() {
    state.protocolReady = false;
    state.gatewayReachable = false;
    state.frontendTools.clear();
    state.backendCapabilities.clear();
    state.targetProjection = null;
    state.session = null;
    state.sessionId = null;
    state.conversationId = null;
    cancelProjectionRenders();
    state.responses.clear();
    state.messageElements.clear();
    state.userItemIds.clear();
    state.recentUserTurns = [];
    state.speechDeliveryQueueDepth = 0;
    state.pendingClientEvents.clear();
    state.requestSummaries.clear();
    state.delegations.clear();
    state.latestDelegationId = null;
    state.completedPlaybackResponses.clear();
    state.suppressedPlaybackResponses.clear();
    state.muteTruncationReported.clear();
    state.reportedPlaybackReceipts.clear();
    state.automaticTurnDetection = false;
    state.automaticResponseCreation = false;
    state.automaticResponseInterruption = false;
    state.vadSpeechActive = false;
    state.stopAfterSpeech = false;
    state.inputMode = "idle";
    elements.conversationLog.replaceChildren(elements.emptyConversation);
    elements.emptyConversation.hidden = false;
    elements.backendBadge.className = "backend-badge";
    elements.backendBadge.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    elements.backendBadge.append(dot, document.createTextNode("Checking backend gateway"));
    elements.targetAvatar.textContent = "A";
    elements.targetName.textContent = "Checking backend gateway";
    elements.targetReference.textContent = "No durable attachment advertised";
    setStatusPill(elements.targetStatus, "Not checked", "neutral");
    elements.capabilityList.replaceChildren();
    const placeholder = document.createElement("span");
    placeholder.className = "capability muted";
    placeholder.textContent = "Waiting for server-projected route operations";
    elements.capabilityList.append(placeholder);
    updateListeningHelp();
    updateTalkControl();
    renderDelegations();
    updateQueue();
  }

  function handleSocketMessage(message) {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      logWire("inbound", { type: "invalid_json", body: String(message.data).slice(0, 300) });
      showNotice("The server sent an invalid JSON event.", "error");
      return;
    }
    logWire("inbound", event);
    handleServerEvent(event);
  }

  function handleServerEvent(event) {
    switch (event.type) {
      case "session.created":
        onSessionCreated(event.session || {});
        break;
      case "session.updated":
        forgetFirstPending("session_update");
        state.session = { ...(state.session || {}), ...(event.session || {}) };
        if (!usesPcm24(state.session?.audio?.input?.format) || !usesPcm24(state.session?.audio?.output?.format)) {
          state.protocolReady = false;
          setConnection("connected", "Unsupported audio format");
          showNotice("This UI requires negotiated 24 kHz PCM input and output audio.", "error", 0);
          break;
        }
        applyNegotiatedTurnDetection(state.session);
        state.protocolReady = true;
        setConnection("connected", "Session ready");
        updateListeningHelp();
        updateTalkControl();
        break;
      case "conversation.created":
        state.conversationId = event.conversation?.id || null;
        break;
      case "input_audio_buffer.speech_started":
        state.vadSpeechActive = true;
        if (state.automaticResponseInterruption) {
          if (state.activeResponseId) state.suppressedPlaybackResponses.add(state.activeResponseId);
          dispatchPlaybackTruncations(stopPlayback(true, true));
        }
        setSignal("input", "Listening", true);
        break;
      case "input_audio_buffer.speech_stopped":
        state.vadSpeechActive = false;
        setSignal("input", "Transcribing", true);
        finishAutomaticStop();
        break;
      case "input_audio_buffer.committed":
        forgetFirstPending("audio_commit");
        setSignal("input", "Transcribing", true);
        finishAutomaticStop();
        if (state.automaticTurnDetection && !state.automaticResponseCreation) {
          const requested = sendEvent(
            { type: "response.create", response: {} },
            { kind: "response_create" },
          );
          if (requested) setSignal("model", "Waiting for transcript", true);
        }
        break;
      case "conversation.item.input_audio_transcription.delta":
        updateUserTranscript(event.item_id, event.delta || "", false);
        break;
      case "conversation.item.input_audio_transcription.completed":
        completeUserTranscript(event.item_id, event.transcript || "");
        if (!state.capturing) state.inputMode = "idle";
        setSignal("input", state.capturing ? "Listening" : "Ready", state.capturing);
        updateTalkControl();
        break;
      case "conversation.item.input_audio_transcription.failed":
        if (!state.capturing) state.inputMode = "idle";
        setSignal("input", state.capturing ? "Listening" : "Ready", state.capturing);
        showNotice(event.error?.message || "Input audio transcription failed.", "error");
        updateTalkControl();
        break;
      case "conversation.item.added":
      case "conversation.item.created":
      case "conversation.item.done":
        if (event.item?.id) forgetFirstPending("user_item", (pending) => pending.itemId === event.item.id);
        handleConversationItem(event.item, event.response_id);
        break;
      case "conversation.item.truncated":
        forgetFirstPending("playback_truncate", (pending) => pending.itemId === event.item_id);
        break;
      case "response.created":
        beginResponse(event.response || {});
        break;
      case "response.output_item.added":
        bindOutputItem(event.response_id, event.item);
        break;
      case "response.output_text.delta":
      case "response.output_audio_transcript.delta":
        appendResponseText(event.response_id, event.delta || "", event.type);
        break;
      case "response.output_text.done":
      case "response.output_audio_transcript.done":
        setResponseText(event.response_id, event.text ?? event.transcript ?? "", event.type);
        break;
      case "response.output_audio.delta":
        if (event.delta) enqueueAudioDelta(event);
        break;
      case "response.output_audio.done":
        markPlaybackStreamDone(event);
        break;
      case "response.function_call_arguments.delta":
        setSignal("model", "Preparing delegation", true);
        break;
      case "response.function_call_arguments.done":
        setSignal("model", "Waiting for backend", true);
        break;
      case "response.done":
        finishResponse(event.response || {});
        break;
      case "error":
        handleProtocolError(event.error || {});
        break;
      case "input_audio_buffer.cleared":
      case "response.content_part.added":
      case "response.content_part.done":
      case "response.output_item.done":
      case "rate_limits.updated":
        break;
      default:
        break;
    }
  }

  function onSessionCreated(session) {
    state.session = session;
    state.sessionId = typeof session.id === "string" ? session.id : null;
    state.protocolReady = false;
    setConnection("connected", "Applying session settings");
    elements.targetName.textContent = "Checking backend gateway";
    elements.targetReference.textContent = "No durable attachment advertised";
    setStatusPill(elements.targetStatus, "Checking gateway", "waiting");
    applyNegotiatedTurnDetection(session);
    updateListeningHelp();

    const patch = {
      type: "realtime",
      output_modalities: ["audio"],
      audio: {
        input: {
          format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE },
        },
        output: { format: { type: "audio/pcm", rate: INPUT_SAMPLE_RATE } },
      },
    };
    const instructions = elements.instructions.value.trim();
    if (instructions) patch.instructions = instructions;
    sendEvent({ type: "session.update", session: patch }, { kind: "session_update" });
  }

  function handleProtocolError(error) {
    const message = error.message || error.code || "Realtime request failed";
    const pending = typeof error.event_id === "string"
      ? state.pendingClientEvents.get(error.event_id)
      : null;
    if (pending) state.pendingClientEvents.delete(error.event_id);
    const benignTerminalCancel = pending?.kind === "response_cancel"
      && error.code === "invalid_request"
      && /response id is not owned by this session|already (?:completed|terminal)|no active response/i.test(message);
    if (benignTerminalCancel) {
      if (state.activeResponseId === pending.responseId) state.activeResponseId = null;
      if (!state.activeResponseId) setSignal("model", "Idle");
      updateQueue();
      return;
    }
    showNotice(message, "error", 0);
    addActivity("Realtime request failed", message, "failure");
    if (pending?.kind === "user_item") {
      removeMessage(pending.itemId);
      state.userItemIds.delete(pending.itemId);
    } else if (pending?.kind === "response_create") {
      if (!state.activeResponseId) setSignal("model", "Idle");
      updateQueue();
    } else if (pending?.kind === "audio_commit") {
      state.inputMode = "idle";
      setSignal("input", "Ready");
      if (!state.activeResponseId) setSignal("model", "Idle");
      updateTalkControl();
    } else if (pending?.kind === "session_update") {
      state.protocolReady = false;
      setConnection("connected", "Session configuration failed");
    }
    if (error.code === "unsupported_capability" && error.param?.includes("turn_detection")) {
      showNotice("This frontend profile does not support click-to-record audio turns. Typed turns remain available.", "error", 0);
      elements.talkButton.disabled = true;
    }
    updateQueue();
  }

  function createMessage(role, id, text = "", streaming = false) {
    elements.emptyConversation.hidden = true;
    const wrapper = document.createElement("article");
    wrapper.className = `message ${role}${streaming ? " streaming" : ""}`;
    wrapper.dataset.messageId = id;
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = `${role === "user" ? "You" : "VoiceClaw"} · ${streaming ? "streaming" : timeLabel()}`;
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    wrapper.append(meta, bubble);
    elements.conversationLog.append(wrapper);
    state.messageElements.set(id, wrapper);
    pruneConversationMessages();
    scrollConversation();
    return wrapper;
  }

  function updateMessage(id, role, text, streaming) {
    const wrapper = state.messageElements.get(id) || createMessage(role, id, "", streaming);
    wrapper.classList.toggle("streaming", Boolean(streaming));
    const bubble = wrapper.querySelector(".message-bubble");
    const meta = wrapper.querySelector(".message-meta");
    bubble.textContent = text;
    meta.textContent = `${role === "user" ? "You" : "VoiceClaw"} · ${streaming ? "streaming" : timeLabel()}`;
    scrollConversation();
    return wrapper;
  }

  function removeMessage(id) {
    state.messageElements.get(id)?.remove();
    state.messageElements.delete(id);
    if (!state.messageElements.size) elements.emptyConversation.hidden = false;
  }

  function pruneConversationMessages() {
    while (state.messageElements.size > MAX_CONVERSATION_MESSAGES) {
      let retired = false;
      for (const [id, element] of state.messageElements) {
        if (id === state.activeResponseId || state.userItemIds.has(id)) continue;
        element.remove();
        state.messageElements.delete(id);
        retired = true;
        break;
      }
      if (!retired) break;
    }
  }

  function rememberUserItemId(itemId) {
    state.userItemIds.add(itemId);
    while (state.userItemIds.size > MAX_USER_ITEM_IDS) {
      state.userItemIds.delete(state.userItemIds.values().next().value);
    }
  }

  function scrollConversation() {
    window.requestAnimationFrame(() => {
      elements.conversationLog.scrollTop = elements.conversationLog.scrollHeight;
    });
  }

  function updateUserTranscript(itemId, delta, completed) {
    const id = itemId || "pending-audio-turn";
    const element = state.messageElements.get(id);
    const existing = element?.querySelector(".message-bubble")?.textContent || "";
    updateMessage(id, "user", `${existing}${delta}`, !completed);
  }

  function normalizeTranscriptWhitespace(value) {
    return typeof value === "string" ? value.replaceAll(/\s+/g, " ").trim() : "";
  }

  function completeUserTranscript(itemId, transcript) {
    const id = itemId || "pending-audio-turn";
    const existing = state.messageElements.get(id)?.querySelector(".message-bubble")?.textContent || "";
    const finalized = normalizeTranscriptWhitespace(transcript) || normalizeTranscriptWhitespace(existing);
    updateMessage(id, "user", finalized || "[Voice turn]", false);
    rememberUserTurn(id, finalized);
  }

  function textFromItem(item) {
    if (!item || !Array.isArray(item.content)) return "";
    return item.content
      .map((part) => part?.text ?? part?.transcript ?? "")
      .filter((part) => typeof part === "string")
      .join("");
  }

  function spokenTextFromItem(item) {
    if (!item || !Array.isArray(item.content)) return "";
    return item.content
      .filter((part) => part?.type === "output_audio" || part?.type === "audio")
      .map((part) => part?.transcript ?? "")
      .filter((part) => typeof part === "string")
      .join("");
  }

  function handleConversationItem(item, responseId) {
    if (!item || !item.id) return;
    if (item.type === "function_call") {
      setSignal("model", "Waiting for backend", true);
      return;
    }
    if (item.type === "function_call_output") {
      // Protected tool outputs are transport receipts, not backend results.
      // Their authoritative lifecycle is rendered from server projections.
      return;
    }
    if (item.type !== "message") return;
    const text = textFromItem(item);
    if (item.role === "user") {
      if (state.userItemIds.has(item.id)) {
        state.userItemIds.delete(item.id);
        pruneConversationMessages();
        if (state.messageElements.has(item.id)) return;
      }
      const hasAudio = Array.isArray(item.content) && item.content.some((part) => part?.type === "input_audio");
      const displayText = hasAudio && item.status === "completed" ? normalizeTranscriptWhitespace(text) : text;
      if (displayText) updateMessage(item.id, "user", displayText, item.status !== "completed");
      return;
    }
    if (item.role !== "assistant") return;
    const spokenText = spokenTextFromItem(item);
    if (!spokenText) return;
    // Conversation item events omit response_id, so correlate the item that
    // response.output_item.added already bound instead of rendering it twice.
    const track = responseFor(responseId, false) || responseForItem(item.id);
    if (track?.projection) return;
    const id = track?.id || responseId || `assistant-${item.id}`;
    if (!state.messageElements.has(id)) updateMessage(id, "assistant", spokenText, item.status !== "completed");
  }

  function responseFor(responseId, create = true) {
    const id = responseId || state.activeResponseId;
    if (!id) return null;
    let track = state.responses.get(id);
    if (!track && create) {
      track = {
        id,
        text: "",
        textStreams: { outputText: "", audioTranscript: "" },
        metadata: null,
        projection: null,
        ignoredProjection: false,
        projectionRenderFrame: null,
        projectionRenderTimer: null,
        projectionLastRenderAt: 0,
        itemIds: new Set(),
      };
      state.responses.set(id, track);
    }
    return track;
  }

  function responseForItem(itemId) {
    if (!itemId) return null;
    return Array.from(state.responses.values()).find((track) => track.itemIds.has(itemId)) || null;
  }

  function beginResponse(response) {
    const id = response.id || makeId("resp");
    const projection = projectionFromMetadata(response.metadata);
    const ignoredProjection = projection !== null && !projectionBelongsToActiveSession(projection);
    state.responses.set(id, {
      id,
      text: "",
      textStreams: { outputText: "", audioTranscript: "" },
      metadata: response.metadata || null,
      projection,
      ignoredProjection,
      projectionRenderFrame: null,
      projectionRenderTimer: null,
      projectionLastRenderAt: 0,
      itemIds: new Set(),
    });
    if (!projection && !ignoredProjection) {
      forgetFirstPending("response_create");
      state.activeResponseId = id;
      setSignal("model", "Thinking", true);
    }
  }

  function bindOutputItem(responseId, item) {
    const track = responseFor(responseId);
    if (track && item?.id) {
      track.itemIds.add(item.id);
      bindConversationMessage(track.id, item.id);
    }
    if (item?.type === "function_call") {
      setSignal("model", "Preparing delegation", true);
    }
  }

  function bindConversationMessage(responseId, itemId) {
    const fallbackId = `assistant-${itemId}`;
    const fallback = state.messageElements.get(fallbackId);
    if (!fallback) return;
    const responseMessage = state.messageElements.get(responseId);
    if (responseMessage) fallback.remove();
    else {
      fallback.dataset.messageId = responseId;
      state.messageElements.set(responseId, fallback);
    }
    state.messageElements.delete(fallbackId);
  }

  function responseTextStream(eventType) {
    return eventType.includes("audio_transcript") ? "audioTranscript" : "outputText";
  }

  function selectResponseText(track) {
    return track.textStreams.outputText || track.textStreams.audioTranscript;
  }

  function projectionCarriesDisplay(projection) {
    const kind = String(firstValue(projection?.kind, ""));
    const phase = String(firstValue(projection?.phase, projection?.state, ""));
    return (kind === "result_display" && phase !== "discarded")
      || kind === "work_result";
  }

  function cancelProjectionRender(track) {
    if (!track) return;
    if (track.projectionRenderTimer !== null && track.projectionRenderTimer !== undefined) {
      window.clearTimeout(track.projectionRenderTimer);
      track.projectionRenderTimer = null;
    }
    if (track.projectionRenderFrame !== null && track.projectionRenderFrame !== undefined) {
      window.cancelAnimationFrame(track.projectionRenderFrame);
      track.projectionRenderFrame = null;
    }
  }

  function cancelProjectionRenders() {
    for (const track of state.responses.values()) cancelProjectionRender(track);
  }

  function scheduleProjectionRender(track) {
    if (
      track.ignoredProjection
      || !track.projection
      || !projectionCarriesDisplay(track.projection)
      || track.projectionRenderFrame !== null
      || track.projectionRenderTimer !== null
    ) return;
    const now = window.performance?.now?.() ?? Date.now();
    const delay = Math.max(
      0,
      STREAMING_MARKDOWN_RENDER_INTERVAL_MS - (now - track.projectionLastRenderAt),
    );
    const requestRenderFrame = () => {
      track.projectionRenderTimer = null;
      track.projectionRenderFrame = window.requestAnimationFrame(() => {
        track.projectionRenderFrame = null;
        if (state.responses.get(track.id) !== track || !track.text) return;
        track.projectionLastRenderAt = window.performance?.now?.() ?? Date.now();
        const result = { ...objectValue(track.projection.result ?? track.projection.delivery) };
        result.state = firstValue(result.state, track.projection.phase, "received");
        result.title = firstValue(result.title, track.projection.title, "Response");
        result.display = track.text;
        updateResult(result, track.projection, track.text, { streaming: true });
      });
    };
    if (delay === 0) requestRenderFrame();
    else track.projectionRenderTimer = window.setTimeout(requestRenderFrame, delay);
  }

  function appendResponseText(responseId, delta, eventType) {
    if (!delta) return;
    const track = responseFor(responseId);
    if (!track) return;
    const stream = responseTextStream(eventType);
    track.textStreams[stream] += delta;
    track.text = selectResponseText(track);
    if (track.projection && stream === "outputText") {
      scheduleProjectionRender(track);
    } else if (!track.projection && stream === "audioTranscript") {
      updateMessage(track.id, "assistant", track.textStreams.audioTranscript, true);
    }
  }

  function setResponseText(responseId, text, eventType) {
    if (typeof text !== "string" || !text) return;
    const track = responseFor(responseId);
    if (!track) return;
    track.textStreams[responseTextStream(eventType)] = text;
    track.text = selectResponseText(track);
    if (track.projection && responseTextStream(eventType) === "outputText") {
      scheduleProjectionRender(track);
    } else if (!track.projection && responseTextStream(eventType) === "audioTranscript") {
      updateMessage(track.id, "assistant", track.textStreams.audioTranscript, true);
    }
  }

  function textFromResponse(response) {
    if (!Array.isArray(response?.output)) return "";
    return response.output.map(textFromItem).join("");
  }

  function spokenTextFromResponse(response) {
    if (!Array.isArray(response?.output)) return "";
    return response.output.map(spokenTextFromItem).join("");
  }

  function finishResponse(response) {
    const track = responseFor(response.id);
    if (!track) return;
    cancelProjectionRender(track);
    forgetPending("response_cancel", (pending) => pending.responseId === track.id);
    track.metadata = response.metadata || track.metadata;
    const completedProjection = projectionFromMetadata(track.metadata);
    if (completedProjection) {
      track.projection = completedProjection;
      track.ignoredProjection = !projectionBelongsToActiveSession(completedProjection);
    }
    track.text = track.text || textFromResponse(response);
    if (track.ignoredProjection) {
      const completedActiveResponse = state.activeResponseId === track.id;
      if (completedActiveResponse) state.activeResponseId = null;
      completePlaybackResponse(track.id);
      removeMessage(track.id);
      state.responses.delete(track.id);
      if (completedActiveResponse) setSignal("model", "Idle");
      return;
    }
    const status = response.status || "completed";
    if (track.projection) {
      removeMessage(track.id);
      applyProjection(track.projection, track.text);
    } else {
      const spokenText = track.textStreams.audioTranscript || spokenTextFromResponse(response);
      if (spokenText) updateMessage(track.id, "assistant", spokenText, false);
      else removeMessage(track.id);
    }

    if (!track.projection) {
      if (status === "failed") {
        const errorCode = response.status_details?.error?.code;
        const emptyAudio = errorCode === "input_audio_transcription_empty";
        const message = emptyAudio
          ? "No speech was detected. Please try again."
          : response.status_details?.error?.message || "The frontend model failed to generate this response.";
        showNotice(message, emptyAudio ? "neutral" : "error");
        addActivity(emptyAudio ? "Empty audio turn ignored" : "Response failed", message, emptyAudio ? "neutral" : "failure");
      } else if (status === "cancelled") {
        // Barge-in is normal voice behavior; retain the wire event in diagnostics
        // without adding noise to the user-facing lifecycle.
      }
    }

    const completedActiveResponse = state.activeResponseId === track.id;
    if (completedActiveResponse) state.activeResponseId = null;
    completePlaybackResponse(track.id);
    state.responses.delete(track.id);
    if (!track.projection && completedActiveResponse) setSignal("model", "Idle");
  }

  function parseMetadataValue(value) {
    if (typeof value !== "string") return value;
    const trimmed = value.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      try {
        return JSON.parse(trimmed);
      } catch {
        return value;
      }
    }
    if (trimmed === "true") return true;
    if (trimmed === "false") return false;
    if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
    return value;
  }

  function projectionFromMetadata(metadata) {
    if (!metadata || typeof metadata !== "object") return null;
    if (metadata.voiceclaw_schema !== "voiceclaw.projection.v1") return null;
    const projection = {};
    for (const [rawKey, rawValue] of Object.entries(metadata)) {
      if (rawKey === "voiceclaw_schema" || !rawKey.startsWith("voiceclaw_")) continue;
      const key = rawKey.slice("voiceclaw_".length);
      projection[key] = key === "request_summary" ? rawValue : parseMetadataValue(rawValue);
    }
    return projection;
  }

  function projectionBelongsToActiveSession(projection) {
    const activeSessionId = state.sessionId;
    if (!activeSessionId) return true;
    return projection.session_id === activeSessionId;
  }

  function firstValue(...values) {
    return values.find((value) => value !== undefined && value !== null && value !== "");
  }

  function objectValue(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function arrayValue(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === "string") return value.split(",").map((entry) => entry.trim()).filter(Boolean);
    return [];
  }

  function appendPlainCodeBlock(parent, value, language = "") {
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = value;
    if (language) code.className = `language-${language}`;
    pre.append(code);
    parent.append(pre);
  }

  function sanitizeMarkedHtml(html) {
    const template = document.createElement("template");
    template.innerHTML = html;
    for (const element of [...template.content.querySelectorAll("*")]) {
      if (!MARKDOWN_TAGS.has(element.tagName)) {
        const fallback = element.tagName === "IMG" ? element.getAttribute("alt") || "" : element.textContent || "";
        element.replaceWith(document.createTextNode(fallback));
        continue;
      }

      const href = element.tagName === "A" ? element.getAttribute("href") : null;
      const languageClass = element.tagName === "CODE" ? element.getAttribute("class") : null;
      const start = element.tagName === "OL" ? element.getAttribute("start") : null;
      const align = ["TD", "TH"].includes(element.tagName) ? element.getAttribute("align") : null;
      for (const attribute of [...element.attributes]) element.removeAttribute(attribute.name);

      if (element.tagName === "A" && href) {
        try {
          const url = new URL(href, window.location.href);
          if (!["http:", "https:"].includes(url.protocol)) throw new Error("unsupported Markdown link protocol");
          element.href = url.toString();
          element.target = "_blank";
          element.rel = "noopener noreferrer";
          element.referrerPolicy = "no-referrer";
        } catch {
          element.replaceWith(document.createTextNode(element.textContent || ""));
        }
      } else if (element.tagName === "CODE" && /^language-[A-Za-z0-9_+.-]{1,64}$/.test(languageClass || "")) {
        element.className = languageClass;
      } else if (element.tagName === "OL" && /^\d{1,6}$/.test(start || "")) {
        element.setAttribute("start", start);
      } else if (["TD", "TH"].includes(element.tagName) && ["left", "center", "right"].includes(align || "")) {
        element.className = `align-${align}`;
      }
    }
    for (const table of [...template.content.querySelectorAll("table")]) {
      if (!table.parentNode) continue;
      const scroller = document.createElement("div");
      scroller.className = "markdown-table-scroll";
      table.replaceWith(scroller);
      scroller.append(table);
    }
    return template.content;
  }

  function renderMarkdown(parent, value) {
    parent.replaceChildren();
    if (typeof value !== "string") {
      appendPlainCodeBlock(parent, JSON.stringify(value, null, 2), "json");
      return;
    }
    try {
      const html = window.marked?.parse?.(value, MARKDOWN_OPTIONS);
      if (typeof html !== "string") throw new Error("Marked parser is unavailable");
      parent.append(sanitizeMarkedHtml(html));
    } catch {
      appendPlainCodeBlock(parent, value);
    }
  }

  function resultPresentation(value, fallbackHeading, streaming = false) {
    const supplied = typeof fallbackHeading === "string" ? fallbackHeading.trim() : "";
    return {
      heading: supplied || "Response",
      body: value,
      streaming,
    };
  }

  function stableRequestSummary(projection, requestId) {
    const supplied = projection.request_summary;
    const normalized = typeof supplied === "string" ? supplied.trim() : "";
    if (requestId !== "not-issued" && normalized && !state.requestSummaries.has(requestId)) {
      state.requestSummaries.set(requestId, normalized);
      while (state.requestSummaries.size > MAX_REQUEST_SUMMARIES) {
        state.requestSummaries.delete(state.requestSummaries.keys().next().value);
      }
    }
    return state.requestSummaries.get(requestId) || normalized || "Delegated goal summary unavailable";
  }

  function projectionCorrelation(projection) {
    const correlation = { ...objectValue(projection.correlation) };
    for (const key of [
      "local_request_id", "commit_id", "backend_session_id", "turn_id", "response_id",
      "identity_authority",
    ]) {
      if (projection[key] !== undefined && projection[key] !== null && projection[key] !== "") {
        correlation[key] = projection[key];
      }
    }
    return correlation;
  }

  function applyProjection(projection, responseText) {
    const kind = String(firstValue(projection.kind, "projection"));
    const phase = String(firstValue(projection.phase, projection.state, "updated"));
    const ephemeralLifecycle = kind === "backend_turn";
    const stateOnly = kind === "delivery_queue";
    const target = objectValue(projection.target);
    const backend = objectValue(projection.backend);
    const durableWork = { ...objectValue(projection.work) };
    let request = durableWork;
    let identityKind = Object.keys(durableWork).length || projection.work_id || projection.work_state
      ? "work"
      : "request";
    const result = { ...objectValue(projection.result ?? projection.delivery) };
    const activity = objectValue(projection.activity);
    const voice = objectValue(projection.voice ?? projection.runtime);

    if (kind === "backend_target" || kind === "runtime_attachment") {
      state.gatewayReachable = phase === "reachable" || (kind === "runtime_attachment" && phase === "ready");
      state.frontendTools = new Set(arrayValue(projection.frontend_tools).map(String));
      state.backendCapabilities = new Set(
        arrayValue(firstValue(target.capabilities, projection.capabilities)).map(String),
      );
      if (!state.gatewayReachable) {
        showNotice("The configured backend gateway is unavailable. Direct realtime conversation remains available.", "error", 0);
      }
    }

    if (kind === "backend_turn") {
      const correlation = projectionCorrelation(projection);
      const mappedPhase = [
        "locally_queued", "dispatching", "waiting_for_response", "succeeded", "failed",
      ].includes(phase)
        ? phase
        : "unknown";
      const localRequestId = String(firstValue(
        projection.local_request_id,
        projection.commit_id,
        correlation.local_request_id,
        correlation.commit_id,
        "not-issued",
      ));
      identityKind = "local_request";
      request = {
        state: mappedPhase,
        id: localRequestId,
        summary: stableRequestSummary(projection, localRequestId),
        steps: [
        {
          label: "Finalized request",
          state: "completed",
        },
        {
          label: "Terminal backend response",
          state: ["succeeded", "failed"].includes(mappedPhase) ? mappedPhase : "waiting",
        },
        ],
      };
    }

    if (kind === "result_display" && phase === "discarded") {
      const correlation = projectionCorrelation(projection);
      const localRequestId = String(firstValue(
        projection.local_request_id,
        correlation.local_request_id,
        state.latestDelegationId,
        "pending",
      ));
      const record = delegationRecord(localRequestId, "local_request");
      record.result = null;
      record.resultStatus = null;
      record.updatedAt = new Date();
      renderDelegations();
    }

    if (projectionCarriesDisplay(projection)) {
      result.state = firstValue(result.state, phase, "received");
      result.title = firstValue(result.title, projection.title, "Response");
      result.display = firstValue(result.display, responseText, projection.title);
    } else if (ephemeralLifecycle && phase === "failed") {
      result.state = firstValue(result.state, phase);
      result.title = firstValue(result.title, projection.title, "Backend response");
      result.display = firstValue(result.display, responseText, projection.title);
    }

    const backendName = firstValue(
      backend.label,
      backend.name,
      projection.backend_name,
      projection.adapter,
      projection.profile,
      target.backend,
      ephemeralLifecycle ? "Configured backend" : undefined,
    );
    if (backendName) updateBackend(String(backendName), phase !== "unavailable");

    const targetProjection = kind === "backend_target"
      || kind === "runtime_attachment"
      || Object.keys(target).length > 0
      || Boolean(
        projection.target_name
        || projection.target_ref
        || projection.target_state
        || projection.agent_readiness
      );
    if (targetProjection) updateTarget(target, projection);
    if (Object.keys(request).length || projection.work_id || projection.work_state) {
      updateRequest(request, projection, identityKind);
    }
    if (Object.keys(result).length || projection.result_state || projection.display) updateResult(result, projection, responseText);

    const queueDepth = firstValue(
      voice.speech_waiting_depth,
      projection.speech_waiting_depth,
      stateOnly ? projection.waiting_depth : undefined,
      voice.speech_queue_depth,
      projection.speech_queue_depth,
      stateOnly ? projection.queue_depth : undefined,
    );
    const normalizedQueueDepth = Number(queueDepth);
    if (Number.isSafeInteger(normalizedQueueDepth) && normalizedQueueDepth >= 0 && normalizedQueueDepth <= 1025) {
      state.speechDeliveryQueueDepth = normalizedQueueDepth;
      updateQueue();
    }
    if (voice.input_state) setSignal("input", titleCase(String(voice.input_state)), voice.input_state !== "idle");
    if (voice.model_state) setSignal("model", titleCase(String(voice.model_state)), voice.model_state !== "idle");
    if (voice.output_state && !hasLocalPlaybackActivity()) {
      setSignal(
        "output",
        state.muted ? "Muted" : titleCase(String(voice.output_state)),
        !state.muted && voice.output_state !== "idle",
      );
    }

    const activityLabel = firstValue(activity.label, activity.message, projection.activity_label, projection.event);
    const activityState = firstValue(activity.state, activity.status, projection.state, "neutral");
    if (activityLabel && !stateOnly && !ephemeralLifecycle) {
      addActivity(String(activityLabel), String(firstValue(activity.detail, activityState, "")), String(activityState));
    } else if (!stateOnly && !ephemeralLifecycle) {
      const label = String(firstValue(projection.title, projection.kind, "VoiceClaw state updated"));
      const detail = phase || request.state || result.state || "";
      addActivity(titleCase(label), detail, String(phase || request.state || result.state || "neutral"));
    }
  }

  function updateBackend(name, reachable = true) {
    elements.backendBadge.className = `backend-badge${reachable ? " reachable" : ""}`;
    elements.backendBadge.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "badge-dot";
    elements.backendBadge.append(dot, document.createTextNode(name));
  }

  function updateTarget(target, root) {
    const previous = state.targetProjection || {};
    const nextCapabilities = firstValue(target.capabilities, root.capabilities);
    const nextFrontendTools = firstValue(target.frontend_tools, root.frontend_tools);
    const snapshot = {
      name: firstValue(target.label, target.name, root.target_name, previous.name, "Configured target"),
      reference: firstValue(
        target.ref,
        target.id,
        target.target_ref,
        root.target_ref,
        previous.reference,
        "No durable attachment advertised",
      ),
      status: String(firstValue(
        target.state,
        target.status,
        root.target_state,
        root.kind === "backend_target" ? root.phase : undefined,
        previous.status,
        state.gatewayReachable ? "gateway_reachable" : undefined,
        "unknown",
      )),
      capabilities: nextCapabilities === undefined
        ? arrayValue(previous.capabilities)
        : arrayValue(nextCapabilities),
      frontendTools: nextFrontendTools === undefined
        ? arrayValue(previous.frontendTools)
        : arrayValue(nextFrontendTools),
      durability: firstValue(root.durability, previous.durability),
      eventDelivery: firstValue(root.event_delivery, previous.eventDelivery),
      agentReadiness: firstValue(root.agent_readiness, previous.agentReadiness),
      maxParallelWork: firstValue(root.max_parallel_work, previous.maxParallelWork),
    };
    state.targetProjection = snapshot;
    elements.targetAvatar.textContent = String(snapshot.name).trim().charAt(0).toUpperCase() || "A";
    elements.targetName.textContent = String(snapshot.name);
    elements.targetReference.textContent = compactId(String(snapshot.reference));
    setStatusPill(elements.targetStatus, titleCase(snapshot.status), snapshot.status);

    const capabilities = snapshot.capabilities;
    const frontendTools = snapshot.frontendTools;
    const contractFacts = [];
    if (snapshot.durability) contractFacts.push(`Durability · ${titleCase(String(snapshot.durability))}`);
    if (snapshot.eventDelivery) contractFacts.push(`Events · ${titleCase(String(snapshot.eventDelivery))}`);
    if (snapshot.agentReadiness) contractFacts.push(`Target readiness · ${titleCase(String(snapshot.agentReadiness))}`);
    const maxParallel = Number(snapshot.maxParallelWork);
    if (Number.isSafeInteger(maxParallel) && maxParallel >= 0) contractFacts.push(`Concurrency · ${maxParallel}`);
    if (capabilities.length || frontendTools.length || contractFacts.length) {
      elements.capabilityList.replaceChildren();
      const note = document.createElement("span");
      note.className = "capability muted";
      note.textContent = "Server-projected, not backend-advertised";
      elements.capabilityList.append(note);
      for (const fact of contractFacts) {
        const chip = document.createElement("span");
        chip.className = "capability";
        chip.textContent = fact;
        elements.capabilityList.append(chip);
      }
      for (const capability of capabilities) {
        const chip = document.createElement("span");
        chip.className = "capability";
        chip.textContent = typeof capability === "string" ? titleCase(capability) : titleCase(capability?.name || "capability");
        elements.capabilityList.append(chip);
      }
      for (const frontendTool of frontendTools) {
        const chip = document.createElement("span");
        chip.className = "capability";
        chip.textContent = `Tool · ${titleCase(String(frontendTool))}`;
        elements.capabilityList.append(chip);
      }
    }
  }

  function rememberUserTurn(id, text) {
    const normalized = normalizeTranscriptWhitespace(text);
    if (!id || !normalized) return;
    const existing = state.recentUserTurns.find((turn) => turn.id === id);
    if (existing) existing.text = normalized;
    else state.recentUserTurns.push({ id, text: normalized });
    state.recentUserTurns.splice(0, Math.max(0, state.recentUserTurns.length - MAX_REQUEST_SUMMARIES));
  }

  function delegationRecord(id, identityKind = "request") {
    const key = String(id || "pending");
    let record = state.delegations.get(key);
    if (!record) {
      for (const existing of state.delegations.values()) {
        if (!existing.userToggled) existing.expanded = false;
      }
      record = {
        id: key,
        identityKind,
        summary: "Delegated goal summary unavailable",
        query: "Delegated goal summary unavailable",
        status: "locally_queued",
        resultStatus: null,
        updatedAt: new Date(),
        phases: [],
        expanded: true,
        userToggled: false,
        result: null,
        artifacts: [],
        correlation: {},
      };
      state.delegations.set(key, record);
      while (state.delegations.size > MAX_DELEGATIONS) {
        state.delegations.delete(state.delegations.keys().next().value);
      }
    }
    state.latestDelegationId = key;
    return record;
  }

  function delegationPhaseLabel(phase) {
    return {
      locally_queued: "Queued locally in VoiceClaw",
      dispatching: "Sending to backend",
      waiting_for_response: "Waiting on response-only exchange · acceptance not confirmed",
      succeeded: "Response received",
      completed: "Response received",
      failed: "Request failed",
      cancelled: "Request cancelled",
      canceled: "Request cancelled",
      outcome_unknown: "Connection closed · outcome unavailable",
    }[phase] || titleCase(phase);
  }

  function delegationStatusLabel(phase) {
    return {
      succeeded: "Response received",
      completed: "Response received",
      failed: "Failed",
      cancelled: "Cancelled",
      canceled: "Cancelled",
      outcome_unknown: "Outcome unknown",
    }[phase] || titleCase(phase);
  }

  function recordDelegationPhase(record, phase, label) {
    const normalized = String(phase || "updated");
    const existing = record.phases.find((entry) => entry.phase === normalized);
    if (existing) {
      existing.label = label || existing.label;
      existing.timestamp = new Date();
      return;
    }
    record.phases.push({
      phase: normalized,
      label: label || delegationPhaseLabel(normalized),
      timestamp: new Date(),
    });
    elements.delegationAnnouncement.textContent = `Backend turn: ${label || delegationPhaseLabel(normalized)}`;
  }

  function artifactRow(rawArtifact) {
    const artifact = typeof rawArtifact === "string" ? { label: rawArtifact } : objectValue(rawArtifact);
    const row = document.createElement("div");
    row.className = "artifact";
    const label = document.createElement("span");
    label.textContent = String(firstValue(artifact.label, artifact.name, artifact.type, "Artifact"));
    row.append(label);
    if (typeof artifact.url === "string") {
      try {
        const url = new URL(artifact.url, window.location.href);
        if (["http:", "https:"].includes(url.protocol)) {
          const link = document.createElement("a");
          link.href = url.toString();
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = "Open";
          row.append(link);
        }
      } catch {
        // Malformed artifact URLs remain visible as labels without becoming links.
      }
    }
    return row;
  }

  function delegationMetaFact(label, rawValue) {
    const value = String(rawValue);
    const fact = document.createElement("span");
    fact.className = "delegation-meta-fact";
    const name = document.createElement("small");
    name.textContent = label;
    const code = document.createElement("code");
    code.textContent = value;
    code.title = value;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy-id";
    copy.textContent = "Copy";
    copy.setAttribute("aria-label", `Copy ${label}`);
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(value);
        copy.textContent = "Copied";
        window.setTimeout(() => {
          copy.textContent = "Copy";
        }, 1200);
      } catch {
        showNotice(`${label} could not be copied.`, "error");
      }
    });
    fact.append(name, code, copy);
    return fact;
  }

  function renderDelegations() {
    const focusedRequestId = document.activeElement?.closest?.(".delegation-card")?.dataset.requestId;
    const records = [...state.delegations.values()].reverse();
    const sharesLiveContext = state.backendCapabilities.has("live_delegated_context");
    elements.delegationsNote.textContent = sharesLiveContext
      ? "Each card is one response-only backend exchange. Delegated follow-ups can share this live agent context, but it is not durable or recoverable after reconnect."
      : "Each card is one non-durable backend exchange. Do not assume context carries between delegated turns unless the target advertises that capability.";
    elements.delegationEmpty.hidden = records.length > 0;
    elements.delegationCount.textContent = `${records.length} ${records.length === 1 ? "turn" : "turns"}`;
    const hasActive = records.some((record) => (
      ["locally_queued", "dispatching", "waiting_for_response", "running", "queued"].includes(record.status)
    ));
    const countState = hasActive
      ? "waiting"
      : ["failed", "cancelled", "canceled"].includes(records[0]?.status) ? "failed"
        : records.length && records[0]?.status !== "outcome_unknown" ? "completed" : "neutral";
    elements.delegationCount.className = `status-pill ${countState}`;
    elements.delegationList.replaceChildren();

    for (const record of records) {
      const card = document.createElement("details");
      card.className = "delegation-card";
      card.dataset.requestId = record.id;
      card.open = record.expanded;
      card.addEventListener("toggle", () => {
        record.expanded = card.open;
      });

      const summary = document.createElement("summary");
      summary.addEventListener("click", () => {
        record.userToggled = true;
      });
      const summaryCopy = document.createElement("span");
      summaryCopy.className = "delegation-summary-copy";
      const title = document.createElement("strong");
      title.textContent = record.query;
      const subtitle = document.createElement("span");
      const contextLabel = record.identityKind === "work"
        ? "durable Work projection"
        : sharesLiveContext ? "live-context delegated turn" : "non-durable delegated turn";
      subtitle.textContent = `${timeLabel(record.updatedAt)} · ${contextLabel}`;
      summaryCopy.append(title, subtitle);
      const status = document.createElement("span");
      setStatusPill(status, delegationStatusLabel(record.status), record.status);
      const chevron = document.createElement("i");
      chevron.className = "delegation-chevron";
      chevron.setAttribute("aria-hidden", "true");
      summary.append(summaryCopy, status, chevron);
      card.append(summary);

      const body = document.createElement("div");
      body.className = "delegation-card-body";
      const querySection = document.createElement("div");
      querySection.className = "delegation-query";
      const queryLabel = document.createElement("span");
      queryLabel.textContent = "Delegated goal summary";
      const queryText = document.createElement("p");
      queryText.textContent = record.query;
      querySection.append(queryLabel, queryText);
      body.append(querySection);

      const timeline = document.createElement("ol");
      timeline.className = "delegation-timeline";
      for (const phase of record.phases) {
        const item = document.createElement("li");
        item.className = normalizeStateClass(phase.phase);
        const phaseLabel = document.createElement("strong");
        phaseLabel.textContent = phase.label;
        phaseLabel.title = phase.label;
        const phaseTime = document.createElement("span");
        phaseTime.textContent = timeLabel(phase.timestamp);
        item.append(phaseLabel, phaseTime);
        timeline.append(item);
      }
      body.append(timeline);

      if (record.result) {
        const failed = ["failed", "cancelled", "canceled"].some(
          (terminal) => terminal === record.status || terminal === record.resultStatus,
        );
        const result = document.createElement("div");
        result.className = `delegation-result${failed ? " failure" : ""}`;
        const resultHeader = document.createElement("div");
        resultHeader.className = "delegation-result-header";
        const resultLabel = document.createElement("span");
        resultLabel.textContent = failed ? "Failure" : "Response";
        const resultTitle = document.createElement("strong");
        resultTitle.textContent = record.result.heading;
        resultHeader.append(resultLabel, resultTitle);
        const resultBody = document.createElement("div");
        resultBody.className = "result-body";
        resultBody.classList.toggle("streaming", Boolean(record.result.streaming));
        resultBody.setAttribute("aria-busy", String(Boolean(record.result.streaming)));
        renderMarkdown(resultBody, record.result.body);
        result.append(resultHeader, resultBody);
        if (record.artifacts.length) {
          const artifacts = document.createElement("div");
          artifacts.className = "artifact-list";
          for (const artifact of record.artifacts) artifacts.append(artifactRow(artifact));
          result.append(artifacts);
        }
        body.append(result);
      }

      const identityLabel = record.identityKind === "work" ? "Work ID" : "Local request ID";
      const facts = [
        [identityLabel, record.id],
        ["Temporary backend session", record.correlation.backend_session_id],
        ["Backend turn", record.correlation.turn_id],
        ["Backend response", record.correlation.response_id],
      ];
      if (facts.some(([, value]) => value)) {
        const developer = document.createElement("details");
        developer.className = "delegation-developer-details";
        const developerSummary = document.createElement("summary");
        developerSummary.textContent = "Developer details";
        const meta = document.createElement("div");
        meta.className = "delegation-meta";
        for (const [label, value] of facts) {
          if (value) meta.append(delegationMetaFact(label, value));
        }
        developer.append(developerSummary, meta);
        body.append(developer);
      }
      card.append(body);
      elements.delegationList.append(card);
    }
    if (focusedRequestId) {
      const focusedCard = [...elements.delegationList.children]
        .find((card) => card.dataset.requestId === focusedRequestId);
      focusedCard?.querySelector("summary")?.focus({ preventScroll: true });
    }
  }

  function updateRequest(request, root, identityKind = "request") {
    const id = firstValue(request.id, request.work_id, root.work_id, "pending");
    const status = String(firstValue(request.state, request.status, root.work_state, "unknown"));
    const summary = String(firstValue(
      request.summary,
      request.objective,
      request.title,
      root.request_summary,
      root.work_summary,
      "Delegated request",
    ));
    const record = delegationRecord(id, identityKind);
    record.identityKind = identityKind;
    record.summary = summary;
    record.query = record.query === "Delegated goal summary unavailable" ? summary : record.query;
    record.status = status;
    record.updatedAt = new Date(firstValue(request.updated_at, request.updatedAt, new Date()));
    record.correlation = { ...record.correlation, ...projectionCorrelation(root) };
    recordDelegationPhase(record, status, delegationPhaseLabel(status));
    if (["succeeded", "completed", "failed", "cancelled", "canceled", "outcome_unknown"].includes(status)
      && !record.userToggled) record.expanded = record.id === state.latestDelegationId;
    renderDelegations();
  }

  function updateResult(result, root, responseText, { streaming = false } = {}) {
    const correlation = projectionCorrelation(root);
    const id = firstValue(
      result.work_id,
      root.work_id,
      root.local_request_id,
      correlation.local_request_id,
      state.latestDelegationId,
      "pending",
    );
    const status = String(firstValue(result.state, result.status, root.result_state, root.phase, "received"));
    const body = firstValue(
      result.display,
      result.body,
      result.text,
      root.display,
      root.result_text,
      responseText,
      "Display result received.",
    );
    const record = delegationRecord(id, root.work_id ? "work" : "local_request");
    record.resultStatus = status;
    record.updatedAt = new Date();
    record.correlation = { ...record.correlation, ...correlation };
    record.result = resultPresentation(
      body,
      firstValue(result.heading, result.title, root.result_title, root.title),
      streaming,
    );
    record.artifacts = arrayValue(firstValue(result.artifacts, root.artifacts));
    // Result availability is independent of the admitted request lifecycle.
    // Only backend-turn/Work projections may mutate the request status or its
    // request timeline.
    if (!record.userToggled && record.id === state.latestDelegationId) record.expanded = true;
    renderDelegations();
  }

  function submitText(text) {
    const trimmed = text.trim();
    if (!trimmed || !state.protocolReady) return;
    interruptActiveResponse();
    const itemId = makeId("item");
    rememberUserItemId(itemId);
    rememberUserTurn(itemId, trimmed);
    updateMessage(itemId, "user", trimmed, false);
    const created = sendEvent({
      type: "conversation.item.create",
      item: { id: itemId, type: "message", role: "user", content: [{ type: "input_text", text: trimmed }] },
    }, { kind: "user_item", itemId });
    if (created) {
      const requested = sendEvent(
        { type: "response.create", response: {} },
        { kind: "response_create" },
      );
      if (requested) setSignal("model", "Thinking", true);
    } else {
      state.userItemIds.delete(itemId);
      removeMessage(itemId);
    }
  }

  function interruptActiveResponse() {
    const responseId = state.activeResponseId;
    if (responseId) state.suppressedPlaybackResponses.add(responseId);
    const truncations = stopPlayback(true, true);
    if (responseId) {
      const cancelled = sendEvent(
        { type: "response.cancel", response_id: responseId },
        { kind: "response_cancel", responseId },
      );
      if (cancelled) {
        dispatchPlaybackTruncations(truncations);
      }
    } else {
      dispatchPlaybackTruncations(truncations);
    }
  }

  function secureMicrophoneContext() {
    const hostname = window.location.hostname;
    return window.isSecureContext || hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
  }

  async function ensureMicrophone() {
    if (!secureMicrophoneContext()) {
      throw new Error("Microphone capture requires HTTPS or localhost. You can continue using typed turns.");
    }
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone capture is not available in this browser.");
    await prepareAudioContext();
    if (state.micStream?.active) return;

    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
    state.micSource = state.audioContext.createMediaStreamSource(state.micStream);
    state.micProcessor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.micSink = state.audioContext.createGain();
    state.micSink.gain.value = 0;
    state.micProcessor.onaudioprocess = (event) => {
      if (!state.capturing || !state.connected) return;
      const input = event.inputBuffer.getChannelData(0);
      const resampled = resample(input, state.audioContext.sampleRate, INPUT_SAMPLE_RATE);
      const pcm = floatToPcm16(resampled);
      state.captureBytes += pcm.byteLength;
      sendEvent({ type: "input_audio_buffer.append", audio: bytesToBase64(new Uint8Array(pcm)) });
    };
    state.micSource.connect(state.micProcessor);
    state.micProcessor.connect(state.micSink);
    state.micSink.connect(state.audioContext.destination);
  }

  function releaseMicrophone() {
    window.clearTimeout(state.stopListeningTimer);
    state.stopListeningTimer = null;
    state.captureRequested = false;
    state.capturing = false;
    state.captureBytes = 0;
    state.vadSpeechActive = false;
    state.stopAfterSpeech = false;
    state.micProcessor?.disconnect();
    state.micSource?.disconnect();
    state.micSink?.disconnect();
    if (state.micProcessor) state.micProcessor.onaudioprocess = null;
    for (const track of state.micStream?.getTracks?.() || []) track.stop();
    state.micStream = null;
    state.micSource = null;
    state.micProcessor = null;
    state.micSink = null;
    updateTalkControl();
  }

  async function startListening() {
    if (!state.protocolReady || state.inputMode !== "idle" || state.captureRequested || state.capturing) return;
    state.captureRequested = true;
    state.inputMode = "starting";
    updateTalkControl();
    clearNotice();
    try {
      await ensureMicrophone();
      if (!state.captureRequested || !state.protocolReady) {
        releaseMicrophone();
        return;
      }
      if (!state.automaticTurnDetection) {
        interruptActiveResponse();
        sendEvent({ type: "input_audio_buffer.clear" });
      }
      state.captureBytes = 0;
      state.capturing = true;
      state.inputMode = "capturing";
      updateTalkControl();
      setSignal("input", "Listening", true);
    } catch (error) {
      state.captureRequested = false;
      state.capturing = false;
      state.inputMode = "idle";
      updateTalkControl();
      showNotice(error.message, "error", 0);
      addActivity("Microphone unavailable", error.message, "failure");
    }
  }

  function stopListening() {
    if (!state.captureRequested && !state.capturing) return;
    const capturedBytes = state.captureBytes;
    const hadCapture = state.capturing;
    if (state.automaticTurnDetection && state.vadSpeechActive) {
      state.stopAfterSpeech = true;
      state.inputMode = "finishing";
      setSignal("input", "Finish speaking", true);
      updateTalkControl();
      window.clearTimeout(state.stopListeningTimer);
      state.stopListeningTimer = window.setTimeout(() => {
        if (!state.stopAfterSpeech) return;
        state.inputMode = "idle";
        releaseMicrophone();
        setSignal("input", "Ready");
        showNotice("Listening stopped before the server detected the end of speech. The final turn may be incomplete.");
      }, AUTOMATIC_STOP_TIMEOUT_MS);
      return;
    }
    state.captureRequested = false;
    state.capturing = false;
    if (state.automaticTurnDetection) {
      state.inputMode = "idle";
      releaseMicrophone();
      setSignal("input", "Ready");
      return;
    }
    if (!hadCapture || capturedBytes < MIN_CAPTURE_BYTES) {
      sendEvent({ type: "input_audio_buffer.clear" });
      state.inputMode = "idle";
      releaseMicrophone();
      setSignal("input", "Ready");
      if (hadCapture) showNotice("No speech was sent. Listen a little longer, then choose Stop & send.");
      return;
    }
    const committed = sendEvent(
      { type: "input_audio_buffer.commit" },
      { kind: "audio_commit" },
    );
    if (committed) {
      state.inputMode = "transcribing";
      const requested = sendEvent(
        { type: "response.create", response: {} },
        { kind: "response_create" },
      );
      if (requested) setSignal("model", "Waiting for transcript", true);
      setSignal("input", "Transcribing", true);
    } else {
      state.inputMode = "idle";
    }
    releaseMicrophone();
    updateTalkControl();
  }

  function finishAutomaticStop() {
    if (!state.stopAfterSpeech) return;
    state.inputMode = "transcribing";
    releaseMicrophone();
    updateTalkControl();
  }

  async function toggleListening() {
    if (state.captureRequested || state.capturing) stopListening();
    else await startListening();
  }

  function setMuted(muted) {
    const nextMuted = Boolean(muted);
    if (nextMuted === state.muted) return;
    if (nextMuted) {
      const activeResponseId = state.activeResponseId;
      if (activeResponseId) state.suppressedPlaybackResponses.add(activeResponseId);
      const truncations = stopPlayback(false, true);
      for (const truncation of truncations) {
        if (!truncation.responseId) continue;
        state.suppressedPlaybackResponses.add(truncation.responseId);
        state.muteTruncationReported.add(truncation.responseId);
      }
      dispatchPlaybackTruncations(truncations);
    }
    state.muted = nextMuted;
    if (state.outputGain) state.outputGain.gain.value = state.muted ? 0 : 1;
    elements.muteButton.classList.toggle("muted", state.muted);
    elements.muteButton.setAttribute("aria-pressed", String(state.muted));
    elements.muteButton.setAttribute("aria-label", state.muted ? "Unmute assistant audio" : "Mute assistant audio");
    elements.muteLabel.textContent = state.muted ? "Unmute" : "Mute";
    refreshLocalOutputSignal();
  }

  function resample(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input.slice();
    const outputLength = Math.max(1, Math.floor((input.length * targetRate) / sourceRate));
    const output = new Float32Array(outputLength);
    const ratio = sourceRate / targetRate;
    for (let index = 0; index < outputLength; index += 1) {
      const position = index * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const fraction = position - left;
      output[index] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }

  function floatToPcm16(input) {
    const buffer = new ArrayBuffer(input.length * 2);
    const view = new DataView(buffer);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    return buffer;
  }

  function bytesToBase64(bytes) {
    let binary = "";
    const stride = 0x8000;
    for (let start = 0; start < bytes.length; start += stride) {
      binary += String.fromCharCode(...bytes.subarray(start, start + stride));
    }
    return window.btoa(binary);
  }

  function usesPcm24(format) {
    if (format === undefined || format === null) return true;
    if (typeof format !== "object" || format.type !== "audio/pcm") return false;
    const rate = format.rate ?? format.sample_rate ?? format.sampleRate ?? INPUT_SAMPLE_RATE;
    return Number(rate) === INPUT_SAMPLE_RATE;
  }

  function outputSampleRate() {
    const format = state.session?.audio?.output?.format;
    if (!usesPcm24(format)) throw new Error("The negotiated output format is not 24 kHz PCM.");
    return INPUT_SAMPLE_RATE;
  }

  function playbackKey(itemId, contentIndex) {
    return `${itemId}:${contentIndex}`;
  }

  function playbackTruncationsAt(now) {
    const truncations = [];
    for (const [key, progress] of state.playbackProgress) {
      let heardThroughMs = progress.playedThroughMs;
      for (const metadata of state.playbackSourceMetadata.values()) {
        if (metadata.key !== key || now <= metadata.startAt) continue;
        const elapsedMs = Math.floor((Math.min(now, metadata.endAt) - metadata.startAt) * 1000);
        heardThroughMs = Math.max(
          heardThroughMs,
          Math.min(metadata.audioEndMs, metadata.audioStartMs + Math.max(0, elapsedMs)),
        );
      }
      heardThroughMs = Math.max(0, Math.min(progress.audioEndMs, Math.floor(heardThroughMs)));
      if (
        !progress.receiptSent
        && !state.reportedPlaybackReceipts.has(progress.receiptId)
        && progress.audioEndMs > 0
      ) {
        truncations.push({
          responseId: progress.responseId,
          itemId: progress.itemId,
          contentIndex: progress.contentIndex,
          audioEndMs: heardThroughMs,
          receiptId: progress.receiptId,
        });
      }
    }
    return truncations;
  }

  function dispatchPlaybackTruncations(truncations) {
    for (const truncation of truncations) {
      if (!truncation.itemId) continue;
      const sent = sendEvent({
        ...(truncation.receiptId ? { event_id: truncation.receiptId } : {}),
        type: "conversation.item.truncate",
        item_id: truncation.itemId,
        content_index: truncation.contentIndex,
        audio_end_ms: truncation.audioEndMs,
      }, {
        kind: "playback_truncate",
        itemId: truncation.itemId,
      });
      if (sent) {
        rememberPlaybackReceipt(truncation.receiptId);
        const key = playbackKey(truncation.itemId, truncation.contentIndex);
        const progress = state.playbackProgress.get(key);
        if (progress) progress.receiptSent = true;
        // Truncation is playback bookkeeping. The full event remains available
        // in diagnostics without duplicating it in the user-facing lifecycle.
      }
    }
  }

  function playbackIdentity(event) {
    const responseId = typeof event.response_id === "string"
      ? event.response_id
      : state.activeResponseId || "";
    const itemId = typeof event.item_id === "string" ? event.item_id : "";
    const contentIndex = Number.isSafeInteger(event.content_index) && event.content_index >= 0
      ? event.content_index
      : 0;
    const track = responseFor(responseId, false);
    const receiptRequired = track?.metadata?.voiceclaw_playback_receipt_required === "true";
    const receiptId = receiptRequired && typeof track?.metadata?.voiceclaw_playback_receipt_id === "string"
      ? track.metadata.voiceclaw_playback_receipt_id
      : null;
    return {
      responseId,
      itemId,
      contentIndex,
      receiptId,
      receiptRequired,
      key: playbackKey(itemId || `response:${responseId || "unknown"}`, contentIndex),
    };
  }

  function rememberPlaybackReceipt(receiptId) {
    if (!receiptId) return;
    state.reportedPlaybackReceipts.add(receiptId);
    while (state.reportedPlaybackReceipts.size > 128) {
      state.reportedPlaybackReceipts.delete(state.reportedPlaybackReceipts.values().next().value);
    }
  }

  function streamHasScheduledSource(key) {
    return Array.from(state.playbackSourceMetadata.values()).some((metadata) => metadata.key === key);
  }

  function prunePlaybackStream(stream) {
    if (!stream.terminal || stream.pendingDeltas > 0 || streamHasScheduledSource(stream.key)) return;
    if (state.playbackStreams.get(stream.key) === stream) state.playbackStreams.delete(stream.key);
  }

  function reportCompletedPlayback(key, progress) {
    if (
      progress.receiptSent
      || state.reportedPlaybackReceipts.has(progress.receiptId)
      || !progress.responseComplete
      || streamHasScheduledSource(key)
      || progress.playedThroughMs < progress.audioEndMs
    ) return false;
    if (!progress.receiptRequired) return true;
    if (!progress.receiptId || !progress.itemId || progress.audioEndMs <= 0) return false;
    const sent = sendEvent({
      event_id: progress.receiptId,
      type: "conversation.item.truncate",
      item_id: progress.itemId,
      content_index: progress.contentIndex,
      audio_end_ms: progress.audioEndMs,
    }, {
      kind: "playback_truncate",
      itemId: progress.itemId,
    });
    if (!sent) return false;
    progress.receiptSent = true;
    rememberPlaybackReceipt(progress.receiptId);
    return true;
  }

  function reportFailedPlayback(stream) {
    if (
      !stream.receiptRequired
      || !stream.receiptId
      || !stream.itemId
      || state.reportedPlaybackReceipts.has(stream.receiptId)
    ) return;
    const progress = state.playbackProgress.get(stream.key);
    const heardThroughMs = Math.max(
      0,
      Math.min(progress?.audioEndMs || 0, Math.floor(progress?.playedThroughMs || 0)),
    );
    const sent = sendEvent({
      event_id: stream.receiptId,
      type: "conversation.item.truncate",
      item_id: stream.itemId,
      content_index: stream.contentIndex,
      audio_end_ms: heardThroughMs,
    }, {
      kind: "playback_truncate",
      itemId: stream.itemId,
    });
    if (!sent) return;
    rememberPlaybackReceipt(stream.receiptId);
    if (progress) progress.receiptSent = true;
    if (stream.responseId) state.suppressedPlaybackResponses.add(stream.responseId);
  }

  function hasLocalPlaybackActivity() {
    if (state.muted) return false;
    for (const metadata of state.playbackSourceMetadata.values()) {
      if (!state.suppressedPlaybackResponses.has(metadata.responseId)) return true;
    }
    for (const stream of state.playbackStreams.values()) {
      if (state.suppressedPlaybackResponses.has(stream.responseId)) continue;
      if (!stream.terminal || stream.pendingDeltas > 0 || streamHasScheduledSource(stream.key)) return true;
    }
    return false;
  }

  function refreshLocalOutputSignal() {
    const active = hasLocalPlaybackActivity();
    setSignal("output", state.muted ? "Muted" : active ? "Speaking" : "Idle", !state.muted && active);
  }

  function markPlaybackStreamDone(event) {
    const identity = playbackIdentity(event);
    let streams = [state.playbackStreams.get(identity.key)].filter(Boolean);
    if (!streams.length && identity.responseId) {
      streams = Array.from(state.playbackStreams.values()).filter((stream) => (
        stream.responseId === identity.responseId
        && (!identity.itemId || stream.itemId === identity.itemId)
        && stream.contentIndex === identity.contentIndex
      ));
    }
    for (const stream of streams) {
      stream.terminal = true;
      prunePlaybackStream(stream);
    }
    refreshLocalOutputSignal();
  }

  function completePlaybackResponse(responseId) {
    state.suppressedPlaybackResponses.delete(responseId);
    state.muteTruncationReported.delete(responseId);
    state.completedPlaybackResponses.add(responseId);
    while (state.completedPlaybackResponses.size > 128) {
      state.completedPlaybackResponses.delete(state.completedPlaybackResponses.values().next().value);
    }
    for (const stream of state.playbackStreams.values()) {
      if (stream.responseId !== responseId) continue;
      stream.terminal = true;
      prunePlaybackStream(stream);
    }
    for (const [key, progress] of state.playbackProgress) {
      if (progress.responseId !== responseId) continue;
      progress.responseComplete = true;
      if (reportCompletedPlayback(key, progress)) state.playbackProgress.delete(key);
    }
    refreshLocalOutputSignal();
  }

  function rebufferLeadSeconds() {
    const browserLead = Number(state.audioContext?.baseLatency) * 4;
    return Math.min(
      INITIAL_PLAYOUT_LEAD_SECONDS,
      Math.max(MIN_REBUFFER_LEAD_SECONDS, Number.isFinite(browserLead) ? browserLead : 0),
    );
  }

  function enqueueAudioDelta(event) {
    const identity = playbackIdentity(event);
    let stream = state.playbackStreams.get(identity.key);
    if (!stream) {
      stream = { ...identity, terminal: false, pendingDeltas: 0 };
      state.playbackStreams.set(identity.key, stream);
    }
    stream.terminal = false;
    stream.pendingDeltas += 1;
    const epoch = state.playbackEpoch;
    refreshLocalOutputSignal();
    const scheduled = state.playbackScheduleTail.then(async () => {
      try {
        await playAudioDelta(event, epoch, stream);
      } finally {
        if (state.playbackStreams.get(stream.key) === stream) {
          stream.pendingDeltas = Math.max(0, stream.pendingDeltas - 1);
          prunePlaybackStream(stream);
          refreshLocalOutputSignal();
        }
      }
    });
    state.playbackScheduleTail = scheduled.catch(() => undefined);
  }

  async function playAudioDelta(event, epoch, stream) {
    try {
      await prepareAudioContext();
      if (epoch !== state.playbackEpoch) return;
      const { responseId, itemId, contentIndex, key } = stream;
      if (state.muted) {
        if (responseId) state.suppressedPlaybackResponses.add(responseId);
        if (itemId && responseId && !state.muteTruncationReported.has(responseId)) {
          const reported = sendEvent({
            ...(stream.receiptId ? { event_id: stream.receiptId } : {}),
            type: "conversation.item.truncate",
            item_id: itemId,
            content_index: contentIndex,
            audio_end_ms: 0,
          }, {
            kind: "playback_truncate",
            itemId,
          });
          if (reported) state.muteTruncationReported.add(responseId);
          if (reported) rememberPlaybackReceipt(stream.receiptId);
        }
        return;
      }
      if (state.suppressedPlaybackResponses.has(responseId)) return;
      const encoded = event.delta;
      const binary = window.atob(encoded);
      const evenLength = binary.length - (binary.length % 2);
      if (!evenLength) return;
      const samples = new Float32Array(evenLength / 2);
      for (let index = 0; index < samples.length; index += 1) {
        const low = binary.charCodeAt(index * 2);
        const high = binary.charCodeAt(index * 2 + 1);
        let value = (high << 8) | low;
        if (value >= 0x8000) value -= 0x10000;
        samples[index] = value / 0x8000;
      }

      const rate = outputSampleRate();
      const audioBuffer = state.audioContext.createBuffer(1, samples.length, rate);
      audioBuffer.copyToChannel(samples, 0);
      const source = state.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(state.outputGain || state.audioContext.destination);
      let progress = state.playbackProgress.get(key);
      if (!progress || progress.rate !== rate) {
        progress = {
          responseId,
          itemId,
          contentIndex,
          rate,
          scheduledSamples: 0,
          audioEndMs: 0,
          playedThroughMs: 0,
          responseComplete: state.completedPlaybackResponses.has(responseId),
          receiptId: stream.receiptId,
          receiptRequired: stream.receiptRequired,
          receiptSent: false,
        };
        state.playbackProgress.set(key, progress);
      }
      const now = state.audioContext.currentTime;
      if (state.playbackCursor <= now + PLAYBACK_CURSOR_EPSILON_SECONDS) {
        const lead = progress.scheduledSamples > 0
          ? rebufferLeadSeconds()
          : INITIAL_PLAYOUT_LEAD_SECONDS;
        state.playbackCursor = now + lead;
      }
      const startAt = state.playbackCursor;
      state.playbackCursor = startAt + audioBuffer.duration;
      const audioStartMs = Math.floor((progress.scheduledSamples * 1000) / rate);
      progress.scheduledSamples += samples.length;
      progress.audioEndMs = Math.floor((progress.scheduledSamples * 1000) / rate);
      const metadata = {
        key,
        responseId: progress.responseId,
        itemId,
        contentIndex,
        startAt,
        endAt: startAt + audioBuffer.duration,
        audioStartMs,
        audioEndMs: progress.audioEndMs,
        interrupted: false,
      };
      state.playbackSources.add(source);
      state.playbackSourceMetadata.set(source, metadata);
      source.addEventListener("ended", () => {
        const endedMetadata = state.playbackSourceMetadata.get(source);
        state.playbackSources.delete(source);
        state.playbackSourceMetadata.delete(source);
        if (endedMetadata && !endedMetadata.interrupted) {
          const endedProgress = state.playbackProgress.get(endedMetadata.key);
          if (endedProgress) {
            endedProgress.playedThroughMs = Math.max(endedProgress.playedThroughMs, endedMetadata.audioEndMs);
            if (reportCompletedPlayback(endedMetadata.key, endedProgress)) {
              state.playbackProgress.delete(endedMetadata.key);
              const sameResponsePending = Array.from(state.playbackProgress.values())
                .some((candidate) => candidate.responseId === endedProgress.responseId);
              if (!sameResponsePending) state.completedPlaybackResponses.delete(endedProgress.responseId);
            }
          }
        }
        const endedStream = endedMetadata ? state.playbackStreams.get(endedMetadata.key) : null;
        if (endedStream) prunePlaybackStream(endedStream);
        if (!hasLocalPlaybackActivity()) {
          state.playbackCursor = state.audioContext.currentTime;
        }
        refreshLocalOutputSignal();
      }, { once: true });
      source.start(startAt);
      refreshLocalOutputSignal();
    } catch (error) {
      reportFailedPlayback(stream);
      showNotice(`Assistant audio could not be played: ${error.message}`, "error");
      refreshLocalOutputSignal();
    }
  }

  function stopPlayback(updateSignal = true, collectTruncations = false) {
    const truncations = collectTruncations && state.audioContext
      ? playbackTruncationsAt(state.audioContext.currentTime)
      : [];
    state.playbackEpoch += 1;
    state.playbackScheduleTail = Promise.resolve();
    for (const source of state.playbackSources) {
      const metadata = state.playbackSourceMetadata.get(source);
      if (metadata) metadata.interrupted = true;
      try {
        source.stop();
      } catch {
        // An already-ended Web Audio source needs no further action.
      }
    }
    state.playbackSources.clear();
    state.playbackSourceMetadata.clear();
    state.playbackProgress.clear();
    state.playbackStreams.clear();
    state.playbackCursor = state.audioContext?.currentTime || 0;
    if (updateSignal) refreshLocalOutputSignal();
    return truncations;
  }

  elements.connectionToggle.addEventListener("click", () => {
    const opening = elements.connectionPanel.hidden;
    elements.connectionPanel.hidden = !opening;
    elements.connectionToggle.setAttribute("aria-expanded", String(opening));
  });
  elements.connectButton.addEventListener("click", connect);
  elements.connectionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    connect();
  });
  elements.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = elements.messageInput.value;
    elements.messageInput.value = "";
    elements.messageInput.style.height = "auto";
    submitText(value);
  });
  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.composer.requestSubmit();
    }
  });
  elements.messageInput.addEventListener("input", () => {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 132)}px`;
  });
  elements.talkButton.addEventListener("click", () => void toggleListening());
  elements.muteButton.addEventListener("click", () => setMuted(!state.muted));
  elements.clearActivity.addEventListener("click", () => {
    state.activities = [];
    renderActivities();
  });
  elements.clearEvents.addEventListener("click", () => {
    state.wireEvents = [];
    state.totalEvents = 0;
    renderWireEvents();
  });
  elements.diagnostics.addEventListener("toggle", () => {
    if (elements.diagnostics.open) renderWireEvents();
  });
  window.addEventListener("beforeunload", () => {
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.close(1000, "page closed");
    releaseMicrophone();
  });

  elements.endpoint.value = defaultEndpoint();
  void loadDeploymentPolicy();
  setConnection("disconnected", "Disconnected");
  setSignal("input", "Ready");
  setSignal("model", "Idle");
  setSignal("output", "Idle");
  renderDelegations();
  resetDisconnectedView();
})();
