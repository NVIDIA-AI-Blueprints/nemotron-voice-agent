// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import {
  usePipecatConversation,
  useRTVIClientEvent,
  filterEmptyMessages,
  type ConversationMessage,
  type ConversationMessagePart,
} from "@pipecat-ai/client-react";
import { uploadAttachment } from "../../api";
import { useApp } from "../../context/useApp";
import { useStickToBottom } from "../../hooks/useStickToBottom";
import { isRecord, stringField } from "../../utils";
import { TranscriptMessage } from "./TranscriptMessage";

type MediaKind = "image" | "audio" | "video";
type UploadStatus = "uploading" | "uploaded" | "failed";

type LocalAttachment = {
  id: string;
  kind: MediaKind;
  name: string;
  createdAt: string;
  anchorCreatedAt: string;
  status: UploadStatus;
  previewUrl: string;
  error?: string;
};

type AgentTask = {
  id: string;
  agent: string;
  status: string;
  stage: string;
  detail: string;
  query: string;
  reasoning: string;
  response: string;
  createdAt: string;
  updatedAt: string;
  attachmentName: string;
};

type AssistantTurn = {
  id: string;
  text: string;
  createdAt: string;
};

const renderPartText = (part: ConversationMessagePart): string => {
  const { text } = part;
  if (text === null || text === undefined) return "";
  if (typeof text === "string") return text;
  if (typeof text === "number" || typeof text === "boolean") return String(text);
  if (
    typeof text === "object" &&
    text !== null &&
    "spoken" in text &&
    "unspoken" in text
  ) {
    const { spoken, unspoken } = text as { spoken: string; unspoken: string };
    return `${spoken}${unspoken}`;
  }
  return "";
};

const renderMessageText = (message: ConversationMessage): string =>
  message.parts.map(renderPartText).join("");

const stripAssistantTurnText = (text: string, assistantTurns: AssistantTurn[]) =>
  assistantTurns
    .reduce((current, turn) => current.replace(turn.text, ""), text)
    .trim();

const isUserOrAssistant = (m: ConversationMessage) =>
  m.role === "user" || m.role === "assistant";

const findLatestUserMessage = (messages: ConversationMessage[]) => {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") return messages[i];
  }
  return undefined;
};

const findLatestUnanchoredUser = (
  messages: ConversationMessage[],
  anchors: { has: (key: string) => boolean }
) => {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role !== "user") continue;
    return anchors.has(messages[i].createdAt) ? undefined : messages[i];
  }
  return undefined;
};

const normalizeTranscript = (text?: string | null) => (text ?? "").trim().replace(/\s+/g, " ");

const findUserMessageByTranscript = (
  messages: ConversationMessage[],
  transcript: string | null | undefined,
  finalizedCreatedAts: { has: (key: string) => boolean }
) => {
  const normalizedTranscript = normalizeTranscript(transcript);
  if (!normalizedTranscript) return undefined;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user") continue;
    if (message.final || finalizedCreatedAts.has(message.createdAt)) continue;
    if (normalizeTranscript(renderMessageText(message)) === normalizedTranscript) return message;
  }
  return undefined;
};

function mediaKindFromFile(file: File): MediaKind | null {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("audio/")) return "audio";
  if (file.type.startsWith("video/")) return "video";
  return null;
}

function attachmentNameFromMessage(message: Record<string, unknown>) {
  const attachment = message.attachment;
  if (!isRecord(attachment)) return "";
  return stringField(attachment, "name");
}

function AgentTaskCard({ task }: Readonly<{ task: AgentTask }>) {
  const status = task.status || task.stage || "running";
  const statusTone = status === "done" ? "done" : status === "failed" || status === "error" ? "failed" : "running";
  return (
    <li className={`agent-task-card agent-task-card-${statusTone}`}>
      <details open={task.status !== "done"}>
        <summary>
          <span className={`agent-task-indicator agent-task-indicator-${statusTone}`} aria-label={`Task ${statusTone}`} />
          <span className="agent-task-title">Agent task</span>
          <span className="agent-task-agent">{task.agent || "agent"}</span>
          <span className="agent-task-status">{status}</span>
        </summary>
        <div className="agent-task-body">
          {task.attachmentName && <p><strong>Attachment:</strong> {task.attachmentName}</p>}
          {task.query && <p><strong>Query:</strong> {task.query}</p>}
          {task.stage && <p><strong>Stage:</strong> {task.stage}</p>}
          {task.detail && <p>{task.detail}</p>}
          {task.reasoning && (
            <div className="agent-task-stream">
              <strong>Reasoning</strong>
              <pre>{task.reasoning}</pre>
            </div>
          )}
          {task.response && (
            <div className="agent-task-stream">
              <strong>Response</strong>
              <pre>{task.response}</pre>
            </div>
          )}
        </div>
      </details>
    </li>
  );
}

function AttachmentPreview({ attachment }: Readonly<{ attachment: LocalAttachment }>) {
  return (
    <li className={`attachment-preview attachment-preview-${attachment.status}`}>
      <div className="attachment-preview-media">
        {attachment.kind === "image" && <img src={attachment.previewUrl} alt={attachment.name} />}
        {attachment.kind === "audio" && <audio src={attachment.previewUrl} controls />}
        {attachment.kind === "video" && <video src={attachment.previewUrl} controls />}
      </div>
      <div className="attachment-preview-meta">
        <strong>{attachment.name}</strong>
        {attachment.status === "uploading" && <small>Uploading...</small>}
      </div>
      {attachment.error && <small>{attachment.error}</small>}
    </li>
  );
}

function AttachMediaButton({ onClick }: Readonly<{ onClick: () => void }>) {
  return (
    <li className="attachment-upload-row">
      <button className="btn-icon attachment-icon-button" type="button" onClick={onClick} title="Attach media" aria-label="Attach media">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M7.5 12.5 13 7a3.2 3.2 0 0 1 4.5 4.5l-7.1 7.1a4.6 4.6 0 0 1-6.5-6.5l7.4-7.4" />
          <path d="m8.7 15.3 7.1-7.1" />
        </svg>
      </button>
    </li>
  );
}

export function ConversationPanel() {
  const { currentSessionId, selectedExample, setCurrentSessionId } = useApp();
  const { messages } = usePipecatConversation();
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [attachments, setAttachments] = useState<LocalAttachment[]>([]);
  const attachmentsRef = useRef<LocalAttachment[]>([]);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  // Workaround for the lack of an explicit assistant-turn boundary in
  // @pipecat-ai/client-react: assistant bubbles are finalized ~2.5s after
  // BotStoppedSpeaking, so analyzer follow-ups are surfaced as standalone
  // turns recorded here. Server side mirrors this with
  // ``_ANALYZER_FOLLOWUP_TURN_DELAY_SECS`` in
  // ``examples.omni_assistant_subagents.subagents.transport``.
  const [assistantTurns, setAssistantTurns] = useState<AssistantTurn[]>([]);
  const canUploadAttachments = selectedExample?.capabilities?.includes("attachments") ?? false;
  const [currentUserTurnActive, setCurrentUserTurnActive] = useState(false);
  const [userTurnAnchors, setUserTurnAnchors] =
    useState<Map<string, string>>(new Map());
  const userTurnAnchorsRef = useRef(userTurnAnchors);
  useEffect(() => {
    userTurnAnchorsRef.current = userTurnAnchors;
  }, [userTurnAnchors]);
  // Holds a finalize signal that arrived before its turn's user bubble exists
  // (the Omni model emits the user transcript late). Applied to the bubble once
  // it appears.
  const pendingUserAnchorRef = useRef<string | null>(null);
  const anchorUserTurn = useCallback((createdAt: string, anchorISO: string) => {
    setUserTurnAnchors((prev) => {
      if (prev.has(createdAt)) return prev;
      const next = new Map(prev);
      next.set(createdAt, anchorISO);
      return next;
    });
  }, []);

  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  const resetConversationExtras = useCallback(() => {
    setCurrentUserTurnActive(false);
    setUserTurnAnchors(new Map());
    pendingUserAnchorRef.current = null;
    setAttachments((prev) => {
      prev.forEach((attachment) => URL.revokeObjectURL(attachment.previewUrl));
      return [];
    });
    setAgentTasks([]);
    setAssistantTurns([]);
  }, []);

  const visibleMessages = useMemo(
    () => filterEmptyMessages(messages).filter(isUserOrAssistant),
    [messages]
  );

  const visibleMessagesRef = useRef<ConversationMessage[]>(visibleMessages);
  useEffect(() => {
    visibleMessagesRef.current = visibleMessages;
  }, [visibleMessages]);

  useRTVIClientEvent(
    RTVIEvent.UserStartedSpeaking,
    useCallback(() => {
      setCurrentUserTurnActive(true);
      pendingUserAnchorRef.current = null;
    }, [])
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((message: unknown) => {
      if (!isRecord(message)) return;
      const type = stringField(message, "type");

      if (type === "agent-task-update") {
        const taskId = stringField(message, "task_id");
        if (!taskId) return;
        const now = new Date().toISOString();
        const spokenResponse = stringField(message, "spoken_response");
        setAgentTasks((prev) => {
          const previous = prev.find((task) => task.id === taskId);
          const next: AgentTask = {
            id: taskId,
            agent: stringField(message, "agent") || previous?.agent || "agent",
            status: stringField(message, "status") || previous?.status || "running",
            stage: stringField(message, "stage") || previous?.stage || "",
            detail: stringField(message, "detail") || previous?.detail || "",
            query: stringField(message, "query") || previous?.query || "",
            reasoning:
              stringField(message, "reasoning") ||
              `${previous?.reasoning || ""}${stringField(message, "reasoning_delta")}`,
            response:
              stringField(message, "response") ||
              `${previous?.response || ""}${stringField(message, "response_delta")}`,
            attachmentName: attachmentNameFromMessage(message) || previous?.attachmentName || "",
            createdAt: previous?.createdAt || now,
            updatedAt: now,
          };
          return [...prev.filter((task) => task.id !== taskId), next].slice(-20);
        });
        if (stringField(message, "status") === "done" && spokenResponse) {
          setAssistantTurns((prev) => [
            ...prev.filter((turn) => turn.id !== taskId),
            { id: taskId, text: spokenResponse, createdAt: now },
          ].slice(-20));
        }
        return;
      }

      if (type !== "user-turn-finalized") return;
      setCurrentUserTurnActive(false);
      const anchorISO = new Date().toISOString();
      const anchors = userTurnAnchorsRef.current;
      const target =
        findUserMessageByTranscript(visibleMessagesRef.current, stringField(message, "transcript"), anchors) ??
        findLatestUnanchoredUser(visibleMessagesRef.current, anchors);
      if (target) anchorUserTurn(target.createdAt, anchorISO);
      else pendingUserAnchorRef.current = anchorISO;
    }, [anchorUserTurn])
  );

  useRTVIClientEvent(
    RTVIEvent.Disconnected,
    useCallback(() => {
      resetConversationExtras();
      setCurrentSessionId("");
    }, [resetConversationExtras, setCurrentSessionId])
  );

  useEffect(() => () => {
    attachmentsRef.current.forEach((attachment) => URL.revokeObjectURL(attachment.previewUrl));
  }, []);

  useEffect(() => {
    if (!pendingUserAnchorRef.current) return;
    const target = findLatestUnanchoredUser(visibleMessages, userTurnAnchors);
    if (!target) return;
    const anchorISO = pendingUserAnchorRef.current;
    pendingUserAnchorRef.current = null;
    anchorUserTurn(target.createdAt, anchorISO);
  }, [visibleMessages, userTurnAnchors, anchorUserTurn]);

  const latestUserCreatedAt = useMemo(
    () => findLatestUserMessage(visibleMessages)?.createdAt,
    [visibleMessages]
  );

  const computeStreaming = (message: ConversationMessage): boolean => {
    if (message.role !== "user") return !message.final;
    if (message.final) return false;
    if (userTurnAnchors.has(message.createdAt)) return false;
    const isLatestUser = message.createdAt === latestUserCreatedAt;
    return isLatestUser && currentUserTurnActive;
  };

  const handleAttachmentSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !currentSessionId || !canUploadAttachments) return;
    const kind = mediaKindFromFile(file);
    if (!kind) return;

    const localId = crypto.randomUUID();
    const anchorCreatedAt = visibleMessagesRef.current.at(-1)?.createdAt ?? "";
    const createdAt = new Date().toISOString();
    const previewUrl = URL.createObjectURL(file);
    setAttachments((prev) => [
      ...prev,
      {
        id: localId,
        kind,
        name: file.name,
        status: "uploading",
        createdAt,
        anchorCreatedAt,
        previewUrl,
      },
    ]);
    try {
      const uploaded = await uploadAttachment(currentSessionId, file, kind);
      setAttachments((prev) =>
        prev.map((attachment) =>
          attachment.id === localId
            ? { ...attachment, id: String(uploaded.id || localId), status: "uploaded" }
            : attachment
        )
      );
    } catch (err) {
      setAttachments((prev) =>
        prev.map((attachment) =>
          attachment.id === localId
            ? { ...attachment, status: "failed", error: err instanceof Error ? err.message : "Upload failed" }
            : attachment
        )
      );
    }
  };

  const conversationItems = useMemo(() => {
    const messageItems = visibleMessages.flatMap((message, idx) => {
      const text =
        message.role === "assistant"
          ? stripAssistantTurnText(renderMessageText(message), assistantTurns)
          : renderMessageText(message);
      if (!text) return [];
      return [{
        type: "message" as const,
        id: `${message.createdAt}-${idx}`,
        createdAt: message.createdAt,
        message,
        text,
        index: idx,
      }];
    });
    const taskItems = agentTasks.map((task) => ({
      type: "task" as const,
      id: task.id,
      createdAt: task.createdAt,
      task,
      index: 101,
    }));
    const assistantTurnItems = assistantTurns.map((turn) => ({
      type: "assistant-turn" as const,
      id: turn.id,
      createdAt: turn.createdAt,
      turn,
      index: 102,
    }));
    const attachmentItems = attachments.map((attachment) => ({
      type: "attachment" as const,
      id: attachment.id,
      createdAt: attachment.createdAt,
      attachment,
      index: 103,
    }));
    const orderTimeMs = (createdAt: string) => {
      const created = new Date(createdAt).getTime();
      const anchor = userTurnAnchors.get(createdAt);
      return anchor ? Math.min(created, new Date(anchor).getTime()) : created;
    };
    const sorted = [...messageItems, ...taskItems, ...assistantTurnItems, ...attachmentItems].sort((a, b) => {
      const timeDelta = orderTimeMs(a.createdAt) - orderTimeMs(b.createdAt);
      return timeDelta || a.index - b.index;
    });

    // TODO: Remove once @pipecat-ai/client-react stops finalizing each assistant
    // sentence as its own message. Merge only assistant bubbles that are adjacent
    // in the display stream, so interleaved tasks/turns/attachments stay between
    // the sentences they landed in.
    const items: typeof sorted = [];
    const mergedChunks: string[][] = [];
    for (const item of sorted) {
      const previous = items.at(-1);
      if (
        item.type === "message" && item.message.role === "assistant" &&
        previous?.type === "message" && previous.message.role === "assistant"
      ) {
        mergedChunks[items.length - 1].push(item.text);
        previous.message = { ...previous.message, final: item.message.final };
        continue;
      }
      items.push(item);
      mergedChunks.push(
        item.type === "message" && item.message.role === "assistant" ? [item.text] : []
      );
    }
    return items.map((item, idx) =>
      item.type === "message" && mergedChunks[idx].length > 1
        ? { ...item, text: mergedChunks[idx].join(" ") }
        : item
    );
  }, [agentTasks, assistantTurns, attachments, visibleMessages, userTurnAnchors]);

  const showAttachmentControl = Boolean(currentSessionId) && canUploadAttachments && visibleMessages.length > 0;
  const bottomAnchorRef = useStickToBottom(conversationItems);

  return (
    <div className="p-4">
      <ul className="d-flex flex-col gap-2" style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {conversationItems.map((item) => {
          if (item.type === "task") return <AgentTaskCard key={item.id} task={item.task} />;
          if (item.type === "attachment") return <AttachmentPreview key={item.id} attachment={item.attachment} />;
          if (item.type === "assistant-turn") {
            return (
              <TranscriptMessage
                key={`assistant-turn-${item.id}`}
                role="bot"
                text={item.turn.text}
                timestamp={item.turn.createdAt}
                streaming={false}
              />
            );
          }

          const msg = item.message;
          return (
            <Fragment key={item.id}>
              <TranscriptMessage
                role={msg.role === "assistant" ? "bot" : "user"}
                text={item.text}
                timestamp={msg.createdAt}
                streaming={computeStreaming(msg)}
              />
            </Fragment>
          );
        })}
        {showAttachmentControl && <AttachMediaButton onClick={() => uploadInputRef.current?.click()} />}
      </ul>
      <div ref={bottomAnchorRef} className="conversation-bottom-spacer" aria-hidden="true" />
      <input
        ref={uploadInputRef}
        type="file"
        accept="image/*,audio/*,video/*"
        hidden
        onChange={handleAttachmentSelected}
      />
    </div>
  );
}
