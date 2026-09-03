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
  anchorCreatedAt: string;
};

type AssistantTurn = {
  id: string;
  text: string;
  createdAt: string;
  anchorCreatedAt: string;
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

const isUserOrAssistant = (m: ConversationMessage) =>
  m.role === "user" || m.role === "assistant";

const normalizeTranscript = (text?: string | null) => (text ?? "").trim().replace(/\s+/g, " ");

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

const START_ANCHOR = "__start__";

type ExtraItem =
  | { kind: "task"; id: string; anchor: string; createdAt: string; sortKey: number; task: AgentTask }
  | { kind: "assistant-turn"; id: string; anchor: string; createdAt: string; sortKey: number; turn: AssistantTurn }
  | { kind: "attachment"; id: string; anchor: string; createdAt: string; sortKey: number; attachment: LocalAttachment };

export function ConversationPanel() {
  const { currentSessionId, selectedExample, setCurrentSessionId } = useApp();
  const { messages } = usePipecatConversation();
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [attachments, setAttachments] = useState<LocalAttachment[]>([]);
  const attachmentsRef = useRef<LocalAttachment[]>([]);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  const agentTasksRef = useRef<AgentTask[]>([]);
  const [assistantTurns, setAssistantTurns] = useState<AssistantTurn[]>([]);
  const canUploadAttachments = selectedExample?.capabilities?.includes("attachments") ?? false;

  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  useEffect(() => {
    agentTasksRef.current = agentTasks;
  }, [agentTasks]);

  const resetConversationExtras = useCallback(() => {
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
    RTVIEvent.ServerMessage,
    useCallback((message: unknown) => {
      if (!isRecord(message)) return;
      const type = stringField(message, "type");

      if (type === "agent-task-update") {
        const taskId = stringField(message, "task_id");
        if (!taskId) return;
        const now = new Date().toISOString();
        const spokenResponse = stringField(message, "spoken_response");
        const existing = agentTasksRef.current.find((task) => task.id === taskId);
        const anchorCreatedAt =
          existing?.anchorCreatedAt ?? (visibleMessagesRef.current.at(-1)?.createdAt ?? "");
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
            anchorCreatedAt: previous?.anchorCreatedAt || anchorCreatedAt,
          };
          return [...prev.filter((task) => task.id !== taskId), next].slice(-20);
        });
        if (stringField(message, "status") === "done" && spokenResponse) {
          setAssistantTurns((prev) => [
            ...prev.filter((turn) => turn.id !== taskId),
            { id: taskId, text: spokenResponse, createdAt: now, anchorCreatedAt },
          ].slice(-20));
        }
      }
    }, [])
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

  const assistantMessageTexts = useMemo(
    () =>
      new Set(
        visibleMessages
          .filter((m) => m.role === "assistant")
          .map((m) => normalizeTranscript(renderMessageText(m)))
          .filter(Boolean)
      ),
    [visibleMessages]
  );

  const extras = useMemo<ExtraItem[]>(() => {
    const list: ExtraItem[] = [];
    agentTasks.forEach((task) =>
      list.push({ kind: "task", id: task.id, anchor: task.anchorCreatedAt, createdAt: task.createdAt, sortKey: 1, task })
    );
    assistantTurns
      .filter((turn) => !assistantMessageTexts.has(normalizeTranscript(turn.text)))
      .forEach((turn) =>
        list.push({ kind: "assistant-turn", id: turn.id, anchor: turn.anchorCreatedAt, createdAt: turn.createdAt, sortKey: 2, turn })
      );
    attachments.forEach((attachment) =>
      list.push({ kind: "attachment", id: attachment.id, anchor: attachment.anchorCreatedAt, createdAt: attachment.createdAt, sortKey: 3, attachment })
    );
    return list;
  }, [agentTasks, assistantTurns, assistantMessageTexts, attachments]);

  const { startExtras, extrasByAnchor } = useMemo(() => {
    const messageIds = new Set(visibleMessages.map((m) => m.createdAt));
    const lastCreatedAt = visibleMessages.at(-1)?.createdAt ?? "";
    const resolve = (anchor: string) => {
      if (!anchor) return START_ANCHOR;
      if (messageIds.has(anchor)) return anchor;
      return lastCreatedAt || START_ANCHOR;
    };
    const byAnchor = new Map<string, ExtraItem[]>();
    for (const extra of extras) {
      const key = resolve(extra.anchor);
      const bucket = byAnchor.get(key) ?? [];
      bucket.push(extra);
      byAnchor.set(key, bucket);
    }
    for (const bucket of byAnchor.values()) {
      bucket.sort((a, b) => a.sortKey - b.sortKey || a.createdAt.localeCompare(b.createdAt));
    }
    return { startExtras: byAnchor.get(START_ANCHOR) ?? [], extrasByAnchor: byAnchor };
  }, [extras, visibleMessages]);

  const renderExtra = (extra: ExtraItem) => {
    if (extra.kind === "task") return <AgentTaskCard key={extra.id} task={extra.task} />;
    if (extra.kind === "attachment") return <AttachmentPreview key={extra.id} attachment={extra.attachment} />;
    return (
      <TranscriptMessage
        key={`assistant-turn-${extra.id}`}
        role="bot"
        text={extra.turn.text}
        timestamp={extra.turn.createdAt}
        streaming={false}
      />
    );
  };

  const showAttachmentControl = Boolean(currentSessionId) && canUploadAttachments && visibleMessages.length > 0;
  const stickSignal = useMemo(() => ({ messages: visibleMessages, extras }), [visibleMessages, extras]);
  const bottomAnchorRef = useStickToBottom(stickSignal);

  return (
    <div className="p-4">
      <ul className="d-flex flex-col gap-2" style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {startExtras.map(renderExtra)}
        {visibleMessages.map((message, idx) => {
          const text = renderMessageText(message);
          const bucket = extrasByAnchor.get(message.createdAt);
          return (
            <Fragment key={`${message.createdAt}-${idx}`}>
              {text && (
                <TranscriptMessage
                  role={message.role === "assistant" ? "bot" : "user"}
                  text={text}
                  timestamp={message.createdAt}
                  streaming={!message.final}
                />
              )}
              {bucket?.map(renderExtra)}
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
