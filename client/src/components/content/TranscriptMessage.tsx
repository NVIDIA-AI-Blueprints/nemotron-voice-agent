// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TranscriptMessageProps {
  role: "user" | "bot";
  text: string;
  timestamp: string;
  streaming?: boolean;
}

const formatTime = (timestamp: string) => {
  if (!timestamp) return "—";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

export function TranscriptMessage({ role, text, timestamp, streaming }: Readonly<TranscriptMessageProps>) {
  const roleClass = role === "user" ? "message-user" : "message-bot";

  return (
    <li className={`transcript-message ${roleClass} ${streaming ? "message-streaming" : ""}`}>
      <span className="message-timestamp">{formatTime(timestamp)}</span>
      <div className="message-content">
        <span className="message-role">{role === "user" ? "You" : "Bot"}:</span>{" "}
        <span>{text}</span>
      </div>
    </li>
  );
}
